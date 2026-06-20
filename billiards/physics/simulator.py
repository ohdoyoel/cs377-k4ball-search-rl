"""Event-driven shot simulator.

`simulate_shot(state, action)` applies a cue impulse, then steps the table
forward in (event ∨ dt_max)-bounded substeps until every ball is at rest
(or t_max). Cushion bounces and ball-ball collisions are resolved
analytically when their TOI lies inside the next dt_max window.

Returns a dict with:
    trajectory:    list[(t, (N,7) ndarray)] — snapshots at each event /
                   dt_max boundary
    events:        list[dict] — typed events {'t', 'type', 'detail'}
    score:         int (0/1)  — Korean 4-ball "made" iff cue ball hits
                   BOTH reds and never the opponent ball
    fouled:        bool       — cue ball touched the opponent ball
    cushion_hits:  int        — number of cue-ball cushion contacts
    duration:      float      — total simulated time in seconds

Event types:
    'cue_hit_red'      cue ball ↔ red ball
    'cue_hit_opp'      cue ball ↔ opponent ball
    'cue_hit_cushion'  cue ball ↔ rail
    'opp_cushion'      opponent ball ↔ rail
    'red_cushion'      red ball ↔ rail
    'red_red'          red ↔ red
    'red_opp'          red ↔ opponent
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .collisions import (
    resolve_ball_ball,
    resolve_ball_cushion,
    toi_ball_ball,
    toi_ball_cushion,
)
from .cue_impact import V_MAX_DEFAULT, apply_cue
from .dynamics import step_free
from .state import BallRole, CueAction, TableState

INF = float("inf")
EVENT_EPS = 1e-9   # don't re-trigger the same contact within this window
BOUNDARY_EPS = 1e-10
MISSED_CUSHION_DEPTH_EPS = 1e-4
BINARY_SEARCH_ITERS = 48


def _ball_kind(idx: int, cue_id: int) -> str:
    if idx == cue_id:
        return "cue"
    if idx == int(BallRole.RED_1) or idx == int(BallRole.RED_2):
        return "red"
    return "opp"


def _pair_event_name(k1: str, k2: str) -> str:
    kinds = tuple(sorted([k1, k2]))
    if kinds == ("cue", "red"):
        return "cue_hit_red"
    if kinds == ("cue", "opp"):
        return "cue_hit_opp"
    if kinds == ("red", "red"):
        return "red_red"
    if kinds == ("opp", "red"):
        return "red_opp"
    return f"{k1}_{k2}"


def _cushion_event_name(kind: str) -> str:
    if kind == "cue":
        return "cue_hit_cushion"
    if kind == "opp":
        return "opp_cushion"
    return "red_cushion"


def _outside_wall(ball, spec, normal: tuple[float, float], eps: float = BOUNDARY_EPS) -> bool:
    r = spec.ball_radius
    if normal == (1.0, 0.0):
        return ball.x < r - eps
    if normal == (-1.0, 0.0):
        return ball.x > spec.width - r + eps
    if normal == (0.0, 1.0):
        return ball.y < r - eps
    if normal == (0.0, -1.0):
        return ball.y > spec.height - r + eps
    return False


def _deepest_wall_violation(
    ball,
    spec,
    min_depth: float = BOUNDARY_EPS,
) -> tuple[float, tuple[float, float]] | None:
    r = spec.ball_radius
    candidates = [
        (r - ball.x, (1.0, 0.0)),
        (ball.x - (spec.width - r), (-1.0, 0.0)),
        (r - ball.y, (0.0, 1.0)),
        (ball.y - (spec.height - r), (0.0, -1.0)),
    ]
    depth, normal = max(candidates, key=lambda item: item[0])
    if depth <= min_depth:
        return None
    return depth, normal


def _snap_ball_to_cushion(ball, normal: tuple[float, float], spec) -> None:
    r = spec.ball_radius
    if normal == (1.0, 0.0):
        ball.x = r
    elif normal == (-1.0, 0.0):
        ball.x = spec.width - r
    elif normal == (0.0, 1.0):
        ball.y = r
    elif normal == (0.0, -1.0):
        ball.y = spec.height - r


def _find_missed_cushion_crossing(
    state: TableState,
    max_dt: float,
) -> tuple[float, int, tuple[float, float]] | None:
    """Find a wall crossing missed by approximate cushion TOI projection.

    Cushion TOI is intentionally cheap and uses a rolling approximation, while
    `step_free` may still be in the slip regime. Strong spin can therefore
    cross a rail inside the current substep even when TOI predicts the hit just
    outside it. This fallback probes the actual free-flight dynamics and uses
    bisection to schedule the missed cushion collision.
    """
    if max_dt <= EVENT_EPS:
        return None

    end_state = state.copy()
    step_free(end_state, max_dt)

    best: tuple[float, int, tuple[float, float]] | None = None
    for i, end_ball in enumerate(end_state.balls):
        violation = _deepest_wall_violation(
            end_ball,
            state.spec,
            min_depth=MISSED_CUSHION_DEPTH_EPS,
        )
        if violation is None:
            continue
        _, normal = violation

        start_ball = state.balls[i]
        if _outside_wall(start_ball, state.spec, normal):
            crossing_t = 0.0
        else:
            low = 0.0
            high = max_dt
            for _ in range(BINARY_SEARCH_ITERS):
                mid = 0.5 * (low + high)
                probe = state.copy()
                step_free(probe, mid)
                if _outside_wall(probe.balls[i], state.spec, normal):
                    high = mid
                else:
                    low = mid
            crossing_t = high

        if best is None or crossing_t < best[0]:
            best = (crossing_t, i, normal)

    return best


def simulate_shot(
    state: TableState,
    action: CueAction,
    t_max: float = 30.0,
    dt_max: float = 0.1,
    v_max: float = V_MAX_DEFAULT,
) -> dict[str, Any]:
    """Run a single shot to completion (or t_max). Mutates `state`.

    Algorithm:
        apply_cue(state, action)
        while not all_at_rest and t < t_max:
            t_evt = min(toi over all balls/pairs, dt_max)
            step_free(state, t_evt)
            if t_evt < dt_max - eps: resolve the responsible collision
            record snapshot

    The smallest TOI is found by enumerating all balls (cushion) and all
    unordered pairs (ball-ball). N=4 keeps this cheap.
    """
    apply_cue(state, action, v_max=v_max)

    spec = state.spec
    n = len(state.balls)
    cue_id = state.cue_id

    trajectory: list[tuple[float, np.ndarray]] = [(state.t, state.to_array())]
    events: list[dict[str, Any]] = []
    cue_reds_hit: set[int] = set()
    cue_opp_touch = False
    cushion_hits = 0
    t0 = state.t

    while state.t - t0 < t_max and not state.all_at_rest():
        # ----- find earliest event -----
        best_t = dt_max
        best_kind = "tick"  # 'cushion' | 'pair' | 'tick'
        best_payload: Any = None

        for i, b in enumerate(state.balls):
            res = toi_ball_cushion(b, spec)
            if res is None:
                continue
            t_c, normal = res
            if t_c < best_t - EVENT_EPS:
                best_t = t_c
                best_kind = "cushion"
                best_payload = (i, normal)

        for i in range(n):
            for j in range(i + 1, n):
                t_p = toi_ball_ball(state.balls[i], state.balls[j], spec)
                if t_p < best_t - EVENT_EPS:
                    best_t = t_p
                    best_kind = "pair"
                    best_payload = (i, j)

        # Don't overshoot remaining time budget
        remaining = t_max - (state.t - t0)
        step_budget = min(best_t, remaining)

        missed_cushion = _find_missed_cushion_crossing(state, step_budget)
        if missed_cushion is not None:
            missed_t, missed_i, missed_normal = missed_cushion
            if best_kind == "tick" or missed_t < step_budget - EVENT_EPS:
                best_t = missed_t
                best_kind = "cushion"
                best_payload = (missed_i, missed_normal, True)
                step_budget = min(best_t, remaining)

        step = max(EVENT_EPS, step_budget)

        step_free(state, step)

        if best_kind == "cushion" and step >= best_t - EVENT_EPS and best_payload is not None:
            i, normal, *payload_flags = best_payload
            from_missed_cushion = bool(payload_flags and payload_flags[0])
            kind = _ball_kind(i, cue_id)
            large_violation = (
                _deepest_wall_violation(
                    state.balls[i],
                    spec,
                    min_depth=MISSED_CUSHION_DEPTH_EPS,
                )
                is not None
            )
            if from_missed_cushion or large_violation:
                _snap_ball_to_cushion(state.balls[i], normal, spec)
            resolve_ball_cushion(state.balls[i], normal, spec)
            ev = {
                "t": state.t,
                "type": _cushion_event_name(kind),
                "detail": {"ball": i, "normal": normal},
            }
            events.append(ev)
            if kind == "cue":
                cushion_hits += 1
        elif best_kind == "pair" and step >= best_t - EVENT_EPS and best_payload is not None:
            i, j = best_payload
            resolve_ball_ball(state.balls[i], state.balls[j], spec)
            k1 = _ball_kind(i, cue_id)
            k2 = _ball_kind(j, cue_id)
            etype = _pair_event_name(k1, k2)
            events.append({
                "t": state.t,
                "type": etype,
                "detail": {"balls": (i, j)},
            })
            if etype == "cue_hit_red":
                red_idx = j if k2 == "red" else i
                cue_reds_hit.add(red_idx)
            elif etype == "cue_hit_opp":
                cue_opp_touch = True

        trajectory.append((state.t, state.to_array()))

    # Clean residuals so the snapshot looks fully at rest.
    for b in state.balls:
        if b.is_at_rest(spec):
            b.vx = b.vy = 0.0
            b.wx = b.wy = 0.0
            b.wz = 0.0

    fouled = bool(cue_opp_touch)
    score = 1 if (len(cue_reds_hit) >= 2 and not cue_opp_touch) else 0

    return {
        "trajectory": trajectory,
        "events": events,
        "score": score,
        "fouled": fouled,
        "cushion_hits": cushion_hits,
        "duration": state.t - t0,
    }
