import math

import numpy as np

from billiards.physics import CueAction, TableSpec, TableState, simulate_shot


def _assert_trajectory_inside_table(result, spec: TableSpec, tol: float = 1e-7) -> None:
    r = spec.ball_radius
    for _, snap in result["trajectory"]:
        xs = snap[:, 0]
        ys = snap[:, 1]
        assert xs.min() >= r - tol
        assert xs.max() <= spec.width - r + tol
        assert ys.min() >= r - tol
        assert ys.max() <= spec.height - r + tol


def test_strong_slip_cushion_crossing_is_resolved() -> None:
    spec = TableSpec()
    state = TableState.initial_4ball(spec=spec)
    state.balls[state.cue_id].x = 1.086808
    state.balls[state.cue_id].y = 0.303237

    action = CueAction(
        theta=4.959742361369275,
        power=0.7304272409903989,
        a=-0.059305820896470136,
        b=0.6267404843347654,
    )
    assert action.is_valid()

    result = simulate_shot(state, action, t_max=12.0)

    _assert_trajectory_inside_table(result, spec)


def _random_state(rng: np.random.Generator, spec: TableSpec) -> TableState:
    r = spec.ball_radius
    positions: list[tuple[float, float]] = []
    while len(positions) < 4:
        x = float(rng.uniform(r + 0.02, spec.width - r - 0.02))
        y = float(rng.uniform(r + 0.02, spec.height - r - 0.02))
        if all(math.hypot(x - px, y - py) > 2.0 * r + 0.01 for px, py in positions):
            positions.append((x, y))

    state = TableState.initial_4ball(spec=spec)
    for ball, (x, y) in zip(state.balls, positions):
        ball.x = x
        ball.y = y
    return state


def _random_valid_action(rng: np.random.Generator) -> CueAction:
    tip_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    tip_radius = float(math.sqrt(rng.uniform(0.0, 0.98 * 0.98)))
    return CueAction(
        theta=float(rng.uniform(0.0, 2.0 * math.pi)),
        power=float(rng.uniform(0.55, 1.0)),
        a=tip_radius * math.cos(tip_angle),
        b=tip_radius * math.sin(tip_angle),
    )


def test_random_strong_shots_stay_inside_table() -> None:
    spec = TableSpec()
    rng = np.random.default_rng(123)

    for _ in range(80):
        state = _random_state(rng, spec)
        action = _random_valid_action(rng)
        assert action.is_valid()

        result = simulate_shot(state, action, t_max=6.0)

        _assert_trajectory_inside_table(result, spec, tol=2e-4)
