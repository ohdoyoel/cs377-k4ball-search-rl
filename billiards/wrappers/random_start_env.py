"""Random-start wrapper for the multi-shot inning env.

On every ``reset()``, replace the canonical 4-ball layout with a random
valid layout: each ball drawn uniformly inside the play area (with a
margin from the cushions), all velocities and spins set to zero, and
pairwise center-distances enforced to ``2R + safety_margin``. We use
rejection sampling and fall back to the canonical layout if the budget
is exhausted.

The wrapped env is expected to expose ``unwrapped._state`` (a
``TableState``) and ``unwrapped._cumulative_score`` / friends — i.e. it's
a ``Billiards4BallInningEnv``. We do not touch the env's observation
space, action space, episode termination, or info dict; we only swap the
state at reset time and re-emit the standard flattened observation.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from billiards.physics.state import (
    BallRole,
    BallState,
    TableSpec,
    TableState,
)


class RandomStartInningEnv(gym.Wrapper):
    """Wrap an inning env so each reset randomizes ball positions.

    Constraints enforced per layout:
        - x ∈ Uniform(R + margin, W - R - margin)
        - y ∈ Uniform(R + margin, H - R - margin)
        - vx = vy = wx = wy = wz = 0
        - For all i ≠ j: ||p_i - p_j|| ≥ 2R + safety_margin

    Parameters
    ----------
    env : gym.Env
        Underlying inning env (``Billiards4BallInningEnv``).
    margin : float
        Cushion margin in meters (default 0.005 m).
    safety_margin : float
        Extra ball-pair clearance in meters (default 0.002 m).
    max_retries : int
        Rejection-sampling budget per reset (default 100). On exhaustion
        we fall back to the canonical 4-ball layout.
    """

    def __init__(
        self,
        env: gym.Env,
        margin: float = 0.005,
        safety_margin: float = 0.002,
        max_retries: int = 100,
    ) -> None:
        super().__init__(env)
        self._margin = float(margin)
        self._safety = float(safety_margin)
        self._max_retries = int(max_retries)
        self._rng: np.random.Generator = np.random.default_rng(0)

    # ------------------------------------------------------------------ rng

    def _seed_rng(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ sampling

    def _spec(self) -> TableSpec:
        # The unwrapped env stores the spec as ``_spec``.
        return self.env.unwrapped._spec  # type: ignore[attr-defined]

    def _sample_layout(self, spec: TableSpec) -> list[BallState] | None:
        R = float(spec.ball_radius)
        W = float(spec.width)
        H = float(spec.height)
        lo_x = R + self._margin
        hi_x = W - R - self._margin
        lo_y = R + self._margin
        hi_y = H - R - self._margin
        if hi_x <= lo_x or hi_y <= lo_y:
            return None
        min_d = 2.0 * R + self._safety
        min_d2 = min_d * min_d

        n_balls = 4
        positions: list[tuple[float, float]] = []
        # Per-ball rejection: try up to max_retries placements for each
        # ball, but reset the whole layout if any single ball can't be
        # placed within its budget.
        for _ in range(n_balls):
            placed = False
            for _trial in range(self._max_retries):
                x = float(self._rng.uniform(lo_x, hi_x))
                y = float(self._rng.uniform(lo_y, hi_y))
                ok = True
                for px, py in positions:
                    dx = x - px
                    dy = y - py
                    if dx * dx + dy * dy < min_d2:
                        ok = False
                        break
                if ok:
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                return None
        return [BallState(x=px, y=py) for px, py in positions]

    # ------------------------------------------------------------------ Gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Always advance our own RNG when a seed is supplied so the
        # layout is reproducible per-seed.
        if seed is not None:
            self._seed_rng(seed)

        # Reset the underlying env so it builds the canonical layout, sets
        # cumulative_score=0, etc. We then overwrite the state.
        _, info = self.env.reset(seed=seed, options=options)

        spec = self._spec()
        layout = self._sample_layout(spec)
        used_fallback = False
        if layout is None:
            # Exhausted budget — fall back to canonical (already set by
            # the underlying reset).
            used_fallback = True
            inner = self.env.unwrapped
            cue_id = int(getattr(inner, "_cue_id", BallRole.CUE_WHITE))
            new_state = TableState.initial_4ball(cue_id=cue_id, spec=spec)
        else:
            inner = self.env.unwrapped
            cue_id = int(getattr(inner, "_cue_id", BallRole.CUE_WHITE))
            new_state = TableState(balls=layout, cue_id=cue_id, spec=spec, t=0.0)

        # Inject and reset bookkeeping.
        inner = self.env.unwrapped
        inner._state = new_state  # type: ignore[attr-defined]
        inner._cumulative_score = 0  # type: ignore[attr-defined]
        inner._shot_index = 0  # type: ignore[attr-defined]
        inner._cumulative_t = 0.0  # type: ignore[attr-defined]
        inner._shot_trajectories = []  # type: ignore[attr-defined]
        inner._shot_offsets = []  # type: ignore[attr-defined]
        inner._inning_log_records = []  # type: ignore[attr-defined]
        inner._last_info = None  # type: ignore[attr-defined]

        # Use the inner env's _obs() so extra_features is respected.
        obs = inner._obs()  # type: ignore[attr-defined]
        info = dict(info) if info is not None else {}
        info["random_start"] = True
        info["random_start_fallback"] = bool(used_fallback)
        return obs, info
