"""Curriculum-start wrapper for the multi-shot inning env.

Like ``RandomStartInningEnv`` but the random layout is biased to be
*easy* at low difficulty (cue close to a red, reds close to each other,
opponent kept far away) and converges to fully-random at difficulty 1.

Difficulty is a scalar in [0, 1] stored on the wrapper. The training
loop is expected to update it (e.g. by calling ``set_difficulty``
periodically from a callback). On ``reset()`` the wrapper samples a
layout under the current difficulty's constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from billiards.physics.state import BallRole, BallState, TableSpec, TableState
from billiards.wrappers.random_start_env import RandomStartInningEnv


class CurriculumStartInningEnv(RandomStartInningEnv):
    """Random-start wrapper with a single difficulty knob.

    Constraints at difficulty ``d`` ∈ [0, 1]:
        - red2 within ``red_pair_max(d)`` of red1
        - cue within ``cue_red_max(d)`` of red1
        - opponent at least ``opp_min(d)`` away from cue

    All thresholds linearly interpolate to ``+∞`` at d=1, recovering the
    plain random layout.
    """

    def __init__(
        self,
        env,
        difficulty: float = 0.0,
        red_pair_max_easy: float = 0.30,
        cue_red_max_easy: float = 0.40,
        opp_min_easy: float = 0.70,
        margin: float = 0.005,
        safety_margin: float = 0.002,
        max_retries: int = 200,
    ) -> None:
        super().__init__(env, margin=margin, safety_margin=safety_margin,
                         max_retries=max_retries)
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self._red_pair_easy = float(red_pair_max_easy)
        self._cue_red_easy = float(cue_red_max_easy)
        self._opp_min_easy = float(opp_min_easy)

    # ----------------------------------------------------------------- knob

    def set_difficulty(self, d: float) -> None:
        self._difficulty = float(np.clip(d, 0.0, 1.0))

    def get_difficulty(self) -> float:
        return float(self._difficulty)

    # ----------------------------------------------------------------- sampling

    def _difficulty_caps(self, spec: TableSpec) -> tuple[float, float, float]:
        """Return (red_pair_max, cue_red_max, opp_min) under current d.

        Linearly interpolate easy bound → table diagonal at d=1.
        opp_min interpolates downward to 0 at d=1.
        """
        d = self._difficulty
        diag = float(np.hypot(spec.width, spec.height))
        red_pair = self._red_pair_easy + d * (diag - self._red_pair_easy)
        cue_red = self._cue_red_easy + d * (diag - self._cue_red_easy)
        opp_min = (1.0 - d) * self._opp_min_easy
        return red_pair, cue_red, opp_min

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

        red_pair_max, cue_red_max, opp_min = self._difficulty_caps(spec)

        # Ball roles: index 0 = CUE_WHITE, 1 = CUE_YELLOW (opponent),
        # 2 = RED_1, 3 = RED_2. We sample in order red1, red2, cue, opp.
        # Re-arrange at the end into role-indexed positions.
        for _attempt in range(self._max_retries):
            # red1: anywhere
            r1 = (
                float(self._rng.uniform(lo_x, hi_x)),
                float(self._rng.uniform(lo_y, hi_y)),
            )

            # red2: within red_pair_max of red1, not overlapping
            r2 = None
            for _ in range(self._max_retries):
                x = float(self._rng.uniform(lo_x, hi_x))
                y = float(self._rng.uniform(lo_y, hi_y))
                dx, dy = x - r1[0], y - r1[1]
                if dx * dx + dy * dy < min_d2:
                    continue
                if np.hypot(dx, dy) > red_pair_max:
                    continue
                r2 = (x, y)
                break
            if r2 is None:
                continue

            # cue: within cue_red_max of red1 (nearest), not overlapping
            cue = None
            for _ in range(self._max_retries):
                x = float(self._rng.uniform(lo_x, hi_x))
                y = float(self._rng.uniform(lo_y, hi_y))
                ok = True
                for px, py in (r1, r2):
                    dx, dy = x - px, y - py
                    if dx * dx + dy * dy < min_d2:
                        ok = False
                        break
                if not ok:
                    continue
                dx, dy = x - r1[0], y - r1[1]
                if np.hypot(dx, dy) > cue_red_max:
                    continue
                cue = (x, y)
                break
            if cue is None:
                continue

            # opponent: at least opp_min from cue, not overlapping any
            opp = None
            for _ in range(self._max_retries):
                x = float(self._rng.uniform(lo_x, hi_x))
                y = float(self._rng.uniform(lo_y, hi_y))
                ok = True
                for px, py in (r1, r2, cue):
                    dx, dy = x - px, y - py
                    if dx * dx + dy * dy < min_d2:
                        ok = False
                        break
                if not ok:
                    continue
                if opp_min > 0:
                    dx, dy = x - cue[0], y - cue[1]
                    if np.hypot(dx, dy) < opp_min:
                        continue
                opp = (x, y)
                break
            if opp is None:
                continue

            # Compose role-indexed positions (CUE_WHITE, CUE_YELLOW, RED_1, RED_2)
            positions_by_role = [None, None, None, None]
            positions_by_role[int(BallRole.CUE_WHITE)] = cue
            positions_by_role[int(BallRole.CUE_YELLOW)] = opp
            positions_by_role[int(BallRole.RED_1)] = r1
            positions_by_role[int(BallRole.RED_2)] = r2
            return [BallState(x=px, y=py) for (px, py) in positions_by_role]

        return None
