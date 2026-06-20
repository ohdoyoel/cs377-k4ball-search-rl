r"""Action-stage curriculum wrapper.

Inner env exposes a 5D continuous action $(\sin\theta, \cos\theta, p, a, b)$
(requires the inner env to be constructed with ``angle_sincos=True``).
This wrapper accepts the same 5D action from the policy, but before
forwarding it to the inner env it overrides some components with fixed
defaults, simulating a smaller action space:

  stage=1 -> only $(\sin\theta, \cos\theta)$ from the policy.
             p=1, a=0, b=0 are forced. Effective 2D direction-only.
  stage=2 -> $(\sin\theta, \cos\theta, p)$ from the policy.
             a=0, b=0 are forced. Effective 3D (direction + power).
  stage=3 -> all five dims used (no override). Full 5D.

Rationale: the policy still emits 5D so a single SAC model can be saved
and reloaded across stages without changing its actor head. Once stage
is bumped, the previously-frozen action dims are simply released back to
the policy.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ActionStageWrapper(gym.Wrapper):
    """Override later action dims by stage; expects inner env to use
    ``angle_sincos=True`` (5D actions ordered as sin, cos, p, a, b)."""

    DEFAULT_P = 1.0
    DEFAULT_A = 0.0
    DEFAULT_B = 0.0

    def __init__(self, env: gym.Env, stage: int = 1) -> None:
        super().__init__(env)
        if env.action_space.shape != (5,):
            raise ValueError(
                f"ActionStageWrapper requires 5D action space, got {env.action_space.shape}"
            )
        self._stage = int(stage)
        if self._stage not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2, or 3 (got {self._stage})")

    @property
    def stage(self) -> int:
        return self._stage

    def set_stage(self, stage: int) -> None:
        if stage not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2, or 3")
        self._stage = int(stage)

    def _apply_mask(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if a.shape != (5,):
            raise ValueError(f"expected 5D action, got shape {a.shape}")
        if self._stage == 1:
            a[2] = self.DEFAULT_P  # p
            a[3] = self.DEFAULT_A  # a
            a[4] = self.DEFAULT_B  # b
        elif self._stage == 2:
            a[3] = self.DEFAULT_A
            a[4] = self.DEFAULT_B
        # stage 3: nothing forced
        return a

    def step(self, action):
        masked = self._apply_mask(action)
        obs, r, term, trunc, info = self.env.step(masked)
        info = dict(info)
        info["action_stage"] = self._stage
        return obs, r, term, trunc, info
