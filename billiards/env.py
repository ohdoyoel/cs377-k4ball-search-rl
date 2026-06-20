"""Gymnasium environment wrapping the 4-ball carom simulator.

Single-shot episodes: each `step` applies one cue impulse, runs the
simulation to rest (or t_max), then terminates. Reward is the integer
4-ball score (1 if both reds were hit and the opponent ball was not, else 0).

Observation (28-dim float32):
    flatten of TableState.to_array() — 4 balls × 7 floats.

Action (4-dim float32):
    [theta, power, a, b]
        theta ∈ [0, 2π)        cue direction
        power ∈ [0, 1]         shot strength
        (a, b) ∈ unit square,  contact offset; (a²+b²) is projected to ≤ 1
                               via radial clip (a,b) ← (a,b)/r when r > 1.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np

from .physics import (
    BallRole,
    CueAction,
    TableSpec,
    TableState,
    simulate_shot,
)

OBS_DIM = 4 * 7  # 28


def _spec_to_dict(spec: TableSpec) -> dict[str, float]:
    """Serialize TableSpec as plain dict (renderer-friendly)."""
    return asdict(spec)


def _project_action(a: np.ndarray) -> CueAction:
    """Map a raw action vector into a valid CueAction.

    - theta: wrapped to [0, 2π)
    - power: clipped to [0, 1]
    - (a, b): radial-projected into the unit disk (only shrinks vectors
      that already fall outside; tip stays in the same direction).
    """
    theta = float(a[0]) % (2.0 * np.pi)
    power = float(np.clip(a[1], 0.0, 1.0))
    ax = float(np.clip(a[2], -1.0, 1.0))
    ay = float(np.clip(a[3], -1.0, 1.0))
    r = np.hypot(ax, ay)
    if r > 1.0:
        ax /= r
        ay /= r
    return CueAction(theta=theta, power=power, a=ax, b=ay)


class Billiards4BallEnv(gym.Env):
    """Korean 4-ball carom: one shot per episode."""

    metadata = {"render_modes": ["trajectory"]}

    def __init__(
        self,
        spec: TableSpec | None = None,
        cue_id: int = int(BallRole.CUE_WHITE),
        t_max: float = 12.0,
    ) -> None:
        super().__init__()
        self._spec = spec or TableSpec()
        self._cue_id = cue_id
        self._t_max = t_max

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0 * np.pi, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._state: TableState | None = None
        self._last_info: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers

    def _obs(self) -> np.ndarray:
        assert self._state is not None
        return self._state.to_array().reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------ Gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = TableState.initial_4ball(cue_id=self._cue_id, spec=self._spec)
        self._last_info = None
        info: dict[str, Any] = {"spec": _spec_to_dict(self._spec)}
        return self._obs(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("env.step called before env.reset")
        cue_action = _project_action(np.asarray(action, dtype=np.float64))
        result = simulate_shot(self._state, cue_action, t_max=self._t_max)
        info: dict[str, Any] = {
            "event_log": result["events"],
            "cushion_hits": result["cushion_hits"],
            "fouled": result["fouled"],
            "score": result["score"],
            "duration": result["duration"],
            "trajectory": result["trajectory"],
            "spec": _spec_to_dict(self._spec),
            "cue_id": self._cue_id,
        }
        self._last_info = info
        reward = float(result["score"])
        return self._obs(), reward, True, False, info

    def render(self) -> Any:
        if self._last_info is None:
            return None
        return self._last_info["trajectory"]
