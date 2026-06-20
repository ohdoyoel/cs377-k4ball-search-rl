"""Reward wrapper that adds a scoring-potential bonus from a learned RM.

The wrapped env's per-step reward is replaced by

    r_t = score_t + lambda * RM(s_{t+1})

where RM(s) is a small MLP trained offline to predict
V(s) = E_{a~Uniform} [P(score=1 | s, a)] (see
``experiments/build_scoring_potential_rm.py``).

If the inning ended this step (terminated/truncated), the RM bonus is
zero --- there is no next state to evaluate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class _RMNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ScoringPotentialRMEnv(gym.Wrapper):
    """Augment reward with a scoring-potential bonus from a trained RM."""

    def __init__(
        self,
        env: gym.Env,
        rm_path: str | Path,
        lam: float = 10.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(env)
        ckpt = torch.load(str(rm_path), map_location=device, weights_only=False)
        self._rm = _RMNet(obs_dim=int(ckpt["obs_dim"]),
                          hidden=int(ckpt.get("hidden", 128))).to(device)
        self._rm.load_state_dict(ckpt["state_dict"])
        self._rm.eval()
        self._device = device
        self._lam = float(lam)
        self._v_stats = ckpt.get("v_stats", {})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0.0
        if not (terminated or truncated):
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self._device).unsqueeze(0)
                bonus = float(self._rm(t).item())
        new_reward = float(reward) + self._lam * bonus
        info = dict(info)
        info["env_reward"] = float(reward)
        info["rm_bonus"] = bonus
        info["rm_lambda"] = self._lam
        return obs, new_reward, terminated, truncated, info
