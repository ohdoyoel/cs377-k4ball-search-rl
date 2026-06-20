"""Reward wrapper that adds a learned V(s, a) bonus.

Replaces the env reward by

    r_t = score_t + lambda * sigmoid(RM(s_t, a_t))

where RM predicts the logit of P(score=1 | s_t, a_t). Unlike the
state-only ScoringPotentialRMEnv, this signal credits the *action*
directly so SAC's policy gradient sees ∂RM/∂a, not just ∂RM/∂s'.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class _VSaNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class VSaRMEnv(gym.Wrapper):
    """Augment reward with sigmoid(V(s, a)) from a trained RM."""

    def __init__(self, env: gym.Env, rm_path: str | Path, lam: float = 1.0,
                 device: str = "cpu") -> None:
        super().__init__(env)
        ckpt = torch.load(str(rm_path), map_location=device, weights_only=False)
        self._rm = _VSaNet(obs_dim=int(ckpt["obs_dim"]),
                           act_dim=int(ckpt["act_dim"]),
                           hidden=int(ckpt.get("hidden", 256))).to(device)
        self._rm.load_state_dict(ckpt["state_dict"])
        self._rm.eval()
        self._device = device
        self._lam = float(lam)
        self._prev_obs: np.ndarray | None = None
        self._rm_act_dim = int(ckpt["act_dim"])
        self._env_act_dim = int(env.action_space.shape[0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def _to_rm_action(self, a_env: np.ndarray) -> np.ndarray:
        """Convert env action to RM-compatible action. If the env emits the
        5D (sin, cos, p, a, b) representation but the RM was trained on the
        4D (theta, p, a, b) representation, collapse the sin/cos pair via
        atan2."""
        if self._env_act_dim == self._rm_act_dim:
            return a_env.astype(np.float32)
        if self._env_act_dim == 5 and self._rm_act_dim == 4:
            theta = float(np.arctan2(a_env[0], a_env[1])) % (2.0 * np.pi)
            return np.array([theta, float(a_env[2]), float(a_env[3]),
                             float(a_env[4])], dtype=np.float32)
        raise ValueError(
            f"Unsupported act-dim mismatch: env {self._env_act_dim}, "
            f"RM {self._rm_act_dim}"
        )

    def step(self, action):
        s_t = self._prev_obs
        a_t = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = 0.0
        if s_t is not None:
            a_rm = self._to_rm_action(a_t)
            with torch.no_grad():
                st = torch.as_tensor(s_t, device=self._device).unsqueeze(0)
                at = torch.as_tensor(a_rm, device=self._device).unsqueeze(0)
                bonus = float(torch.sigmoid(self._rm(st, at)).item())
        new_reward = float(reward) + self._lam * bonus
        self._prev_obs = np.asarray(obs, dtype=np.float32)
        info = dict(info)
        info["env_reward"] = float(reward)
        info["rm_bonus"] = bonus
        info["rm_lambda"] = self._lam
        return obs, new_reward, terminated, truncated, info
