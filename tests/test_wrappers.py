"""Tests for billiards.wrappers.RewardModelEnv.

These tests need a trained reward model on disk
(``models/reward_model.pt``) — produced by Phase D. They are skipped
gracefully when the model file is absent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from billiards.env import Billiards4BallEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "reward_model.pt"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"reward model missing at {MODEL_PATH} (Phase D not yet run)",
)


def _make_wrapped() -> "RewardModelEnv":  # noqa: F821 - imported lazily under skipif
    from billiards.wrappers import RewardModelEnv

    return RewardModelEnv(Billiards4BallEnv(), model_path=str(MODEL_PATH))


def test_spaces_unchanged() -> None:
    base = Billiards4BallEnv()
    wrapped = _make_wrapped()
    assert wrapped.observation_space == base.observation_space
    assert wrapped.action_space == base.action_space


def test_reset_returns_valid_obs() -> None:
    wrapped = _make_wrapped()
    obs, info = wrapped.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (28,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_reward_is_float_and_env_reward_preserved() -> None:
    wrapped = _make_wrapped()
    wrapped.reset(seed=42)
    action = np.array([1.5, 0.6, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = wrapped.step(action)
    # Reward must be a Python float, not int — distinguishes the model
    # output from the original env's integer score.
    assert isinstance(reward, float)
    assert "env_reward" in info
    assert isinstance(info["env_reward"], float)
    # The original env reward is the carom score: 0 or 1.
    assert info["env_reward"] in (0.0, 1.0)
    assert obs.shape == (28,)
    assert terminated is True


def test_reward_model_called_once_per_step() -> None:
    wrapped = _make_wrapped()
    wrapped.reset(seed=7)
    real_model = wrapped.model
    call_count = {"n": 0}

    def counting_call(*args, **kwargs):
        call_count["n"] += 1
        return real_model(*args, **kwargs)

    with patch.object(wrapped, "model", side_effect=counting_call,
                      wraps=real_model) as mocked:
        # Replacement object must still be callable and produce a tensor.
        # We can't easily wrap a Module via patch.object(side_effect),
        # so instead substitute a callable manually.
        pass

    # Rebuild with a hand-rolled counter wrapper around the original
    # model to count __call__ invocations.
    counter = {"n": 0}
    original_model = wrapped.model

    class _CountingModel(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            counter["n"] += 1
            return self.inner(s, a)

    wrapped.model = _CountingModel(original_model).to(wrapped.device).eval()

    action = np.array([2.0, 0.5, 0.1, -0.1], dtype=np.float32)
    wrapped.step(action)
    assert counter["n"] == 1, f"expected 1 reward-model call per step, got {counter['n']}"
    # Second step should add exactly one more call.
    wrapped.reset(seed=8)
    wrapped.step(action)
    assert counter["n"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
