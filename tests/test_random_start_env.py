"""Tests for RandomStartInningEnv (Phase I)."""

from __future__ import annotations

import numpy as np
import pytest

from billiards.inning_env import Billiards4BallInningEnv
from billiards.wrappers.random_start_env import RandomStartInningEnv


def _ball_positions(env) -> np.ndarray:
    state = env.unwrapped._state
    return np.stack([np.array([b.x, b.y]) for b in state.balls])


def _spec(env):
    return env.unwrapped._spec


def test_random_reset_produces_valid_layouts():
    """100 random resets all yield non-overlapping in-bounds balls."""
    env = RandomStartInningEnv(Billiards4BallInningEnv(max_shots=10))
    spec = _spec(env)
    R = spec.ball_radius
    margin = 0.005
    safety = 0.002
    min_d = 2.0 * R + safety - 1e-9  # tolerance on the equality

    for i in range(100):
        env.reset(seed=1000 + i)
        positions = _ball_positions(env)
        # In-bounds with margin (allow tiny numerical slack).
        assert np.all(positions[:, 0] >= R + margin - 1e-9), positions
        assert np.all(positions[:, 0] <= spec.width - R - margin + 1e-9), positions
        assert np.all(positions[:, 1] >= R + margin - 1e-9), positions
        assert np.all(positions[:, 1] <= spec.height - R - margin + 1e-9), positions
        # Pairwise non-overlap.
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                d = float(np.linalg.norm(positions[a] - positions[b]))
                assert d >= min_d, (
                    f"reset {i}: balls {a}/{b} too close: d={d:.5f} < {min_d:.5f}"
                )
        # All velocities and spins zero.
        state = env.unwrapped._state
        for b in state.balls:
            assert b.vx == 0.0 and b.vy == 0.0
            assert b.wx == 0.0 and b.wy == 0.0 and b.wz == 0.0


def test_same_seed_same_layout():
    """A given seed must produce the same layout every time."""
    env = RandomStartInningEnv(Billiards4BallInningEnv(max_shots=10))
    env.reset(seed=42)
    p1 = _ball_positions(env)
    env.reset(seed=42)
    p2 = _ball_positions(env)
    np.testing.assert_allclose(p1, p2, atol=1e-12)

    # And once more, after a different seed has perturbed the rng.
    env.reset(seed=99)
    env.reset(seed=42)
    p3 = _ball_positions(env)
    np.testing.assert_allclose(p1, p3, atol=1e-12)


def test_different_seeds_different_layouts():
    """Different seeds produce non-identical layouts (not all the same)."""
    env = RandomStartInningEnv(Billiards4BallInningEnv(max_shots=10))
    layouts = []
    for s in range(10):
        env.reset(seed=s)
        layouts.append(_ball_positions(env).copy())

    # All 10 layouts must not collapse to a single layout.
    seen_distinct = False
    for i in range(1, len(layouts)):
        if not np.allclose(layouts[0], layouts[i], atol=1e-9):
            seen_distinct = True
            break
    assert seen_distinct, "all sampled layouts were identical — RNG not working"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
