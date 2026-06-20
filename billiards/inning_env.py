"""Multi-shot inning environment for the 4-ball carom simulator.

Single-shot ``Billiards4BallEnv`` terminates after one cue impulse. This
wrapper keeps the table state alive across shots so a policy plays an
*inning* — a streak of shots that ends on a miss, foul, or shot-cap.

Termination semantics:
    terminated:  the inning ended due to game rules (no score or foul)
    truncated:   the per-inning shot budget was exhausted
"""

from __future__ import annotations

import copy
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
EXTRA_FEATURE_DIM = 4  # d_red1, d_red2, sin(phi), cos(phi)


def _spec_to_dict(spec: TableSpec) -> dict[str, float]:
    return asdict(spec)


def _project_action(a: np.ndarray) -> CueAction:
    """Same projection as Billiards4BallEnv: wrap theta, clip power,
    radial-clip (a, b) into the unit disk."""
    theta = float(a[0]) % (2.0 * np.pi)
    power = float(np.clip(a[1], 0.0, 1.0))
    ax = float(np.clip(a[2], -1.0, 1.0))
    ay = float(np.clip(a[3], -1.0, 1.0))
    r = np.hypot(ax, ay)
    if r > 1.0:
        ax /= r
        ay /= r
    return CueAction(theta=theta, power=power, a=ax, b=ay)


class Billiards4BallInningEnv(gym.Env):
    """Korean 4-ball carom: continue shooting until a miss / foul / cap."""

    metadata = {"render_modes": ["trajectory"]}

    def __init__(
        self,
        spec: TableSpec | None = None,
        cue_id: int = int(BallRole.CUE_WHITE),
        t_max: float = 12.0,
        max_shots: int = 50,
        continue_on_miss: bool = False,
        foul_penalty: float = 0.1,
        ignore_opponent: bool = False,
        constrain_aim: bool = False,
        aim_alpha_cap_deg: float = 80.0,
        wide_aim: bool = False,
        aim_width_mult: float = 3.0,
        wide_aim_cap_deg: float = 60.0,
        extra_features: bool = False,
        gentle_shot: bool = False,
        gentle_alpha: float = 0.2,
        gentle_d_target: float = 0.2,
        gentle_sigma: float = 0.1,
        setup_shaping: bool = False,
        setup_alpha: float = 0.05,
        setup_scale: float = 0.3,
        near_miss_shaping: bool = False,
        near_miss_alpha: float = 0.05,
        near_miss_scale: float = 0.03,
        robust_reward: bool = False,
        robust_eps: float = 0.05,
        robust_n: int = 8,
        robust_alpha: float = 0.2,
        angle_sincos: bool = False,
    ) -> None:
        super().__init__()
        self._spec = spec or TableSpec()
        self._cue_id = cue_id
        self._t_max = t_max
        self._max_shots = int(max_shots)
        # When True, the episode never terminates on miss/foul; the policy
        # keeps shooting from whatever ball configuration the previous shot
        # produced until ``max_shots`` is reached. Used to expose the policy
        # to a wide distribution of in-play states during training.
        self._continue_on_miss = bool(continue_on_miss)
        self._foul_penalty = float(foul_penalty)
        # Curriculum stage 1: pretend the opponent ball doesn't exist for
        # scoring/foul purposes. Physics still simulates all 4 balls (so the
        # observation space is unchanged), but score depends only on the cue
        # ball hitting both reds and ``fouled`` is forced False.
        self._ignore_opponent = bool(ignore_opponent)
        # Geometric aim constraint: map agent's theta into the ±alpha window
        # that geometrically guarantees first contact with the nearest red
        # ball. alpha = arcsin(2r/d), capped at ``aim_alpha_cap_deg`` to keep
        # the projection well-defined when d is close to 2r.
        self._constrain_aim = bool(constrain_aim)
        self._aim_alpha_cap = float(np.radians(aim_alpha_cap_deg))
        self._wide_aim = bool(wide_aim)
        self._aim_width_mult = float(aim_width_mult)
        self._wide_aim_cap = float(np.radians(wide_aim_cap_deg))
        # Augment obs with hand-crafted geometric features about the two reds:
        # distance to each red, plus the polar angle of the *other* red as
        # seen from the nearest red with cue→nearest as the polar axis. The
        # polar angle is split into (sin, cos) to avoid the 2π wrap.
        self._extra_features = bool(extra_features)
        # Gaussian leave-position bonus: reward the cue ball ending up near
        # the second red ball after a scoring shot (encourages position play).
        # Only applied on non-foul scoring shots.
        self._gentle_shot = bool(gentle_shot)
        self._gentle_alpha = float(gentle_alpha)
        self._gentle_d_target = float(gentle_d_target)
        self._gentle_sigma = float(gentle_sigma)
        # Dense per-shot shaping: bonus = setup_alpha * exp(-d_min/setup_scale)
        # applied after every non-fouled shot. d_min is the post-shot distance
        # from cue ball to its nearest red. Encourages the cue ball to end the
        # shot close to a red so the next attempt has a good aim, even when
        # the current shot didn't score. Keep setup_alpha small (~0.05) so the
        # +1 score reward dominates the per-inning sum.
        self._setup_shaping = bool(setup_shaping)
        self._setup_alpha = float(setup_alpha)
        self._setup_scale = float(setup_scale)
        # Sparse carom progress shaping: on non-foul misses where the cue ball
        # contacted exactly one red, reward how closely the cue passed the
        # missing red over the full trajectory. This keeps the bonus tied to
        # "almost completed the carom" rather than generic post-shot position.
        self._near_miss_shaping = bool(near_miss_shaping)
        self._near_miss_alpha = float(near_miss_alpha)
        self._near_miss_scale = float(near_miss_scale)
        # Action-space robustness bonus. On scoring (non-foul) shots, sample
        # ``robust_n`` perturbed actions ~ N(0, robust_eps) around the raw
        # action, replay each from the pre-shot state, and add
        # ``robust_alpha * success_fraction`` to the reward. Encourages choices
        # that tolerate motor noise (margin to failure).
        self._robust_reward = bool(robust_reward)
        self._robust_eps = float(robust_eps)
        self._robust_n = int(robust_n)
        self._robust_alpha = float(robust_alpha)
        # Action parameterisation. When False (default), the policy emits 4
        # dims (theta in [0, 2pi], power in [0, 1], a, b in [-1, 1]). When
        # True, the policy emits 5 dims with the first two being unnormalised
        # (sin theta, cos theta) hints, both in [-1, 1]; theta is recovered
        # via atan2 so the agent doesn't have to learn around the 0/2pi wrap.
        self._angle_sincos = bool(angle_sincos)
        obs_dim = OBS_DIM + (EXTRA_FEATURE_DIM if self._extra_features else 0)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        if self._angle_sincos:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([2.0 * np.pi, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self._state: TableState | None = None
        self._cumulative_score: int = 0
        self._shot_index: int = 0
        self._cumulative_t: float = 0.0
        # Per-shot trajectories with their original (local) timestamps.
        self._shot_trajectories: list[list[tuple[float, np.ndarray]]] = []
        # Per-shot offsets so we can reconstruct an inning-global timeline.
        self._shot_offsets: list[float] = []
        self._inning_log_records: list[dict[str, Any]] = []
        self._last_info: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers

    def _obs(self) -> np.ndarray:
        assert self._state is not None
        base = self._state.to_array().reshape(-1).astype(np.float32)
        if not self._extra_features:
            return base
        cue = self._state.balls[self._cue_id]
        red1 = self._state.balls[int(BallRole.RED_1)]
        red2 = self._state.balls[int(BallRole.RED_2)]
        d1 = float(np.hypot(red1.x - cue.x, red1.y - cue.y))
        d2 = float(np.hypot(red2.x - cue.x, red2.y - cue.y))
        if d1 <= d2:
            nearest, other = red1, red2
        else:
            nearest, other = red2, red1
        axis = float(np.arctan2(cue.y - nearest.y, cue.x - nearest.x))
        other_dir = float(np.arctan2(other.y - nearest.y, other.x - nearest.x))
        phi = other_dir - axis
        extras = np.array(
            [d1, d2, float(np.sin(phi)), float(np.cos(phi))],
            dtype=np.float32,
        )
        return np.concatenate([base, extras], axis=0)

    def _apply_aim_constraint(self, cue_action: CueAction) -> CueAction:
        """Remap agent's theta into the angular tolerance that guarantees the
        cue ball will first-contact the nearest red. theta is interpreted as
        an offset in [-1, 1] (linearly from [0, 2π]) and scaled by alpha."""
        assert self._state is not None
        cue = self._state.balls[self._cue_id]
        r = self._state.spec.ball_radius
        red1 = self._state.balls[int(BallRole.RED_1)]
        red2 = self._state.balls[int(BallRole.RED_2)]
        d1 = float(np.hypot(red1.x - cue.x, red1.y - cue.y))
        d2 = float(np.hypot(red2.x - cue.x, red2.y - cue.y))
        if d1 <= d2:
            nearest, d = red1, d1
        else:
            nearest, d = red2, d2
        target_dir = float(np.arctan2(nearest.y - cue.y, nearest.x - cue.x))
        if d <= 2.0 * r:
            alpha = self._aim_alpha_cap
        else:
            sin_a = min(2.0 * r / d, float(np.sin(self._aim_alpha_cap)))
            alpha = float(np.arcsin(sin_a))
        if self._wide_aim:
            alpha = min(self._aim_width_mult * alpha, self._wide_aim_cap)
        offset = (cue_action.theta - np.pi) / np.pi  # [-1, 1]
        theta_final = float((target_dir + offset * alpha) % (2.0 * np.pi))
        return CueAction(
            theta=theta_final,
            power=cue_action.power,
            a=cue_action.a,
            b=cue_action.b,
        )

    def _estimate_robustness(
        self,
        pre_shot_state: TableState,
        raw_action: np.ndarray,
    ) -> float:
        """Fraction of N perturbed actions that still score (non-foul) when
        replayed from ``pre_shot_state``. Noise is N(0, robust_eps) added to
        the raw 4-d action; ``_project_action`` then re-projects it onto the
        valid action manifold, and ``_apply_aim_constraint`` is honored so the
        perturbation lives in the same constrained-aim window the policy sees.

        Restores ``self._state`` to whatever it pointed to on entry before
        returning, so the caller's post-shot state is preserved.
        """
        saved_state = self._state
        successes = 0
        for _ in range(self._robust_n):
            # Fresh copy per trial: simulate_shot mutates the state.
            self._state = copy.deepcopy(pre_shot_state)
            delta = self.np_random.normal(
                0.0, self._robust_eps, size=4
            ).astype(np.float64)
            perturbed_raw = np.asarray(raw_action, dtype=np.float64) + delta
            pert_action = _project_action(perturbed_raw)
            if self._constrain_aim or self._wide_aim:
                pert_action = self._apply_aim_constraint(pert_action)
            try:
                result = simulate_shot(
                    self._state, pert_action, t_max=self._t_max,
                )
                if int(result["score"]) > 0 and not bool(result["fouled"]):
                    successes += 1
            except Exception:  # noqa: BLE001 - treat any sim error as fail
                pass
        self._state = saved_state
        return successes / max(1, self._robust_n)

    def _calc_gentle_reward(self, events: list) -> float:
        """Gaussian bonus based on cue-ball distance to the second red hit.

        Tracks unique reds in contact order; returns 0 if fewer than two
        distinct reds were hit (shouldn't happen on a scoring shot, but safe).
        """
        reds_hit: list[int] = []
        for ev in events:
            if ev["type"] == "cue_hit_red":
                i, j = ev["detail"]["balls"]
                red_idx = j if i == self._cue_id else i
                if red_idx not in reds_hit:
                    reds_hit.append(red_idx)
        if len(reds_hit) < 2:
            return 0.0
        second_red = self._state.balls[reds_hit[1]]
        cue = self._state.balls[self._cue_id]
        dist = float(np.hypot(second_red.x - cue.x, second_red.y - cue.y))
        return float(
            self._gentle_alpha
            * np.exp(-(dist - self._gentle_d_target) ** 2
                     / (2.0 * self._gentle_sigma ** 2))
        )

    def _calc_near_miss_reward(self, events: list, trajectory: list) -> float:
        """Small bonus for almost hitting the second red on a missed carom."""
        reds_hit: set[int] = set()
        for ev in events:
            if ev["type"] == "cue_hit_red":
                i, j = ev["detail"]["balls"]
                red_idx = j if i == self._cue_id else i
                reds_hit.add(red_idx)
        if len(reds_hit) != 1:
            return 0.0

        hit_red = next(iter(reds_hit))
        missing_red = (
            int(BallRole.RED_2)
            if hit_red == int(BallRole.RED_1)
            else int(BallRole.RED_1)
        )
        r_contact = 2.0 * self._state.spec.ball_radius
        min_gap = float("inf")
        for _, arr in trajectory:
            cue_xy = arr[self._cue_id, :2]
            red_xy = arr[missing_red, :2]
            gap = float(np.hypot(*(cue_xy - red_xy)) - r_contact)
            min_gap = min(min_gap, max(0.0, gap))
        if not np.isfinite(min_gap):
            return 0.0
        scale = max(1e-9, self._near_miss_scale)
        return float(self._near_miss_alpha * np.exp(-min_gap / scale))

    # ------------------------------------------------------------------ Gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = TableState.initial_4ball(cue_id=self._cue_id, spec=self._spec)
        self._cumulative_score = 0
        self._shot_index = 0
        self._cumulative_t = 0.0
        self._shot_trajectories = []
        self._shot_offsets = []
        self._inning_log_records = []
        self._last_info = None
        info: dict[str, Any] = {
            "spec": _spec_to_dict(self._spec),
            "shot_index": 0,
            "cumulative_score": 0,
        }
        return self._obs(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("env.step called before env.reset")

        raw_action = np.asarray(action, dtype=np.float64)
        if self._angle_sincos:
            # Convert (sin, cos, power, a, b) -> (theta, power, a, b).
            sin_t = float(raw_action[0])
            cos_t = float(raw_action[1])
            theta = float(np.arctan2(sin_t, cos_t)) % (2.0 * np.pi)
            raw_action = np.array(
                [theta, raw_action[2], raw_action[3], raw_action[4]],
                dtype=np.float64,
            )
        cue_action = _project_action(raw_action)
        if self._constrain_aim or self._wide_aim:
            cue_action = self._apply_aim_constraint(cue_action)

        # Snapshot the pre-shot state so we can replay perturbed actions from
        # the same position when computing the action-space robustness bonus.
        # Skipped when robust_reward is off to avoid an unnecessary deepcopy.
        pre_shot_snapshot = (
            copy.deepcopy(self._state) if self._robust_reward else None
        )

        # simulate_shot mutates state in place and resets per-shot t origin
        # via state.t (which is currently whatever we left it at). We snap
        # the local clock to 0 before each shot so each trajectory starts at
        # t=0 locally; the inning timeline is rebuilt via _shot_offsets.
        offset = self._cumulative_t
        self._state.t = 0.0
        result = simulate_shot(self._state, cue_action, t_max=self._t_max)

        # Defensive: simulate_shot can return with residual velocity if the
        # inner loop hit t_max before balls settled. Without snapping, the
        # next apply_cue raises ("cue ball must be at rest"). Snap here so
        # downstream code is safe regardless of continue_on_miss mode.
        if not self._state.all_at_rest():
            for b in self._state.balls:
                b.vx = b.vy = 0.0
                b.wx = b.wy = b.wz = 0.0

        # Record per-shot trajectory (raw, local-time).
        shot_traj: list[tuple[float, np.ndarray]] = list(result["trajectory"])
        self._shot_trajectories.append(shot_traj)
        self._shot_offsets.append(offset)
        self._cumulative_t = offset + float(result["duration"])

        if self._ignore_opponent:
            reds: set[int] = set()
            for ev in result["events"]:
                if ev["type"] == "cue_hit_red":
                    i, j = ev["detail"]["balls"]
                    red_idx = j if i == self._cue_id else i
                    reds.add(red_idx)
            score = 1 if len(reds) >= 2 else 0
            fouled = False
        else:
            score = int(result["score"])
            fouled = bool(result["fouled"])
        self._cumulative_score += score
        self._shot_index += 1

        # Stash a result record sans heavyweight trajectory for analysis.
        record = {
            "shot_index": self._shot_index,
            "score": score,
            "fouled": fouled,
            "cushion_hits": int(result["cushion_hits"]),
            "duration": float(result["duration"]),
            "events": result["events"],
            "offset": offset,
        }
        self._inning_log_records.append(record)

        # Foul negates the score: a fouled shot earns -foul_penalty regardless
        # of whether the cue ball happened to hit both reds.
        if fouled:
            reward = -self._foul_penalty
        else:
            reward = float(score)
            if score > 0 and self._gentle_shot:
                reward += self._calc_gentle_reward(result["events"])
            if self._setup_shaping:
                cue = self._state.balls[self._cue_id]
                r1 = self._state.balls[int(BallRole.RED_1)]
                r2 = self._state.balls[int(BallRole.RED_2)]
                d1 = float(np.hypot(cue.x - r1.x, cue.y - r1.y))
                d2 = float(np.hypot(cue.x - r2.x, cue.y - r2.y))
                d_min = min(d1, d2)
                reward += self._setup_alpha * float(np.exp(-d_min / self._setup_scale))
            if score == 0 and self._near_miss_shaping:
                near_miss_reward = self._calc_near_miss_reward(
                    result["events"],
                    shot_traj,
                )
                reward += near_miss_reward
                record["near_miss_reward"] = near_miss_reward
            if score > 0 and self._robust_reward and pre_shot_snapshot is not None:
                robustness = self._estimate_robustness(
                    pre_shot_snapshot, raw_action,
                )
                reward += self._robust_alpha * robustness
                record["robustness"] = robustness

        if self._continue_on_miss:
            terminated = False
            truncated = self._shot_index >= self._max_shots
            # simulate_shot only zero-clamps balls that ended at rest. When
            # t_max truncates a still-rolling shot the cue ball keeps a
            # residual velocity and the next apply_cue() raises. In
            # continue_on_miss mode we force the table to a halt between
            # shots so the next shot is always well-defined.
            if not truncated:
                for b in self._state.balls:
                    b.vx = b.vy = 0.0
                    b.wx = b.wy = 0.0
                    b.wz = 0.0
        else:
            terminated = (score == 0) or fouled
            truncated = (self._shot_index >= self._max_shots) and not terminated

        # Slim info for VecEnv IPC: trajectory + event_log + spec are heavy
        # to pickle every step and never consumed by training (Monitor only
        # touches the scalar keys). Keep the full record in self._last_info
        # so render() / notebooks still work.
        slim_info: dict[str, Any] = {
            "cushion_hits": result["cushion_hits"],
            "fouled": fouled,
            "score": score,
            "duration": result["duration"],
            "shot_index": self._shot_index,
            "cumulative_score": self._cumulative_score,
        }
        self._last_info = {
            **slim_info,
            "event_log": result["events"],
            "trajectory": shot_traj,
            "spec": _spec_to_dict(self._spec),
            "cue_id": self._cue_id,
        }
        return self._obs(), reward, terminated, truncated, slim_info

    def render(self) -> Any:
        if self._last_info is None:
            return None
        return self._last_info["trajectory"]

    # ------------------------------------------------------------------ inning-level views

    @property
    def full_trajectory(self) -> list[tuple[float, np.ndarray]]:
        """All shots stitched into one (t_global, (4,7)) timeline.

        Each shot's local t is shifted by its inning offset so timestamps
        are globally monotonic. The very first sample is forced to t=0.
        """
        out: list[tuple[float, np.ndarray]] = []
        for traj, offset in zip(self._shot_trajectories, self._shot_offsets):
            for t_local, arr in traj:
                out.append((float(t_local) + float(offset), arr))
        out.sort(key=lambda tp: tp[0])
        return out

    @property
    def inning_log(self) -> list[dict[str, Any]]:
        """Per-shot result records (no trajectory payload)."""
        return list(self._inning_log_records)

    @property
    def shot_trajectories(self) -> list[list[tuple[float, np.ndarray]]]:
        """Raw per-shot trajectories with local-time origins (one list per shot)."""
        return [list(traj) for traj in self._shot_trajectories]

    @property
    def cumulative_score(self) -> int:
        return self._cumulative_score

    @property
    def shot_index(self) -> int:
        return self._shot_index
