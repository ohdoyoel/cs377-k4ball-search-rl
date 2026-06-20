"""Train + evaluate SAC or TD3 on the multi-shot inning env.

Reward per env step = score for that shot (0 or 1). The episode is a
Korean-4-ball *inning*: cumulative reward across shots until the policy
misses, fouls, or hits the per-inning cap. Compared to the single-shot
``Billiards4BallEnv`` used in Phase F/G, this exposes credit for
*chaining* scoring shots in one possession.

PPO is kept in commented-out form for reference (Phase II_b excludes it
from training since SAC dominated 100% vs PPO 0% in the prior matrix).

Outputs under ``{out_dir}/{run_id}/``:
    training_curve.csv  per-rollout aggregates (ep_return = inning score)
    eval.parquet        one row per evaluation inning (200 by default)
    policy.zip
    config.json
    run.log

Usage:
    uv run python experiments/run_inning_sac.py \\
        --algo sac --seed 0 --total_steps 50000 --eval_episodes 200 \\
        --out_dir experiments/runs_inning/sac_s0

    uv run python experiments/run_inning_sac.py \\
        --algo td3 --seed 0 --total_steps 50000 --eval_episodes 200 \\
        --out_dir experiments/runs_inning/td3_s0
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import PPO, SAC, TD3  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402
from experiments.eval_policy import run_standard_eval  # noqa: E402


# ---- shared hyperparameters ----------------------------------------------
T_MAX = 12.0
GAMMA = 0.99
LR = 3e-4

# SAC
SAC_BATCH = 256
SAC_BUFFER = 200_000
SAC_LEARNING_STARTS = 1_000

# TD3
TD3_BATCH = 256
TD3_BUFFER = 200_000
TD3_LEARNING_STARTS = 1_000
TD3_ACTION_NOISE_SIGMA = 0.1  # exploration noise stddev (action space scale)

# PPO (on-policy; re-enabled for the §1 algorithm comparison)
PPO_N_STEPS = 512
PPO_BATCH = 512
PPO_N_EPOCHS = 4
PPO_GAE = 0.95
PPO_CLIP = 0.2
PPO_VF = 0.5
PPO_ENT = 0.01


# ---------------------------------------------------------------- IO helpers


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:  # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- env factory


def _env_factory(
    max_shots: int,
    seed: int,
    continue_on_miss: bool,
    ignore_opponent: bool,
    constrain_aim: bool,
    wide_aim: bool,
    aim_width_mult: float,
    wide_aim_cap_deg: float,
    extra_features: bool,
    random_start: bool,
    foul_penalty: float,
    gentle_shot: bool,
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
):
    """Build a thunk that constructs one Monitor-wrapped env. Used by both
    DummyVecEnv (n_envs=1) and SubprocVecEnv (n_envs>1)."""
    def _thunk():
        env = Billiards4BallInningEnv(
            t_max=T_MAX,
            max_shots=max_shots,
            continue_on_miss=continue_on_miss,
            ignore_opponent=ignore_opponent,
            constrain_aim=constrain_aim,
            wide_aim=wide_aim,
            aim_width_mult=aim_width_mult,
            wide_aim_cap_deg=wide_aim_cap_deg,
            extra_features=extra_features,
            foul_penalty=foul_penalty,
            gentle_shot=gentle_shot,
            setup_shaping=setup_shaping,
            setup_alpha=setup_alpha,
            setup_scale=setup_scale,
            near_miss_shaping=near_miss_shaping,
            near_miss_alpha=near_miss_alpha,
            near_miss_scale=near_miss_scale,
            robust_reward=robust_reward,
            robust_eps=robust_eps,
            robust_n=robust_n,
            robust_alpha=robust_alpha,
        )
        if random_start:
            env = RandomStartInningEnv(env)
        env = Monitor(env, info_keywords=("cushion_hits", "fouled", "score"))
        env.reset(seed=seed)
        return env
    return _thunk


def _make_train_env(
    max_shots: int,
    seed: int,
    continue_on_miss: bool = False,
    ignore_opponent: bool = False,
    constrain_aim: bool = False,
    wide_aim: bool = False,
    aim_width_mult: float = 3.0,
    wide_aim_cap_deg: float = 60.0,
    extra_features: bool = False,
    random_start: bool = False,
    foul_penalty: float = 0.1,
    gentle_shot: bool = False,
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
    n_envs: int = 1,
):
    """Vectorized training env. Uses SubprocVecEnv when n_envs>1 so multiple
    physics simulations run on parallel CPU cores; falls back to DummyVecEnv
    for n_envs=1 (single-process, no spawn overhead)."""
    factories = [
        _env_factory(
            max_shots=max_shots,
            seed=seed + i,
            continue_on_miss=continue_on_miss,
            ignore_opponent=ignore_opponent,
            constrain_aim=constrain_aim,
            wide_aim=wide_aim,
            aim_width_mult=aim_width_mult,
            wide_aim_cap_deg=wide_aim_cap_deg,
            extra_features=extra_features,
            random_start=random_start,
            foul_penalty=foul_penalty,
            gentle_shot=gentle_shot,
            setup_shaping=setup_shaping,
            setup_alpha=setup_alpha,
            setup_scale=setup_scale,
            near_miss_shaping=near_miss_shaping,
            near_miss_alpha=near_miss_alpha,
            near_miss_scale=near_miss_scale,
            robust_reward=robust_reward,
            robust_eps=robust_eps,
            robust_n=robust_n,
            robust_alpha=robust_alpha,
        )
        for i in range(n_envs)
    ]
    if n_envs <= 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)


# ---------------------------------------------------------------- callbacks


class InningCurveCallback(BaseCallback):
    """Append rollout aggregates to ``training_curve.csv``.

    The Monitor wraps the inning env with terminated=True at the end of
    each inning, so SB3's ``ep_info_buffer`` collects one entry per
    inning (with ``r`` = cumulative inning score = inning_score and ``l``
    = number of shots in the inning). We additionally average our
    ``info_keywords`` across the steps that fed this inning into the
    buffer; with Monitor these are recorded only at episode end as the
    *last* shot's value, so we additionally tally them ourselves on
    ``_on_step``.
    """

    HEADER = [
        "timesteps",
        "ep_return_mean", "ep_length_mean",
        "ep_cushion_mean", "ep_foul_rate",
        "n_episodes_in_window",
    ]

    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(self.HEADER)
        # Per-shot tallies, flushed at end of each inning.
        self._cur_cushions: list[int] = []
        self._cur_score: int = 0
        self._cur_shots: int = 0
        self._cur_fouls: int = 0
        # Completed-inning stats since last write (window).
        self._win_returns: list[float] = []
        self._win_lengths: list[int] = []
        self._win_cushion_means: list[float] = []
        self._win_foul_any: list[float] = []
        self._next_log = 0
        self._log_every = 2048  # flush every ~2k env steps

    def _on_step(self) -> bool:  # noqa: D401
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            dones = [False] * len(infos)
        for info, done in zip(infos, dones):
            cushion = int(info.get("cushion_hits", 0)) if info else 0
            self._cur_cushions.append(cushion)
            self._cur_score += int(info.get("score", 0)) if info else 0
            self._cur_shots += 1
            if info and bool(info.get("fouled", False)):
                self._cur_fouls += 1
            if done:
                self._win_returns.append(float(self._cur_score))
                self._win_lengths.append(int(self._cur_shots))
                self._win_cushion_means.append(
                    float(np.mean(self._cur_cushions)) if self._cur_cushions else 0.0
                )
                self._win_foul_any.append(1.0 if self._cur_fouls > 0 else 0.0)
                self._cur_cushions = []
                self._cur_score = 0
                self._cur_shots = 0
                self._cur_fouls = 0

        if self.num_timesteps - self._next_log >= self._log_every:
            self._flush_window()
            self._next_log = int(self.num_timesteps)
        return True

    def _flush_window(self) -> None:
        if not self._win_returns:
            return
        row = [
            int(self.num_timesteps),
            f"{float(np.mean(self._win_returns)):.4f}",
            f"{float(np.mean(self._win_lengths)):.4f}",
            f"{float(np.mean(self._win_cushion_means)):.4f}",
            f"{float(np.mean(self._win_foul_any)):.4f}",
            int(len(self._win_returns)),
        ]
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        self._win_returns.clear()
        self._win_lengths.clear()
        self._win_cushion_means.clear()
        self._win_foul_any.clear()

    def _on_training_end(self) -> None:
        self._flush_window()


# ---------------------------------------------------------------- evaluation


def _evaluate(
    model,
    n_episodes: int,
    seed_base: int,
    max_shots: int,
    continue_on_miss: bool = False,
    ignore_opponent: bool = False,
    constrain_aim: bool = False,
    wide_aim: bool = False,
    aim_width_mult: float = 3.0,
    wide_aim_cap_deg: float = 60.0,
    extra_features: bool = False,
    random_start: bool = False,
) -> pd.DataFrame:
    """Run ``n_episodes`` deterministic innings; one row per inning."""
    base = Billiards4BallInningEnv(
        t_max=T_MAX,
        max_shots=max_shots,
        continue_on_miss=continue_on_miss,
        ignore_opponent=ignore_opponent,
        constrain_aim=constrain_aim,
        wide_aim=wide_aim,
        aim_width_mult=aim_width_mult,
        wide_aim_cap_deg=wide_aim_cap_deg,
        extra_features=extra_features,
    )
    env = RandomStartInningEnv(base) if random_start else base
    rows: list[dict] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        cushions = 0
        n_shots = 0
        n_fouls = 0
        n_scores = 0
        fouled = False
        total_dur = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            obs, _, terminated, truncated, info = env.step(action)
            cushions += int(info.get("cushion_hits", 0))
            total_dur += float(info.get("duration", 0.0))
            n_shots += 1
            shot_score = int(info.get("score", 0))
            shot_foul = bool(info.get("fouled", False))
            if shot_foul:
                fouled = True
                n_fouls += 1
            elif shot_score > 0:
                n_scores += 1
            if terminated or truncated:
                break
        inner = env.unwrapped if random_start else env
        rows.append({
            "ep_idx": int(ep),
            "seed": int(seed_base + ep),
            "inning_score": int(inner.cumulative_score),
            "n_shots": int(n_shots),
            "n_fouls": int(n_fouls),
            "n_scores": int(n_scores),
            "mean_cushions": float(cushions / max(1, n_shots)),
            "fouled": bool(fouled),
            "total_duration": float(total_dur),
            "truncated": bool(truncated),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Inning-env SAC/PPO baseline.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--algo", type=str, choices=("sac", "td3", "ppo"), default="sac")
    parser.add_argument("--out_dir", type=str, default="experiments/runs_inning")
    parser.add_argument(
        "--continue_on_miss",
        action="store_true",
        help="Keep shooting until max_shots regardless of miss/foul "
             "(exposes the policy to diverse mid-rack states).",
    )
    parser.add_argument(
        "--ignore_opponent",
        action="store_true",
        help="Curriculum stage 1: score only requires hitting both reds; "
             "opponent ball contact is not a foul.",
    )
    parser.add_argument(
        "--load_policy",
        type=str,
        default=None,
        help="Path to a policy .zip to warm-start from (stage 2 fine-tune).",
    )
    parser.add_argument(
        "--constrain_aim",
        action="store_true",
        help="Map theta into the ±arcsin(2r/d) window around the nearest red "
             "so the cue ball geometrically must first-contact a red.",
    )
    parser.add_argument(
        "--extra_features",
        action="store_true",
        help="Augment obs with d(cue,red1), d(cue,red2), and the polar "
             "angle (sin,cos) of the other red as seen from the nearest red.",
    )
    parser.add_argument(
        "--wide_aim",
        action="store_true",
        help="Use a widened nearest-red aim window instead of the exact contact window.",
    )
    parser.add_argument("--aim_width_mult", type=float, default=3.0)
    parser.add_argument("--wide_aim_cap_deg", type=float, default=60.0)
    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Randomize ball positions on each reset via RandomStartInningEnv.",
    )
    parser.add_argument(
        "--foul_penalty",
        type=float,
        default=0.1,
        help="Penalty subtracted from reward on foul (only in continue_on_miss mode).",
    )
    parser.add_argument(
        "--gentle_shot",
        action="store_true",
        help="Add Gaussian leave-position bonus on scoring shots "
             "(alpha=0.2, d_target=0.2m, sigma=0.1m).",
    )
    parser.add_argument(
        "--setup_shaping",
        action="store_true",
        help="Dense per-shot bonus: setup_alpha*exp(-d_min/setup_scale) "
             "after every non-foul shot. Encourages cue ball to end near a "
             "red even on a miss, so next attempt has a good aim.",
    )
    parser.add_argument("--setup_alpha", type=float, default=0.05)
    parser.add_argument("--setup_scale", type=float, default=0.3)
    parser.add_argument(
        "--near_miss_shaping",
        action="store_true",
        help="Small dense bonus on non-foul misses that hit one red and pass "
             "near the second red (near_miss_alpha*exp(-gap/near_miss_scale)).",
    )
    parser.add_argument("--near_miss_alpha", type=float, default=0.05)
    parser.add_argument("--near_miss_scale", type=float, default=0.03)
    parser.add_argument(
        "--robust_reward",
        action="store_true",
        help="On scoring (non-foul) shots, sample N perturbed actions ~ "
             "N(0, robust_eps) and add robust_alpha * (success fraction) to "
             "the reward. Encourages action choices with margin to failure.",
    )
    parser.add_argument("--robust_eps", type=float, default=0.05,
                        help="Stddev of action-space perturbation noise.")
    parser.add_argument("--robust_n", type=int, default=8,
                        help="Perturbations per scoring shot.")
    parser.add_argument("--robust_alpha", type=float, default=0.2,
                        help="Bonus weight for robustness fraction.")
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel envs (SubprocVecEnv) for training. "
             "Eval always runs single-env.",
    )
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=1,
        help="SAC/TD3 gradient steps per env step (-1 = match n_envs). "
             "With n_envs>1 you usually want this to equal n_envs/k to "
             "match a single-env update budget.",
    )
    parser.add_argument(
        "--net_arch",
        type=str,
        default="",
        help="Comma-separated MLP hidden sizes, e.g. '400,300'. "
             "Empty = SB3 default ([256,256] for SAC).",
    )
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--buffer_size", type=int, default=SAC_BUFFER)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_f = log_path.open("w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_f)
    err_tee = _Tee(sys.__stderr__, log_f)

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t0 = time.perf_counter()
        config = {
            "algo": args.algo,
            "seed": int(args.seed),
            "total_steps": int(args.total_steps),
            "max_shots": int(args.max_shots),
            "continue_on_miss": bool(args.continue_on_miss),
            "ignore_opponent": bool(args.ignore_opponent),
            "constrain_aim": bool(args.constrain_aim),
            "wide_aim": bool(args.wide_aim),
            "aim_width_mult": float(args.aim_width_mult),
            "wide_aim_cap_deg": float(args.wide_aim_cap_deg),
            "extra_features": bool(args.extra_features),
            "random_start": bool(args.random_start),
            "foul_penalty": float(args.foul_penalty),
            "gentle_shot": bool(args.gentle_shot),
            "setup_shaping": bool(args.setup_shaping),
            "setup_alpha": float(args.setup_alpha),
            "setup_scale": float(args.setup_scale),
            "near_miss_shaping": bool(args.near_miss_shaping),
            "near_miss_alpha": float(args.near_miss_alpha),
            "near_miss_scale": float(args.near_miss_scale),
            "robust_reward": bool(args.robust_reward),
            "robust_eps": float(args.robust_eps),
            "robust_n": int(args.robust_n),
            "robust_alpha": float(args.robust_alpha),
            "n_envs": int(args.n_envs),
            "gradient_steps": int(args.gradient_steps),
            "net_arch": str(args.net_arch),
            "load_policy": args.load_policy,
            "eval_episodes": int(args.eval_episodes),
            "t_max": T_MAX,
            "gamma": GAMMA,
            "learning_rate": LR,
        }
        if args.algo == "sac":
            config.update({
                "batch_size": SAC_BATCH,
                "buffer_size": SAC_BUFFER,
                "learning_starts": SAC_LEARNING_STARTS,
            })
        elif args.algo == "td3":
            config.update({
                "batch_size": TD3_BATCH,
                "buffer_size": TD3_BUFFER,
                "learning_starts": TD3_LEARNING_STARTS,
                "action_noise_sigma": TD3_ACTION_NOISE_SIGMA,
            })
        elif args.algo == "ppo":
            config.update({
                "n_steps": PPO_N_STEPS,
                "batch_size": PPO_BATCH,
                "n_epochs": PPO_N_EPOCHS,
                "gae_lambda": PPO_GAE,
                "clip_range": PPO_CLIP,
                "vf_coef": PPO_VF,
                "ent_coef": PPO_ENT,
            })
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[run_inning] algo={args.algo} seed={args.seed} "
              f"total_steps={args.total_steps} max_shots={args.max_shots} "
              f"continue_on_miss={args.continue_on_miss} "
              f"ignore_opponent={args.ignore_opponent} "
              f"constrain_aim={args.constrain_aim} "
              f"wide_aim={args.wide_aim} "
              f"extra_features={args.extra_features} "
              f"random_start={args.random_start} "
              f"foul_penalty={args.foul_penalty} "
              f"gentle_shot={args.gentle_shot} "
              f"near_miss_shaping={args.near_miss_shaping} "
              f"robust_reward={args.robust_reward} "
              f"n_envs={args.n_envs} "
              f"load_policy={args.load_policy} "
              f"out_dir={out_dir}")

        set_random_seed(int(args.seed))
        env = _make_train_env(
            max_shots=int(args.max_shots),
            seed=int(args.seed),
            continue_on_miss=bool(args.continue_on_miss),
            ignore_opponent=bool(args.ignore_opponent),
            constrain_aim=bool(args.constrain_aim),
            wide_aim=bool(args.wide_aim),
            aim_width_mult=float(args.aim_width_mult),
            wide_aim_cap_deg=float(args.wide_aim_cap_deg),
            extra_features=bool(args.extra_features),
            random_start=bool(args.random_start),
            foul_penalty=float(args.foul_penalty),
            gentle_shot=bool(args.gentle_shot),
            setup_shaping=bool(args.setup_shaping),
            setup_alpha=float(args.setup_alpha),
            setup_scale=float(args.setup_scale),
            near_miss_shaping=bool(args.near_miss_shaping),
            near_miss_alpha=float(args.near_miss_alpha),
            near_miss_scale=float(args.near_miss_scale),
            robust_reward=bool(args.robust_reward),
            robust_eps=float(args.robust_eps),
            robust_n=int(args.robust_n),
            robust_alpha=float(args.robust_alpha),
            n_envs=int(args.n_envs),
        )

        policy_kwargs = None
        if args.net_arch:
            sizes = [int(x) for x in args.net_arch.split(",") if x.strip()]
            policy_kwargs = {"net_arch": sizes}
            print(f"[run_inning] custom net_arch={sizes}")

        if args.algo == "sac":
            if args.load_policy:
                model = SAC.load(args.load_policy, env=env, device="cpu")
                print(f"[run_inning] loaded SAC policy <- {args.load_policy}")
            else:
                model = SAC(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=LR,
                    buffer_size=int(args.buffer_size),
                    batch_size=SAC_BATCH,
                    gamma=float(args.gamma),
                    learning_starts=SAC_LEARNING_STARTS,
                    gradient_steps=int(args.gradient_steps),
                    policy_kwargs=policy_kwargs,
                    seed=int(args.seed),
                    verbose=0,
                    device="cpu",
                )
        elif args.algo == "td3":
            if args.load_policy:
                model = TD3.load(args.load_policy, env=env, device="cpu")
                print(f"[run_inning] loaded TD3 policy <- {args.load_policy}")
            else:
                n_actions = env.action_space.shape[0]
                action_noise = NormalActionNoise(
                    mean=np.zeros(n_actions, dtype=np.float32),
                    sigma=TD3_ACTION_NOISE_SIGMA * np.ones(n_actions, dtype=np.float32),
                )
                model = TD3(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=LR,
                    buffer_size=TD3_BUFFER,
                    batch_size=TD3_BATCH,
                    gamma=GAMMA,
                    learning_starts=TD3_LEARNING_STARTS,
                    action_noise=action_noise,
                    seed=int(args.seed),
                    verbose=0,
                    device="cpu",
                )
        elif args.algo == "ppo":
            if args.load_policy:
                model = PPO.load(args.load_policy, env=env, device="cpu")
                print(f"[run_inning] loaded PPO policy <- {args.load_policy}")
            else:
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=LR,
                    n_steps=PPO_N_STEPS,
                    batch_size=PPO_BATCH,
                    n_epochs=PPO_N_EPOCHS,
                    gamma=float(args.gamma),
                    gae_lambda=PPO_GAE,
                    clip_range=PPO_CLIP,
                    vf_coef=PPO_VF,
                    ent_coef=PPO_ENT,
                    policy_kwargs=policy_kwargs,
                    seed=int(args.seed),
                    verbose=0,
                    device="cpu",
                )

        try:
            cb = InningCurveCallback(out_dir / "training_curve.csv")
            t_train0 = time.perf_counter()
            model.learn(total_timesteps=int(args.total_steps), callback=cb,
                        progress_bar=False)
            train_wall = time.perf_counter() - t_train0
            print(f"[run_inning] train done in {train_wall:.1f}s")

            policy_path = out_dir / "policy.zip"
            model.save(str(policy_path))
            print(f"[run_inning] saved policy -> {policy_path}")

            t_eval0 = time.perf_counter()
            eval_df = _evaluate(
                model,
                n_episodes=int(args.eval_episodes),
                seed_base=int(args.seed) + 10_000,
                max_shots=int(args.max_shots),
                continue_on_miss=bool(args.continue_on_miss),
                ignore_opponent=bool(args.ignore_opponent),
                constrain_aim=bool(args.constrain_aim),
                wide_aim=bool(args.wide_aim),
                aim_width_mult=float(args.aim_width_mult),
                wide_aim_cap_deg=float(args.wide_aim_cap_deg),
                extra_features=bool(args.extra_features),
                random_start=bool(args.random_start),
            )
            eval_wall = time.perf_counter() - t_eval0
            eval_path = out_dir / "eval.parquet"
            eval_df.to_parquet(eval_path, engine="pyarrow", index=False)
            print(f"[run_inning] eval n={len(eval_df)} done in {eval_wall:.1f}s "
                  f"-> {eval_path}")

            mean_inning = float(eval_df["inning_score"].mean())
            max_inning = int(eval_df["inning_score"].max())
            p_ge1 = 100.0 * float((eval_df["inning_score"] >= 1).mean())
            p_ge3 = 100.0 * float((eval_df["inning_score"] >= 3).mean())
            p_ge5 = 100.0 * float((eval_df["inning_score"] >= 5).mean())
            mean_shots = float(eval_df["n_shots"].mean())
            total_shots = float(eval_df["n_shots"].sum())
            # Per-shot probabilities (pooled over all shots in all innings),
            # not the inning-level "any foul this inning" rate.
            foul_rate = 100.0 * float(eval_df["n_fouls"].sum()) / max(1.0, total_shots)
            success_rate = 100.0 * float(eval_df["n_scores"].sum()) / max(1.0, total_shots)
            mean_cushions = float(eval_df["mean_cushions"].mean())
            wall = time.perf_counter() - t0
            summary = {
                "algo": args.algo,
                "seed": int(args.seed),
                "train_wall_s": float(train_wall),
                "eval_wall_s": float(eval_wall),
                "wall_s": float(wall),
                "mean_inning_score": mean_inning,
                "max_inning_score": max_inning,
                "p_score_ge1": p_ge1,
                "p_score_ge3": p_ge3,
                "p_score_ge5": p_ge5,
                "mean_shots": mean_shots,
                "foul_rate": foul_rate,
                "success_rate": success_rate,
                "mean_cushions": mean_cushions,
            }
            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(
                f"[run_inning] DONE algo={args.algo} seed={args.seed} "
                f"mean_inning={mean_inning:.3f} max={max_inning} "
                f"p>=1={p_ge1:.1f}% p>=3={p_ge3:.1f}% p>=5={p_ge5:.1f}% "
                f"mean_shots={mean_shots:.2f} foul%/shot={foul_rate:.2f} "
                f"success%/shot={success_rate:.2f} "
                f"wall={wall:.1f}s"
            )

            print("[eval] running standard eval (canonical + random, continue_on_miss=False) ...")
            run_standard_eval(
                model,
                out_dir=out_dir,
                constrain_aim=bool(args.constrain_aim),
                wide_aim=bool(args.wide_aim),
                aim_width_mult=float(args.aim_width_mult),
                wide_aim_cap_deg=float(args.wide_aim_cap_deg),
                extra_features=bool(args.extra_features),
            )
        finally:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
            log_f.close()


if __name__ == "__main__":
    main()
