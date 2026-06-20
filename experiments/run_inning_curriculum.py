"""Smoke trainer for SAC + curriculum-start, plain (no aim/shape/gentle).

Goal: verify whether progressively easier initial layouts let plain SAC
reach a meaningful inning score on the same architecture and reward used
in run_inning_random.py.

Curriculum schedule is a fixed step-based linear ramp on difficulty
∈ [0, 1] managed by a callback; auto-advance based on rolling reward is
intentionally avoided to keep behaviour deterministic for smoke runs.
"""

from __future__ import annotations

import argparse
import contextlib
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

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.curriculum_start_env import CurriculumStartInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402


T_MAX = 12.0
GAMMA = 0.99
LR = 3e-4
SAC_BATCH = 256
SAC_BUFFER = 200_000
SAC_LEARNING_STARTS = 1_000
EVAL_SEED_OFFSET = 50_000


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s); st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


def _make_train_env(seed: int, max_shots: int, difficulty: float):
    base = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots)
    env = CurriculumStartInningEnv(base, difficulty=difficulty)
    env = Monitor(env, info_keywords=("cushion_hits", "fouled", "score"))
    env.reset(seed=seed)
    return env


def _make_eval_env(env_kind: str, max_shots: int):
    base = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots)
    if env_kind == "random":
        return RandomStartInningEnv(base)
    if env_kind.startswith("curriculum"):
        # parse "curriculum:0.3" form; default to 0.0
        d = 0.0
        if ":" in env_kind:
            d = float(env_kind.split(":", 1)[1])
        return CurriculumStartInningEnv(base, difficulty=d)
    raise ValueError(f"unknown env_kind={env_kind}")


class CurriculumRampCallback(BaseCallback):
    """Linearly ramp the wrapped env's difficulty from d0 to d1 over the run."""

    def __init__(self, env, d0: float, d1: float, total_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self._env = env
        self._d0 = float(d0)
        self._d1 = float(d1)
        self._total = int(total_steps)
        # cache the wrapper instance (Monitor -> CurriculumStartInningEnv)
        self._wrapper = None
        cur = env
        while cur is not None:
            if isinstance(cur, CurriculumStartInningEnv):
                self._wrapper = cur
                break
            cur = getattr(cur, "env", None)

    def _on_step(self) -> bool:
        if self._wrapper is None:
            return True
        frac = min(1.0, max(0.0, self.num_timesteps / max(1, self._total)))
        d = self._d0 + (self._d1 - self._d0) * frac
        self._wrapper.set_difficulty(d)
        return True


def evaluate(model, env_kind: str, n: int, seed_base: int, max_shots: int) -> dict:
    env = _make_eval_env(env_kind, max_shots)
    rows = []
    for ep in range(n):
        obs, _ = env.reset(seed=seed_base + ep)
        score, shots, fouled = 0, 0, False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(np.asarray(action, np.float32).reshape(-1))
            shots += 1
            if bool(info.get("fouled", False)):
                fouled = True
            if term or trunc:
                break
        rows.append({
            "inning_score": int(env.unwrapped.cumulative_score),
            "n_shots": int(shots),
            "fouled": bool(fouled),
        })
    df = pd.DataFrame(rows)
    return {
        "n": int(n),
        "mean_inning_score": float(df["inning_score"].mean()),
        "max_inning_score": int(df["inning_score"].max()),
        "p_score_ge1": float((df["inning_score"] >= 1).mean() * 100.0),
        "p_score_ge3": float((df["inning_score"] >= 3).mean() * 100.0),
        "mean_shots": float(df["n_shots"].mean()),
        "foul_rate": float(df["fouled"].mean() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Curriculum SAC smoke runner.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--d_start", type=float, default=0.0,
                        help="initial difficulty (0 = easiest)")
    parser.add_argument("--d_end", type=float, default=1.0,
                        help="final difficulty (1 = full random)")
    parser.add_argument("--eval_envs", type=str,
                        default="curriculum:0.0,curriculum:0.5,random")
    parser.add_argument("--out_dir", type=str,
                        default="experiments/runs_inning_random/sac_curriculum_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_f = log_path.open("w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_f)
    err_tee = _Tee(sys.__stderr__, log_f)

    eval_envs = [e.strip() for e in args.eval_envs.split(",") if e.strip()]

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t0 = time.perf_counter()
        print(f"[curriculum] seed={args.seed} total_steps={args.total_steps} "
              f"d_start={args.d_start} d_end={args.d_end} eval_envs={eval_envs}")

        set_random_seed(int(args.seed))
        env = _make_train_env(int(args.seed), int(args.max_shots),
                              difficulty=float(args.d_start))
        model = SAC("MlpPolicy", env,
                    learning_rate=LR, gamma=GAMMA,
                    batch_size=SAC_BATCH, buffer_size=SAC_BUFFER,
                    learning_starts=SAC_LEARNING_STARTS,
                    seed=int(args.seed), verbose=0)
        cb = CurriculumRampCallback(env, args.d_start, args.d_end, args.total_steps)
        model.learn(total_timesteps=int(args.total_steps), callback=cb,
                    log_interval=50, progress_bar=True)
        train_wall = time.perf_counter() - t0
        print(f"[curriculum] train done in {train_wall:.1f}s")

        model.save(out_dir / "policy.zip")

        results = {"train_wall_s": train_wall, "seed": int(args.seed),
                   "total_steps": int(args.total_steps),
                   "d_start": float(args.d_start), "d_end": float(args.d_end)}
        for ev in eval_envs:
            r = evaluate(model, ev, int(args.eval_episodes),
                         seed_base=int(args.seed) + EVAL_SEED_OFFSET,
                         max_shots=int(args.max_shots))
            results[f"eval_{ev}"] = r
            print(f"[curriculum] eval[{ev}] mean={r['mean_inning_score']:.3f} "
                  f"max={r['max_inning_score']} p>=1={r['p_score_ge1']:.1f}% "
                  f"foul%={r['foul_rate']:.1f}")

        with (out_dir / "summary.json").open("w") as f:
            json.dump(results, f, indent=2)
        print(f"[curriculum] DONE wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
