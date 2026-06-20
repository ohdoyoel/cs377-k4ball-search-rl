"""Train SAC with V(s,a) RM bonus added to the per-step reward.

Reward = score + lambda * sigmoid(V_rm(s_t, a_t)). Evaluation uses
the original env reward (no bonus) so headline numbers are directly
comparable to the plain-SAC and SAC+RM(s) baselines.
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

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402
from billiards.wrappers.vsa_rm_env import VSaRMEnv  # noqa: E402


T_MAX = 12.0
GAMMA = 0.99
LR = 3e-4
SAC_BATCH = 256
SAC_BUFFER = 200_000
SAC_LEARNING_STARTS = 1_000
EVAL_SEED_OFFSET = 50_000


class _Tee(io.TextIOBase):
    def __init__(self, *streams): self._streams = streams
    def write(self, s):
        for st in self._streams:
            try: st.write(s); st.flush()
            except Exception: pass
        return len(s)
    def flush(self):
        for st in self._streams:
            try: st.flush()
            except Exception: pass


def make_train_env(seed: int, max_shots: int, rm_path: str, lam: float):
    base = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots)
    env = RandomStartInningEnv(base)
    env = VSaRMEnv(env, rm_path=rm_path, lam=lam)
    env = Monitor(env, info_keywords=("cushion_hits", "fouled", "score",
                                      "env_reward", "rm_bonus"))
    env.reset(seed=seed)
    return env


def make_eval_env(env_kind: str, max_shots: int):
    base = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots)
    if env_kind == "canonical":
        return base
    if env_kind == "random":
        return RandomStartInningEnv(base)
    raise ValueError(env_kind)


def evaluate(model, env_kind: str, n: int, seed_base: int, max_shots: int) -> dict:
    env = make_eval_env(env_kind, max_shots)
    rows = []
    for ep in range(n):
        obs, _ = env.reset(seed=seed_base + ep)
        shots, fouled = 0, False
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
        "p_score_ge5": float((df["inning_score"] >= 5).mean() * 100.0),
        "mean_shots": float(df["n_shots"].mean()),
        "foul_rate": float(df["fouled"].mean() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--rm_path", type=str,
                        default="experiments/rm_data_vsa/vsa_rm.pt")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str,
                        default="experiments/runs_inning_random/sac_vsarm_200k_s0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "run.log").open("w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_f)
    err_tee = _Tee(sys.__stderr__, log_f)

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t0 = time.perf_counter()
        print(f"[sac+vsarm] seed={args.seed} steps={args.total_steps} "
              f"lam={args.lam} rm={args.rm_path}")
        set_random_seed(int(args.seed))
        env = make_train_env(int(args.seed), int(args.max_shots),
                             rm_path=args.rm_path, lam=float(args.lam))
        model = SAC("MlpPolicy", env,
                    learning_rate=LR, gamma=GAMMA,
                    batch_size=SAC_BATCH, buffer_size=SAC_BUFFER,
                    learning_starts=SAC_LEARNING_STARTS,
                    seed=int(args.seed), verbose=0)
        model.learn(total_timesteps=int(args.total_steps), log_interval=50,
                    progress_bar=True)
        wall = time.perf_counter() - t0
        print(f"[sac+vsarm] train done in {wall:.1f}s")
        model.save(out_dir / "policy.zip")

        results = {"train_wall_s": wall, "seed": int(args.seed),
                   "total_steps": int(args.total_steps),
                   "lam": float(args.lam), "rm_path": args.rm_path}
        for ev in ("canonical", "random"):
            r = evaluate(model, ev, int(args.eval_episodes),
                         seed_base=int(args.seed) + EVAL_SEED_OFFSET,
                         max_shots=int(args.max_shots))
            results[f"eval_{ev}"] = r
            print(f"[sac+vsarm] eval[{ev}] mean={r['mean_inning_score']:.3f} "
                  f"max={r['max_inning_score']} p>=1={r['p_score_ge1']:.1f}% "
                  f"foul%={r['foul_rate']:.1f}")
        with (out_dir / "summary.json").open("w") as f:
            json.dump(results, f, indent=2)
        print(f"[sac+vsarm] DONE wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
