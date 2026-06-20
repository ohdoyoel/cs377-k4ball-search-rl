"""Action-space curriculum SAC.

Trains SAC over the same 5D action space (sin θ, cos θ, p, a, b) but in
three stages, where progressively more action dimensions are released to
the policy:

  stage 1: (sin θ, cos θ) — direction only, p=1, a=b=0 forced
  stage 2: (sin θ, cos θ, p) — direction + power, a=b=0 forced
  stage 3: full 5D

The same SAC model is reused across stages (warm start). Between stages
we save the policy, reload it, and bump the wrapper's stage flag. The
policy head is unchanged, so all stage transitions are weight-preserving.

No reward shaping or aim constraint. Random-start layout throughout.
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
from billiards.wrappers.action_stage_env import ActionStageWrapper  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402


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


def make_train_env(seed: int, max_shots: int, stage: int):
    base = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots,
                                    angle_sincos=True)
    env = RandomStartInningEnv(base)
    env = ActionStageWrapper(env, stage=stage)
    env = Monitor(env, info_keywords=("cushion_hits", "fouled", "score",
                                       "action_stage"))
    env.reset(seed=seed)
    return env


def make_eval_env(env_kind: str, max_shots: int, stage: int):
    base = Billiards4BallInningEnv(t_max=T_MAX, max_shots=max_shots,
                                    angle_sincos=True)
    if env_kind == "canonical":
        inner = base
    elif env_kind == "random":
        inner = RandomStartInningEnv(base)
    else:
        raise ValueError(env_kind)
    return ActionStageWrapper(inner, stage=stage)


def evaluate(model, env_kind: str, n: int, seed_base: int, max_shots: int,
             stage: int) -> dict:
    env = make_eval_env(env_kind, max_shots, stage=stage)
    rows = []
    for ep in range(n):
        obs, _ = env.reset(seed=seed_base + ep)
        shots, fouled = 0, False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(
                np.asarray(action, np.float32).reshape(-1))
            shots += 1
            if bool(info.get("fouled", False)):
                fouled = True
            if term or trunc:
                break
        rows.append({"inning_score": int(env.unwrapped.cumulative_score),
                     "n_shots": int(shots), "fouled": bool(fouled)})
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
    parser.add_argument("--steps_s1", type=int, default=100_000)
    parser.add_argument("--steps_s2", type=int, default=100_000)
    parser.add_argument("--steps_s3", type=int, default=100_000)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--out_dir", type=str,
                        default="experiments/runs_inning_random/sac_act_curr_s0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "run.log").open("w", encoding="utf-8")
    tee = _Tee(sys.__stdout__, log_f)
    err_tee = _Tee(sys.__stderr__, log_f)

    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
        t_total = time.perf_counter()
        set_random_seed(int(args.seed))

        results = {"seed": int(args.seed),
                   "steps_s1": int(args.steps_s1),
                   "steps_s2": int(args.steps_s2),
                   "steps_s3": int(args.steps_s3)}

        model = None
        for stage, steps in ((1, args.steps_s1),
                              (2, args.steps_s2),
                              (3, args.steps_s3)):
            if steps <= 0:
                continue
            print(f"\n[act_curr] === STAGE {stage} ({steps} steps) ===")
            t0 = time.perf_counter()

            env = make_train_env(int(args.seed) + stage * 1000,
                                  int(args.max_shots), stage=stage)
            if model is None:
                model = SAC("MlpPolicy", env,
                            learning_rate=LR, gamma=GAMMA,
                            batch_size=SAC_BATCH, buffer_size=SAC_BUFFER,
                            learning_starts=SAC_LEARNING_STARTS,
                            seed=int(args.seed), verbose=0)
            else:
                # Reuse weights: just point model at the new env (which has
                # the higher-stage wrapper). Replay buffer is preserved.
                model.set_env(env)

            model.learn(total_timesteps=int(steps), log_interval=50,
                        progress_bar=True, reset_num_timesteps=False)
            wall = time.perf_counter() - t0
            print(f"[act_curr] stage {stage} done in {wall:.1f}s")

            stage_dir = out_dir / f"stage{stage}"
            stage_dir.mkdir(exist_ok=True)
            model.save(stage_dir / "policy.zip")

            stage_results = {"stage": stage, "steps": int(steps),
                             "wall_s": wall}
            for ev in ("random",):
                r = evaluate(model, ev, int(args.eval_episodes),
                             seed_base=int(args.seed) + EVAL_SEED_OFFSET,
                             max_shots=int(args.max_shots),
                             stage=stage)
                stage_results[f"eval_{ev}"] = r
                print(f"[act_curr] stage {stage} eval[{ev}] "
                      f"mean={r['mean_inning_score']:.3f} "
                      f"max={r['max_inning_score']} "
                      f"p>=1={r['p_score_ge1']:.1f}% "
                      f"foul%={r['foul_rate']:.1f}")
            results[f"stage{stage}"] = stage_results

        results["total_wall_s"] = time.perf_counter() - t_total
        with (out_dir / "summary.json").open("w") as f:
            json.dump(results, f, indent=2)
        print(f"[act_curr] DONE total wall={results['total_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
