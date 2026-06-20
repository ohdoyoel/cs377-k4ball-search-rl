"""Multi-seed actor + simulator h=2 lookahead — chase infinite billiards.

For each shot t, given current state s_t:
  1. PROPOSE: each of N seed policies emits ``K_per_policy`` candidates
     (1 deterministic + (K_per_policy-1) stochastic samples), giving
     ``K1 = N * K_per_policy`` candidates total.
  2. VERIFY1: simulate each candidate from the snapshot of s_t, recording
     (r1, s') for each. Keep top-M1 by r1.
  3. RECURSE: for each top-M1 candidate a1, propose K2 candidates from
     s' (using the same multi-seed ensemble), simulate each, keep the
     best r2. v(a1) = r1 + gamma * r2_max.
  4. EXECUTE: argmax_a1 v(a1).

No RM is used — the base policy was trained with extra_features=True
(obs_dim=32) and the existing V(s,a) RM v3 was trained with extra
features off (obs_dim=28). They can't share the obs, so the simulator
itself is the verifier.

Env config matches the base policy's training env exactly:
  constrain_aim=True, extra_features=True, foul_penalty=1.0,
  random_start=True, continue_on_miss=True, max_shots=<large>.

Outputs ``out_dir/summary.json`` with per-episode score/shots and
hyperparams, plus ``innings.parquet`` for downstream analysis.
"""

from __future__ import annotations

import argparse
import copy
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

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402

T_MAX = 12.0


def make_env(max_shots: int, continue_on_miss: bool = True,
             random_start: bool = True, foul_penalty: float = 1.0,
             gentle_shot: bool = False, setup_shaping: bool = False):
    base = Billiards4BallInningEnv(
        t_max=T_MAX, max_shots=max_shots,
        continue_on_miss=continue_on_miss,
        constrain_aim=True, extra_features=True,
        foul_penalty=foul_penalty,
        gentle_shot=gentle_shot,
        setup_shaping=setup_shaping,
        setup_alpha=0.05, setup_scale=0.3,
    )
    return RandomStartInningEnv(base) if random_start else base


def _snap(env):
    inner = env.unwrapped
    return (
        copy.deepcopy(inner._state),
        inner._shot_index,
        inner._cumulative_score,
        inner._cumulative_t,
        list(inner._shot_trajectories),
        list(inner._shot_offsets),
        list(inner._inning_log_records),
    )


def _restore(env, s):
    inner = env.unwrapped
    inner._state = copy.deepcopy(s[0])
    inner._shot_index = s[1]
    inner._cumulative_score = s[2]
    inner._cumulative_t = s[3]
    inner._shot_trajectories = list(s[4])
    inner._shot_offsets = list(s[5])
    inner._inning_log_records = list(s[6])


def _propose(policies: list, obs: np.ndarray, k_per_policy: int) -> np.ndarray:
    """Each policy emits 1 deterministic + (k_per_policy-1) stochastic
    candidates; concatenate."""
    out = []
    for m in policies:
        a_det, _ = m.predict(obs, deterministic=True)
        out.append(np.asarray(a_det, np.float32).reshape(-1))
        for _ in range(k_per_policy - 1):
            a_sto, _ = m.predict(obs, deterministic=False)
            out.append(np.asarray(a_sto, np.float32).reshape(-1))
    return np.stack(out, axis=0)


def search_step_h2(env, obs: np.ndarray, policies: list,
                   k1_pp: int, m1: int, k2_pp: int, m2: int, gamma: float):
    cand1 = _propose(policies, obs, k1_pp)
    snap0 = _snap(env)

    # Level 1: simulate every candidate, record (r1, s', terminal flag).
    r1_arr = np.zeros(len(cand1), dtype=np.float64)
    obs1_list: list[np.ndarray | None] = [None] * len(cand1)
    terminal: list[bool] = [False] * len(cand1)
    for i, a in enumerate(cand1):
        _restore(env, snap0)
        try:
            o1, r1, t1, tr1, _ = env.step(a.astype(np.float32))
        except Exception:
            r1, t1, tr1, o1 = 0.0, True, True, None
        r1_arr[i] = float(r1)
        obs1_list[i] = None if o1 is None else np.asarray(o1, np.float32)
        terminal[i] = bool(t1 or tr1)

    # Pick top-M1 by r1 (ties broken by simulation order).
    top1 = np.argsort(-r1_arr)[:m1]

    # Level 2: recurse on each top-M1 leaf.
    best_val = -np.inf
    best_action = cand1[top1[0]]
    for i in top1:
        if terminal[i] or obs1_list[i] is None:
            v = r1_arr[i]
        else:
            cand2 = _propose(policies, obs1_list[i], k2_pp)
            snap1 = _snap(env)  # snap of root s_t
            # Need to land at s' first to simulate a2 from there.
            _restore(env, snap0)
            env.step(cand1[i].astype(np.float32))
            snap_s1 = _snap(env)
            r2_best = 0.0
            for a2 in cand2[:m2 if m2 else len(cand2)]:
                _restore(env, snap_s1)
                try:
                    _, r2, _, _, _ = env.step(a2.astype(np.float32))
                except Exception:
                    r2 = 0.0
                if r2 > r2_best:
                    r2_best = float(r2)
            v = r1_arr[i] + gamma * r2_best
        if v > best_val:
            best_val = v
            best_action = cand1[i]

    _restore(env, snap0)
    return best_action


def run_inning(env, policies, k1_pp, m1, k2_pp, m2, gamma, seed, max_shots,
               log_every: int = 50):
    obs, _ = env.reset(seed=seed)
    shots, fouled, n_fouls = 0, False, 0
    t0 = time.perf_counter()
    while True:
        action = search_step_h2(env, np.asarray(obs, np.float32),
                                policies, k1_pp, m1, k2_pp, m2, gamma)
        obs, _, term, trunc, info = env.step(action.astype(np.float32))
        shots += 1
        if bool(info.get("fouled", False)):
            fouled = True
            n_fouls += 1
        if shots % log_every == 0:
            score = int(env.unwrapped.cumulative_score)
            dt = time.perf_counter() - t0
            print(f"  ... shot={shots} score={score} foul={fouled} "
                  f"wall={dt/60:.1f}min {dt/shots:.2f}s/shot",
                  flush=True)
        if term or trunc or shots >= max_shots:
            break
    return int(env.unwrapped.cumulative_score), shots, fouled, n_fouls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+",
                   default=["exp_3_train_step/runs/constrain_extra_foul1/sac_s0/policy_2M.zip",
                            "exp_3_train_step/runs/constrain_extra_foul1/sac_s1/policy_2M.zip",
                            "exp_3_train_step/runs/constrain_extra_foul1/sac_s2/policy_2M.zip"])
    p.add_argument("--k1_per_policy", type=int, default=50,
                   help="Per-policy candidate count at root.")
    p.add_argument("--m1", type=int, default=10,
                   help="Top-M1 kept by r1 for level-2 expansion.")
    p.add_argument("--k2_per_policy", type=int, default=10,
                   help="Per-policy candidate count at level 2.")
    p.add_argument("--m2", type=int, default=5,
                   help="Simulate top-M2 candidates at level 2.")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--max_shots", type=int, default=1000)
    p.add_argument("--seed_base", type=int, default=99000)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Explicit episode seeds; overrides seed_base/n_episodes "
                        "(e.g. re-run only the seeds that hit the shot cap).")
    p.add_argument("--continue_on_miss", action="store_true", default=True)
    p.add_argument("--no_continue_on_miss", action="store_false",
                   dest="continue_on_miss",
                   help="Terminate episode on first miss/foul (canonical eval).")
    p.add_argument("--random_start", action="store_true", default=True)
    p.add_argument("--no_random_start", action="store_false",
                   dest="random_start",
                   help="Use canonical fixed starting rack instead of random.")
    p.add_argument("--foul_penalty", type=float, default=1.0)
    p.add_argument("--gentle_shot", action="store_true",
                   help="Match Brian's fast_long_fp02 training env.")
    p.add_argument("--setup_shaping", action="store_true",
                   help="Match Brian's fast_long_fp02 training env.")
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = [SAC.load(str(p), device="cpu") for p in args.policies]
    env = make_env(max_shots=int(args.max_shots),
                   continue_on_miss=bool(args.continue_on_miss),
                   random_start=bool(args.random_start),
                   foul_penalty=float(args.foul_penalty),
                   gentle_shot=bool(args.gentle_shot),
                   setup_shaping=bool(args.setup_shaping))

    print(f"[multi_h2] policies={len(policies)} K1_pp={args.k1_per_policy} "
          f"M1={args.m1} K2_pp={args.k2_per_policy} M2={args.m2} "
          f"n_ep={args.n_episodes} max_shots={args.max_shots} "
          f"continue_on_miss={args.continue_on_miss} "
          f"random_start={args.random_start}",
          flush=True)

    seeds = (list(args.seeds) if args.seeds
             else [int(args.seed_base) + ep for ep in range(int(args.n_episodes))])

    rows = []
    t0 = time.perf_counter()
    for ep, seed in enumerate(seeds):
        ep_t = time.perf_counter()
        print(f"--- episode {ep} (seed={seed}) ---", flush=True)
        score, shots, fouled, n_fouls = run_inning(
            env, policies,
            k1_pp=int(args.k1_per_policy), m1=int(args.m1),
            k2_pp=int(args.k2_per_policy), m2=int(args.m2),
            gamma=float(args.gamma),
            seed=seed, max_shots=int(args.max_shots),
        )
        ep_wall = time.perf_counter() - ep_t
        rows.append({"ep_idx": ep, "seed": seed, "score": score,
                     "shots": shots, "fouled": fouled, "n_fouls": n_fouls,
                     "wall_s": ep_wall})
        print(f"=== ep{ep}: score={score} shots={shots} foul={fouled} "
              f"wall={ep_wall/60:.1f}min ({ep_wall/shots:.2f}s/shot)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "innings.parquet", engine="pyarrow", index=False)

    summary = {
        "k1_per_policy": int(args.k1_per_policy),
        "m1": int(args.m1),
        "k2_per_policy": int(args.k2_per_policy),
        "m2": int(args.m2),
        "k1_total": int(args.k1_per_policy) * len(policies),
        "k2_total": int(args.k2_per_policy) * len(policies),
        "n_policies": len(policies),
        "gamma": float(args.gamma),
        "n_episodes": len(seeds),
        "seeds": seeds,
        "max_shots": int(args.max_shots),
        "continue_on_miss": bool(args.continue_on_miss),
        "random_start": bool(args.random_start),
        "policies": args.policies,
        "scores": df["score"].tolist(),
        "shots": df["shots"].tolist(),
        "fouled": df["fouled"].tolist(),
        "wall_s_per_episode": df["wall_s"].tolist(),
        "mean_score": float(df["score"].mean()),
        "std_score": float(df["score"].std(ddof=0)),
        "max_score": int(df["score"].max()),
        "mean_shots": float(df["shots"].mean()),
        "max_shots_episode": int(df["shots"].max()),
        "n_fouls": df["n_fouls"].tolist(),
        "foul_rate": float(df["fouled"].mean() * 100),
        "foul_per_shot": float(df["n_fouls"].sum() / max(1, df["shots"].sum()) * 100),
        "success_per_shot": float(df["score"].sum() / max(1, df["shots"].sum()) * 100),
        "wall_s_total": time.perf_counter() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[multi_h2] DONE mean_score={summary['mean_score']:.1f} "
          f"max={summary['max_score']} mean_shots={summary['mean_shots']:.0f} "
          f"max_shots_ep={summary['max_shots_episode']} "
          f"wall={summary['wall_s_total']/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
