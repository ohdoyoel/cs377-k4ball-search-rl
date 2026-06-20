"""Standard post-training evaluation for a single policy.

Always evaluates under two fixed conditions regardless of how the policy
was trained:

  canonical : continue_on_miss=False, random_start=False  (fixed layout)
  random    : continue_on_miss=False, random_start=True   (randomised layout)

Outputs (written into out_dir):
  eval_canonical.parquet   one row per episode
  eval_random.parquet      one row per episode
  eval_summary.json        aggregated stats for both modes

Can be imported and called from run_inning_sac.py, or run standalone:
    uv run python experiments/eval_policy.py \\
        --policy experiments/runs_inning/sac_rs_200k_s0/policy.zip \\
        --algo sac --constrain_aim --extra_features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402

T_MAX = 12.0
MAX_SHOTS = 20   # generous cap; episode ends on first miss/foul anyway


def _roll_episodes(
    model,
    n_episodes: int,
    seed_base: int,
    constrain_aim: bool,
    wide_aim: bool,
    aim_width_mult: float,
    wide_aim_cap_deg: float,
    extra_features: bool,
    random_start: bool,
) -> pd.DataFrame:
    base = Billiards4BallInningEnv(
        t_max=T_MAX,
        max_shots=MAX_SHOTS,
        continue_on_miss=False,
        constrain_aim=constrain_aim,
        wide_aim=wide_aim,
        aim_width_mult=aim_width_mult,
        wide_aim_cap_deg=wide_aim_cap_deg,
        extra_features=extra_features,
    )
    env = RandomStartInningEnv(base) if random_start else base

    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        fouled = False
        n_fouls = 0
        n_scores = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32).reshape(-1)
            )
            shot_score = int(info.get("score", 0))
            shot_foul = bool(info.get("fouled", False))
            if shot_foul:
                fouled = True
                n_fouls += 1
            elif shot_score > 0:
                n_scores += 1
            if terminated or truncated:
                break
        rows.append({
            "ep_idx": ep,
            "seed": seed_base + ep,
            "inning_score": int(base.cumulative_score),
            "n_shots": int(base.shot_index),
            "n_fouls": int(n_fouls),
            "n_scores": int(n_scores),
            "fouled": bool(fouled),
        })
    return pd.DataFrame(rows)


def _stats(df: pd.DataFrame) -> dict:
    s = df["inning_score"].to_numpy()
    total_shots = float(df["n_shots"].sum())
    return {
        "mean_score": float(s.mean()),
        "max_score": int(s.max()),
        "p_ge1": float((s >= 1).mean()),
        "p_ge3": float((s >= 3).mean()),
        "p_ge5": float((s >= 5).mean()),
        # Per-shot probabilities (pooled over all shots), as fractions.
        "foul_rate": float(df["n_fouls"].sum()) / max(1.0, total_shots),
        "success_rate": float(df["n_scores"].sum()) / max(1.0, total_shots),
        "n_episodes": len(df),
    }


def run_standard_eval(
    model,
    out_dir: Path,
    constrain_aim: bool,
    extra_features: bool,
    wide_aim: bool = False,
    aim_width_mult: float = 3.0,
    wide_aim_cap_deg: float = 60.0,
    n_episodes: int = 200,
    seed_base: int = 99000,
) -> dict:
    """Evaluate model under canonical and random-start conditions.

    Saves parquet files and eval_summary.json into out_dir.
    Returns the summary dict.
    """
    out_dir = Path(out_dir)
    summary: dict = {}

    for mode, random_start in (("canonical", False), ("random", True)):
        df = _roll_episodes(
            model,
            n_episodes=n_episodes,
            seed_base=seed_base,
            constrain_aim=constrain_aim,
            wide_aim=wide_aim,
            aim_width_mult=aim_width_mult,
            wide_aim_cap_deg=wide_aim_cap_deg,
            extra_features=extra_features,
            random_start=random_start,
        )
        df.to_parquet(out_dir / f"eval_{mode}.parquet", engine="pyarrow", index=False)
        stats = _stats(df)
        summary[mode] = stats
        print(
            f"[eval] {mode:9s}  mean={stats['mean_score']:.3f}  "
            f"max={stats['max_score']}  "
            f"p1={stats['p_ge1']*100:.1f}%  "
            f"p3={stats['p_ge3']*100:.1f}%  "
            f"p5={stats['p_ge5']*100:.1f}%  "
            f"foul/shot={stats['foul_rate']*100:.1f}%  "
            f"success/shot={stats['success_rate']*100:.1f}%"
        )

    with (out_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------- standalone

def main() -> None:
    parser = argparse.ArgumentParser()
    # convenience: point at a run directory and let it read config.json
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to a run directory (reads config.json automatically)")
    # or specify policy + flags manually
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--algo", type=str, choices=("sac", "td3"), default="sac")
    parser.add_argument("--constrain_aim", action="store_true")
    parser.add_argument("--wide_aim", action="store_true")
    parser.add_argument("--aim_width_mult", type=float, default=3.0)
    parser.add_argument("--wide_aim_cap_deg", type=float, default=60.0)
    parser.add_argument("--extra_features", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=200)
    args = parser.parse_args()

    from stable_baselines3 import SAC, TD3  # noqa: E402

    if args.run_dir:
        run_dir = _REPO_ROOT / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            print(f"[eval] ERROR: no config.json in {run_dir}")
            sys.exit(1)
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        algo = cfg.get("algo", "sac")
        constrain_aim = bool(cfg.get("constrain_aim", False))
        wide_aim = bool(cfg.get("wide_aim", False))
        aim_width_mult = float(cfg.get("aim_width_mult", 3.0))
        wide_aim_cap_deg = float(cfg.get("wide_aim_cap_deg", 60.0))
        extra_features = bool(cfg.get("extra_features", False))
        policy_path = run_dir / "policy.zip"
        out_dir = run_dir
    else:
        if not args.policy:
            print("[eval] ERROR: provide --run_dir or --policy")
            sys.exit(1)
        policy_path = Path(args.policy)
        out_dir = policy_path.parent
        algo = args.algo
        constrain_aim = args.constrain_aim
        wide_aim = args.wide_aim
        aim_width_mult = float(args.aim_width_mult)
        wide_aim_cap_deg = float(args.wide_aim_cap_deg)
        extra_features = args.extra_features

    loader = SAC if algo == "sac" else TD3
    model = loader.load(str(policy_path), device="cpu")

    print(
        f"[eval] {out_dir.name}  algo={algo}  constrain_aim={constrain_aim} "
        f"wide_aim={wide_aim}  extra_features={extra_features}"
    )
    run_standard_eval(
        model,
        out_dir=out_dir,
        constrain_aim=constrain_aim,
        wide_aim=wide_aim,
        aim_width_mult=aim_width_mult,
        wide_aim_cap_deg=wide_aim_cap_deg,
        extra_features=extra_features,
        n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    main()
