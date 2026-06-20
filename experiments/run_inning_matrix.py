"""Sequential 6-config matrix runner for the multi-shot inning baseline.

6 configs = {sac, ppo} × {seed 0, 1, 2}. After each successful run, appends
a row to ``experiments/results/inning_summary.parquet``. Subprocesses are
foreground; do not run in background. Each SAC run is ~10 min wall.

Usage:
    uv run python experiments/run_inning_matrix.py
    uv run python experiments/run_inning_matrix.py --skip_existing
    uv run python experiments/run_inning_matrix.py --start 0 --end 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


ALGOS = ["sac", "ppo"]
SEEDS = [0, 1, 2]


def get_configs() -> list[dict]:
    """6 configs (algo outer, seed inner)."""
    return [
        {"algo": a, "seed": s, "run_id": f"{a}_s{s}"}
        for a in ALGOS for s in SEEDS
    ]


def _row_from_summary(cfg: dict, summary: dict) -> dict:
    return {
        "method": str(cfg["algo"]),
        "algo": str(cfg["algo"]),
        "seed": int(cfg["seed"]),
        "run_id": str(cfg["run_id"]),
        "mean_inning_score": float(summary.get("mean_inning_score", float("nan"))),
        "max_inning_score": int(summary.get("max_inning_score", 0)),
        "p_score_ge1": float(summary.get("p_score_ge1", float("nan"))),
        "p_score_ge3": float(summary.get("p_score_ge3", float("nan"))),
        "p_score_ge5": float(summary.get("p_score_ge5", float("nan"))),
        "mean_shots": float(summary.get("mean_shots", float("nan"))),
        "foul_rate": float(summary.get("foul_rate", float("nan"))),
        "mean_cushions": float(summary.get("mean_cushions", float("nan"))),
        "train_wall_s": float(summary.get("train_wall_s", float("nan"))),
        "wall_s": float(summary.get("wall_s", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the inning 2x3 matrix.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=6)
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--max_shots", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--out_dir", type=str, default="experiments/runs_inning")
    parser.add_argument(
        "--summary_path", type=str,
        default="experiments/results/inning_summary.parquet",
    )
    args = parser.parse_args()

    runs_dir = Path(args.out_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    full_configs = get_configs()
    total_full = len(full_configs)
    end_idx = max(args.start, min(args.end, total_full))
    configs = full_configs[args.start:end_idx]
    total = len(configs)
    print(f"[run_inning_matrix] running {total}/{total_full} (idx {args.start}:{end_idx}). "
          f"runs_dir={runs_dir} summary={summary_path}")

    rows: list[dict] = []
    if summary_path.exists():
        try:
            existing = pd.read_parquet(summary_path).to_dict("records")
            rows.extend(existing)
            print(f"[run_inning_matrix] resumed with {len(rows)} prior rows")
        except Exception:  # noqa: BLE001
            pass

    t_matrix0 = time.perf_counter()
    for i, cfg in enumerate(configs, start=1):
        run_id = cfg["run_id"]
        out_dir = runs_dir / run_id
        summary_json = out_dir / "summary.json"

        if args.skip_existing and summary_json.exists():
            with summary_json.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            rows = [r for r in rows if r.get("run_id") != run_id]
            rows.append(_row_from_summary(cfg, summary))
            print(f"[{i}/{total}] {run_id} SKIPPED (exists)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "experiments/run_inning_sac.py",
            "--algo", str(cfg["algo"]),
            "--seed", str(cfg["seed"]),
            "--total_steps", str(args.total_steps),
            "--max_shots", str(args.max_shots),
            "--eval_episodes", str(args.eval_episodes),
            "--out_dir", str(out_dir),
        ]
        t0 = time.perf_counter()
        elapsed_so_far = time.perf_counter() - t_matrix0
        avg = elapsed_so_far / max(1, i - 1) if i > 1 else float("nan")
        eta = avg * (total - i + 1) if i > 1 else float("nan")
        eta_str = f"{eta/60:.1f}min" if eta == eta else "?"
        print(f"[{i}/{total}] {run_id} starting (eta_remaining={eta_str})...",
              flush=True)

        try:
            proc = subprocess.run(cmd, check=False)
            wall = time.perf_counter() - t0
            ok = (proc.returncode == 0) and summary_json.exists()
        except Exception as e:  # noqa: BLE001
            wall = time.perf_counter() - t0
            print(f"[{i}/{total}] subprocess error: {e}")
            ok = False

        if not ok:
            print(f"[{i}/{total}] {run_id} FAILED, wall={wall:.1f}s")
            continue

        with summary_json.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        rows = [r for r in rows if r.get("run_id") != run_id]
        rows.append(_row_from_summary(cfg, summary))
        df = pd.DataFrame(rows)
        df.to_parquet(summary_path, engine="pyarrow", index=False)
        print(
            f"[{i}/{total}] {run_id} done. "
            f"mean_inning={summary['mean_inning_score']:.3f} "
            f"max={summary['max_inning_score']} "
            f"p>=1={summary['p_score_ge1']:.1f}% "
            f"p>=3={summary['p_score_ge3']:.1f}% "
            f"foul%={summary['foul_rate']:.2f} wall={wall:.1f}s",
            flush=True,
        )

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(summary_path, engine="pyarrow", index=False)

    elapsed = time.perf_counter() - t_matrix0
    print(f"[run_inning_matrix] {total} runs complete in {elapsed/60:.1f} min")
    print(f"[run_inning_matrix] summary -> {summary_path}")


if __name__ == "__main__":
    main()
