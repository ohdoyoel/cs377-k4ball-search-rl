"""Smoke test: run_inning_sac.py end-to-end with a tiny budget."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def test_run_inning_sac_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "smoke"
    cmd = [
        sys.executable,
        str(REPO / "experiments" / "run_inning_sac.py"),
        "--algo", "sac",
        "--seed", "0",
        "--total_steps", "2048",
        "--max_shots", "5",
        "--eval_episodes", "3",
        "--out_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=REPO)
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"

    for fname in ("config.json", "summary.json", "training_curve.csv",
                  "eval.parquet", "policy.zip", "run.log"):
        assert (out_dir / fname).exists(), f"missing {fname}"

    with (out_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["algo"] == "sac"
    assert summary["seed"] == 0
    assert "mean_inning_score" in summary
    assert "p_score_ge1" in summary

    eval_df = pd.read_parquet(out_dir / "eval.parquet")
    assert len(eval_df) == 3
    for col in ("inning_score", "n_shots", "mean_cushions",
                "fouled", "total_duration"):
        assert col in eval_df.columns
