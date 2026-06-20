"""Summarize VALIDATION §3 domain-knowledge ablation runs.

Reads existing summary.json files and prints the two report tables:

  3.1 plain vs constrain vs wide
  3.2 constrain vs reward-shaping variants
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("experiments/runs_inning_v2")
PLAIN_ROOT = ROOT / "baseline_plain"
DOMAIN_ROOT = ROOT / "domain_knowledge"
SEEDS = (0, 1, 2)


def _load(run_dir: Path) -> dict | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _series(run_name: str, root: Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        row = _load(root / f"{run_name}_s{seed}")
        if row is not None:
            rows.append(row)
    return rows


def _plain_series() -> list[dict]:
    rows = []
    for seed in SEEDS:
        row = _load(PLAIN_ROOT / f"sac_s{seed}")
        if row is not None:
            rows.append(row)
    return rows


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "NA"
    return f"{mean(xs):.3f}+/-{pstdev(xs):.3f}"


def _print_table(title: str, variants: list[tuple[str, list[dict]]]) -> None:
    print(f"\n{title}")
    print(
        "variant,n,seed_scores,mean+/-std,max,p3_mean,p5_mean,"
        "foul_per_shot_mean,success_per_shot_mean"
    )
    for name, rows in variants:
        if not rows:
            print(f"{name},0,NA,NA,NA,NA,NA,NA,NA")
            continue
        scores = [float(r["mean_inning_score"]) for r in rows]
        maxes = [float(r["max_inning_score"]) for r in rows]
        p3 = [float(r["p_score_ge3"]) for r in rows]
        p5 = [float(r["p_score_ge5"]) for r in rows]
        foul = [float(r["foul_rate"]) for r in rows]
        # success_rate added by newer eval runs; fall back to NA if absent.
        success = [float(r["success_rate"]) for r in rows if "success_rate" in r]
        seed_scores = ";".join(f"{v:.2f}" for v in scores)
        success_str = f"{mean(success):.1f}" if success else "NA"
        print(
            f"{name},{len(rows)},{seed_scores},{_fmt(scores)},"
            f"{max(maxes):.0f},{mean(p3):.1f},{mean(p5):.1f},"
            f"{mean(foul):.1f},{success_str}"
        )


def main() -> None:
    plain = _plain_series()
    variants = {
        "constrain": _series("constrain", DOMAIN_ROOT),
        "wide3": _series("wide3", DOMAIN_ROOT),
        "constrain_extra": _series("constrain_extra", DOMAIN_ROOT),
        "constrain_extra_gentle": _series("constrain_extra_gentle", DOMAIN_ROOT),
        "constrain_extra_nearmiss": _series("constrain_extra_nearmiss", DOMAIN_ROOT),
        "constrain_extra_foul1": _series("constrain_extra_foul1", DOMAIN_ROOT),
        "constrain_gentle": _series("constrain_gentle", DOMAIN_ROOT),
        "constrain_foul1": _series("constrain_foul1", DOMAIN_ROOT),
        "constrain_nearmiss": _series("constrain_nearmiss", DOMAIN_ROOT),
        "constrain_all": _series("constrain_all", DOMAIN_ROOT),
    }

    _print_table(
        "VALIDATION 3.1 action constraint",
        [
            ("plain", plain),
            ("constrain", variants["constrain"]),
            ("wide3", variants["wide3"]),
        ],
    )
    _print_table(
        "VALIDATION 3.2 extra features",
        [
            ("constrain", variants["constrain"]),
            ("constrain_extra", variants["constrain_extra"]),
        ],
    )
    _print_table(
        "VALIDATION 3.3 reward shaping under constrain + extra features",
        [
            ("constrain_extra", variants["constrain_extra"]),
            ("constrain_extra_gentle", variants["constrain_extra_gentle"]),
            ("constrain_extra_nearmiss", variants["constrain_extra_nearmiss"]),
            ("constrain_extra_foul1", variants["constrain_extra_foul1"]),
        ],
    )
    _print_table(
        "VALIDATION 3.4 reward shaping under constrained aim (constrain-only, SUPERSEDED)",
        [
            ("constrain", variants["constrain"]),
            ("constrain_gentle", variants["constrain_gentle"]),
            ("constrain_foul1", variants["constrain_foul1"]),
            ("constrain_nearmiss", variants["constrain_nearmiss"]),
            ("constrain_all", variants["constrain_all"]),
        ],
    )


if __name__ == "__main__":
    main()
