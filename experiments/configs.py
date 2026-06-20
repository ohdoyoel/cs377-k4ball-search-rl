"""Phase F alpha-sweep run configurations.

Cross-product of alpha grid x seeds. Run id ``a{alpha:.1f}_s{seed}`` is
stable across runs so output directories are deterministic.
"""

from __future__ import annotations

ALPHA_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
SEEDS = [0, 1, 2]


def get_configs() -> list[dict]:
    """Return the full 18-run config list (alpha outer, seed inner)."""
    return [
        {"alpha": a, "seed": s, "run_id": f"a{a:.1f}_s{s}"}
        for a in ALPHA_GRID
        for s in SEEDS
    ]
