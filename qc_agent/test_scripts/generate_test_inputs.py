#!/usr/bin/env python3
"""
Generate synthetic test h5ad files from the base file.
Each variant modifies one known property so ground truth is exact.

Cases:
  base             – original file (all steps done, raw accessible)
  raw_counts_in_X  – X replaced with raw counts (normalisation pending)
  no_hvg           – highly_variable removed (feature_selection pending)
  no_raw_counts    – raw=None, X still log-normalised (raw warning expected)
  scaled_x         – X z-scored, raw=None (scaled warning + raw warning)

Usage:
    python generate_test_inputs.py --base /path/to/file.h5ad
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BASE = (
    _PROJECT_ROOT / "code/05_scenicplus/data_inputs/SEAAD_MTG_microglia_rna.h5ad"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=str(_DEFAULT_BASE),
        help="Path to the base h5ad file",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "test_inputs"),
        help="Output directory",
    )
    args = parser.parse_args()

    try:
        import anndata as ad
    except ImportError:
        print("Error: anndata not installed.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {args.base}")
    base = ad.read_h5ad(args.base)
    print(f"  Shape: {base.shape} | raw.X: {base.raw is not None} | "
          f"X min={float(base.X.min()):.3f} max={float(base.X.max()):.3f}")

    # ------------------------------------------------------------------
    # Case 1: base — copy as-is
    # Expected: normalisation=already_done, HVG=already_done, raw accessible
    # ------------------------------------------------------------------
    print("\nCase 1: base")
    base.write_h5ad(out / "base.h5ad")
    print("  Saved base.h5ad")

    # ------------------------------------------------------------------
    # Case 2: raw_counts_in_X — X = raw counts, raw.X still present
    # Expected: normalisation=pending, HVG=already_done, raw accessible
    # ------------------------------------------------------------------
    print("\nCase 2: raw_counts_in_X")
    v2 = base.copy()
    # raw.X has more genes than X (pre-filter), so we can't copy it directly.
    # Instead simulate raw counts by undoing log1p + rescaling — gives integer-like
    # values with high max, which the inspector classifies as "likely_raw_counts".
    X = v2.X.toarray() if sp.issparse(v2.X) else np.asarray(v2.X, dtype=np.float64)
    X_counts = np.round(np.expm1(X) * 1e4).astype(np.float32)  # undo log1p, rescale
    v2.X = sp.csr_matrix(X_counts)
    print(f"  Simulated raw counts via expm1*1e4 | "
          f"min={float(v2.X.min()):.0f} max={float(v2.X.max()):.0f} "
          f"(raw.X still present for accessibility check)")
    v2.write_h5ad(out / "raw_counts_in_X.h5ad")
    print("  Saved raw_counts_in_X.h5ad")

    # ------------------------------------------------------------------
    # Case 3: no_hvg — remove highly_variable annotation
    # Expected: feature_selection=pending, everything else same as base
    # ------------------------------------------------------------------
    print("\nCase 3: no_hvg")
    v3 = base.copy()
    removed = []
    for col in ["highly_variable", "highly_variable_rank", "means", "variances",
                "dispersions", "dispersions_norm"]:
        if col in v3.var.columns:
            del v3.var[col]
            removed.append(col)
    print(f"  Removed var columns: {removed}")
    v3.write_h5ad(out / "no_hvg.h5ad")
    print("  Saved no_hvg.h5ad")

    # ------------------------------------------------------------------
    # Case 4: no_raw_counts — raw=None, X stays log-normalised
    # Expected: normalisation=already_done, data_integrity_warning about raw
    # ------------------------------------------------------------------
    print("\nCase 4: no_raw_counts")
    v4 = base.copy()
    v4.raw = None
    for key in list(v4.layers.keys()):
        if key.lower() in ("counts", "raw", "counts_raw"):
            del v4.layers[key]
            print(f"  Removed layer: {key}")
    print(f"  raw=None, remaining layers: {list(v4.layers.keys())}")
    v4.write_h5ad(out / "no_raw_counts.h5ad")
    print("  Saved no_raw_counts.h5ad")

    # ------------------------------------------------------------------
    # Case 5: scaled_x — X is z-scored, raw=None
    # Expected: x_state=likely_scaled_or_z_scored, data_integrity_warning
    # ------------------------------------------------------------------
    print("\nCase 5: scaled_x")
    v5 = base.copy()
    v5.raw = None
    for key in list(v5.layers.keys()):
        if key.lower() in ("counts", "raw", "counts_raw"):
            del v5.layers[key]
    X = v5.X.toarray() if sp.issparse(v5.X) else np.asarray(v5.X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_scaled = ((X - mean) / std).astype(np.float32)
    v5.X = sp.csr_matrix(X_scaled)
    print(f"  Z-scored X | min={X_scaled.min():.2f} max={X_scaled.max():.2f} "
          f"(negative values present: {(X_scaled < 0).any()})")
    v5.write_h5ad(out / "scaled_x.h5ad")
    print("  Saved scaled_x.h5ad")

    print(f"\nAll 5 test inputs written to: {out}")


if __name__ == "__main__":
    main()
