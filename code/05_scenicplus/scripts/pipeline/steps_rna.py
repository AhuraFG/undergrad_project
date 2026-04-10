"""Step 1–2: RNA preparation and verification."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse import issparse, csr_matrix

from .paths import CONTAMINANT_MARKERS, DATA_DIR, DATA_INPUTS, GEX_PATH, MICROGLIA_PATTERNS
from .rna_helpers import get_tf_symbols, has_real_counts, safe_get_vec


def step_prepare_rna(args):
    import scanpy as sc

    rna_path = args.rna_path or os.path.join(DATA_DIR, "rna_subset.h5ad")
    os.makedirs(DATA_INPUTS, exist_ok=True)
    print("[Step 1] Preparing microglia RNA...")
    rna = sc.read_h5ad(rna_path)

    have_counts = has_real_counts(rna)
    if have_counts:
        if rna.raw is not None:
            raw = rna.raw.to_adata()
            if list(raw.var_names) == list(rna.var_names):
                rna.layers["counts"] = raw.X.copy()
            else:
                common = rna.var_names[rna.var_names.isin(raw.var_names)].tolist()
                if len(common) == rna.n_vars:
                    rna.layers["counts"] = raw[:, common].X.copy()
                else:
                    rna.layers["counts"] = rna.X.copy()
        elif "counts" not in rna.layers:
            rna.layers["counts"] = rna.X.copy()

    name_col = "feature_names" if "feature_names" in rna.var.columns else "feature_name"
    if name_col in rna.var.columns:
        sym = rna.var[name_col].astype("string").str.strip()
        sym = sym.mask(sym.isna() | (sym == ""), rna.var_names)
        rna.var["symbol"] = sym.str.upper()
    else:
        rna.var["symbol"] = rna.var_names.astype(str).str.upper()
    rna.var_names = rna.var["symbol"].to_numpy()
    rna.var_names_make_unique()

    for col in ["Subclass", "subclass", "cell_type", "CellType", "celltype"]:
        if col in rna.obs.columns:
            vals = rna.obs[col].astype(str).str.strip()
            mask = np.zeros(rna.n_obs, dtype=bool)
            for pat in MICROGLIA_PATTERNS:
                mask |= vals.str.lower().str.contains(pat.lower(), na=False)
            if mask.sum() > 0:
                rna = rna[mask].copy()
                print(f"  Subset to microglia: {rna.n_obs} cells.")
            break

    exclude_file = getattr(args, "exclude_barcodes_file", None)
    if exclude_file and os.path.isfile(exclude_file):
        with open(exclude_file) as f:
            exclude_barcodes = set(line.strip() for line in f if line.strip())
        before = rna.n_obs
        keep = ~rna.obs_names.isin(exclude_barcodes)
        rna = rna[keep].copy()
        print(f"  Excluded {before - rna.n_obs} barcodes from {exclude_file}. Remaining: {rna.n_obs} cells.")

    if args.remove_contaminants:
        all_m = []
        for v in CONTAMINANT_MARKERS.values():
            all_m.extend(v)
        scores = np.zeros(rna.n_obs)
        for g in all_m:
            v = safe_get_vec(rna, g)
            if v is not None:
                v = np.log1p(v)
                scores += (v - np.mean(v)) / (np.std(v) + 1e-8)
        keep = scores <= np.percentile(scores, 95)
        rna = rna[keep].copy()

    if have_counts and "counts" in rna.layers:
        counts = rna.layers["counts"]
        counts = np.asarray(counts.toarray() if issparse(counts) else counts).astype(float)
        counts = np.maximum(counts, 0)
        lib = np.asarray(counts.sum(axis=1)).ravel()
        lib[lib == 0] = 1
        norm = (counts.T / lib).T * 1e4
        rna.X = csr_matrix(np.log1p(norm)) if rna.n_obs * rna.n_vars > 1e8 else np.log1p(norm)
        rna.layers["counts"] = csr_matrix(counts) if rna.n_obs * rna.n_vars > 1e8 else counts
    else:
        print("  No integer counts detected; leaving rna.X unchanged (no re-normalisation).")

    if "counts" in getattr(rna, "layers", {}) and rna.layers["counts"] is not None:
        import anndata as ad

        rna.raw = ad.AnnData(
            X=rna.layers["counts"].copy(),
            obs=rna.obs.copy(),
            var=rna.var.copy(),
        )

    tf_symbols = get_tf_symbols()
    if args.gene_strategy == "full":
        rna.var["highly_variable"] = True
    else:
        sc.pp.highly_variable_genes(
            rna,
            n_top_genes=min(15000, rna.n_vars - 1),
            batch_key="donor_id" if "donor_id" in rna.obs.columns else None,
            flavor="seurat",
        )
        hv = rna.var["highly_variable"].to_numpy()
        for i, g in enumerate(rna.var_names):
            if str(g).upper() in tf_symbols:
                hv[i] = True
        rna.var["highly_variable"] = hv
        rna = rna[:, rna.var["highly_variable"]].copy()
    rna.write_h5ad(GEX_PATH)
    print("  Wrote", GEX_PATH)


def step_verify_rna():
    import scanpy as sc

    print("[Step 2] Verifying RNA...")
    rna = sc.read_h5ad(GEX_PATH)
    errors = []
    if issparse(rna.X):
        has_negative = (rna.X.data < 0).any() if hasattr(rna.X, "data") and rna.X.data.size else False
        x_median = None
    else:
        X = np.asarray(rna.X).astype(float)
        has_negative = np.any(X < 0)
        x_median = float(np.median(X)) if rna.n_obs * rna.n_vars <= 1e7 else None
    if has_negative:
        errors.append("rna.X has negative values.")
    if "highly_variable" not in rna.var.columns or rna.var["highly_variable"].sum() == 0:
        errors.append("rna.var['highly_variable'] missing or empty.")
    has_counts_or_raw = (rna.raw is not None) or ("counts" in getattr(rna, "layers", {}))
    if not has_counts_or_raw and x_median is not None and x_median > 50:
        errors.append("No raw/counts layer and rna.X median > 50 (unclear if counts or normalised).")
    if errors:
        for e in errors:
            print("  ERROR:", e)
        sys.exit(1)
    print("  All checks passed.")
