"""Step 0a–0b: RNA/ATAC subsets and cisTopic raw object."""
from __future__ import annotations

import os
import pickle
import sys

import pandas as pd
from scipy.sparse import issparse, csr_matrix

from .paths import DATA_DIR, DATA_INPUTS, OUT_DIR


def step_create_subsets(args):
    """From full ATAC and microglia RNA, subset ATAC to microglia, intersect cells, write rna_subset + atac_subset."""
    import scanpy as sc

    if not (getattr(args, "full_rna", None) and getattr(args, "full_atac", None)):
        return
    full_rna = args.full_rna
    full_atac = args.full_atac
    if not os.path.isfile(full_rna) or not os.path.isfile(full_atac):
        print("[Step 0a] ERROR: --full-rna and --full-atac must be paths to existing files.")
        sys.exit(1)
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[Step 0a] Creating RNA/ATAC subsets (intersection of cells)...")
    atac = sc.read_h5ad(full_atac)
    for col in ["Subclass", "subclass"]:
        if col in atac.obs.columns:
            mask = atac.obs[col].astype(str).str.contains("Microglia-PVM", case=False, na=False)
            if mask.sum() > 0:
                atac = atac[mask].copy()
                break
    rna = sc.read_h5ad(full_rna)
    intersection = rna.obs_names.intersection(atac.obs_names)
    if len(intersection) == 0:
        print("  ERROR: No common cell IDs between RNA and ATAC.")
        sys.exit(1)
    rna_subset = rna[intersection].copy()
    atac_subset = atac[intersection].copy()
    rna_path = os.path.join(DATA_DIR, "rna_subset.h5ad")
    atac_path = os.path.join(DATA_DIR, "atac_subset.h5ad")
    rna_subset.write_h5ad(rna_path)
    atac_subset.write_h5ad(atac_path)
    print(f"  Wrote {rna_path}, {atac_path} ({len(intersection)} cells).")


def step_create_cistopic_raw(args):
    """Build cisTopic object from atac_subset.h5ad and save to outs/cistopic_obj_raw.pkl."""
    from pycisTopic.cistopic_class import create_cistopic_object

    atac_path = os.path.join(DATA_DIR, "atac_subset.h5ad")
    raw_path = os.path.join(OUT_DIR, "cistopic_obj_raw.pkl")
    if not os.path.isfile(atac_path):
        return
    if os.path.isfile(raw_path) and not getattr(args, "force_cistopic_raw", False):
        return
    import scanpy as sc

    os.makedirs(OUT_DIR, exist_ok=True)
    print("[Step 0b] Building cisTopic object from atac_subset (sparse)...")
    adata_atac = sc.read_h5ad(atac_path)
    exclude_file = getattr(args, "exclude_barcodes_file", None)
    if exclude_file and os.path.isfile(exclude_file):
        with open(exclude_file) as f:
            exclude_barcodes = set(line.strip() for line in f if line.strip())
        before = adata_atac.n_obs
        keep = ~adata_atac.obs_names.isin(exclude_barcodes)
        adata_atac = adata_atac[keep].copy()
        print(f"  Excluded {before - adata_atac.n_obs} barcodes from {exclude_file}. Remaining: {adata_atac.n_obs} cells.")
    X = adata_atac.X
    if not issparse(X):
        X = csr_matrix(X)
    X = X.T.tocsr()
    try:
        count_df = pd.DataFrame.sparse.from_spmatrix(X, index=adata_atac.var_names, columns=adata_atac.obs_names)
    except Exception:
        print("  WARNING: Sparse DataFrame failed; using dense (high RAM).")
        count_df = pd.DataFrame(X.toarray(), index=adata_atac.var_names, columns=adata_atac.obs_names)
    cistopic_obj = create_cistopic_object(count_df)
    cistopic_obj.add_cell_data(adata_atac.obs)
    with open(raw_path, "wb") as f:
        pickle.dump(cistopic_obj, f)
    print("  Wrote", raw_path)
