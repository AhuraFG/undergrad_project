"""Align AnnData regulon matrix from SCENIC+ MuData (shared by post-analysis scripts)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import issparse


def align_regulon_obsm_from_mudata(adata, mdata_path: Path | None) -> None:
    """If ``adata.obsm['X_regulon']`` is missing, fill it from ``direct_gene_based_AUC`` in MuData."""
    if "X_regulon" in adata.obsm:
        return
    if not mdata_path or not mdata_path.exists():
        return
    import mudata as md

    mdata = md.read_h5mu(mdata_path)
    auc_mod = mdata.mod.get("direct_gene_based_AUC")
    if auc_mod is None:
        return
    X_auc = np.asarray(auc_mod.X.toarray() if issparse(auc_mod.X) else auc_mod.X)
    regulon_names = list(auc_mod.var_names)
    mdata_obs = list(auc_mod.obs_names)
    mdata_to_idx = {str(n): i for i, n in enumerate(mdata_obs)}
    idx_in_mdata = []
    for name in adata.obs_names:
        j = (
            mdata_to_idx.get(name)
            or mdata_to_idx.get(name + "___cisTopic")
            or mdata_to_idx.get(name + "_cisTopic")
        )
        idx_in_mdata.append(j)
    X_aligned = np.full((adata.n_obs, len(regulon_names)), np.nan, dtype=float)
    for i, j in enumerate(idx_in_mdata):
        if j is not None:
            X_aligned[i, :] = X_auc[j, :]
    adata.obsm["X_regulon"] = X_aligned
    adata.uns["regulon_names"] = regulon_names
