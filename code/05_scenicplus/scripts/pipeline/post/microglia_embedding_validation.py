#!/usr/bin/env python
"""
Microglia embedding + regulon validation for SCENIC+ outputs.

Creates: (1) global + microglia-only embeddings, (2) UMAP + regulon + marker plots,
(3) cluster × regulon heatmap/dotplot, (4) TBX21 validation triptych,
(5) Nature Neuroscience cross-reference table.

Run from code/05_scenicplus:
  python -m scripts.pipeline.post.microglia_embedding_validation [--full-atlas path/to/atlas.h5ad]

Outputs: validation_figures/ (PNGs), data_inputs/adata_microglia_embedding*.h5ad, results/nn_paper_regulon_mapping*.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Paths: 05_scenicplus root (same as scripts.pipeline.paths)
from ..paths import PIPELINE_ROOT

ROOT = Path(PIPELINE_ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = ROOT  # for compatibility with save_adata etc.
DATA_INPUTS = ROOT / "data_inputs"
SNAKE_DIR = ROOT / "scplus_pipeline" / "Snakemake"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "validation_figures"

# Default paths
MICROGLIA_H5AD = DATA_INPUTS / "SEAAD_MTG_microglia_rna.h5ad"
MDATA_PATH = SNAKE_DIR / "scplusmdata.h5mu"
REGULON_TARGETS_RSS = RESULTS_DIR / "regulon_targets_by_rss.csv"


def _load_top_regulons_by_rss(rss_path: Path, regulon_names: set, top_n: int = 15):
    """Load regulon_targets_by_rss.csv; return list of top regulon names by RSS (desc) that exist in regulon_names."""
    if not rss_path.exists():
        return None
    df = pd.read_csv(rss_path)
    if "RSS" not in df.columns or "regulon" not in df.columns:
        return None
    df = df[df["regulon"].isin(regulon_names)].sort_values("RSS", ascending=False)
    return df["regulon"].head(top_n).tolist()


# Top regulons for feature plots (fallback if no RSS file)
TOP_REGULONS = [
    "MEF2A_direct_+/+_(11g)", "ETS1_direct_+/+_(37g)", "KLF12_direct_-/+_(94g)",
    "IKZF3_direct_-/+_(113g)", "MAFB_direct_+/+_(10g)", "CEBPB_direct_+/+_(26g)",
    "CEBPD_direct_+/+_(17g)", "FOS_direct_+/+_(20g)", "JUNB_direct_+/+_(19g)",
    "FOSL2_direct_+/+_(25g)", "ATF3_direct_+/+_(20g)", "TBX21_direct_-/+_(18g)",
]
# Step 3: Regulons that truly vary (annotation figure); TBX21 as footnote
ANNOTATION_REGULONS = [
    "MEF2A_direct_+/+_(11g)", "KLF12_direct_-/+_(94g)", "IKZF3_direct_-/+_(113g)",
    "FOS_direct_+/+_(20g)", "JUNB_direct_+/+_(19g)", "FOSL2_direct_+/+_(25g)",
    "CEBPB_direct_+/+_(26g)", "CEBPD_direct_+/+_(17g)", "TBX21_direct_-/+_(18g)",
]

# Marker gene panels
MARKER_PANELS = {
    "microglia_identity": ["P2RY12", "TMEM119", "CX3CR1", "AIF1"],
    "DAM": ["TREM2", "APOE", "LPL", "CST7"],
    "antigen_presentation": ["CD74", "HLA-DRA", "HLA-DRB1"],
    "inflammatory": ["IL1B", "TNF", "CCL3", "CCL4"],
    "interferon": ["ISG15", "IFITM3", "IFI6", "IFIT3"],
    "T_NK_contamination": ["TRAC", "CD3D", "CD3E", "NKG7", "GNLY", "PRF1"],
}

# Nature Neuroscience early microglia upregulated (for cross-reference)
NN_PAPER_GENES = ["CSF1R", "FCGR1A", "CD74", "HLA-DRA", "C1QA", "C1QB", "APOE", "LPL", "TREM2", "CST7"]

# Step 1: Cluster 9 identity panel (marker-based)
CLUSTER9_MARKER_PANEL = {
    "microglia_identity": ["P2RY12", "TMEM119", "CX3CR1", "AIF1", "CSF1R"],
    "DAM-ish": ["TREM2", "APOE", "LPL", "CST7", "TYROBP"],
    "antigen_presentation": ["CD74", "HLA-DRA", "HLA-DRB1"],
    "astro": ["AQP4", "ALDH1L1"],
    "OPC": ["PDGFRA", "CSPG4"],
    "oligo": ["MBP", "PLP1", "MOG"],
    "endothelial": ["PECAM1", "VWF", "CLDN5"],
    "pericyte": ["RGS5", "PDGFRB"],
    "T_NK_contamination": ["TRAC", "CD3D", "CD3E", "NKG7", "GNLY", "PRF1"],
}


def _safe_imports():
    """Import scanpy, mudata, matplotlib; set backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc
        import mudata as md
        return plt, sc, md
    except ImportError as e:
        print("Required: scanpy, mudata, matplotlib. Install with conda/pip.", file=sys.stderr)
        raise SystemExit(1) from e


# -----------------------------------------------------------------------------
# A) Load data and set up
# -----------------------------------------------------------------------------
def load_and_verify(adata_path: Path, full_atlas_path: Path | None):
    """Load microglia AnnData; optionally full atlas. Verify .obs and global UMAP."""
    plt, sc, md = _safe_imports()
    adata = sc.read_h5ad(adata_path)
    print("[A] Loaded microglia:", adata_path)
    print("    .obs columns:", list(adata.obs.columns))
    print("    .obs shape:", adata.obs.shape)
    print("    .var shape:", adata.var.shape)

    # Batch / donor
    batch_candidates = ["donor_id", "Donor ID", "donor", "batch", "sample_id"]
    batch_col = None
    for c in batch_candidates:
        if c in adata.obs.columns:
            batch_col = c
            break
    print("    Batch/donor column:", batch_col or "(none found)")

    # Cluster labels
    cluster_candidates = ["leiden", "louvain", "cluster", "Subclass", "Supertype", "Class"]
    cluster_col = None
    for c in cluster_candidates:
        if c in adata.obs.columns:
            cluster_col = c
            break
    print("    Existing cluster column:", cluster_col or "(none)")

    # Global UMAP
    has_global_umap = "X_umap" in adata.obsm
    print("    .obsm['X_umap'] (global):", has_global_umap)

    adata_global = None
    if full_atlas_path and full_atlas_path.exists():
        adata_atlas = sc.read_h5ad(full_atlas_path)
        common = adata.obs_names.intersection(adata_atlas.obs_names)
        if len(common) > 0:
            adata_global = adata_atlas[common].copy()
            print("[B] Subset full atlas to microglia cells:", len(common))
        else:
            print("[B] No overlap between microglia and atlas obs_names; skipping global embedding.")
    else:
        if full_atlas_path:
            print("[B] Full atlas path not found:", full_atlas_path)
        else:
            print("[B] No full atlas path provided; skipping global embedding.")

    return adata, adata_global, batch_col, cluster_col


# -----------------------------------------------------------------------------
# C) Microglia-only embedding
# -----------------------------------------------------------------------------
def compute_microglia_embedding(adata, batch_col: str | None, n_hvg=4000, n_pcs=30, n_neighbors=20, leiden_res=0.6):
    """Recompute HVG, PCA, neighbors, Leiden, UMAP on microglia. Writes embedding into adata and returns it."""
    plt, sc, _ = _safe_imports()
    # Preprocess full adata in place so we keep all genes for later marker plots
    if "counts" in adata.layers:
        from scipy.sparse import issparse, csr_matrix
        counts = adata.layers["counts"]
        counts = np.asarray(counts.toarray() if issparse(counts) else counts).astype(float)
        lib = np.array(counts.sum(axis=1)).ravel()
        lib[lib == 0] = 1
        norm = (counts.T / lib).T * 1e4
        adata.X = csr_matrix(np.log1p(norm)) if adata.n_obs * adata.n_vars > 1e7 else np.log1p(norm)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_hvg, adata.n_vars - 1),
        batch_key=batch_col if batch_col else None,
        flavor="seurat",
    )
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=min(n_pcs, adata_hvg.n_obs - 1, adata_hvg.n_vars - 1))
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=min(20, adata_hvg.obsm["X_pca"].shape[1]))
    sc.tl.leiden(adata_hvg, resolution=leiden_res, key_added="leiden_micro")
    sc.tl.umap(adata_hvg)

    # Copy embedding back to full adata (same cell order)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    adata.obs["leiden_micro"] = adata_hvg.obs["leiden_micro"]
    # Step 2: Keep full gene space in .raw for all marker/gene plots (scanpy uses raw when present).
    # HVGs were used only for PCA/neighbors/UMAP; raw = full log1p so TBX21, P2RY12, etc. are plottable.
    adata.raw = adata.copy()
    print("[C] Microglia-only embedding: UMAP + leiden_micro done (embedding copied to full adata).")
    return adata


def compute_microglia_embedding_scvi(adata, scvi_key: str = "X_scVI", n_neighbors: int = 20, leiden_res: float = 0.6, exclude_clusters: list[str] | None = None):
    """Microglia-only embedding using precomputed scVI latent in .obsm[scvi_key].

    Typical workflow: build kNN graph from scVI latent → Leiden on that graph → UMAP for visualisation.
    If exclude_clusters is set (e.g. ["6"]), remove those clusters then recompute
    neighbors/Leiden/UMAP on the cleaned set (locked microglia-only baseline).
    """
    plt, sc, _ = _safe_imports()
    if scvi_key not in adata.obsm.keys():
        raise KeyError(f"scVI latent '{scvi_key}' not found in adata.obsm keys: {list(adata.obsm.keys())}")
    sc.pp.neighbors(adata, use_rep=scvi_key, n_neighbors=n_neighbors)
    sc.tl.leiden(adata, resolution=leiden_res, key_added="leiden_micro")
    sc.tl.umap(adata)
    # Ensure raw holds full gene expression for marker plots; overwrite any HVG-limited raw
    adata.raw = adata.copy()

    if exclude_clusters:
        exclude_set = set(str(c) for c in exclude_clusters)
        keep = ~adata.obs["leiden_micro"].astype(str).isin(exclude_set)
        n_remove = (~keep).sum()
        adata = adata[keep].copy()
        print(f"[C-scVI] Removed {n_remove} cells in excluded clusters {sorted(exclude_set)}; recomputing neighbors/Leiden/UMAP on cleaned set.")
        sc.pp.neighbors(adata, use_rep=scvi_key, n_neighbors=n_neighbors)
        sc.tl.leiden(adata, resolution=leiden_res, key_added="leiden_micro")
        sc.tl.umap(adata)
        adata.raw = adata.copy()

    print(f"[C-scVI] Microglia-only embedding: Leiden + UMAP using {scvi_key} (n_obs={adata.n_obs}).")
    return adata


# -----------------------------------------------------------------------------
# D) Align AUCell to microglia object
# -----------------------------------------------------------------------------
def align_regulon_activity(adata_micro, mdata_path: Path):
    """Load AUCell from mudata; align by cell IDs; add to adata_micro.obsm and .obs."""
    _, _, md = _safe_imports()
    mdata = md.read_h5mu(mdata_path)
    auc_mod = mdata.mod.get("direct_gene_based_AUC")
    if auc_mod is None:
        raise FileNotFoundError("direct_gene_based_AUC not found in mudata")
    from scipy.sparse import issparse
    X_auc = np.asarray(auc_mod.X.toarray() if issparse(auc_mod.X) else auc_mod.X)
    regulon_names = list(auc_mod.var_names)
    mdata_obs_names = list(auc_mod.obs_names)

    # Align: mudata may have barcode suffix (e.g. ___cisTopic); microglia may not
    mdata_to_idx = {str(n): i for i, n in enumerate(mdata_obs_names)}
    idx_in_mdata = []
    for name in adata_micro.obs_names:
        if name in mdata_to_idx:
            idx_in_mdata.append(mdata_to_idx[name])
        else:
            found = None
            for suffix in ["___cisTopic", "_cisTopic"]:
                key = name + suffix
                if key in mdata_to_idx:
                    found = mdata_to_idx[key]
                    break
            idx_in_mdata.append(found)

    if sum(1 for i in idx_in_mdata if i is None) > 0:
        n_miss = sum(1 for i in idx_in_mdata if i is None)
        print(f"[D] Warning: {n_miss} cells not found in mudata; filling regulon with NaN for those.")
    else:
        print("[D] All cells aligned to mudata AUCell.")

    X_aligned = np.full((adata_micro.n_obs, len(regulon_names)), np.nan, dtype=float)
    for i, j in enumerate(idx_in_mdata):
        if j is not None:
            X_aligned[i, :] = X_auc[j, :]

    adata_micro.obsm["X_regulon"] = X_aligned
    adata_micro.uns["regulon_names"] = regulon_names
    top_n = min(20, len(regulon_names))
    for k in range(top_n):
        key = regulon_names[k].replace("(", "_").replace(")", "_").replace("/", "_")[:60]
        adata_micro.obs[f"regulon_{key}"] = X_aligned[:, k]
    print("[D] Stored X_regulon and regulon_names.")
    return adata_micro, regulon_names


# -----------------------------------------------------------------------------
# E) Regulon + marker UMAP plots
# -----------------------------------------------------------------------------
def plot_regulon_umaps(adata_micro, regulon_names: list, top_regulons: list[str], fig_dir: Path, suffix: str = ""):
    """UMAP feature plots for selected regulons (consistent scale)."""
    plt, sc, _ = _safe_imports()
    if "X_regulon" not in adata_micro.obsm:
        return
    X = adata_micro.obsm["X_regulon"]
    name_to_col = {n: i for i, n in enumerate(regulon_names)}
    vmin, vmax = np.nanpercentile(X, [1, 99])
    if np.isnan(vmin):
        vmin, vmax = 0, 1
    to_plot = [r for r in top_regulons if r in name_to_col][:12]
    n_plot = len(to_plot)
    ncols = 4
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, reg in enumerate(to_plot):
        j = name_to_col[reg]
        ax = axes.flat[idx]
        # Use a temp obs column for this regulon
        adata_micro.obs["_reg_plot"] = X[:, j]
        sc.pl.umap(adata_micro, color="_reg_plot", ax=ax, show=False, vmin=vmin, vmax=vmax, title=reg[:50])
    for idx in range(n_plot, axes.size):
        axes.flat[idx].set_visible(False)
    # Shared colorbar for AUCell scale (all panels use same vmin/vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, label="AUCell")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out = fig_dir / f"umap_regulons_topRSS{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out)


# Panels to save (only these three + combined); all use gene expression log1p from .raw
MARKER_PANELS_FOR_FIGURES = ("microglia_identity", "DAM", "antigen_presentation")


def plot_marker_panels(adata_micro, fig_dir: Path, suffix: str = ""):
    """UMAP panels for marker gene groups (gene expression from .raw). Only microglia_identity, DAM, antigen_presentation + combined."""
    plt, sc, _ = _safe_imports()
    if adata_micro.raw is not None:
        gene_names = adata_micro.raw.var_names
    else:
        gene_names = adata_micro.var_names
    try:
        var_upper = set(gene_names.str.upper())
    except Exception:
        var_upper = set(str(x).upper() for x in gene_names)
    fig_list = []
    for panel_name in MARKER_PANELS_FOR_FIGURES:
        genes = MARKER_PANELS.get(panel_name, [])
        present = [g for g in genes if g in gene_names or g.upper() in var_upper]
        if not present:
            continue
        n_plot = len(present)
        ncols = min(3, n_plot)
        nrows = (n_plot + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        for idx, g in enumerate(present):
            ax = axes.flat[idx]
            sc.pl.umap(adata_micro, color=g, ax=ax, show=False, title=g)
        for idx in range(n_plot, axes.size):
            axes.flat[idx].set_visible(False)
        plt.suptitle(panel_name, y=1.02)
        # Shared colorbar scale for expression (approximate: log1p range)
        from scipy.sparse import issparse
        _X = adata_micro.raw.X if adata_micro.raw is not None else adata_micro.X
        _x = np.asarray(_X.A).ravel() if issparse(_X) else np.asarray(_X).ravel()
        vmin_m, vmax_m = np.nanpercentile(_x, [1, 99])
        if not np.isfinite(vmin_m):
            vmin_m, vmax_m = 0, 1
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin_m, vmax=vmax_m))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cbar_ax, label="Expression (log1p)")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        out = fig_dir / f"umap_markers_{panel_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        fig_list.append(out)
    if fig_list:
        print("    Saved marker panels:", [str(p) for p in fig_list])
    # Combined multi-panel (only genes from the three panels above)
    all_genes = []
    for panel_name in MARKER_PANELS_FOR_FIGURES:
        for g in MARKER_PANELS.get(panel_name, []):
            if g in gene_names or g.upper() in var_upper:
                all_genes.append(g)
    if all_genes:
        n_plot = min(18, len(all_genes))
        ncols = 6
        nrows = (n_plot + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes)
        for idx, g in enumerate(all_genes[:n_plot]):
            sc.pl.umap(adata_micro, color=g, ax=axes.flat[idx], show=False, title=g)
        for idx in range(n_plot, axes.size):
            axes.flat[idx].set_visible(False)
        from scipy.sparse import issparse
        _X = adata_micro.raw.X if adata_micro.raw is not None else adata_micro.X
        _x = np.asarray(_X.A).ravel() if issparse(_X) else np.asarray(_X).ravel()
        vmin_m, vmax_m = np.nanpercentile(_x, [1, 99])
        if not np.isfinite(vmin_m):
            vmin_m, vmax_m = 0, 1
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin_m, vmax=vmax_m))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cbar_ax, label="Expression (log1p)")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        out = fig_dir / f"umap_marker_panels{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("    Saved", out)


# -----------------------------------------------------------------------------
# F) Cluster × regulon heatmap / dotplot (ordering by RSS; values = AUCell)
# -----------------------------------------------------------------------------
def plot_regulon_cluster_summary(adata_micro, regulon_names: list, fig_dir: Path, top_n_regulons: int = 15, suffix: str = "", rss_path: Path | None = None):
    """Mean AUCell per cluster; dotplot (dot color = mean AUCell, size = % active). Order regulons by RSS."""
    plt, sc, _ = _safe_imports()
    if "X_regulon" not in adata_micro.obsm or "leiden_micro" not in adata_micro.obs:
        return
    X = adata_micro.obsm["X_regulon"]
    clusters = adata_micro.obs["leiden_micro"].astype(str)
    uniq = sorted(clusters.unique(), key=lambda x: (int(x) if x.isdigit() else 999, x))
    mean_per_cluster = pd.DataFrame(index=regulon_names, columns=uniq, dtype=float)
    pct_active_per_cluster = pd.DataFrame(index=regulon_names, columns=uniq, dtype=float)
    q75 = np.nanpercentile(X, 75, axis=0)  # regulon-specific threshold for "active"
    for c in uniq:
        mask = clusters == c
        mean_per_cluster[c] = np.nanmean(X[mask], axis=0)
        pct_active_per_cluster[c] = np.nanmean(X[mask] >= q75, axis=0) * 100
    mean_per_cluster = mean_per_cluster.astype(float)
    pct_active_per_cluster = pct_active_per_cluster.astype(float)

    reg_set = set(regulon_names)
    top_reg = _load_top_regulons_by_rss(rss_path or REGULON_TARGETS_RSS, reg_set, top_n=top_n_regulons)
    if not top_reg:
        var_per_reg = mean_per_cluster.var(axis=1)
        top_reg = var_per_reg.nlargest(top_n_regulons).index.tolist()
    M = mean_per_cluster.loc[top_reg]
    P = pct_active_per_cluster.loc[top_reg]

    # Dotplot: dot color = mean AUCell, dot size = % active
    fig, ax = plt.subplots(figsize=(max(6, M.shape[1] * 0.7), max(5, M.shape[0] * 0.3)))
    x_labels = list(M.columns)
    y_labels = [r[:35] for r in top_reg]
    vmax = M.values.max() or 1
    for i, reg in enumerate(top_reg):
        for j, cl in enumerate(x_labels):
            s = P.loc[reg, cl]
            c = M.loc[reg, cl]
            ax.scatter(j, i, s=20 + s * 2, c=[plt.cm.viridis(c / vmax)], alpha=0.8, edgecolors="gray")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Regulon")
    ax.set_title("Dotplot: size = % active (>75th pct), color = mean AUCell (ordered by RSS)")
    # Only the colorbar scale above/near the colorbar (no size legend)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Mean AUCell", shrink=0.6)
    plt.tight_layout()
    out_dp = fig_dir / f"regulon_cluster_dotplot{suffix}.png"
    plt.savefig(out_dp, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out_dp)


# -----------------------------------------------------------------------------
# Step 1 — Cluster 9 marker-based identity (dotplot + heatmap + decision)
# -----------------------------------------------------------------------------
def plot_cluster9_markers(adata_micro, fig_dir: Path, cluster_col: str = "leiden_micro", suffix: str = ""):
    """Dotplot and heatmap for cluster identity panel; decision summary for cluster 9."""
    plt, sc, _ = _safe_imports()
    from scipy.sparse import issparse

    gene_names = adata_micro.raw.var_names if adata_micro.raw is not None else adata_micro.var_names
    var_set = set(gene_names)
    # Flatten panel with category for ordering/labels
    genes_ordered = []
    for cat, genes in CLUSTER9_MARKER_PANEL.items():
        for g in genes:
            if g in var_set:
                genes_ordered.append((g, cat))

    if not genes_ordered:
        print("    [Step 1] No cluster-9 panel genes found in adata; skipping.")
        return

    genes_only = [g for g, _ in genes_ordered]
    if cluster_col not in adata_micro.obs.columns:
        print("    [Step 1] No cluster column", cluster_col)
        return

    # Dotplot (scanpy uses raw when present); group by category if supported
    from collections import OrderedDict
    var_groups = OrderedDict()
    for cat, genes in CLUSTER9_MARKER_PANEL.items():
        present = [g for g in genes if g in var_set]
        if present:
            var_groups[cat] = present
    try:
        fig = sc.pl.dotplot(adata_micro, genes_only, groupby=cluster_col, dendrogram=False, return_fig=True,
                            var_group_labels=list(var_groups.keys()))
    except TypeError:
        fig = sc.pl.dotplot(adata_micro, genes_only, groupby=cluster_col, dendrogram=False, return_fig=True)
    out_dp = fig_dir / f"cluster9_identity_dotplot{suffix}.png"
    if fig is not None:
        fig.savefig(out_dp, dpi=150, bbox_inches="tight")
    else:
        plt.gcf().savefig(out_dp, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out_dp)

    # Heatmap (scanpy heatmap does not support return_fig)
    sc.pl.heatmap(adata_micro, genes_only, groupby=cluster_col, dendrogram=False)
    out_hm = fig_dir / f"cluster9_identity_heatmap{suffix}.png"
    plt.gcf().savefig(out_hm, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out_hm)

    # Per-cluster mean expression (from raw or X) for decision rule
    X = adata_micro.raw.X if adata_micro.raw is not None else adata_micro.X
    if issparse(X):
        X = np.asarray(X.toarray())
    else:
        X = np.asarray(X)
    clusters = adata_micro.obs[cluster_col].astype(str)
    uniq = sorted(clusters.unique(), key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    gene_to_idx = {g: list(gene_names).index(g) for g in genes_only}
    microglia_genes = [g for g, c in genes_ordered if c == "microglia_identity"]
    dam_genes = [g for g, c in genes_ordered if c == "DAM-ish"]
    contamination_genes = [g for g, c in genes_ordered if c == "T_NK_contamination"]
    sanity_genes = [g for g, c in genes_ordered if c in ("astro", "OPC", "oligo", "endothelial", "pericyte")]

    means = {}
    for c in uniq:
        mask = clusters == c
        means[c] = {g: float(np.nanmean(X[mask, gene_to_idx[g]])) for g in genes_only}

    cluster9 = "9"
    if cluster9 not in uniq:
        # Find numeric 9 or largest cluster id
        for c in uniq:
            if c == "9" or (c.isdigit() and int(c) == 9):
                cluster9 = c
                break
        else:
            cluster9 = uniq[-1] if uniq else None

    lines = [
        "Step 1 — Cluster 9 (or target cluster) identity summary",
        "=" * 60,
        "",
    ]
    if cluster9 is None:
        lines.append("No clusters found.")
    else:
        m_mean = np.mean([means[cluster9][g] for g in microglia_genes]) if microglia_genes else 0
        d_mean = np.mean([means[cluster9][g] for g in dam_genes]) if dam_genes else 0
        c_mean = np.mean([means[cluster9][g] for g in contamination_genes]) if contamination_genes else 0
        s_mean = np.mean([means[cluster9][g] for g in sanity_genes]) if sanity_genes else 0
        global_micro = np.mean([np.mean([means[c][g] for g in microglia_genes]) for c in uniq]) if microglia_genes else 0
        global_cont = np.mean([np.mean([means[c][g] for g in contamination_genes]) for c in uniq]) if contamination_genes else 0

        lines.append(f"Cluster: {cluster9}")
        lines.append(f"  Microglia identity (mean expr): {m_mean:.3f}  (global mean: {global_micro:.3f})")
        lines.append(f"  DAM-ish (mean expr): {d_mean:.3f}")
        lines.append(f"  T/NK contamination (mean expr): {c_mean:.3f}  (global mean: {global_cont:.3f})")
        lines.append(f"  Astro/OPC/oligo/endo/pericyte (mean expr): {s_mean:.3f}")
        lines.append("")
        if m_mean < global_micro * 0.5 and (c_mean > global_cont * 1.5 or s_mean > 0.5):
            lines.append("Decision: NOT MICROGLIA — cluster fails microglia markers and/or shows contamination/sanity.")
            lines.append("  → Remove this cluster from the microglia object and redo UMAP/clustering.")
        elif m_mean >= global_micro * 0.5 and (c_mean > global_cont * 1.2 or s_mean > 0.3):
            lines.append("Decision: LIKELY CONTAMINATION — consider removing and redoing embedding.")
        else:
            lines.append("Decision: MICROGLIA — cluster passes identity; if distinct program (e.g. high MHC/IFN/DAM), keep and label.")
    out_txt = fig_dir / f"cluster9_decision_summary{suffix}.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print("    Saved", out_txt)


def _plot_cluster_proportions_by_condition(adata_micro, cluster_col: str, condition_col: str, fig_dir: Path, suffix: str, donor_col: str | None = "donor_id"):
    """Stacked bar: x = condition level (e.g. ADNC), y = proportion, segments = cluster. If donor_col in obs, x-labels show N donors per group."""
    plt, sc, _ = _safe_imports()
    clusters = adata_micro.obs[cluster_col].astype(str)
    cond = adata_micro.obs[condition_col].astype(str)
    df = pd.DataFrame({"cluster": clusters, "condition": cond})
    if donor_col and donor_col in adata_micro.obs.columns:
        df["donor"] = adata_micro.obs[donor_col].astype(str)
    cross = pd.crosstab(df["condition"], df["cluster"])
    prop = cross.div(cross.sum(axis=1), axis=0)
    prop = prop.loc[:, (prop > 0).any(axis=0)]  # drop empty clusters
    n_cond = prop.shape[0]
    n_cluster = prop.shape[1]
    if n_cond == 0 or n_cluster == 0:
        return
    # N donors per condition level (for x-labels / caption)
    n_donors_per_cond = None
    if donor_col and donor_col in adata_micro.obs.columns:
        n_donors_per_cond = df.groupby("condition")["donor"].nunique().reindex(prop.index)
    fig, ax = plt.subplots(figsize=(max(8, n_cond * 0.4), 5))
    x = np.arange(n_cond)
    bottom = np.zeros(n_cond)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_cluster, 1)))
    for i, cl in enumerate(prop.columns):
        ax.bar(x, prop[cl].values, bottom=bottom, label=str(cl), color=colors[i % len(colors)], width=0.7)
        bottom += prop[cl].values
    ax.set_xticks(x)
    if n_donors_per_cond is not None:
        n_donors_per_cond = n_donors_per_cond.fillna(0).astype(int)
        x_labels = [f"{lev} (n={n_donors_per_cond[lev]})" for lev in prop.index]
    else:
        x_labels = prop.index.tolist()
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(condition_col)
    ax.set_title(f"Cluster proportions by {condition_col}")
    ax.legend(title=cluster_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = fig_dir / f"cluster_proportions_by_{condition_col.replace(' ', '_')}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out)


def _condition_cols_for_proportions(adata_micro, batch_col: str | None) -> list[str]:
    """Return list of condition columns for proportion plots: donor + disease/disease stage (first available)."""
    cols = []
    if batch_col and batch_col in adata_micro.obs.columns:
        cols.append(batch_col)
    for c in ("ADNC", "disease", "Braak stage", "Cognitive status"):
        if c in adata_micro.obs.columns:
            cols.append(c)
            break
    return cols


# -----------------------------------------------------------------------------
# Donor-level analysis: fraction of microglia per cluster per donor, by ADNC
# -----------------------------------------------------------------------------
def plot_donor_level_cluster_fractions_by_adnc(
    adata_micro,
    fig_dir: Path,
    results_dir: Path,
    suffix: str = "",
    cluster_col: str = "leiden_micro",
    donor_col: str = "donor_id",
    group_col: str = "ADNC",
):
    """
    For each donor, compute fraction of their cells in each cluster. Plot box/violin + jitter
    of donor fractions by ADNC group (one panel per cluster). Test: Kruskal-Wallis (fraction ~ ADNC).
    Optionally run ordinal/beta regression. Saves figure and CSV of fractions + test results.
    """
    plt, sc, _ = _safe_imports()
    from scipy import stats

    if cluster_col not in adata_micro.obs.columns or donor_col not in adata_micro.obs.columns or group_col not in adata_micro.obs.columns:
        print(f"    [Donor-level] Missing {cluster_col}, {donor_col}, or {group_col}; skipping.")
        return

    clusters = adata_micro.obs[cluster_col].astype(str)
    donors = adata_micro.obs[donor_col].astype(str)
    groups = adata_micro.obs[group_col].astype(str)

    # Per-donor cluster counts
    df_cells = pd.DataFrame({"donor": donors, "cluster": clusters, "group": groups})
    count_per_donor_cluster = df_cells.groupby(["donor", "cluster"]).size().unstack(fill_value=0)
    n_cells_per_donor = count_per_donor_cluster.sum(axis=1)
    # Fractions (each donor sums to 1)
    frac_per_donor_cluster = count_per_donor_cluster.div(n_cells_per_donor, axis=0)

    # One group (ADNC) per donor (take first occurrence)
    donor_to_group = df_cells.groupby("donor")["group"].first()
    frac_per_donor_cluster["_group"] = frac_per_donor_cluster.index.map(donor_to_group)
    # Drop donors with missing group
    frac_per_donor_cluster = frac_per_donor_cluster.dropna(subset=["_group"])
    if frac_per_donor_cluster.empty or frac_per_donor_cluster["_group"].nunique() < 2:
        print(f"    [Donor-level] Need at least 2 {group_col} groups with donors; skipping.")
        return

    # Disease-severity order (Not AD < Low < Intermediate < High) for ordinal regression
    ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
    group_levels = [g for g in ADNC_ORDER if g in frac_per_donor_cluster["_group"].unique()]
    cluster_cols = [c for c in frac_per_donor_cluster.columns if c != "_group"]
    cluster_cols = sorted(cluster_cols, key=lambda x: (int(x) if x.isdigit() else 999, x))

    # Kruskal-Wallis per cluster
    kw_results = []
    for cl in cluster_cols:
        series = frac_per_donor_cluster[cl]
        by_group = [frac_per_donor_cluster.loc[frac_per_donor_cluster["_group"] == g, cl].values for g in group_levels]
        by_group = [x for x in by_group if len(x) > 0]
        if len(by_group) < 2:
            kw_results.append({"cluster": cl, "kw_statistic": np.nan, "kw_pvalue": np.nan})
            continue
        stat, p = stats.kruskal(*by_group)
        kw_results.append({"cluster": cl, "kw_statistic": stat, "kw_pvalue": p})

    kw_df = pd.DataFrame(kw_results)

    # FDR correction (Benjamini-Hochberg) for multiple clusters
    try:
        from statsmodels.stats.multitest import multipletests
        _p = kw_df["kw_pvalue"].values.astype(float)
        _valid = np.isfinite(_p)
        kw_q = np.full_like(_p, np.nan)
        if _valid.sum() > 0:
            _, qvals, _, _ = multipletests(_p[_valid], method="fdr_bh")
            kw_q[_valid] = qvals
        kw_df["kw_qvalue"] = kw_q
    except Exception:
        kw_df["kw_qvalue"] = np.nan

    # Optional: ordinal predictor on logit(fraction) as parametric alternative
    # Fit logit(fraction) ~ group_ordinal; get predicted mean fraction per ADNC level for overlay plot.
    fitted_means_per_cluster = {}
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.families.links import Logit
        group_ordinal = frac_per_donor_cluster["_group"].map({g: i for i, g in enumerate(group_levels)})
        logit_pvalues = []
        for cl in cluster_cols:
            y = np.clip(frac_per_donor_cluster[cl].values, 1e-6, 1 - 1e-6)
            X = sm.add_constant(group_ordinal.values)
            model = GLM(y, X, family=Binomial(link=Logit()))
            res = model.fit(disp=0)
            p = res.pvalues.iloc[1] if len(res.pvalues) > 1 else np.nan
            logit_pvalues.append(p)
            X_pred = sm.add_constant(np.arange(len(group_levels)))
            fitted_means_per_cluster[cl] = res.predict(X_pred)
        kw_df["logit_ordinal_pvalue"] = logit_pvalues
    except Exception:
        # Fallback: OLS on logit(fraction) ~ group_ordinal, then inverse-logit for predicted fraction
        try:
            import statsmodels.api as sm
            group_ordinal = frac_per_donor_cluster["_group"].map({g: i for i, g in enumerate(group_levels)})
            logit_pvalues = []
            for cl in cluster_cols:
                y = np.clip(frac_per_donor_cluster[cl].values, 1e-6, 1 - 1e-6)
                y_logit = np.log(y / (1 - y))
                X = sm.add_constant(group_ordinal.values)
                res = sm.OLS(y_logit, X).fit()
                p = res.pvalues.iloc[1] if len(res.pvalues) > 1 else np.nan
                logit_pvalues.append(p)
                X_pred = sm.add_constant(np.arange(len(group_levels)))
                pred_logit = res.predict(X_pred)
                fitted_means_per_cluster[cl] = 1 / (1 + np.exp(-pred_logit))
            kw_df["logit_ordinal_pvalue"] = logit_pvalues
        except Exception:
            pass
    if "logit_ordinal_pvalue" in kw_df.columns:
        try:
            from statsmodels.stats.multitest import multipletests
            _p = kw_df["logit_ordinal_pvalue"].values.astype(float)
            _valid = np.isfinite(_p)
            logit_q = np.full_like(_p, np.nan)
            if _valid.sum() > 0:
                _, qvals, _, _ = multipletests(_p[_valid], method="fdr_bh")
                logit_q[_valid] = qvals
            kw_df["logit_ordinal_qvalue"] = logit_q
        except Exception:
            kw_df["logit_ordinal_qvalue"] = np.nan
    else:
        kw_df["logit_ordinal_qvalue"] = np.nan
    if not fitted_means_per_cluster:
        # No regression fit: use observed mean fraction per ADNC group as trend overlay
        for cl in cluster_cols:
            means = [frac_per_donor_cluster.loc[frac_per_donor_cluster["_group"] == g, cl].mean() for g in group_levels]
            fitted_means_per_cluster[cl] = np.array(means)
        if "logit_ordinal_pvalue" not in kw_df.columns:
            kw_df["logit_ordinal_pvalue"] = np.nan

    # Save CSV: donor-level fractions + test results
    results_dir.mkdir(parents=True, exist_ok=True)
    frac_long = frac_per_donor_cluster.reset_index()
    frac_long = frac_long.rename(columns={"_group": group_col})
    frac_long.to_csv(results_dir / f"donor_cluster_fractions_by_{group_col}{suffix}.csv", index=False)
    kw_df.to_csv(results_dir / f"donor_cluster_fractions_kw_tests{suffix}.csv", index=False)
    print("    Saved donor-level fractions and Kruskal-Wallis results to", results_dir)

    # Plot: one panel per cluster, box + violin + jitter by ADNC
    n_clusters = len(cluster_cols)
    ncols = min(4, n_clusters)
    nrows = (n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, cl in enumerate(cluster_cols):
        ax = axes.flat[idx]
        kw_row = kw_df[kw_df["cluster"] == cl].iloc[0]
        pval = kw_row["kw_pvalue"]
        qval = kw_row.get("kw_qvalue", np.nan)
        p_str = f"K-W p={pval:.3g}" if pd.notna(pval) else "K-W n/a"
        if pd.notna(qval):
            p_str += f", q={qval:.3g}"

        data_for_plot = []
        positions = []
        for i, g in enumerate(group_levels):
            vals = frac_per_donor_cluster.loc[frac_per_donor_cluster["_group"] == g, cl].values
            if len(vals) > 0:
                data_for_plot.append(vals)
                positions.append(i)

        if not data_for_plot:
            ax.set_title(f"Cluster {cl}\n{p_str}")
            continue

        parts = ax.violinplot(data_for_plot, positions=positions, showmeans=False, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("lightsteelblue")
            pc.set_alpha(0.7)
        bp = ax.boxplot(data_for_plot, positions=positions, widths=0.15, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.9)
        # Jitter
        for i, (pos, vals) in enumerate(zip(positions, data_for_plot)):
            jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(pos + jitter, vals, alpha=0.6, s=20, c="black", zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(group_levels, rotation=45, ha="right")
        ax.set_ylabel("Donor fraction")
        ax.set_title(f"Cluster {cl}\n{p_str}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=5, label="Donors", linestyle="")], loc="upper right", fontsize=8)
    for idx in range(n_clusters, axes.size):
        axes.flat[idx].set_visible(False)
    plt.suptitle(f"Donor-level cluster fractions by {group_col} (one point = one donor)", y=1.02, fontsize=11)
    plt.tight_layout()
    out_fig = fig_dir / f"donor_cluster_fractions_by_{group_col}{suffix}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out_fig)

    # Second figure: same layout with ordinal (logit) regression fit overlaid
    if fitted_means_per_cluster:
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes2 = np.atleast_2d(axes2)
        for idx, cl in enumerate(cluster_cols):
            ax = axes2.flat[idx]
            logit_p = kw_df.loc[kw_df["cluster"] == cl, "logit_ordinal_pvalue"].iloc[0]
            logit_q = kw_df.loc[kw_df["cluster"] == cl, "logit_ordinal_qvalue"].iloc[0]
            p_str = f"Ordinal (logit) p={logit_p:.3g}" if pd.notna(logit_p) else "Observed mean trend"
            if pd.notna(logit_q):
                p_str += f", q={logit_q:.3g}"

            data_for_plot = []
            positions = []
            for i, g in enumerate(group_levels):
                vals = frac_per_donor_cluster.loc[frac_per_donor_cluster["_group"] == g, cl].values
                if len(vals) > 0:
                    data_for_plot.append(vals)
                    positions.append(i)

            if not data_for_plot:
                ax.set_title(f"Cluster {cl}\n{p_str}")
                continue

            parts = ax.violinplot(data_for_plot, positions=positions, showmeans=False, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor("lightsteelblue")
                pc.set_alpha(0.7)
            bp = ax.boxplot(data_for_plot, positions=positions, widths=0.15, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("white")
                patch.set_alpha(0.9)
            for i, (pos, vals) in enumerate(zip(positions, data_for_plot)):
                jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(pos + jitter, vals, alpha=0.6, s=20, c="black", zorder=3)

            # Overlay ordinal regression fitted means (line + points)
            if cl in fitted_means_per_cluster:
                pred = np.asarray(fitted_means_per_cluster[cl]).ravel()
                x_fit = np.arange(len(group_levels))
                ax.plot(x_fit, pred, color="darkred", linewidth=2, label="Ordinal (logit) fit", zorder=4)
                ax.scatter(x_fit, pred, color="darkred", s=80, zorder=5, edgecolors="white", linewidths=1.5)

            ax.set_xticks(positions)
            ax.set_xticklabels(group_levels, rotation=45, ha="right")
            ax.set_ylabel("Donor fraction")
            ax.set_title(f"Cluster {cl}\n{p_str}")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="y", alpha=0.3)
        for idx in range(n_clusters, axes2.size):
            axes2.flat[idx].set_visible(False)
        plt.suptitle(f"Donor-level cluster fractions by {group_col} with ordinal (logit) regression fit", y=1.02, fontsize=11)
        plt.tight_layout()
        out_fig2 = fig_dir / f"donor_cluster_fractions_by_{group_col}_ordinal_fit{suffix}.png"
        plt.savefig(out_fig2, dpi=150, bbox_inches="tight")
        plt.close()
        print("    Saved", out_fig2)
def plot_cluster_annotation_figures(adata_micro, regulon_names: list, fig_dir: Path, suffix: str = "", clusters_only: bool = True, condition_col: str | None = None, condition_cols: list[str] | None = None):
    """UMAP colored by annotated cluster (or leiden). If condition_col/condition_cols set, also plot cluster proportions by each condition."""
    plt, sc, _ = _safe_imports()
    cluster_col = "cluster_annotation" if "cluster_annotation" in adata_micro.obs.columns else "leiden_micro"
    sc.pl.umap(adata_micro, color=cluster_col, legend_loc="on data", show=False)
    out_umap = fig_dir / f"umap_annotated_clusters{suffix}.png"
    plt.gcf().savefig(out_umap, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out_umap, "(colored by", cluster_col + ")")

    # Cluster proportions by condition(s): stacked bar per condition level
    cols = list(condition_cols) if condition_cols else ([condition_col] if condition_col else [])
    for col in cols:
        if col and col in adata_micro.obs.columns:
            _plot_cluster_proportions_by_condition(adata_micro, cluster_col, col, fig_dir, suffix)
        elif col:
            print("    [Step 3] condition", col, "not in .obs; skipping proportions plot.")

    if clusters_only:
        return
    name_to_col = {n: i for i, n in enumerate(regulon_names)}
    to_plot = [r for r in ANNOTATION_REGULONS if r in name_to_col]
    if not to_plot:
        return
    X = adata_micro.obsm["X_regulon"]
    vmin, vmax = np.nanpercentile(X, [1, 99])
    if np.isnan(vmin):
        vmin, vmax = 0, 1
    n_plot = len(to_plot)
    ncols = 3
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, reg in enumerate(to_plot):
        j = name_to_col[reg]
        adata_micro.obs["_reg"] = X[:, j]
        title = "TBX21 (global)" if "TBX21" in reg else reg[:45]
        sc.pl.umap(adata_micro, color="_reg", ax=axes.flat[idx], show=False, vmin=vmin, vmax=vmax, title=title)
    for idx in range(n_plot, axes.size):
        axes.flat[idx].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="AUCell")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_panel = fig_dir / f"umap_regulons_annotation_panel{suffix}.png"
    plt.savefig(out_panel, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out_panel)


# -----------------------------------------------------------------------------
# Cluster barcode export (for exclusion / contamination inspection)
# -----------------------------------------------------------------------------
def write_cluster_barcodes(adata_micro, cluster_col: str, results_dir: Path, suffix: str = ""):
    """Write one text file per cluster containing cell barcodes. Also write a summary CSV of cluster sizes."""
    results_dir.mkdir(parents=True, exist_ok=True)
    clusters = adata_micro.obs[cluster_col].astype(str)
    summary_rows = []
    for cl in sorted(clusters.unique(), key=lambda x: (int(x) if x.isdigit() else 999, x)):
        mask = clusters == cl
        barcodes = adata_micro.obs_names[mask].tolist()
        out = results_dir / f"cluster_{cl}_barcodes{suffix}.txt"
        with open(out, "w") as f:
            f.write("\n".join(barcodes))
        summary_rows.append({"cluster": cl, "n_cells": len(barcodes)})
    pd.DataFrame(summary_rows).to_csv(results_dir / f"cluster_sizes{suffix}.csv", index=False)
    print(f"    Wrote per-cluster barcode files and cluster_sizes{suffix}.csv to {results_dir}")


# -----------------------------------------------------------------------------
# Step 4 — Per-cluster regulon ranking (top 5 by mean AUCell, top 5 by % active)
# -----------------------------------------------------------------------------
def write_per_cluster_regulon_ranking(adata_micro, regulon_names: list, fig_dir: Path, cluster_col: str = "leiden_micro", suffix: str = ""):
    """For each cluster: top 5 regulons by mean AUCell and top 5 by % active. CSV + short summary."""
    if "X_regulon" not in adata_micro.obsm or cluster_col not in adata_micro.obs.columns:
        print("    [Step 4] Missing X_regulon or cluster column; skipping ranking.")
        return
    X = adata_micro.obsm["X_regulon"]
    clusters = adata_micro.obs[cluster_col].astype(str)
    uniq = sorted(clusters.unique(), key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    q75 = np.nanpercentile(X, 75, axis=0)
    pct_active = np.mean(X >= q75, axis=0) * 100  # global; per-cluster below

    rows = []
    for c in uniq:
        mask = clusters == c
        mean_c = np.nanmean(X[mask], axis=0)
        pct_c = np.nanmean(X[mask] >= q75, axis=0) * 100
        top_mean = np.argsort(-mean_c)[:5]
        top_pct = np.argsort(-pct_c)[:5]
        rows.append({
            "cluster": c,
            "top5_by_mean_AUCell": ";".join(regulon_names[j] for j in top_mean),
            "top5_by_pct_active": ";".join(regulon_names[j] for j in top_pct),
        })
    df = pd.DataFrame(rows)
    out_csv = fig_dir / f"cluster_regulon_ranking{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print("    Saved", out_csv)


# -----------------------------------------------------------------------------
# G) TBX21 validation triptych
# -----------------------------------------------------------------------------
def plot_tbx21_validation(adata_micro, regulon_names: list, fig_dir: Path, suffix: str = ""):
    """Three panels: TBX21 gene, TBX21_direct_-/+ regulon, T/NK score."""
    plt, sc, _ = _safe_imports()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Scanpy uses adata.raw for color when raw exists; TBX21 must be in that source.
    _gene_names = adata_micro.raw.var_names if adata_micro.raw is not None else adata_micro.var_names
    if "TBX21" in _gene_names:
        sc.pl.umap(adata_micro, color="TBX21", ax=axes[0], show=False, title="TBX21 (gene)")
    else:
        axes[0].set_title("TBX21 (gene) – not in var")
        axes[0].axis("off")

    tb_reg = next((r for r in regulon_names if "TBX21" in r and "direct" in r and "-/+" in r), None)
    if tb_reg and "X_regulon" in adata_micro.obsm:
        j = regulon_names.index(tb_reg)
        adata_micro.obs["_tbx21_reg"] = adata_micro.obsm["X_regulon"][:, j]
        sc.pl.umap(adata_micro, color="_tbx21_reg", ax=axes[1], show=False, title=f"Regulon: {tb_reg[:30]}...")
    else:
        axes[1].set_title("TBX21 regulon – not found")
        axes[1].axis("off")

    tnk_genes = ["TRAC", "CD3D", "CD3E", "NKG7", "GNLY", "PRF1"]
    _gene_names = adata_micro.raw.var_names if adata_micro.raw is not None else adata_micro.var_names
    present = [g for g in tnk_genes if g in _gene_names]
    if present:
        from scipy.sparse import issparse
        expr = []
        for g in present:
            j = list(_gene_names).index(g)
            x = adata_micro.raw.X[:, j] if adata_micro.raw is not None else adata_micro.X[:, j]
            expr.append(np.asarray(x.A).ravel() if issparse(x) else np.asarray(x).ravel())
        adata_micro.obs["T_NK_score"] = np.mean(expr, axis=0)
        sc.pl.umap(adata_micro, color="T_NK_score", ax=axes[2], show=False, title="T/NK score (TRAC, CD3D, ...)")
    else:
        axes[2].set_title("T/NK score – genes not in var")
        axes[2].axis("off")

    # Add colorbars so each continuous panel has a scale (scanpy may omit when ax= is used)
    for ax in axes:
        if not ax.get_visible():
            continue
        colls = [c for c in ax.collections if hasattr(c, "get_array") and c.get_array() is not None]
        if colls:
            plt.colorbar(colls[0], ax=ax, shrink=0.7)
    plt.tight_layout()
    out = fig_dir / f"tbx21_validation_triptych{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved", out)


# -----------------------------------------------------------------------------
# H) Nature Neuroscience cross-reference table
# -----------------------------------------------------------------------------
def write_nn_paper_mapping(regulon_targets_rss_path: Path, fig_dir: Path, suffix: str = ""):
    """Which regulons have NN paper genes as targets -> CSV."""
    if not regulon_targets_rss_path.exists():
        print("[H] regulon_targets_by_rss.csv not found; skipping NN mapping.")
        return
    df = pd.read_csv(regulon_targets_rss_path)
    rows = []
    for _, row in df.iterrows():
        reg = row["regulon"]
        targets = set(str(row["targets"]).split(";"))
        overlap = [g for g in NN_PAPER_GENES if g in targets]
        if overlap:
            rows.append({"regulon": reg, "NN_paper_genes_in_targets": ";".join(overlap), "n_overlap": len(overlap)})
    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        out_df = out_df.sort_values("n_overlap", ascending=False)
    out_path = fig_dir / f"nn_paper_regulon_mapping{suffix}.csv"
    out_df.to_csv(out_path, index=False)
    print("[H] Saved", out_path)
    print("    Note: NN paper TFs (RUNX1/IKZF1/NFATC2/MAF) from snATAC GRNs; ours are SCENIC+ eRegulons – partial overlap expected.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _clear_post_snakemake_outputs(fig_dir: Path, results_dir: Path) -> None:
    """Remove all validation figures and results produced by the post-snakemake (embedding/progression/enrichment) section."""
    # Clear validation_figures: all files and pathway_enrichment/
    if fig_dir.exists():
        for f in fig_dir.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                for g in f.rglob("*"):
                    if g.is_file():
                        g.unlink()
                f.rmdir()
    # Clear post-snakemake results (keep pipeline outputs like regulon_targets_by_rss.csv, regulon_credibility_table.csv, etc.)
    patterns = [
        "donor_cluster_fractions_by_*.csv",
        "donor_cluster_fractions_kw_tests*.csv",
        "cluster_sizes*.csv",
        "nn_paper_regulon_mapping*.csv",
        "cluster_regulon_ranking*.csv",
        "regulon_progression_adnc.csv",
        "leiden_param_sweep_scvi_ari_silhouette.csv",
        "best_leiden_params_scvi.txt",
        "cluster_*_barcodes*.txt",
    ]
    for pat in patterns:
        for f in results_dir.glob(pat):
            if f.is_file():
                f.unlink()
    # cluster_regulon_ranking is written to fig_dir, not results_dir; already cleared above
    print("[validation] Cleared validation_figures/ and post-snakemake results/ outputs.")


def main(args=None):
    parser = argparse.ArgumentParser(description="Microglia embedding + regulon validation")
    parser.add_argument("--microglia-h5ad", type=Path, default=MICROGLIA_H5AD, help="Microglia AnnData path")
    parser.add_argument("--full-atlas", type=Path, default=None, help="Full atlas AnnData for global UMAP context")
    parser.add_argument("--mdata", type=Path, default=MDATA_PATH, help="MuData with AUCell")
    parser.add_argument("--out-fig-dir", type=Path, default=FIG_DIR, help="Output figure directory")
    parser.add_argument("--save-adata", type=Path, default=DATA_INPUTS / "adata_microglia_embedding.h5ad", help="Save microglia adata here")
    parser.add_argument("--n-hvg", type=int, default=4000)
    parser.add_argument("--leiden-resolution", type=float, default=0.6)
    parser.add_argument("--use-scvi", action="store_true", default=True, help="Use scVI latent in .obsm['X_scVI'] for neighbors/UMAP (default)")
    parser.add_argument("--no-scvi", action="store_true", help="Use HVG+PCA instead of scVI")
    parser.add_argument("--scvi-key", type=str, default="X_scVI", help="Key in .obsm with scVI latent (default: X_scVI)")
    parser.add_argument("--sweep-leiden", action="store_true", help="Run Leiden parameter sweep first (n_neighbors × resolution), then use best params for rest; implies --use-scvi")
    parser.add_argument("--leiden-params", type=Path, default=RESULTS_DIR / "best_leiden_params_scvi.txt", help="JSON file with n_neighbors and resolution (used when --use-scvi and not --sweep-leiden)")
    parser.add_argument("--exclude-scvi-clusters", type=str, default=None, help="Comma-separated cluster IDs to remove before recomputing embedding (e.g. 6 for contamination); only with --use-scvi")
    parser.add_argument("--skip-progression", action="store_true", help="Do not run regulon vs ADNC progression analysis")
    parser.add_argument("--skip-enrichment", action="store_true", help="(deprecated, enrichment removed from pipeline)")
    args = parser.parse_args() if args is None else args
    if getattr(args, "no_scvi", False):
        args.use_scvi = False

    warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)
    _clear_post_snakemake_outputs(args.out_fig_dir, RESULTS_DIR)
    plt, sc, _ = _safe_imports()

    if args.sweep_leiden:
        args.use_scvi = True

    adata, adata_global, batch_col, _ = load_and_verify(args.microglia_h5ad, args.full_atlas)

    # Decide suffix for output filenames and embedding strategy
    suffix = "_scvi" if args.use_scvi else ""
    # If saving to default path and using scVI, redirect to a *_scvi file
    default_save = DATA_INPUTS / "adata_microglia_embedding.h5ad"
    if args.use_scvi and args.save_adata == default_save:
        args.save_adata = DATA_INPUTS / "adata_microglia_embedding_scvi.h5ad"

    adata_micro = None
    ran_sweep = False
    adata_best_path = DATA_INPUTS / "adata_micro_scvi_best.h5ad"

    if args.sweep_leiden and args.scvi_key in adata.obsm:
        from scripts.utils.leiden_param_sweep_scvi import run_sweep
        adata_sweep = adata.copy()
        adata_sweep, _ = align_regulon_activity(adata_sweep, args.mdata)
        run_sweep(
            adata_sweep,
            out_csv=RESULTS_DIR / "leiden_param_sweep_scvi_ari_silhouette.csv",
            out_best_txt=args.leiden_params,
            out_fig=args.out_fig_dir / "leiden_param_sweep_diagnostics_ari_silhouette.png",
            fig_dir=args.out_fig_dir,
            adata_best_path=adata_best_path,
        )
        adata_micro = sc.read_h5ad(adata_best_path)
        regulon_names = list(adata_micro.uns["regulon_names"])
        ran_sweep = True
        print("Using clustering from Leiden sweep (adata_micro_scvi_best.h5ad). Regulon dotplot and top-RSS regulon UMAPs will use these Leiden labels.")

    if adata_micro is None and args.use_scvi:
        best_k, best_res = 20, args.leiden_resolution
        if args.leiden_params.exists():
            try:
                import json
                with open(args.leiden_params) as f:
                    best_params = json.load(f)
                best_k = int(best_params.get("n_neighbors", 20))
                best_res = float(best_params.get("resolution", args.leiden_resolution))
                print(f"Using Leiden params from {args.leiden_params}: n_neighbors={best_k}, resolution={best_res}")
            except Exception:
                pass
        exclude_clusters = None
        if args.exclude_scvi_clusters:
            exclude_clusters = [s.strip() for s in args.exclude_scvi_clusters.split(",") if s.strip()]
        adata_micro = compute_microglia_embedding_scvi(
            adata,
            scvi_key=args.scvi_key,
            n_neighbors=best_k,
            leiden_res=best_res,
            exclude_clusters=exclude_clusters,
        )
    elif adata_micro is None:
        adata_micro = compute_microglia_embedding(
            adata,
            batch_col,
            n_hvg=args.n_hvg,
            leiden_res=args.leiden_resolution,
        )

    # Write per-cluster barcode files for inspection / exclusion (run regardless of --exclude-scvi-clusters)
    cluster_col = "leiden_micro" if "leiden_micro" in adata_micro.obs.columns else "leiden" if "leiden" in adata_micro.obs.columns else None
    if cluster_col:
        write_cluster_barcodes(adata_micro, cluster_col, RESULTS_DIR, suffix=suffix)

    if not ran_sweep:
        adata_micro, regulon_names = align_regulon_activity(adata_micro, args.mdata)

    # QC plots: only umap_donor (umap_clusters not kept in final 10)
    if batch_col:
        sc.pl.umap(adata_micro, color=batch_col, legend_loc="right margin", show=False)
        plt.gcf().savefig(args.out_fig_dir / f"umap_donor{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Saved umap_donor" + suffix + ".png")

    # UMAP side-by-side: clusters and ADNC
    cluster_col = "leiden_micro" if "leiden_micro" in adata_micro.obs.columns else "leiden" if "leiden" in adata_micro.obs.columns else None
    if cluster_col and "ADNC" in adata_micro.obs.columns:
        ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
        adata_micro.obs["ADNC"] = pd.Categorical(
            adata_micro.obs["ADNC"].astype(str),
            categories=[c for c in ADNC_ORDER if c in adata_micro.obs["ADNC"].astype(str).unique()],
            ordered=True,
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc.pl.umap(adata_micro, color=cluster_col, legend_loc="on data", ax=axes[0], show=False)
        axes[0].set_title("Clusters")
        sc.pl.umap(adata_micro, color="ADNC", legend_loc="right margin", ax=axes[1], show=False)
        axes[1].set_title("ADNC")
        plt.tight_layout()
        plt.savefig(args.out_fig_dir / f"umap_clusters_vs_adnc{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Saved umap_clusters_vs_adnc" + suffix + ".png")

    # Selection by RSS; coloring by AUCell
    reg_set = set(regulon_names)
    top_regulons_rss = _load_top_regulons_by_rss(REGULON_TARGETS_RSS, reg_set, top_n=12)
    if not top_regulons_rss:
        top_regulons_rss = [r for r in TOP_REGULONS if r in reg_set][:12]
    plot_regulon_umaps(adata_micro, regulon_names, top_regulons_rss, args.out_fig_dir, suffix=suffix)
    plot_marker_panels(adata_micro, args.out_fig_dir, suffix=suffix)
    plot_regulon_cluster_summary(adata_micro, regulon_names, args.out_fig_dir, top_n_regulons=15, suffix=suffix, rss_path=REGULON_TARGETS_RSS)
    plot_cluster_annotation_figures(
        adata_micro, regulon_names, args.out_fig_dir, suffix=suffix, clusters_only=True,
        condition_cols=_condition_cols_for_proportions(adata_micro, batch_col),
    )
    if batch_col and "ADNC" in adata_micro.obs.columns:
        plot_donor_level_cluster_fractions_by_adnc(
            adata_micro, args.out_fig_dir, RESULTS_DIR, suffix=suffix,
            cluster_col="leiden_micro", donor_col=batch_col, group_col="ADNC",
        )
    plot_tbx21_validation(adata_micro, regulon_names, args.out_fig_dir, suffix=suffix)

    # Regulon progression vs ADNC: write regulon_progression_adnc.csv and sensitivity to results/
    if not getattr(args, "skip_progression", False) and "X_regulon" in adata_micro.obsm and "donor_id" in adata_micro.obs.columns and "ADNC" in adata_micro.obs.columns:
        from .microglia_progression_regulons import run_analysis as run_progression_analysis
        run_progression_analysis(adata_micro, RESULTS_DIR, args.out_fig_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_per_cluster_regulon_ranking(adata_micro, regulon_names, RESULTS_DIR, cluster_col="leiden_micro", suffix=suffix)
    write_nn_paper_mapping(REGULON_TARGETS_RSS, RESULTS_DIR, suffix=suffix)

    adata_micro.write_h5ad(args.save_adata)
    print("Saved", args.save_adata)
    print("Done. Figures in", args.out_fig_dir)


if __name__ == "__main__":
    main()
