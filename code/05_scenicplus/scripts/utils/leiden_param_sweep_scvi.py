#!/usr/bin/env python
"""
Leiden parameter selection on scVI latent for microglia.

Evaluates only two metrics: (i) mean ARI stability across seeds, (ii) silhouette score
in scVI latent space. Uses adata.obsm['X_scVI'] and sc.pp.neighbors(use_rep='X_scVI').

Grid: n_neighbors in {15, 20, 30}, resolution in {0.4, 0.6, 0.8, 1.0}.
Guardrails: reject min_cluster_size < 50 or n_clusters > 15.
Best: mean_ari >= 0.50 and guardrails pass, then highest mean_ari; tie within 0.02 ARI
      broken by higher silhouette.

Outputs: leiden_param_sweep_scvi_ari_silhouette.csv,
         leiden_param_sweep_diagnostics_ari_silhouette.png,
         best_leiden_params_scvi.txt, adata_micro_scvi_best.h5ad,
         umap_leiden_best.png, umap_donor_best.png.
Downstream: main validation pipeline regenerates regulon dotplot and top-RSS regulon UMAPs
            using the new Leiden labels for consistency.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

from .regulon_mudata import align_regulon_obsm_from_mudata

# Paths relative to 05_scenicplus (parent of scripts/utils/)
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "validation_figures"
MDATA_PATH = SCRIPT_DIR / "scplus_pipeline" / "Snakemake" / "scplusmdata.h5mu"

# Grid and constants
N_NEIGHBORS_GRID = [15, 20, 30]
RESOLUTION_GRID = [0.4, 0.6, 0.8, 1.0]
N_SEEDS = 10  # seeds 0 .. N_SEEDS-1
SCVI_KEY = "X_scVI"
DONOR_COL = "donor_id"

# Guardrails (~3k cells: avoid over-splitting)
MIN_CLUSTER_SIZE = 50
MAX_N_CLUSTERS = 15
MEAN_ARI_MIN = 0.50
ARI_TIE_TOL = 0.02


def _safe_imports():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc
    return plt, sc


def _load_adata(adata_path: Path, mdata_path: Path | None):
    """Load adata; ensure X_scVI; optionally align X_regulon from mdata for downstream plots."""
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)
    if SCVI_KEY not in adata.obsm:
        raise KeyError(f"{SCVI_KEY} not in adata.obsm; run scVI first.")
    align_regulon_obsm_from_mudata(adata, mdata_path)
    return adata


def run_sweep(adata, out_csv: Path, out_best_txt: Path, out_fig: Path,
              fig_dir: Path, adata_best_path: Path):
    plt, sc = _safe_imports()
    warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")

    X_scvi = np.asarray(adata.obsm[SCVI_KEY])
    if X_scvi.ndim != 2:
        raise ValueError(f"{SCVI_KEY} must be 2D")

    rows = []
    for k in N_NEIGHBORS_GRID:
        sc.pp.neighbors(adata, use_rep=SCVI_KEY, n_neighbors=k)
        for res in RESOLUTION_GRID:
            labels_per_seed = []
            for seed in range(N_SEEDS):
                sc.tl.leiden(adata, resolution=res, key_added="leiden_sweep", random_state=seed)
                lab = adata.obs["leiden_sweep"].astype(str).values
                labels_per_seed.append(lab)

            # Mean and std pairwise ARI across the N_SEEDS runs
            aris = []
            for i in range(N_SEEDS):
                for j in range(i + 1, N_SEEDS):
                    aris.append(adjusted_rand_score(labels_per_seed[i], labels_per_seed[j]))
            mean_ari = float(np.mean(aris)) if aris else np.nan
            std_ari = float(np.std(aris)) if aris else np.nan

            # Seed-0 partition for silhouette and cluster stats
            labels0 = labels_per_seed[0]
            n_clusters = int(len(np.unique(labels0)))
            sizes = pd.Series(labels0).value_counts()
            min_cluster_size = int(sizes.min()) if len(sizes) else 0

            # Silhouette on X_scVI (numeric labels for sklearn)
            le = LabelEncoder()
            labels_numeric = le.fit_transform(labels0)
            if n_clusters >= 2 and len(np.unique(labels_numeric)) >= 2:
                sil = float(silhouette_score(X_scvi, labels_numeric))
            else:
                sil = np.nan

            # Guardrails
            passes = (min_cluster_size >= MIN_CLUSTER_SIZE and
                     n_clusters <= MAX_N_CLUSTERS and
                     np.isfinite(mean_ari) and mean_ari >= MEAN_ARI_MIN)

            rows.append({
                "n_neighbors": k,
                "resolution": res,
                "n_clusters": n_clusters,
                "min_cluster_size": min_cluster_size,
                "mean_ari": mean_ari,
                "std_ari": std_ari,
                "silhouette": sil,
                "passes_filters": passes,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved", out_csv)

    # Best: (a) keep only passes_filters, (b) highest mean_ari; tie within ARI_TIE_TOL break by silhouette
    df_ok = df[df["passes_filters"]]
    if df_ok.empty:
        df_ok = df
        print("Warning: no combo passed filters; choosing among all.")
    max_ari = df_ok["mean_ari"].max()
    # Among combos within ARI_TIE_TOL of the max, pick highest silhouette
    tied = df_ok[df_ok["mean_ari"] >= max_ari - ARI_TIE_TOL]
    sil_vals = tied["silhouette"].fillna(-np.inf)
    best_idx = sil_vals.idxmax()
    best = tied.loc[best_idx]
    best_k = int(best["n_neighbors"])
    best_res = float(best["resolution"])
    best_params = {"n_neighbors": best_k, "resolution": best_res}
    with open(out_best_txt, "w") as f:
        f.write(json.dumps(best_params, indent=2))
    print("Best Leiden params:", best_params, "->", out_best_txt)

    # Diagnostic: silhouette vs mean_ari, color by n_neighbors, label resolution
    fig, ax = plt.subplots(figsize=(7, 5))
    for ki, k in enumerate(N_NEIGHBORS_GRID):
        sub = df[df["n_neighbors"] == k]
        for _, r in sub.iterrows():
            x, y = r["mean_ari"], r["silhouette"]
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter(x, y, c=f"C{ki}", s=80, alpha=0.8)
                ax.annotate(f"{r['resolution']}", (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axvline(x=MEAN_ARI_MIN, color="gray", linestyle="--", alpha=0.7, label=f"mean_ari≥{MEAN_ARI_MIN}")
    for ki, k in enumerate(N_NEIGHBORS_GRID):
        ax.scatter([], [], c=f"C{ki}", s=60, label=f"n_neighbors={k}")
    ax.set_xlabel("Mean ARI (stability)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette vs mean ARI (label=resolution)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out_fig)

    # Rebuild final clustering with chosen params
    sc.pp.neighbors(adata, use_rep=SCVI_KEY, n_neighbors=best_k)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=best_res, key_added="leiden_micro", random_state=0)
    if adata.raw is None:
        adata.raw = adata.copy()
    adata.write_h5ad(adata_best_path)
    print("Saved", adata_best_path)

    fig_dir.mkdir(parents=True, exist_ok=True)
    sc.pl.umap(adata, color="leiden_micro", legend_loc="on data", show=False)
    plt.gcf().savefig(fig_dir / "umap_leiden_best.png", dpi=150, bbox_inches="tight")
    plt.close()
    if DONOR_COL in adata.obs:
        sc.pl.umap(adata, color=DONOR_COL, legend_loc="right margin", show=False)
        plt.gcf().savefig(fig_dir / "umap_donor_best.png", dpi=150, bbox_inches="tight")
        plt.close()
    print("Saved UMAPs to", fig_dir)
    print("Downstream: run validation pipeline to regenerate regulon dotplot and top-RSS regulon UMAPs with these labels.")

    return best_k, best_res


def main():
    parser = argparse.ArgumentParser(description="Leiden parameter sweep (ARI + silhouette) on scVI latent")
    parser.add_argument("--adata", type=Path, default=SCRIPT_DIR / "data_inputs" / "adata_microglia_embedding_scvi.h5ad", help="AnnData with X_scVI")
    parser.add_argument("--mdata", type=Path, default=MDATA_PATH, help="MuData for AUCell (for downstream regulon plots)")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "leiden_param_sweep_scvi_ari_silhouette.csv")
    parser.add_argument("--out-best", type=Path, default=RESULTS_DIR / "best_leiden_params_scvi.txt")
    parser.add_argument("--out-fig", type=Path, default=FIG_DIR / "leiden_param_sweep_diagnostics_ari_silhouette.png")
    parser.add_argument("--out-adata", type=Path, default=SCRIPT_DIR / "data_inputs" / "adata_micro_scvi_best.h5ad")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adata = _load_adata(args.adata, args.mdata)

    run_sweep(
        adata,
        out_csv=args.out_csv,
        out_best_txt=args.out_best,
        out_fig=args.out_fig,
        fig_dir=FIG_DIR,
        adata_best_path=args.out_adata,
    )
    print("Done.")


if __name__ == "__main__":
    main()
