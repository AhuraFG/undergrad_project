"""Steps 5–6: regulon QC / filter and pipeline plots."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .paths import MDATA_PATH, PLOTS_DIR, PIPELINE_ROOT, RESULTS_DIR, SNAKE_DIR


def regulon_to_tf(name):
    m = re.match(r"^([A-Za-z0-9_-]+)_direct_", str(name))
    return m.group(1) if m else None


def step_spurious_filter(args):
    import mudata as md

    print("[Step 5] Filtering spurious regulons...")
    if not os.path.isfile(MDATA_PATH):
        print("  Skipped (no scplusmdata.h5mu).")
        return
    mdata = md.read_h5mu(MDATA_PATH)
    rna = mdata.mod["scRNA_counts"]
    auc = mdata.mod["direct_gene_based_AUC"]
    X_auc = np.asarray(auc.X.toarray() if issparse(auc.X) else auc.X)
    regulon_names = list(auc.var_names)

    def get_expr(adata, gene):
        if gene not in adata.var_names:
            return None
        j = list(adata.var_names).index(gene)
        x = adata.X[:, j]
        return np.asarray(x.toarray()).ravel() if issparse(x) else np.asarray(x).ravel()

    rows = []
    for i, reg in enumerate(regulon_names):
        tf = regulon_to_tf(reg)
        if tf is None:
            rows.append({"regulon": reg, "TF": None, "pct_cells": 0, "mean_tf_expr": 0, "corr": np.nan})
            continue
        tf_expr = get_expr(rna, tf)
        if tf_expr is None:
            rows.append({"regulon": reg, "TF": tf, "pct_cells": 0, "mean_tf_expr": 0, "corr": np.nan})
            continue
        tf_expr = np.log1p(np.maximum(tf_expr, 0))
        auc_vec = X_auc[:, i].ravel()
        pct = 100 * np.mean(tf_expr > 0)
        mean_expr = float(np.mean(tf_expr))
        corr = np.corrcoef(auc_vec, tf_expr)[0, 1] if np.std(auc_vec) > 1e-10 and np.std(tf_expr) > 1e-10 else np.nan
        rows.append({"regulon": reg, "TF": tf, "pct_cells": pct, "mean_tf_expr": mean_expr, "corr": corr})

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "regulon_credibility_table.csv"), index=False)

    tf2g_path = os.path.join(SNAKE_DIR, "tf_to_gene_adj.tsv")
    if os.path.isfile(tf2g_path):
        tf2g = pd.read_csv(tf2g_path, sep="\t")
        tf2g.to_csv(os.path.join(RESULTS_DIR, "tf_to_gene_adjacency.csv"), index=False)

    min_pct = getattr(args, "credible_min_pct_cells", 5.0)
    cred_idx = [i for i, row in enumerate(rows) if row["pct_cells"] >= min_pct]
    if not cred_idx:
        print("  WARNING: No regulons passed credibility filter; keeping all.")
        cred_idx = list(range(len(regulon_names)))
    n_filtered = len(regulon_names) - len(cred_idx)
    with open(os.path.join(RESULTS_DIR, "regulon_qc_summary.txt"), "w") as f:
        f.write(f"Total regulons: {len(df)}. Kept {len(cred_idx)} (pct_cells >= {min_pct}%); filtered {n_filtered}.\n")
    if cred_idx:
        cred_means = X_auc[:, cred_idx].mean(axis=0)
        order = np.argsort(-cred_means)
        top_df = pd.DataFrame({"regulon": [regulon_names[cred_idx[i]] for i in order], "mean_AUCell": cred_means[order]})
        top_df.to_csv(os.path.join(RESULTS_DIR, "top_tfs.csv"), index=False)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_show = min(30, len(top_df))
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(n_show), top_df["mean_AUCell"].values[:n_show], color="steelblue", alpha=0.8)
            ax.set_yticks(range(n_show))
            ax.set_yticklabels(top_df["regulon"].values[:n_show], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Mean AUCell")
            ax.set_title("Top TFs by mean AUCell")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "top_tfs.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        cluster_col = None
        if hasattr(mdata, "obs") and mdata.obs is not None and mdata.obs.shape[1] > 0:
            for cand in (
                "scRNA_counts:Subclass",
                "scRNA_counts:Supertype",
                "scRNA_counts:Class",
                "Subclass",
                "Supertype",
                "Class",
                "leiden",
                "louvain",
                "cluster",
            ):
                if cand in mdata.obs.columns:
                    cl = mdata.obs[cand].astype(str)
                    if cl.nunique() >= 2:
                        cluster_col = cl.values
                        break
        if cluster_col is not None:
            cred_auc = X_auc[:, cred_idx]
            rss_scores = []
            for j in range(cred_auc.shape[1]):
                x = np.asarray(cred_auc[:, j].ravel())
                means_per_cluster = pd.Series(x).groupby(cluster_col).mean()
                rss_scores.append(float(means_per_cluster.var()) if len(means_per_cluster) >= 2 else 0.0)
            rss_scores = np.array(rss_scores)
            order_rss = np.argsort(-rss_scores)
            top_df_rss = pd.DataFrame(
                {
                    "regulon": [regulon_names[cred_idx[i]] for i in order_rss],
                    "RSS": rss_scores[order_rss],
                }
            )
            top_df_rss.to_csv(os.path.join(RESULTS_DIR, "top_tfs_rss.csv"), index=False)
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                n_show = min(30, len(top_df_rss))
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(range(n_show), top_df_rss["RSS"].values[:n_show], color="darkgreen", alpha=0.8)
                ax.set_yticks(range(n_show))
                ax.set_yticklabels(top_df_rss["regulon"].values[:n_show], fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("RSS (cluster specificity)")
                ax.set_title("Top TFs by cluster specificity (RSS)")
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, "top_tfs_rss.png"), dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                pass
    print("  Saved results/ (CSVs), plots/ (PNGs).")


def step_plots(_args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    import mudata as md
    import seaborn as sns

    print("[Step 6] Generating plots...")
    if not os.path.isfile(MDATA_PATH):
        print("  Skipped (no scplusmdata.h5mu).")
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)

    mdata = md.read_h5mu(MDATA_PATH)
    gene_auc = mdata.mod["direct_gene_based_AUC"]
    region_auc = mdata.mod["direct_region_based_AUC"]
    X_g = np.asarray(gene_auc.X.toarray() if issparse(gene_auc.X) else gene_auc.X)
    X_r = np.asarray(region_auc.X.toarray() if issparse(region_auc.X) else region_auc.X)

    data_mat = pd.DataFrame(
        np.hstack([X_g, X_r]),
        index=gene_auc.obs_names,
        columns=list(gene_auc.var_names) + list(region_auc.var_names),
    )
    data_mat = data_mat.loc[:, data_mat.sum(axis=0) > 0]

    corr = data_mat.corr()
    sim = 1 - corr
    Z = linkage(np.clip(squareform(sim), 0, sim.to_numpy().max()), "average")
    order = np.argsort(fcluster(Z, 0.1))
    clustered = data_mat.iloc[:, order].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(clustered, cmap="viridis", square=True, ax=ax, robust=True, xticklabels=False, yticklabels=True)
    ax.set_title("eRegulon activity correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eregulon_activity_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mean_auc = X_g.mean(axis=0).ravel()
    tf_df = (
        pd.DataFrame({"TF": list(gene_auc.var_names), "mean_AUCell": mean_auc})
        .sort_values("mean_AUCell", ascending=False)
        .head(30)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(tf_df)), tf_df["mean_AUCell"].values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(tf_df)))
    ax.set_yticklabels(tf_df["TF"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean AUCell")
    ax.set_title("eRegulon TF overview (top 30)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eregulon_tf_overview.png"), dpi=150, bbox_inches="tight")
    plt.close()

    rss_df = None
    for candidate in (os.path.join(RESULTS_DIR, "top_tfs_rss.csv"), os.path.join(PIPELINE_ROOT, "top_tfs_after_filter_rss.csv")):
        if os.path.isfile(candidate):
            rss_df = pd.read_csv(candidate)
            if "RSS" not in rss_df.columns or "regulon" not in rss_df.columns:
                if "RSS" in rss_df.columns and len(rss_df.columns) >= 2:
                    rss_df = rss_df.rename(columns={rss_df.columns[0]: "regulon"})
                else:
                    rss_df = None
            break
    if rss_df is None:
        cluster_col = None
        if hasattr(mdata, "obs") and mdata.obs is not None and mdata.obs.shape[1] > 0:
            for cand in (
                "scRNA_counts:Subclass",
                "scRNA_counts:Supertype",
                "scRNA_counts:Class",
                "Subclass",
                "Supertype",
                "Class",
                "leiden",
                "louvain",
                "cluster",
            ):
                if cand in mdata.obs.columns:
                    cl = mdata.obs[cand].astype(str)
                    if cl.nunique() >= 2:
                        cluster_col = cl.values
                        break
        if cluster_col is not None:
            rss_scores = []
            for j in range(X_g.shape[1]):
                x = np.asarray(X_g[:, j].ravel())
                means_per_cluster = pd.Series(x).groupby(cluster_col).mean()
                rss_scores.append(float(means_per_cluster.var()) if len(means_per_cluster) >= 2 else 0.0)
            rss_scores = np.array(rss_scores)
            order_rss = np.argsort(-rss_scores)
            rss_df = pd.DataFrame(
                {
                    "regulon": [gene_auc.var_names[i] for i in order_rss],
                    "RSS": rss_scores[order_rss],
                }
            )
    if rss_df is not None and len(rss_df) > 0:
        n_show = min(30, len(rss_df))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(n_show), rss_df["RSS"].values[:n_show], color="darkgreen", alpha=0.8)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(rss_df["regulon"].values[:n_show], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("RSS (cluster specificity)")
        ax.set_title("Top TFs by RSS (top 30)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "top_tfs_rss.png"), dpi=150, bbox_inches="tight")
        plt.close()

    meta = mdata.uns.get("direct_e_regulon_metadata")
    if meta is not None:
        meta = pd.DataFrame(meta) if not isinstance(meta, pd.DataFrame) else meta
        reg_col = None
        for c in ("eRegulon_name", "Regulon", "TF"):
            if c in meta.columns:
                reg_col = c
                break
        if reg_col is not None and "Gene" in meta.columns:
            targets_per_reg = meta.groupby(reg_col)["Gene"].apply(lambda s: ";".join(sorted(s.astype(str).unique()))).to_dict()
            regulons = list(gene_auc.var_names)
            mean_auc_vec = X_g.mean(axis=0).ravel()

            def _reg_base(r):
                if "_(" in r and "g)" in r:
                    return r.rsplit("_(", 1)[0]
                return r

            rows_rss = []
            rows_auc = []
            rss_map = dict(zip(rss_df["regulon"], rss_df["RSS"])) if rss_df is not None and len(rss_df) > 0 else {}
            for i, reg in enumerate(regulons):
                base = _reg_base(reg)
                targets = targets_per_reg.get(base, targets_per_reg.get(reg, ""))
                if not targets:
                    continue
                m_auc = float(mean_auc_vec[i])
                rows_auc.append({"regulon": reg, "targets": targets, "mean_AUCell": m_auc})
                rss_val = rss_map.get(reg, np.nan)
                rows_rss.append({"regulon": reg, "targets": targets, "RSS": rss_val})
            if rows_rss:
                pd.DataFrame(rows_rss).sort_values("RSS", ascending=False).to_csv(
                    os.path.join(RESULTS_DIR, "regulon_targets_by_rss.csv"), index=False
                )
            if rows_auc:
                pd.DataFrame(rows_auc).sort_values("mean_AUCell", ascending=False).to_csv(
                    os.path.join(RESULTS_DIR, "regulon_targets_by_aucell.csv"), index=False
                )
            if rows_rss or rows_auc:
                print("  Saved results/regulon_targets_by_rss.csv, results/regulon_targets_by_aucell.csv")
        G = nx.DiGraph()
        for _, row in meta.iterrows():
            G.add_edge(row["TF"], row["Gene"], weight=float(row.get("importance_TF2G", 1)))
        top_tfs = {regulon_to_tf(r) for r in tf_df["TF"].head(25)}
        edges = [(u, v) for u, v in G.edges() if u in top_tfs][:600]
        if edges:
            sub = nx.DiGraph()
            for u, v in edges:
                sub.add_edge(u, v, weight=G.edges[u, v].get("weight", 1))
            G = sub
            fig, ax = plt.subplots(figsize=(14, 10))
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, arrows=True, arrowsize=8)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80, node_color="lightblue", alpha=0.9)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
            ax.set_title("TF–gene network (top eRegulons)")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "tf_gene_network.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print("  Saved plots/eregulon_activity_heatmap.png, plots/eregulon_tf_overview.png, plots/tf_gene_network.png, plots/top_tfs_rss.png.")
