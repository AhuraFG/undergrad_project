#!/usr/bin/env python
"""
Microglia progression: donor-level regulon activity vs ADNC severity.

Two weighted OLS models per regulon:
  (1) Unadjusted: donor mean AUCell ~ ADNC ordinal
  (2) Adjusted:   donor mean AUCell ~ ADNC ordinal + age + sex + cluster fractions
Donor-level cluster fractions from leiden_micro; drop the fraction for the largest cluster.
Weights = sqrt(n_cells per donor) so donors with more cells have higher weight.
Age/sex from adata.obs (flexible column names); omitted with warning if missing.
BH FDR on p_adnc_adj → q_adnc; direction from sign(β_adnc_adj).

Outputs:
  - regulon_progression_adnc.csv (regulon, beta_adnc_unadj, p_adnc_unadj, beta_adnc_adj, p_adnc_adj, q_adnc, direction; sorted by q_adnc)
  - top_progressing_regulons_adnc.png (regulons with p_adj < 0.1, mean AUCell > 0.02; subtitle: β/p unadj, adj, and FDR q_adnc; green if q<0.1 else grey)

Run from code/05_scenicplus:
  python -m scripts.pipeline.post.microglia_progression_regulons [--adata path] [--mdata path]
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Paths (05_scenicplus root)
from ...utils.regulon_mudata import align_regulon_obsm_from_mudata
from ..paths import PIPELINE_ROOT

SCRIPT_DIR = Path(PIPELINE_ROOT)
DATA_INPUTS = SCRIPT_DIR / "data_inputs"
SNAKE_DIR = SCRIPT_DIR / "scplus_pipeline" / "Snakemake"
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "validation_figures"

ADATA_DEFAULT = SCRIPT_DIR / "data_inputs" / "adata_micro_scvi_best.h5ad"
MDATA_PATH = SNAKE_DIR / "scplusmdata.h5mu"

ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
DONOR_COL = "donor_id"
GROUP_COL = "ADNC"
CLUSTER_COL = "leiden_micro"

AGE_COL_CANDIDATES = ["Age at death", "age_at_death", "Age", "age"]
SEX_COL_CANDIDATES = ["Sex", "sex", "Gender", "gender"]

TOP_N_REGULONS_FIG1 = 10
P_MAX_FIG = 0.1  # Include regulons with p_adnc_adj below this for the figure; FDR (q_adnc) is shown in subtitle
MIN_DONORS_REGRESSION = 10


def _safe_imports():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc
    return plt, sc


def _find_obs_column(adata, candidates: list[str]):
    """Return first adata.obs column name in candidates that exists; else None."""
    for c in candidates:
        if c in adata.obs.columns:
            return c
    return None


def _load_adata(adata_path: Path, mdata_path: Path | None):
    """Load AnnData; ensure X_regulon and ADNC; align regulons from mdata if needed."""
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)
    align_regulon_obsm_from_mudata(adata, mdata_path)
    if GROUP_COL not in adata.obs or DONOR_COL not in adata.obs:
        raise ValueError(f"Need {GROUP_COL} and {DONOR_COL} in adata.obs")
    return adata


def run_analysis(adata, out_dir: Path, fig_dir: Path):
    plt, sc = _safe_imports()
    warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")

    X = np.asarray(adata.obsm["X_regulon"])
    regulon_names = list(adata.uns["regulon_names"])
    n_reg = len(regulon_names)
    donors = adata.obs[DONOR_COL].astype(str).values
    adnc_raw = adata.obs[GROUP_COL].astype(str).values

    adnc_order = [s.strip() for s in ADNC_ORDER]
    adnc_to_ord = {g: i for i, g in enumerate(adnc_order)}
    adnc_ordinal = np.array([adnc_to_ord.get(g, np.nan) for g in adnc_raw])
    valid = np.isfinite(adnc_ordinal)
    if valid.sum() < adata.n_obs:
        print(f"  Dropping {(~valid).sum()} cells with ADNC not in {adnc_order}")

    donor_ids = np.unique(donors[valid])
    donor_to_idx = {d: i for i, d in enumerate(donor_ids)}
    adnc_per_donor = np.full(len(donor_ids), np.nan)
    mean_auc = np.full((len(donor_ids), n_reg), np.nan)

    for i, d in enumerate(donor_ids):
        mask = (donors == d) & valid
        adnc_per_donor[i] = adnc_ordinal[mask][0]
        for r in range(n_reg):
            vals = X[mask, r].astype(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                mean_auc[i, r] = np.mean(vals)

    # Covariates: age
    age_col = _find_obs_column(adata, AGE_COL_CANDIDATES)
    if age_col is None:
        print("  Warning: no age column found (tried: " + ", ".join(AGE_COL_CANDIDATES) + "); omitting age from model.")
        age_per_donor = None
    else:
        # donor-level: take first value per donor (should be constant per donor)
        age_per_donor = np.full(len(donor_ids), np.nan)
        for i, d in enumerate(donor_ids):
            mask = (donors == d) & valid
            a = adata.obs.loc[mask, age_col].dropna()
            if len(a) > 0:
                try:
                    age_per_donor[i] = float(pd.to_numeric(a.iloc[0], errors="coerce"))
                except Exception:
                    pass

    # Covariates: sex (0/1)
    sex_col = _find_obs_column(adata, SEX_COL_CANDIDATES)
    if sex_col is None:
        print("  Warning: no sex column found (tried: " + ", ".join(SEX_COL_CANDIDATES) + "); omitting sex from model.")
        sex_per_donor = None
    else:
        # Encode 0/1: first unique -> 0, second -> 1 (or F/female -> 0, M/male -> 1)
        raw = adata.obs.loc[valid, sex_col].astype(str).str.strip().str.lower()
        uniq = raw.unique()
        uniq = [u for u in uniq if u and u not in ("nan", "")]
        if len(uniq) >= 2:
            # Map so that 'f'/'female' -> 0 if present, else first -> 0
            f_like = [u for u in uniq if u.startswith("f") or "female" in u]
            if f_like:
                female_val = f_like[0]
                male_vals = [u for u in uniq if u != female_val]
            else:
                female_val = uniq[0]
                male_vals = uniq[1:]
            sex_map = {female_val: 0}
            for v in male_vals:
                sex_map[v] = 1
        else:
            sex_map = {u: 0 for u in uniq} if uniq else {}
        sex_per_donor = np.full(len(donor_ids), np.nan)
        for i, d in enumerate(donor_ids):
            mask = (donors == d) & valid
            s = adata.obs.loc[mask, sex_col].astype(str).str.strip().str.lower()
            s = s.dropna()
            s = s[s.isin(sex_map)]
            if len(s) > 0:
                sex_per_donor[i] = sex_map.get(s.iloc[0], np.nan)

    # Donor-level cluster fractions (leiden_micro)
    if CLUSTER_COL not in adata.obs:
        raise ValueError(f"Need {CLUSTER_COL} in adata.obs for cluster fractions.")
    clusters = adata.obs[CLUSTER_COL].astype(str).values
    df_cells = pd.DataFrame({"donor": donors, "cluster": clusters})
    df_cells = df_cells[valid]
    count_per_donor_cluster = df_cells.groupby(["donor", "cluster"]).size().unstack(fill_value=0)
    n_cells_per_donor = count_per_donor_cluster.sum(axis=1)
    frac_per_donor_cluster = count_per_donor_cluster.div(n_cells_per_donor, axis=0)
    # Drop column for cluster with most cells (to avoid collinearity)
    total_per_cluster = count_per_donor_cluster.sum(axis=0)
    drop_cluster = total_per_cluster.idxmax()
    cluster_cols = [c for c in frac_per_donor_cluster.columns if c != drop_cluster]
    frac_per_donor_cluster = frac_per_donor_cluster[cluster_cols]
    # Align to donor_ids order
    common_donors = [d for d in donor_ids if d in frac_per_donor_cluster.index]
    if len(common_donors) < MIN_DONORS_REGRESSION:
        raise ValueError(f"Too few donors with cluster data ({len(common_donors)}); need >= {MIN_DONORS_REGRESSION}.")
    idx_common = np.array([list(donor_ids).index(d) for d in common_donors])
    X_frac = frac_per_donor_cluster.loc[common_donors].values
    adnc_common = adnc_per_donor[idx_common]
    mean_auc_common = mean_auc[idx_common, :]
    # Donor weights = sqrt(n_cells) for weighted least squares
    n_cells_arr = n_cells_per_donor.reindex(donor_ids, fill_value=0).values.astype(float)
    weights_per_donor = np.sqrt(np.maximum(n_cells_arr, 1.0))
    weights_common = np.sqrt(np.maximum(n_cells_per_donor.loc[common_donors].values.astype(float), 1.0))

    # Build design matrix for adjusted model: [ADNC_ord, age?, sex?, cluster_fracs]
    # If age/sex cause too many missing rows, omit them so we have enough donors.
    design_parts = [adnc_common.astype(float)]
    if age_per_donor is not None:
        design_parts.append(age_per_donor[idx_common])
    if sex_per_donor is not None:
        design_parts.append(sex_per_donor[idx_common])
    design_parts.append(X_frac)
    X_design = np.column_stack(design_parts)
    design_valid = np.isfinite(X_design).all(axis=1)
    n_valid = design_valid.sum()
    if n_valid < MIN_DONORS_REGRESSION and (age_per_donor is not None or sex_per_donor is not None):
        # Retry without age and sex so we don't drop all donors
        design_parts = [adnc_common.astype(float), X_frac]
        X_design = np.column_stack(design_parts)
        design_valid = np.isfinite(X_design).all(axis=1)
        n_valid = design_valid.sum()
        if n_valid < MIN_DONORS_REGRESSION:
            raise ValueError(f"Too few donors with complete design ({n_valid}); need >= {MIN_DONORS_REGRESSION}.")
        print("  Warning: omitting age/sex from model (too many missing); using ADNC + cluster fractions only.")
        age_per_donor = None
        sex_per_donor = None
    elif n_valid < MIN_DONORS_REGRESSION:
        raise ValueError(f"Too few donors with complete design ({n_valid}); need >= {MIN_DONORS_REGRESSION}.")

    import statsmodels.api as sm
    beta_unadj_list = []
    p_unadj_list = []
    beta_adj_list = []
    p_adj_list = []
    ols_failed_once = False
    for r in range(n_reg):
        # Unadjusted: y ~ ADNC (all donors with valid ADNC and y)
        y_all = mean_auc[:, r]
        ok_unadj = np.isfinite(y_all) & np.isfinite(adnc_per_donor)
        if ok_unadj.sum() < MIN_DONORS_REGRESSION:
            beta_unadj_list.append(np.nan)
            p_unadj_list.append(np.nan)
        else:
            try:
                X_u = sm.add_constant(adnc_per_donor[ok_unadj])
                w_u = weights_per_donor[ok_unadj]
                m_u = sm.WLS(y_all[ok_unadj], X_u, weights=w_u).fit()
                beta_unadj_list.append(float(m_u.params[1]))
                p_unadj_list.append(float(m_u.pvalues[1]))
            except Exception as e:
                if not ols_failed_once:
                    print("  WLS (unadjusted) failed for at least one regulon:", e)
                    ols_failed_once = True
                beta_unadj_list.append(np.nan)
                p_unadj_list.append(np.nan)

        # Adjusted: y ~ ADNC + age + sex + cluster fractions (common_donors with full design)
        y = mean_auc_common[:, r]
        ok = np.isfinite(y) & design_valid
        if ok.sum() < MIN_DONORS_REGRESSION:
            beta_adj_list.append(np.nan)
            p_adj_list.append(np.nan)
            continue
        X_ok = sm.add_constant(X_design[ok])
        y_ok = y[ok]
        w_ok = weights_common[ok]
        try:
            m = sm.WLS(y_ok, X_ok, weights=w_ok).fit()
            beta_adj_list.append(float(m.params[1]))
            p_adj_list.append(float(m.pvalues[1]))
        except Exception as e:
            if not ols_failed_once:
                print("  WLS (adjusted) failed for at least one regulon:", e)
                ols_failed_once = True
            beta_adj_list.append(np.nan)
            p_adj_list.append(np.nan)

    p_adj_a = np.array(p_adj_list)
    valid_p = np.isfinite(p_adj_a)
    q_adnc = np.full(n_reg, np.nan)
    if valid_p.sum() > 0:
        _, q_adnc_valid, _, _ = multipletests(p_adj_a[valid_p], method="fdr_bh")
        q_adnc[valid_p] = q_adnc_valid

    beta_adj_a = np.array(beta_adj_list)
    direction = np.where(np.isfinite(beta_adj_a), np.where(beta_adj_a > 0, "up", "down"), "flat")

    df_results = pd.DataFrame({
        "regulon": regulon_names,
        "beta_adnc_unadj": beta_unadj_list,
        "p_adnc_unadj": p_unadj_list,
        "beta_adnc_adj": beta_adj_list,
        "p_adnc_adj": p_adj_list,
        "q_adnc": q_adnc,
        "direction": direction,
    })
    # Add mean AUCell per regulon (used for figure inclusion and by enrichment --top-progressing-only)
    mean_auc_per_reg = np.nanmean(mean_auc, axis=0)
    mean_by_reg = pd.Series(mean_auc_per_reg, index=regulon_names)
    df_results["mean_auc"] = df_results["regulon"].map(mean_by_reg)
    df_results = df_results.sort_values("q_adnc", na_position="last").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_dir / "regulon_progression_adnc.csv", index=False, na_rep="nan")
    print("Saved", out_dir / "regulon_progression_adnc.csv")

    # Plot data: donor × regulon mean AUCell and ADNC label
    donor_adnc_labels = [adnc_order[int(adnc_per_donor[i])] if np.isfinite(adnc_per_donor[i]) else "" for i in range(len(donor_ids))]
    plot_df = []
    for i, d in enumerate(donor_ids):
        adnc_lab = donor_adnc_labels[i]
        for r, name in enumerate(regulon_names):
            if np.isfinite(mean_auc[i, r]):
                plot_df.append({
                    "donor": d,
                    "ADNC": adnc_lab,
                    "ADNC_ordinal": adnc_per_donor[i],
                    "regulon": name,
                    "mean_AUCell": mean_auc[i, r],
                })
    plot_df = pd.DataFrame(plot_df)

    # ---------- Figure: Include regulons with p_adj < P_MAX_FIG (0.1) and mean AUCell > 0.02; show FDR (q_adnc) in subtitle ----------
    mean_above = df_results["mean_auc"] > 0.02
    p_adj_ok = df_results["p_adnc_adj"].fillna(1) < P_MAX_FIG
    sig_mask = p_adj_ok & mean_above
    sig = df_results[sig_mask].copy()
    if len(sig) == 0:
        sig = df_results[df_results["q_adnc"].notna()].head(TOP_N_REGULONS_FIG1)
    sig["abs_beta_adj"] = sig["beta_adnc_adj"].abs()
    sig = sig.sort_values("abs_beta_adj", ascending=False, na_position="last")
    top_regulons = sig["regulon"].tolist()
    if len(top_regulons) == 0:
        top_regulons = df_results["regulon"].head(TOP_N_REGULONS_FIG1).tolist()

    fig1_df = plot_df[plot_df["regulon"].isin(top_regulons)].copy()
    fig1_df["ADNC"] = pd.Categorical(fig1_df["ADNC"], categories=adnc_order, ordered=True)
    n_top = len(top_regulons)
    ncols = min(4, n_top)
    nrows = (n_top + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, reg in enumerate(top_regulons):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        sub = fig1_df[fig1_df["regulon"] == reg]
        res = df_results[df_results["regulon"] == reg].iloc[0]
        beta_u = res["beta_adnc_unadj"]
        p_u = res["p_adnc_unadj"]
        beta_a = res["beta_adnc_adj"]
        p_a = res["p_adnc_adj"]
        q = res["q_adnc"]
        title = reg[:40] + "..." if len(reg) > 40 else reg
        ax.text(0.5, 1.08, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12, clip_on=False)
        def _fmt(b, p):
            if np.isfinite(b) and np.isfinite(p):
                return f"β={b:.4f}, p={p:.4f}"
            return "β=—, p=—"
        subtitle = f"unadj: {_fmt(beta_u, p_u)}  |  adj: {_fmt(beta_a, p_a)}"
        sub_color = "green" if np.isfinite(q) and q < 0.1 else "dimgray"
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=sub_color, fontweight="bold", clip_on=False)
        sub_ord = sub.sort_values("ADNC_ordinal")
        for adnc_cat in adnc_order:
            vals = sub_ord[sub_ord["ADNC"] == adnc_cat]["mean_AUCell"].values
            if len(vals) == 0:
                continue
            x_pos = adnc_order.index(adnc_cat)
            jitter = np.random.RandomState(idx).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(x_pos + jitter, vals, alpha=0.7, s=45, color="black", zorder=3)
        parts = ax.violinplot(
            [sub_ord[sub_ord["ADNC"] == adnc_cat]["mean_AUCell"].values for adnc_cat in adnc_order],
            positions=range(len(adnc_order)),
            showmeans=True,
            showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("lightsteelblue")
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(adnc_order)))
        ax.set_xticklabels(adnc_order, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Donor mean AUCell", fontsize=11)
        ax.set_xlabel("", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if idx == 0:
            from matplotlib.lines import Line2D
            leg_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=8, label="Donors", linestyle="")]
            ax.legend(handles=leg_handles, loc="upper right", fontsize=10)
    for idx in range(len(top_regulons), axes.size):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "top_progressing_regulons_adnc.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fig_dir / "top_progressing_regulons_adnc.png")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Microglia regulon progression vs ADNC (OLS + covariates)")
    parser.add_argument("--adata", type=Path, default=ADATA_DEFAULT, help="AnnData with X_regulon, donor_id, ADNC, leiden_micro")
    parser.add_argument("--mdata", type=Path, default=MDATA_PATH, help="MuData for AUCell if adata has no X_regulon")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    adata = _load_adata(args.adata, args.mdata)
    run_analysis(adata, args.out_dir, args.fig_dir)
    print("Done.")


if __name__ == "__main__":
    main()
