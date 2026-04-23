#!/usr/bin/env python
"""
Microglia progression: donor-level regulon activity vs ADNC severity.

Two donor-level weighted least squares (WLS) models per regulon (not OLS — weights applied):
  (1) Unadjusted: donor mean AUCell ~ ADNC ordinal
  (2) Adjusted:   donor mean AUCell ~ ADNC ordinal + age + sex [+ PMI, RIN, APOE if in obs]
We do not adjust for leiden_micro cluster fractions (within-type states are mediators/outcomes, not confounders).
Weights = sqrt(n_cells per donor) so donors with more cells have higher weight.
Age/sex from adata.obs (flexible column names); optional PMI/RIN/APOE added only if present with enough non-missing donors.
BH FDR on p_adnc_adj → q_adnc; direction from sign(β_adnc_adj).

Outputs:
  - regulon_progression_adnc.csv (regulon, beta_adnc_unadj, p_adnc_unadj, beta_adnc_adj, p_adnc_adj, q_adnc, direction; sorted by q_adnc)
  - top_progressing_regulons_adnc.png (regulons with q_adnc < 0.05 only; subtitle: β_unadj, β_adj, q_FDR)
  - top_progressing_regulon_tfs_adnc.png (donor-mean TF expression by ADNC for TFs whose regulons pass q_adnc < 0.05)

Run from code/05_scenicplus:
  python -m scripts.pipeline.post.microglia_progression_regulons [--adata path] [--mdata path]
"""

from __future__ import annotations

import argparse
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
# Continuous disease progression predictor (SEAAD); used instead of ordinal ADNC in regression
# for higher statistical power. Falls back to ordinal ADNC if column absent.
CPPS_COL = "Continuous Pseudo-progression Score"

AGE_COL_CANDIDATES = ["Age at death", "age_at_death", "Age", "age"]
SEX_COL_CANDIDATES = ["Sex", "sex", "Gender", "gender"]
# Donor-level technical/biological confounders (only included if column exists and enough complete donors)
OPTIONAL_CONFOUNDER_CANDIDATES = [
    ("PMI", ["PMI", "Post-mortem interval", "post_mortem_interval"]),
    ("RIN", ["RIN", "rin", "RIN score"]),
    ("APOE4", ["APOE4 status", "APOE4", "APOE genotype"]),
]

TOP_N_REGULONS_FIG1 = 10
Q_MAX_FIG = 0.05  # Include only regulons with q_adnc below this threshold in the figure
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


def _tf_from_regulon_name(regulon: str) -> str:
    """Extract TF symbol from SCENIC+ regulon name like TF_direct_+/+_(ng)."""
    return str(regulon).split("_", 1)[0]


def _expression_matrix_and_var_names(adata):
    """Return expression matrix and gene names, preferring raw log1p counts when present."""
    if adata.raw is not None:
        return adata.raw.X, adata.raw.var_names.astype(str)
    return adata.X, adata.var_names.astype(str)


def _dense_vector(x) -> np.ndarray:
    from scipy.sparse import issparse

    return np.asarray(x.A).ravel() if issparse(x) else np.asarray(x).ravel()


def _fit_wls_predictor(
    y: np.ndarray,
    predictor: np.ndarray,
    design_valid: np.ndarray,
    x_design: np.ndarray,
    weights: np.ndarray,
    min_donors: int,
):
    import statsmodels.api as sm

    beta_unadj = np.nan
    p_unadj = np.nan
    ok_unadj = np.isfinite(y) & np.isfinite(predictor)
    if int(ok_unadj.sum()) >= min_donors:
        x_u = sm.add_constant(predictor[ok_unadj])
        w_u = weights[ok_unadj]
        m_u = sm.WLS(y[ok_unadj], x_u, weights=w_u).fit()
        beta_unadj = float(m_u.params[1])
        p_unadj = float(m_u.pvalues[1])

    beta_adj = np.nan
    p_adj = np.nan
    ok_adj = np.isfinite(y) & design_valid
    if int(ok_adj.sum()) >= min_donors:
        x_ok = sm.add_constant(x_design[ok_adj])
        y_ok = y[ok_adj]
        w_ok = weights[ok_adj]
        m = sm.WLS(y_ok, x_ok, weights=w_ok).fit()
        beta_adj = float(m.params[1])
        p_adj = float(m.pvalues[1])

    return beta_unadj, p_unadj, beta_adj, p_adj


def _parse_seaad_value(s) -> float:
    """Parse SEAAD categorical metadata strings to numeric.

    Handles:
    - Direct numeric: "75" → 75.0
    - Range bins: "78 to 89 years old" → 83.5 (midpoint)
    - 90+ format: "90+ years old" → 90.0
    - Y/N binary: "Y" → 1.0, "N" → 0.0
    """
    import re
    s = str(s).strip()
    # Direct numeric
    try:
        v = float(s)
        if np.isfinite(v):
            return v
    except (ValueError, TypeError):
        pass
    sl = s.lower()
    # Y/N binary (APOE4 status etc.)
    if sl in ("y", "yes"):
        return 1.0
    if sl in ("n", "no"):
        return 0.0
    # "90+ years old" style
    m = re.match(r"([\d.]+)\+", sl)
    if m:
        return float(m.group(1))
    # "X to Y unit" range → midpoint
    m = re.match(r"([\d.]+)\s+to\s+([\d.]+)", sl)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


def _donor_first_numeric(adata, donor_ids, donors, valid, col: str) -> np.ndarray:
    """One numeric value per donor (first cell); handles SEAAD binned/categorical formats."""
    out = np.full(len(donor_ids), np.nan)
    idx = {d: i for i, d in enumerate(donor_ids)}
    o = adata.obs.loc[valid, col].astype(str)
    d_v = donors[valid]
    seen: set[str] = set()
    for donor, val in zip(d_v, o):
        if donor in seen:
            continue
        seen.add(donor)
        if donor not in idx:
            continue
        parsed = _parse_seaad_value(val)
        if np.isfinite(parsed):
            out[idx[donor]] = parsed
    return out


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

    # Use Continuous Pseudo-progression Score as regression predictor if available
    # (more powerful than ordinal ADNC; constant within donor; pre-computed by SEAAD authors)
    if CPPS_COL in adata.obs.columns:
        cpps_per_donor = _donor_first_numeric(adata, donor_ids, donors, valid, CPPS_COL)
        n_cpps_ok = int(np.isfinite(cpps_per_donor).sum())
        print(f"  Using '{CPPS_COL}' as regression predictor ({n_cpps_ok}/{len(donor_ids)} donors).")
        if n_cpps_ok < MIN_DONORS_REGRESSION:
            print("  Warning: too few CPPS values; falling back to ordinal ADNC.")
            cpps_per_donor = adnc_per_donor.copy()
            using_cpps = False
        else:
            using_cpps = True
    else:
        print(f"  '{CPPS_COL}' not found; using ordinal ADNC as predictor.")
        cpps_per_donor = adnc_per_donor.copy()
        using_cpps = False
    predictor_label = "CPPS" if using_cpps else "ADNC"

    # Covariates: age (handles SEAAD binned strings e.g. "78 to 89 years old", "90+ years old")
    age_col = _find_obs_column(adata, AGE_COL_CANDIDATES)
    if age_col is None:
        print("  Warning: no age column found (tried: " + ", ".join(AGE_COL_CANDIDATES) + "); omitting age from model.")
        age_per_donor = None
    else:
        age_per_donor = _donor_first_numeric(adata, donor_ids, donors, valid, age_col)
        n_age_ok = int(np.isfinite(age_per_donor).sum())
        print(f"  Age parsed from {age_col!r}: {n_age_ok}/{len(donor_ids)} donors have finite values.")

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

    # sqrt(n_cells) weights — counts only (no cluster-fraction covariates)
    d_valid = donors[valid]
    n_cells_series = pd.Series(d_valid).value_counts()
    n_cells_arr = np.array([float(n_cells_series.get(d, 0)) for d in donor_ids], dtype=float)
    weights_per_donor = np.sqrt(np.maximum(n_cells_arr, 1.0))

    optional_cols: list[tuple[str, np.ndarray]] = []
    for label, candidates in OPTIONAL_CONFOUNDER_CANDIDATES:
        col = _find_obs_column(adata, candidates)
        if col is None:
            continue
        vec = _donor_first_numeric(adata, donor_ids, donors, valid, col)
        n_ok = int(np.isfinite(vec).sum())
        if n_ok >= MIN_DONORS_REGRESSION:
            optional_cols.append((label, vec))
        else:
            print(f"  Note: obs column {col!r} ({label}) has {n_ok} donors with finite values; need >= {MIN_DONORS_REGRESSION} to include.")

    def _stack_design(include_age_sex: bool, use_optional: bool) -> np.ndarray:
        parts: list[np.ndarray] = [cpps_per_donor.astype(float)]
        if include_age_sex and age_per_donor is not None:
            parts.append(age_per_donor)
        if include_age_sex and sex_per_donor is not None:
            parts.append(sex_per_donor)
        if use_optional:
            for _lab, arr in optional_cols:
                parts.append(arr)
        return np.column_stack(parts)

    X_design: np.ndarray | None = None
    design_valid: np.ndarray | None = None
    chosen: tuple[bool, bool] | None = None
    for include_age_sex, use_optional in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        if not include_age_sex and use_optional and not optional_cols:
            continue
        x_try = _stack_design(include_age_sex, use_optional)
        dv = np.isfinite(x_try).all(axis=1)
        if int(dv.sum()) >= MIN_DONORS_REGRESSION:
            X_design, design_valid, chosen = x_try, dv, (include_age_sex, use_optional)
            break
    if X_design is None or design_valid is None or chosen is None:
        raise ValueError(
            f"Too few donors with complete design for adjusted model (need >= {MIN_DONORS_REGRESSION})."
        )
    inc_age, inc_opt = chosen
    parts_desc = [predictor_label]
    if inc_age and age_per_donor is not None:
        parts_desc.append("age")
    if inc_age and sex_per_donor is not None:
        parts_desc.append("sex")
    if inc_opt:
        parts_desc.extend(lab for lab, _ in optional_cols)
    print(f"  Adjusted WLS: mean AUCell ~ {' + '.join(parts_desc)} (no leiden cluster fractions).")
    if not inc_age and inc_opt:
        print(f"  Note: age/sex incomplete — using {predictor_label} + PMI/RIN/APOE only (when present).")
    elif not inc_age and not inc_opt:
        print(f"  Warning: omitting age/sex and optional covariates; adjusted model is {predictor_label} only (same as unadjusted).")
    elif inc_age and not inc_opt and optional_cols:
        print(f"  Note: optional PMI/RIN/APOE omitted so enough donors have complete rows ({predictor_label} + age + sex only).")

    # All fits use sm.WLS (heteroskedasticity-aware via sqrt(n_cells) weights), not sm.OLS.
    beta_unadj_list = []
    p_unadj_list = []
    beta_adj_list = []
    p_adj_list = []
    wls_failed_once = False
    for r in range(n_reg):
        # Unadjusted: y ~ predictor (CPPS or ordinal ADNC)
        y_all = mean_auc[:, r]
        ok_unadj = np.isfinite(y_all) & np.isfinite(cpps_per_donor)
        if ok_unadj.sum() < MIN_DONORS_REGRESSION:
            beta_unadj_list.append(np.nan)
            p_unadj_list.append(np.nan)
        else:
            try:
                beta_u, p_u, _, _ = _fit_wls_predictor(
                    y_all,
                    cpps_per_donor,
                    design_valid,
                    X_design,
                    weights_per_donor,
                    MIN_DONORS_REGRESSION,
                )
                beta_unadj_list.append(beta_u)
                p_unadj_list.append(p_u)
            except Exception as e:
                if not wls_failed_once:
                    print("  WLS (unadjusted) failed for at least one regulon:", e)
                    wls_failed_once = True
                beta_unadj_list.append(np.nan)
                p_unadj_list.append(np.nan)

        # Adjusted: y ~ ADNC + covariates (age, sex, optional PMI/RIN/APOE — no cluster fractions)
        y = mean_auc[:, r]
        try:
            _, _, beta_a, p_a = _fit_wls_predictor(
                y,
                cpps_per_donor,
                design_valid,
                X_design,
                weights_per_donor,
                MIN_DONORS_REGRESSION,
            )
            beta_adj_list.append(beta_a)
            p_adj_list.append(p_a)
        except Exception as e:
            if not wls_failed_once:
                print("  WLS (adjusted) failed for at least one regulon:", e)
                wls_failed_once = True
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

    # ---------- Figure: include only regulons with q_adnc < Q_MAX_FIG (0.05) ----------
    sig = df_results[df_results["q_adnc"].fillna(1.0) < Q_MAX_FIG].copy()
    sig["abs_beta_adj"] = sig["beta_adnc_adj"].abs()
    sig = sig.sort_values("abs_beta_adj", ascending=False, na_position="last")
    top_regulons = sig["regulon"].tolist()

    if len(top_regulons) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No regulons passed q_adnc < {Q_MAX_FIG:.2f}", ha="center", va="center", fontsize=13)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "top_progressing_regulons_adnc.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved", fig_dir / "top_progressing_regulons_adnc.png")
        return df_results

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
        beta_a = res["beta_adnc_adj"]
        q = res["q_adnc"]
        title = reg[:40] + "..." if len(reg) > 40 else reg
        ax.text(0.5, 1.08, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12, clip_on=False)

        def _fmt_beta(b):
            return f"{b:.4f}" if np.isfinite(b) else "—"

        def _fmt_q(x):
            return f"{x:.4f}" if np.isfinite(x) else "—"

        # One q only: BH-FDR on p_adnc_adj across regulons (not a separate "adj q").
        subtitle = f"β_unadj={_fmt_beta(beta_u)}, β_adj={_fmt_beta(beta_a)}, q_FDR={_fmt_q(q)}"
        sub_color = "green" if np.isfinite(q) and q < Q_MAX_FIG else "dimgray"
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

    # ---------- Companion figure: donor-mean TF expression for q-significant regulons ----------
    tf_records = []
    expr_matrix, gene_names = _expression_matrix_and_var_names(adata)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    sig["tf"] = sig["regulon"].map(_tf_from_regulon_name)
    tf_hits = sig[["regulon", "tf", "q_adnc"]].drop_duplicates(subset=["tf"]).reset_index(drop=True)
    missing_tfs = []
    for _, row in tf_hits.iterrows():
        tf = row["tf"]
        if tf not in gene_to_idx:
            missing_tfs.append(tf)
            continue
        j = gene_to_idx[tf]
        expr = _dense_vector(expr_matrix[:, j])
        for i, donor in enumerate(donor_ids):
            mask = (donors == donor) & valid & np.isfinite(expr)
            if not np.any(mask):
                continue
            tf_records.append({
                "donor": donor,
                "ADNC": donor_adnc_labels[i],
                "ADNC_ordinal": adnc_per_donor[i],
                "tf": tf,
                "regulon": row["regulon"],
                "q_adnc": row["q_adnc"],
                "mean_expr": float(np.mean(expr[mask])),
            })

    tf_fig_path = fig_dir / "top_progressing_regulon_tfs_adnc.png"
    if not tf_records:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        msg = f"No TF expression panels available for regulons with q_adnc < {Q_MAX_FIG:.2f}"
        if missing_tfs:
            msg += "\nMissing in adata.var_names/raw.var_names: " + ", ".join(sorted(set(missing_tfs)))
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
        fig.savefig(tf_fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved", tf_fig_path)
        return df_results

    tf_df = pd.DataFrame(tf_records)
    tf_df["ADNC"] = pd.Categorical(tf_df["ADNC"], categories=adnc_order, ordered=True)
    tf_panels = tf_hits[tf_hits["tf"].isin(tf_df["tf"].unique())].copy().reset_index(drop=True)
    tf_stats = []
    tf_wls_failed_once = False
    for _, row in tf_panels.iterrows():
        tf = row["tf"]
        sub = tf_df[tf_df["tf"] == tf].copy()
        y = np.full(len(donor_ids), np.nan)
        donor_to_expr = dict(zip(sub["donor"], sub["mean_expr"]))
        for i, donor in enumerate(donor_ids):
            val = donor_to_expr.get(donor, np.nan)
            if pd.notna(val):
                y[i] = float(val)
        try:
            beta_u, p_u, beta_a, p_a = _fit_wls_predictor(
                y,
                cpps_per_donor,
                design_valid,
                X_design,
                weights_per_donor,
                MIN_DONORS_REGRESSION,
            )
        except Exception as e:
            if not tf_wls_failed_once:
                print("  WLS failed for at least one TF expression model:", e)
                tf_wls_failed_once = True
            beta_u, p_u, beta_a, p_a = np.nan, np.nan, np.nan, np.nan
        tf_stats.append({
            "tf": tf,
            "regulon": row["regulon"],
            "regulon_q_adnc": row["q_adnc"],
            "beta_adnc_unadj": beta_u,
            "p_adnc_unadj": p_u,
            "beta_adnc_adj": beta_a,
            "p_adnc_adj": p_a,
            "mean_expr": float(np.nanmean(y)),
        })
    tf_stats_df = pd.DataFrame(tf_stats)
    tf_q = np.full(len(tf_stats_df), np.nan)
    tf_valid = np.isfinite(tf_stats_df["p_adnc_adj"].values)
    if int(tf_valid.sum()) > 0:
        _, tf_q_valid, _, _ = multipletests(tf_stats_df.loc[tf_valid, "p_adnc_adj"].values, method="fdr_bh")
        tf_q[tf_valid] = tf_q_valid
    tf_stats_df["q_adnc"] = tf_q
    tf_stats_df["direction"] = np.where(
        np.isfinite(tf_stats_df["beta_adnc_adj"].values),
        np.where(tf_stats_df["beta_adnc_adj"].values > 0, "up", "down"),
        "flat",
    )
    tf_stats_df = tf_stats_df.sort_values(["q_adnc", "p_adnc_adj"], na_position="last").reset_index(drop=True)
    tf_stats_df.to_csv(out_dir / "top_progressing_tfs_adnc.csv", index=False, na_rep="nan")
    print("Saved", out_dir / "top_progressing_tfs_adnc.csv")
    tf_panels = tf_panels.merge(tf_stats_df, on=["tf", "regulon"], how="left", suffixes=("", "_tf"))
    n_tf = len(tf_panels)
    ncols = min(4, n_tf)
    nrows = (n_tf + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, row in tf_panels.iterrows():
        tf = row["tf"]
        ax = axes[idx // ncols, idx % ncols]
        sub = tf_df[tf_df["tf"] == tf].sort_values("ADNC_ordinal")
        title = f"{tf} expression"
        ax.text(0.5, 1.08, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12, clip_on=False)
        ax.text(
            0.5,
            1.02,
            (
                f"reg_q={row['regulon_q_adnc']:.4f}, "
                f"TF β_adj={row['beta_adnc_adj']:.4f}, "
                f"TF q_FDR={row['q_adnc_tf']:.4f}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="green" if np.isfinite(row["q_adnc_tf"]) and row["q_adnc_tf"] < 0.05 else "dimgray",
            fontweight="bold",
            clip_on=False,
        )
        for adnc_cat in adnc_order:
            vals = sub[sub["ADNC"] == adnc_cat]["mean_expr"].values
            if len(vals) == 0:
                continue
            x_pos = adnc_order.index(adnc_cat)
            jitter = np.random.RandomState(idx + 100).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(x_pos + jitter, vals, alpha=0.7, s=45, color="black", zorder=3)
        parts = ax.violinplot(
            [sub[sub["ADNC"] == adnc_cat]["mean_expr"].values for adnc_cat in adnc_order],
            positions=range(len(adnc_order)),
            showmeans=True,
            showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("mistyrose")
            pc.set_alpha(0.65)
        ax.set_xticks(range(len(adnc_order)))
        ax.set_xticklabels(adnc_order, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Donor mean TF expression (log1p)", fontsize=11)
        ax.set_xlabel("", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    for idx in range(n_tf, axes.size):
        axes[idx // ncols, idx % ncols].set_visible(False)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(tf_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", tf_fig_path)

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Microglia regulon progression vs ADNC (donor-level WLS + covariates)")
    parser.add_argument("--adata", type=Path, default=ADATA_DEFAULT, help="AnnData with X_regulon, donor_id, ADNC (optional: age, sex, PMI, RIN, APOE)")
    parser.add_argument("--mdata", type=Path, default=MDATA_PATH, help="MuData for AUCell if adata has no X_regulon")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    adata = _load_adata(args.adata, args.mdata)
    run_analysis(adata, args.out_dir, args.fig_dir)
    print("Done.")


if __name__ == "__main__":
    main()
