#!/usr/bin/env python
"""
Regulon pathway enrichment: select progressing regulons (≥50 targets, q < 0.1),
run GSEApy Enrichr with microglia background, save combined CSV and dotplot.

Libraries: GO Biological Process 2023, Reactome 2022, KEGG 2021 Human.
Background: var_names from --adata (default: adata_micro_scvi_best.h5ad). If that
file is missing, falls back to the RNA dataset (SEAAD_MTG_microglia_rna.h5ad or
rna_subset.h5ad) so enrichment uses genes from your RNA dataset.

Run from code/05_scenicplus:
  python -m scripts.pipeline.post.regulon_enrichment [--results-dir results] [--fig-dir validation_figures]
  python -m scripts.pipeline.post.regulon_enrichment --adata data_inputs/SEAAD_MTG_microglia_rna.h5ad

Or call run_enrichment(adata, out_dir, fig_dir) from the validation pipeline.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ..paths import PIPELINE_ROOT, PROJECT_ROOT

SCRIPT_DIR = Path(PIPELINE_ROOT)
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR = SCRIPT_DIR / "validation_figures"
ADATA_DEFAULT = SCRIPT_DIR / "data_inputs" / "adata_micro_scvi_best.h5ad"
# Fallback: use genes from RNA dataset (prepared GEX or rna_subset) when embedding adata is missing
ADATA_RNA_FALLBACKS = [
    SCRIPT_DIR / "data_inputs" / "SEAAD_MTG_microglia_rna.h5ad",
    Path(PROJECT_ROOT) / "data" / "sea_ad" / "rna_subset.h5ad",
]

AUCELL_CSV = "regulon_targets_by_aucell.csv"
PROGRESSION_CSV = "regulon_progression_adnc.csv"
MIN_TARGETS = 50
Q_THRESH = 0.1
TOP_N_REGULONS_FALLBACK = 4
# Match microglia_progression_regulons figure inclusion (p_adj < 0.1, mean_auc > 0.02)
P_MAX_FIG = 0.1
MIN_MEAN_AUC_FIG = 0.02
TOP_N_REGULONS_FIG1 = 10
ADJ_P_THRESH = 0.05
TOP_TERMS_PLOT = 10

ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
]


def _safe_imports():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _get_top_progressing_regulons(progression_path: Path) -> list[str]:
    """Return regulon names that appear in the top-progressing figure (same logic as microglia_progression_regulons)."""
    prog = pd.read_csv(progression_path)
    if "mean_auc" in prog.columns:
        mean_above = prog["mean_auc"] > MIN_MEAN_AUC_FIG
        p_adj_ok = prog["p_adnc_adj"].fillna(1) < P_MAX_FIG
        sig = prog[mean_above & p_adj_ok].copy()
    else:
        p_adj_ok = prog["p_adnc_adj"].fillna(1) < P_MAX_FIG
        sig = prog[p_adj_ok].copy()
    if len(sig) == 0:
        sig = prog[prog["q_adnc"].notna()].head(TOP_N_REGULONS_FIG1).copy() if "q_adnc" in prog.columns else prog.head(TOP_N_REGULONS_FIG1).copy()
    sig["abs_beta_adj"] = sig["beta_adnc_adj"].abs()
    sig = sig.sort_values("abs_beta_adj", ascending=False, na_position="last")
    top = sig["regulon"].tolist()
    if len(top) == 0:
        top = prog["regulon"].head(TOP_N_REGULONS_FIG1).tolist()
    return top


def _select_regulons(aucell_path: Path, progression_path: Path) -> pd.DataFrame:
    """Filter to regulons with ≥50 targets AND q < 0.1; if < 4, take top 4 by q-value."""
    aucell = pd.read_csv(aucell_path)
    prog = pd.read_csv(progression_path)
    # Progression CSV may have q_adnc (current) or qvalue (legacy)
    q_col = "q_adnc" if "q_adnc" in prog.columns else "qvalue"
    aucell["n_targets"] = aucell["targets"].fillna("").str.split(";").str.len()
    merged = prog.merge(
        aucell[["regulon", "n_targets", "targets"]],
        on="regulon",
        how="inner",
    )
    passed = merged[(merged["n_targets"] >= MIN_TARGETS) & (merged[q_col] < Q_THRESH)]
    if len(passed) >= 4:
        selected = passed.copy()
    else:
        # Top 4 by q-value (ascending)
        selected = merged.nsmallest(TOP_N_REGULONS_FALLBACK, q_col)
    return selected


def _select_regulons_top_progressing_only(
    aucell_path: Path, progression_path: Path
) -> pd.DataFrame:
    """Build selected table from top-progressing figure regulons only (no 50-target or q threshold)."""
    top_list = _get_top_progressing_regulons(progression_path)
    aucell = pd.read_csv(aucell_path)
    prog = pd.read_csv(progression_path)
    aucell["n_targets"] = aucell["targets"].fillna("").str.split(";").str.len()
    merged = prog.merge(
        aucell[["regulon", "n_targets", "targets"]],
        on="regulon",
        how="inner",
    )
    selected = merged[merged["regulon"].isin(top_list)].copy()
    selected = selected.drop_duplicates(subset=["regulon"], keep="first")
    return selected


def _run_enrichr_one_regulon(
    gene_list: list[str],
    background: list[str],
    libraries: list[str],
    adj_p_thresh: float,
) -> list[pd.DataFrame]:
    """Run Enrichr for one gene list with custom background; return list of result DataFrames per library."""
    background_set = list(background)
    gene_list = [g.strip() for g in gene_list if g.strip()]
    if not gene_list:
        return []
    try:
        from gseapy import enrichr as enrichr_fn
    except ImportError:
        raise ImportError("gseapy is required: pip install gseapy")
    results_per_lib = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for lib in libraries:
            try:
                enr = enrichr_fn(
                    gene_list=gene_list,
                    gene_sets=lib,
                    organism="human",
                    outdir=tmpdir,
                    background=background_set,
                    cutoff=adj_p_thresh,
                    no_plot=True,
                )
                if enr.results is not None and len(enr.results) > 0:
                    res = enr.results.copy()
                    if isinstance(res, list):
                        res = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
                    if not res.empty:
                        res["library"] = lib
                        if "Adjusted P-value" not in res.columns and "Adjusted_P-value" in res.columns:
                            res["Adjusted P-value"] = res["Adjusted_P-value"]
                        results_per_lib.append(res)
            except Exception:
                continue
    return results_per_lib


def _combined_table(selected: pd.DataFrame, results_per_regulon: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Build combined CSV: regulon, direction, library, term, overlap, adjusted p-value."""
    rows = []
    for _, row in selected.iterrows():
        reg = row["regulon"]
        direction = row.get("direction", "up")
        if reg not in results_per_regulon:
            continue
        for res_df in results_per_regulon[reg]:
            lib = res_df["library"].iloc[0] if "library" in res_df.columns else ""
            adj_col = "Adjusted P-value" if "Adjusted P-value" in res_df.columns else "Adjusted_P-value"
            if adj_col not in res_df.columns:
                continue
            res_df = res_df[res_df[adj_col] < ADJ_P_THRESH].copy()
            term_col = "Term" if "Term" in res_df.columns else "term"
            if term_col not in res_df.columns:
                continue
            for _, r in res_df.iterrows():
                term = r[term_col]
                adj_p = r[adj_col]
                # Overlap: from "Genes" column count or "Overlap" (e.g. "5/100")
                if "Overlap" in r:
                    ov = str(r["Overlap"])
                    try:
                        overlap = int(ov.split("/")[0]) if "/" in ov else int(ov)
                    except Exception:
                        overlap = 0
                elif "Genes" in r:
                    overlap = len([x for x in str(r["Genes"]).split(";") if x.strip()])
                else:
                    overlap = 0
                rows.append({
                    "regulon": reg,
                    "direction": direction,
                    "library": lib,
                    "term": term,
                    "overlap": overlap,
                    "adjusted_pvalue": adj_p,
                })
    return pd.DataFrame(rows)


def _strip_term_labels(term: str) -> str:
    """Remove GO accessions (GO:NNNNNNN) and Reactome IDs (R-HSA-NNNNN) from term labels."""
    # GO: (GO:0001234) at end
    term = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term), flags=re.IGNORECASE)
    # Reactome: " R-HSA-12345" or "R-HSA-12345 " anywhere
    term = re.sub(r"\s*R-HSA-\S+", "", term, flags=re.IGNORECASE)
    term = re.sub(r"^R-HSA-\S+\s*", "", term, flags=re.IGNORECASE)
    return term.strip()


PATHWAY_ENRICHMENT_SUBDIR = "pathway_enrichment"


def _sanitize_filename(name: str) -> str:
    """Make regulon name safe for use as a filename."""
    return re.sub(r"[\s/\\()]+", "_", str(name)).strip("_") or "regulon"


def _plot_one_regulon(
    df: pd.DataFrame,
    reg: str,
    fig_path: Path,
    top_n: int = TOP_TERMS_PLOT,
) -> None:
    """Draw a single horizontal bar chart for one regulon and save to fig_path."""
    plt = _safe_imports()
    sub = df[df["regulon"] == reg].copy()
    if sub.empty:
        return
    sub = sub.nsmallest(top_n, "adjusted_pvalue").sort_values("adjusted_pvalue", ascending=False)
    terms = sub["term_clean"].tolist()
    neglogp = sub["neglog10p"].values
    direction = sub["direction"].iloc[0]
    arrow = " ↓" if direction == "down" else " ↑" if direction == "up" else ""
    y_pos = np.arange(len(terms))
    fig, ax = plt.subplots(figsize=(7, max(4, len(terms) * 0.35)))
    ax.barh(y_pos, neglogp, height=0.45, align="center", color="steelblue", edgecolor="gray", linewidth=0.4)
    ax.set_ylim(-0.5, top_n - 0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=9)
    ax.set_xlabel("-log₁₀(adj. p-value)")
    ax.set_title(f"{reg}{arrow}", fontsize=11)
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_dotplot(combined: pd.DataFrame, selected_regulons: list[str], fig_path: Path, top_n: int = TOP_TERMS_PLOT):
    """One subplot per regulon: x = adjusted p-value (-log10), y = pathway terms, horizontal bars."""
    plt = _safe_imports()
    if combined.empty:
        return
    df = combined.copy()
    df["term_clean"] = df["term"].map(_strip_term_labels)
    df["neglog10p"] = -np.log10(df["adjusted_pvalue"].clip(1e-20))

    # One row per (term_clean, regulon): keep best significance
    df = (
        df.sort_values("adjusted_pvalue")
        .groupby(["term_clean", "regulon"], as_index=False)
        .first()
    )

    # Subplot order: down-trending regulons first, then up-trending
    reg_direction = df.groupby("regulon")["direction"].first().reindex(selected_regulons)
    reg_direction = reg_direction.dropna()
    down_regs = reg_direction[reg_direction.eq("down")].index.tolist()
    up_regs = reg_direction[reg_direction.eq("up")].index.tolist()
    flat_regs = reg_direction[~reg_direction.isin(["down", "up"])].index.tolist()
    reg_order = down_regs + flat_regs + up_regs
    if not reg_order:
        reg_order = selected_regulons

    n_reg = len(reg_order)
    ncols = min(2, n_reg)
    nrows = (n_reg + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.ravel()

    for idx, reg in enumerate(reg_order):
        ax = axes[idx]
        sub = df[df["regulon"] == reg].copy()
        if sub.empty:
            ax.set_title(f"{reg}\n(direction: —)", fontsize=10)
            ax.axis("off")
            continue
        sub = sub.nsmallest(top_n, "adjusted_pvalue").sort_values("adjusted_pvalue", ascending=False)
        terms = sub["term_clean"].tolist()
        neglogp = sub["neglog10p"].values
        direction = sub["direction"].iloc[0]
        arrow = " ↓" if direction == "down" else " ↑" if direction == "up" else ""
        y_pos = np.arange(len(terms))
        ax.barh(y_pos, neglogp, height=0.45, align="center", color="steelblue", edgecolor="gray", linewidth=0.4)
        ax.set_ylim(-0.5, top_n - 0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=8)
        ax.set_xlabel("-log₁₀(adj. p-value)")
        ax.set_title(f"{reg}{arrow}", fontsize=10)
        ax.set_xlim(left=0)
        ax.invert_yaxis()
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_enrichment(
    adata=None,
    out_dir: Path | None = None,
    fig_dir: Path | None = None,
    adata_path: Path | None = None,
    aucell_path: Path | None = None,
    progression_path: Path | None = None,
    top_progressing_only: bool = False,
) -> tuple[Path | None, Path | None]:
    """
    Run regulon enrichment and save CSV + dotplot.
    Same pattern as microglia_progression_regulons.run_analysis.

    Parameters
    ----------
    adata : AnnData or None
        If provided, use adata.var_names as background; else load from adata_path.
    out_dir : Path
        Directory for CSV output (default: results).
    fig_dir : Path
        Directory for figure (default: validation_figures).
    adata_path : Path
        Path to adata if adata is None (default: adata_micro_scvi_best.h5ad).
        If that file does not exist, tries RNA dataset paths (SEAAD_MTG_microglia_rna.h5ad, rna_subset.h5ad).
    aucell_path, progression_path : Path
        Paths to regulon_targets_by_aucell.csv and regulon_progression_adnc.csv.

    Returns
    -------
    (path_to_csv, path_to_figure) or (None, None) if nothing to do.
    """
    out_dir = out_dir or RESULTS_DIR
    fig_dir = fig_dir or FIG_DIR
    adata_path = adata_path or ADATA_DEFAULT
    aucell_path = aucell_path or out_dir / AUCELL_CSV
    progression_path = progression_path or out_dir / PROGRESSION_CSV

    if not aucell_path.exists() or not progression_path.exists():
        print(f"[regulon_enrichment] Missing inputs: {aucell_path} or {progression_path}")
        return None, None

    if top_progressing_only:
        selected = _select_regulons_top_progressing_only(aucell_path, progression_path)
        print(f"[regulon_enrichment] Top-progressing only: {len(selected)} regulons from figure: {list(selected['regulon'])}")
    else:
        selected = _select_regulons(aucell_path, progression_path)
    if selected.empty:
        print("[regulon_enrichment] No regulons selected.")
        return None, None

    # Background: microglia var_names (from embedding adata or RNA dataset)
    if adata is not None:
        background = list(adata.var_names.astype(str))
    elif adata_path.exists():
        import scanpy as sc
        adata = sc.read_h5ad(adata_path)
        background = list(adata.var_names.astype(str))
    else:
        # Try RNA dataset fallbacks (prepared GEX or rna_subset)
        adata = None
        for fallback in ADATA_RNA_FALLBACKS:
            if fallback.exists():
                import scanpy as sc
                adata = sc.read_h5ad(fallback)
                background = list(adata.var_names.astype(str))
                print(f"[regulon_enrichment] Using RNA dataset for background: {fallback}")
                break
        if adata is None:
            print(f"[regulon_enrichment] No adata; need {adata_path} or one of {ADATA_RNA_FALLBACKS} for background.")
            return None, None

    print(f"[regulon_enrichment] Background: {len(background)} genes. Selected {len(selected)} regulons.")

    # Run Enrichr per regulon
    results_per_regulon = {}
    for _, row in selected.iterrows():
        reg = row["regulon"]
        targets_str = row.get("targets", "")
        gene_list = [g.strip() for g in str(targets_str).split(";") if g.strip()]
        gene_list = [g for g in gene_list if g in background]
        if len(gene_list) < 5:
            continue
        results_per_regulon[reg] = _run_enrichr_one_regulon(
            gene_list,
            background,
            ENRICHR_LIBRARIES,
            ADJ_P_THRESH,
        )

    if not results_per_regulon:
        print("[regulon_enrichment] No enrichment results.")
        return None, None

    combined = _combined_table(selected, results_per_regulon)
    if combined.empty:
        print("[regulon_enrichment] No significant terms at adj p < 0.05.")
        return None, None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "regulon_enrichment_combined.csv"
    combined.to_csv(csv_path, index=False)
    print("Saved", csv_path)

    # One plot per regulon in validation_figures/pathway_enrichment/
    pathway_dir = fig_dir / PATHWAY_ENRICHMENT_SUBDIR
    pathway_dir.mkdir(parents=True, exist_ok=True)
    for f in pathway_dir.iterdir():
        if f.is_file():
            f.unlink()
    df = combined.copy()
    df["term_clean"] = df["term"].map(_strip_term_labels)
    df["neglog10p"] = -np.log10(df["adjusted_pvalue"].clip(1e-20))
    df = (
        df.sort_values("adjusted_pvalue")
        .groupby(["term_clean", "regulon"], as_index=False)
        .first()
    )
    for reg in selected["regulon"].tolist():
        safe_name = _sanitize_filename(reg)
        one_path = pathway_dir / f"{safe_name}.png"
        _plot_one_regulon(df, reg, one_path, top_n=TOP_TERMS_PLOT)
        print("Saved", one_path)

    return csv_path, pathway_dir


def main():
    parser = argparse.ArgumentParser(description="Regulon pathway enrichment (Enrichr + microglia background)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--adata", type=Path, default=ADATA_DEFAULT, help="AnnData for background genes (var_names). Default: adata_micro_scvi_best.h5ad; if missing, tries RNA dataset: SEAAD_MTG_microglia_rna.h5ad, rna_subset.h5ad")
    parser.add_argument("--top-progressing-only", action="store_true", help="Run enrichment only for regulons in the top-progressing figure. Clears pathway_enrichment/ before saving.")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw dotplot from existing regulon_enrichment_combined.csv")
    args = parser.parse_args()
    if args.plot_only:
        csv_path = args.results_dir / "regulon_enrichment_combined.csv"
        if not csv_path.exists():
            print(f"[regulon_enrichment] --plot-only: {csv_path} not found.")
            return
        combined = pd.read_csv(csv_path)
        selected_regulons = combined["regulon"].unique().tolist()
        pathway_dir = args.fig_dir / PATHWAY_ENRICHMENT_SUBDIR
        pathway_dir.mkdir(parents=True, exist_ok=True)
        for f in pathway_dir.iterdir():
            if f.is_file():
                f.unlink()
        df = combined.copy()
        df["term_clean"] = df["term"].map(_strip_term_labels)
        df["neglog10p"] = -np.log10(df["adjusted_pvalue"].clip(1e-20))
        df = (
            df.sort_values("adjusted_pvalue")
            .groupby(["term_clean", "regulon"], as_index=False)
            .first()
        )
        for reg in selected_regulons:
            safe_name = _sanitize_filename(reg)
            one_path = pathway_dir / f"{safe_name}.png"
            _plot_one_regulon(df, reg, one_path)
            print("Saved", one_path)
        return
    run_enrichment(
        adata=None,
        out_dir=args.results_dir,
        fig_dir=args.fig_dir,
        adata_path=args.adata,
        top_progressing_only=getattr(args, "top_progressing_only", False),
    )
    print("Done.")


if __name__ == "__main__":
    main()
