#!/usr/bin/env python
"""
Single entry point: subset → SCENIC+ pipeline → embedding → progression → enrichment.

Stage selection: set stages in run_config.yaml (run_xxx: true/false), and optionally
add --skip-* on CLI to force skipping specific stages for that run.

  python run_all.py
  python run_all.py --config run_config.yaml
  python run_all.py --skip-lda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(SCRIPT_DIR)
DATA_INPUTS = ROOT / "data_inputs"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "validation_figures"
SNAKE_DIR = ROOT / "scplus_pipeline" / "Snakemake"

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "run_config.yaml")
STAGE_TO_ATTR = {
    "create_subsets": "skip_create_subsets",
    "create_cistopic_raw": "skip_create_cistopic_raw",
    "prep": "skip_prep",
    "verify": "skip_verify",
    "lda": "skip_lda",
    "snakemake": "skip_snakemake",
    "filter": "skip_filter",
    "pipeline_plots": "skip_plots",
    "embedding": "skip_embedding",
    "progression": "skip_progression",
    "enrichment": "skip_enrichment",
}


def load_config(path):
    if not path or not os.path.isfile(path):
        return {}
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def apply_config_to_args(args, config):
    """Resolve skip_* from CLI + config.

    Priority:
      1) Explicit CLI --skip-* flag (force skip)
      2) Config stages.<stage> (run=false => skip)
      3) Default (do not skip)
    """
    stages = config.get("stages") or {}
    for stage_key, skip_attr in STAGE_TO_ATTR.items():
        cli_skip = getattr(args, skip_attr, None)
        if cli_skip is True:
            setattr(args, skip_attr, True)
            continue

        if stage_key in stages:
            run = bool(stages.get(stage_key, True))
            setattr(args, skip_attr, not run)
        else:
            setattr(args, skip_attr, False)


def _parser():
    ap = argparse.ArgumentParser(
        description="Run full SCENIC+ microglia workflow: subset → pipeline → embedding → progression → enrichment. Use --skip-* to exclude any stage.",
    )
    # ----- Pipeline: subset & cisTopic -----
    g = ap.add_argument_group("Subset and cisTopic")
    g.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG_PATH), help="YAML config for stages (stages.run_*: true/false); use --config /dev/null to ignore")
    g.add_argument("--skip-pipeline", action="store_true", help="Skip entire pipeline; only run embedding + progression + enrichment")
    g.add_argument("--full-rna", default=None, help="Full microglia RNA .h5ad (SEA-AD); with --full-atac creates rna/atac subsets")
    g.add_argument("--full-atac", default=None, help="Full ATAC .h5ad (SEA-AD); subset to Microglia-PVM and intersect with RNA")
    g.add_argument("--skip-create-subsets", action="store_true", help="Do not create rna/atac_subset even if --full-rna/--full-atac given")
    g.add_argument("--skip-create-cistopic-raw", action="store_true", help="Do not build cistopic_obj_raw.pkl from atac_subset")
    g.add_argument("--force-cistopic-raw", action="store_true", help="Rebuild cistopic_obj_raw.pkl even if it exists")
    g.add_argument("--exclude-barcodes-file", default=None, help="Text file of cell barcodes (one per line) to exclude from RNA/ATAC subsets")

    # ----- Pipeline: prep, verify, LDA, Snakemake, filter, plots -----
    g = ap.add_argument_group("Pipeline")
    g.add_argument("--skip-prep", action="store_true", help="Skip RNA preparation")
    g.add_argument("--skip-verify", action="store_true", help="Skip RNA verification")
    g.add_argument("--skip-lda", action="store_true", help="Skip cisTopic LDA")
    g.add_argument("--skip-snakemake", action="store_true", help="Skip Snakemake SCENIC+")
    g.add_argument("--skip-filter", action="store_true", help="Skip spurious regulon filter")
    g.add_argument("--skip-plots", action="store_true", help="Skip pipeline plot generation")
    g.add_argument("--cores", type=int, default=4, help="Cores for LDA and Snakemake")
    g.add_argument("--rna-path", default=None, help="Input RNA .h5ad (default: data/sea_ad/rna_subset.h5ad)")
    g.add_argument("--gene-strategy", choices=["full", "hvg_plus_tf"], default="hvg_plus_tf")
    g.add_argument("--remove-contaminants", action="store_true")
    g.add_argument("--credible-min-pct-cells", type=float, default=5.0, help="Min %% cells expressing TF for credible regulon")
    g.add_argument("--topic-grid", type=int, nargs="*", default=None, help="LDA topic numbers to try (e.g. 5 10 15 20 25 30)")
    g.add_argument("--binarize-ntop", type=int, default=3000, help="Top regions per topic for binarisation")
    g.add_argument("--lda-n-iter", type=int, default=500, help="LDA iterations per model")

    # ----- Validation: embedding -----
    g = ap.add_argument_group("Embedding and validation")
    g.add_argument("--skip-embedding", action="store_true", help="Skip embedding/validation (UMAP, Leiden, validation figures)")
    g.add_argument("--microglia-h5ad", type=Path, default=DATA_INPUTS / "SEAAD_MTG_microglia_rna.h5ad", help="Microglia AnnData path")
    g.add_argument("--full-atlas", type=Path, default=None, help="Full atlas AnnData for global UMAP context")
    g.add_argument("--mdata", type=Path, default=SNAKE_DIR / "scplusmdata.h5mu", help="MuData with AUCell")
    g.add_argument("--out-fig-dir", type=Path, default=FIG_DIR, help="Output figure directory")
    g.add_argument("--save-adata", type=Path, default=DATA_INPUTS / "adata_microglia_embedding.h5ad", help="Save microglia adata here")
    g.add_argument("--n-hvg", type=int, default=4000)
    g.add_argument("--leiden-resolution", type=float, default=0.6)
    g.add_argument("--use-scvi", action="store_true", default=True, help="Use scVI latent for neighbors/UMAP (default)")
    g.add_argument("--no-scvi", action="store_true", help="Use HVG+PCA instead of scVI")
    g.add_argument("--scvi-key", type=str, default="X_scVI", help="Key in .obsm with scVI latent")
    g.add_argument("--sweep-leiden", action="store_true", help="Run Leiden parameter sweep before embedding; implies --use-scvi")
    g.add_argument("--leiden-params", type=Path, default=RESULTS_DIR / "best_leiden_params_scvi.txt", help="JSON with n_neighbors and resolution")
    g.add_argument("--exclude-scvi-clusters", type=str, default=None, help="Comma-separated cluster IDs to remove before embedding")

    # ----- Validation: progression & enrichment -----
    g = ap.add_argument_group("Progression and enrichment")
    g.add_argument("--skip-progression", action="store_true", help="Skip regulon vs ADNC progression analysis")
    g.add_argument("--skip-enrichment", action="store_true", help="Skip pathway enrichment (Enrichr)")

    return ap


def main():
    args = _parser().parse_args()
    os.chdir(SCRIPT_DIR)

    # Config file: stage defaults; explicit CLI --skip-* flags override config.
    config_path = str(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    # --skip-pipeline is a meta-flag: force-skip all pipeline stages before config
    # merging so that config cannot re-enable them.
    if getattr(args, "skip_pipeline", False):
        args.skip_create_subsets = True
        args.skip_create_cistopic_raw = True
        args.skip_prep = True
        args.skip_verify = True
        args.skip_lda = True
        args.skip_snakemake = True
        args.skip_filter = True
        args.skip_plots = True

    apply_config_to_args(args, config)

    # ----- Pipeline (subset → prep → LDA → Snakemake → filter → plots) -----
    from scripts.pipeline import run as pipeline_run

    pipeline_run(args)

    print("Done.")


if __name__ == "__main__":
    main()
