"""SCENIC+ microglia pipeline: subset → SCENIC+ → embedding → progression → enrichment."""
from __future__ import annotations

import os

from .paths import PIPELINE_ROOT
from .steps_lda import step_lda_and_copy
from .steps_post import step_plots, step_spurious_filter
from .steps_rna import step_prepare_rna, step_verify_rna
from .steps_snakemake import step_snakemake
from .steps_subsets import step_create_cistopic_raw, step_create_subsets


def run(args):
    """Run the full pipeline with the given parsed arguments."""
    os.chdir(PIPELINE_ROOT)
    if getattr(args, "full_rna", None) and getattr(args, "full_atac", None) and not getattr(args, "skip_create_subsets", False):
        step_create_subsets(args)
    if not getattr(args, "skip_create_cistopic_raw", False):
        step_create_cistopic_raw(args)
    if not args.skip_prep:
        step_prepare_rna(args)
    if not args.skip_verify:
        step_verify_rna()
    if not args.skip_lda:
        step_lda_and_copy(args)
    if not args.skip_snakemake:
        step_snakemake(args)
    if not args.skip_filter:
        step_spurious_filter(args)
    if not args.skip_plots:
        step_plots(args)

    if not getattr(args, "skip_embedding", False):
        if getattr(args, "no_scvi", False):
            args.use_scvi = False
        from .post import microglia_embedding_validation

        microglia_embedding_validation.main(args=args)

    print("Pipeline finished.")
