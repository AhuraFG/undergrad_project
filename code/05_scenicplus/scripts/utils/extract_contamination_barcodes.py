#!/usr/bin/env python
"""
Extract barcodes of cells in specified Leiden clusters for exclusion from the pipeline.

Use after running embedding (scripts.pipeline.post.microglia_embedding_validation) to identify contaminating clusters,
then pass the output file to run_all.py --exclude-barcodes-file.

Example:
  python scripts/utils/extract_contamination_barcodes.py --adata data_inputs/adata_microglia_embedding_scvi.h5ad --clusters 6,7 --out data_inputs/excluded_barcodes.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc


def main():
    p = argparse.ArgumentParser(description="Extract barcodes from specified clusters for exclusion")
    p.add_argument("--adata", type=Path, required=True, help="AnnData with leiden_micro (e.g. adata_microglia_embedding_scvi.h5ad)")
    p.add_argument("--clusters", type=str, required=True, help="Comma-separated cluster IDs to export (e.g. 6 or 6,7)")
    p.add_argument("--out", type=Path, required=True, help="Output text file: one barcode per line")
    args = p.parse_args()

    adata = sc.read_h5ad(args.adata)
    if "leiden_micro" not in adata.obs.columns:
        raise SystemExit("adata has no 'leiden_micro' column; run embedding first.")

    cluster_ids = [s.strip() for s in args.clusters.split(",") if s.strip()]
    clusters = adata.obs["leiden_micro"].astype(str)
    all_barcodes = []
    for cl in cluster_ids:
        mask = clusters == cl
        n = mask.sum()
        barcodes = adata.obs_names[mask].tolist()
        all_barcodes.extend(barcodes)
        print(f"  Cluster {cl}: {n} cells")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(all_barcodes))
    print(f"Wrote {len(all_barcodes)} barcodes to {args.out}")


if __name__ == "__main__":
    main()
