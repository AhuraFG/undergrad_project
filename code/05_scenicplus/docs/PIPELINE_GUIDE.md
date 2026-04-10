# SCENIC+ Microglia Pipeline Guide

**How to run (commands and entry points):** [../README.md](../README.md).

This guide walks through the full workflow from **data inputs** to **final outputs**, explaining what each stage does and where results are written.

---

## Overview

The pipeline infers **eRegulons** (enhancer-driven gene regulatory networks) for microglia using RNA + ATAC data, then validates them with embeddings, progression analysis (vs. ADNC), and pathway enrichment.

**Entry point:** `run_all.py` (optionally controlled by `run_config.yaml`).  
**Implementation:** `scripts/pipeline/` covers the full workflow (SCENIC+ core + `post/` embedding, progression, enrichment), invoked from `run_all.py`.

---

## 1. Data inputs

Inputs live under **`data_inputs/`** (and optionally **`data/sea_ad/`** if you create subsets).

| Input | Description | Used by |
|-------|-------------|--------|
| **RNA .h5ad** | Microglia RNA AnnData (e.g. `SEAAD_MTG_microglia_rna.h5ad` or `rna_subset.h5ad`) | Prep, Snakemake, embedding |
| **ATAC .h5ad** | ATAC AnnData (Microglia-PVM subset, same cells as RNA) — only if creating subsets | Subset creation, cisTopic |
| **Genome annotation** | `genome_annotation.tsv` (SCENIC+ format: chr, start, end, gene_name, etc.) | Snakemake |
| **Region sets** | Topic-based BED/region sets (from cisTopic LDA) in `data_inputs/region_sets/` | Snakemake |
| **Motif databases** | `.feather` (rankings/scores) + motif annotations table | Snakemake (DEM/CTX) |
| **Optional:** full RNA/ATAC paths | With `--full-rna` / `--full-atac`: full SEA-AD files to derive subsets | Subset creation |
| **Optional:** `--exclude-barcodes-file` | Text file, one barcode per line, to drop from ATAC/cisTopic | cisTopic raw, prep |

Helper scripts (run separately): `scripts/utils/fetch_genome_annotation_biomart.py`, `scripts/utils/convert_genome_annotation.py` → produce `genome_annotation.tsv`.

---

## 2. Pipeline stages (in order)

These run when you execute `run_all.py` without `--skip-pipeline` (and with the corresponding stage enabled in `run_config.yaml`).

### 2.1 Create subsets (optional)

- **When:** You pass `--full-rna` and `--full-atac`.
- **What:** Subset ATAC to Microglia-PVM, intersect cell IDs with RNA, write:
  - `data/sea_ad/rna_subset.h5ad`
  - `data/sea_ad/atac_subset.h5ad`
- **Skip:** `--skip-create-subsets` or set `stages.create_subsets: false`.

### 2.2 Create cisTopic raw object

- **What:** Build a cisTopic object from **ATAC** (from `data/sea_ad/atac_subset.h5ad` or the ATAC path you use). Regions × cells, sparse. Optionally exclude barcodes from `--exclude-barcodes-file`.
- **Output:** `outs/cistopic_obj_raw.pkl`
- **Skip:** `--skip-create-cistopic-raw` or `stages.create_cistopic_raw: false`.

### 2.3 RNA preparation (prep)

- **What:** Load RNA AnnData (default `data/sea_ad/rna_subset.h5ad` or `--rna-path`). Optional contaminant removal (T/NK, B markers), gene selection (HVG + TFs or full). Normalize and prepare for SCENIC+.
- **Output:** Written to **`data_inputs/SEAAD_MTG_microglia_rna.h5ad`** (this becomes the GEX input for Snakemake).
- **Skip:** `--skip-prep` or `stages.prep: false`.

### 2.4 RNA verification (verify)

- **What:** Check microglia identity (e.g. marker presence), optional contaminant detection. Can write QC metrics.
- **Output:** Metrics CSV under `outs/` if enabled.
- **Skip:** `--skip-verify` or `stages.verify: false`.

### 2.5 cisTopic LDA

- **What:** Run LDA on the cisTopic object (topic grid search optional via `--topic-grid`). Select best model (e.g. by coherence), binarise topics (top regions per topic), export region sets.
- **Outputs:**
  - `outs/cistopic_obj.pkl` (fitted cisTopic object)
  - `data_inputs/cistopic_obj.pkl` (copy used by Snakemake)
  - `data_inputs/region_sets/` — BED/region sets per topic
- **Skip:** `--skip-lda` or `stages.lda: false`.

### 2.6 Snakemake SCENIC+

- **What:** Run the SCENIC+ Snakemake workflow: integrate GEX + accessibility, run DEM/CTX motif enrichment, infer TF–gene and region–gene links, build eRegulons, run AUCell.
- **Inputs (from config):** `data_inputs/` (GEX h5ad, cisTopic object, region_sets, genome annotation, motif DBs).
- **Outputs (under `scplus_pipeline/Snakemake/`):**
  - `scplusmdata.h5mu` — MuData with AUCell matrices (main downstream artifact)
  - `cistromes_direct.h5ad`, `cistromes_extended.h5ad`
  - `AUCell_direct.h5mu`, `AUCell_extended.h5mu`
  - eRegulon TSVs, adjacencies, etc.
- **Skip:** `--skip-snakemake` or `stages.snakemake: false`.

### 2.7 Regulon filter

- **What:** Filter spurious regulons (e.g. by % cells expressing the TF, lymphoid TF exclusion). Build credibility table.
- **Outputs:**
  - `results/regulon_credibility_table.csv`
  - `results/tf_to_gene_adjacency.csv` (if produced)
  - Filtered regulon list used by pipeline plots and downstream scripts.
- **Skip:** `--skip-filter` or `stages.filter: false`.

### 2.8 Pipeline plots

- **What:** eRegulon activity heatmap, TF overview, top TFs by RSS, regulon–target tables.
- **Outputs:**
  - `plots/eregulon_activity_heatmap.png`, `plots/eregulon_tf_overview.png`
  - `plots/top_tfs.png`, `plots/top_tfs_rss.png` (if applicable)
  - `results/top_tfs.csv`, `results/top_tfs_rss.csv`
  - `results/regulon_targets_by_rss.csv`, `results/regulon_targets_by_aucell.csv`
- **Skip:** `--skip-plots` or `stages.pipeline_plots: false`.

---

## 3. Embedding and validation

After the SCENIC+ core steps (or if you skip them and use existing Snakemake outputs), embedding and validation run from **`scripts/pipeline/post/microglia_embedding_validation.py`** (or via `run_all.py` → `scripts.pipeline.run`).

- **Inputs:** Microglia AnnData (e.g. `data_inputs/SEAAD_MTG_microglia_rna.h5ad`), **`scplus_pipeline/Snakemake/scplusmdata.h5mu`** (for AUCell).
- **What:**  
  - Embedding: HVG + PCA or **scVI** (`--use-scvi`). Typical workflow: build kNN graph from latent → **Leiden** on that graph → **UMAP** for visualisation (optionally after a Leiden parameter sweep with `scripts/utils/leiden_param_sweep_scvi.py`).  
  - Validation figures: UMAP (clusters, donor, ADNC), regulon dotplots/heatmaps, marker panels, donor × cluster fractions, optional Nature Neuroscience cross-reference table.
- **Outputs:**
  - **`data_inputs/adata_microglia_embedding.h5ad`** or **`adata_microglia_embedding_scvi.h5ad`** (saved AnnData with embeddings and clusters)
  - **`data_inputs/adata_micro_scvi_best.h5ad`** (if Leiden sweep was run — used as default for progression and enrichment)
  - **`validation_figures/*.png`** (UMAPs, donor fractions, regulon heatmaps, etc.)
  - **`results/donor_cluster_fractions_by_*.csv`**, **`results/cluster_sizes*.csv`**, **`results/nn_paper_regulon_mapping*.csv`** (if generated)

---

## 4. Progression vs. ADNC

- **Script:** `scripts/pipeline/post/microglia_progression_regulons.py` (or via `run_all.py`).
- **Inputs:**  
  - AnnData with AUCell (e.g. `data_inputs/adata_microglia_embedding_scvi.h5ad` or `adata_micro_scvi_best.h5ad`)  
  - MuData `scplusmdata.h5mu` (for AUCell and regulon names)  
  - Donor-level metadata including ADNC (and optionally age, sex, cluster fractions).
- **What:** Donor-level weighted OLS: mean AUCell per regulon ~ ADNC (unadjusted and adjusted for age, sex, cluster fractions). Weights = sqrt(n_cells). BH FDR on adjusted p → **q_adnc**.
- **Outputs:**
  - **`results/regulon_progression_adnc.csv`** — regulon, beta/p (unadj & adj), q_adnc, direction, etc.
  - **`validation_figures/top_progressing_regulons_adnc.png`** — violins by ADNC; regulons with p_adj < 0.1 (and mean AUCell > 0.02) with unadj/adj stats in subtitle; green/dimgray by q_adnc < 0.1.

---

## 5. Pathway enrichment

- **Script:** `scripts/pipeline/post/regulon_enrichment.py` (or via `run_all.py`).
- **Inputs:**  
  - **`results/regulon_progression_adnc.csv`** (for selecting regulons by q_adnc)  
  - **`results/regulon_targets_by_aucell.csv`** (target genes per regulon)  
  - AnnData for background genes (default: **`data_inputs/adata_micro_scvi_best.h5ad`** — microglia var_names).
- **What:** Select regulons (e.g. ≥50 targets and q_adnc < 0.1; fallback to top 4 by q). Run **Enrichr** (GSEApy) with GO BP, Reactome, KEGG; microglia background.
- **Outputs:**
  - **`results/regulon_enrichment_combined.csv`** — combined Enrichr results per regulon.
  - **`validation_figures/pathway_enrichment/<regulon>.png`** — bar chart per regulon (top terms).

---

## 6. Final outputs summary

| Location | Contents |
|----------|----------|
| **`results/`** | `regulon_progression_adnc.csv`, `regulon_enrichment_combined.csv`, `regulon_targets_by_aucell.csv`, `regulon_targets_by_rss.csv`, `regulon_credibility_table.csv`, donor/cluster CSVs, top_tfs*.csv, nn_paper_regulon_mapping*.csv |
| **`validation_figures/`** | UMAPs, top_progressing_regulons_adnc.png, donor/cluster figures, **pathway_enrichment/*.png** |
| **`plots/`** | Pipeline plots: eRegulon heatmap, TF overview, top_tfs*.png |
| **`scplus_pipeline/Snakemake/`** | **scplusmdata.h5mu**, AUCell h5mu, cistromes, eRegulon TSVs, adjacencies |
| **`outs/`** | cisTopic objects (cistopic_obj_raw.pkl, cistopic_obj.pkl), region BEDs, QC metrics |
| **`data_inputs/`** | Prepared GEX h5ad, cisTopic object copy, region_sets, **adata_microglia_embedding*.h5ad**, **adata_micro_scvi_best.h5ad** |

---

## Quick reference: data flow

```
[Full RNA/ATAC] ──► create_subsets ──► rna_subset.h5ad, atac_subset.h5ad
       │
       ▼
atac_subset ──► create_cistopic_raw ──► outs/cistopic_obj_raw.pkl
       │
rna_subset ──► prep ──► data_inputs/SEAAD_MTG_microglia_rna.h5ad
       │
       ▼
verify ──► (metrics)
       │
cistopic_obj_raw ──► LDA ──► outs/cistopic_obj.pkl, data_inputs/region_sets/
       │
       ▼
Snakemake (GEX + cisTopic + region_sets + genome + motifs) ──► scplusmdata.h5mu, AUCell, eRegulons
       │
       ▼
filter ──► results/regulon_credibility_table.csv, filtered regulons
       │
       ▼
pipeline_plots ──► plots/*.png, results/regulon_targets_by_*.csv
       │
       ▼
Embedding (pipeline/post/microglia_embedding_validation.py) ──► adata_microglia_embedding*.h5ad, validation_figures/*.png
       │
       ▼
Progression (pipeline/post/microglia_progression_regulons.py) ──► regulon_progression_adnc.csv, top_progressing_regulons_adnc.png
       │
       ▼
Enrichment (pipeline/post/regulon_enrichment.py) ──► regulon_enrichment_combined.csv, pathway_enrichment/*.png
```

For run options and stage toggles, see the main **README.md** in this directory.
