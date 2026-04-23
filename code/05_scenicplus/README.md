# SCENIC+ (microglia pipeline)

**SCENIC+** workflow for microglia **eRegulon** inference and downstream validation: RNA/ATAC prep, cisTopic LDA, Snakemake SCENIC+, regulon QC, embedding validation, **ADNC progression** analysis, and **pathway enrichment**.

**Full stage-by-stage narrative** (data inputs → files on disk): **[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)**.

---

## How to run

All commands below assume your current directory is **`code/05_scenicplus/`** so relative paths resolve correctly.

### Full pipeline (recommended entry point)

From **`code/05_scenicplus/`**:

```bash
cd code/05_scenicplus
python run_all.py
```

Behaviour is controlled by **`run_config.yaml`** (which stages are on/off). Override with:

```bash
python run_all.py --config /path/to/other_run_config.yaml
```

Common skips (CLI always wins over the YAML):

```bash
python run_all.py --skip-pipeline          # only post-Snakemake stages (embedding, progression, …)
python run_all.py --skip-embedding
python run_all.py --skip-progression --skip-enrichment
python run_all.py --sweep-leiden           # Leiden sweep on scVI latent before embedding
```

To build subsets from full SEA-AD objects and then run:

```bash
python run_all.py --full-rna /path/to/microglia_rna.h5ad --full-atac /path/to/atac.h5ad --cores 4
```

**Embedding / validation only** (expects microglia `.h5ad` + Snakemake MuData):

```bash
python -m scripts.pipeline.post.microglia_embedding_validation [--no-scvi] \
  [--microglia-h5ad data_inputs/SEAAD_MTG_microglia_rna.h5ad]
```

**Progression (regulon vs ADNC) only:**

```bash
python -m scripts.pipeline.post.microglia_progression_regulons \
  --adata data_inputs/adata_micro_scvi_best.h5ad \
  --out-dir results --fig-dir validation_figures
```

**Pathway enrichment only:**

```bash
python -m scripts.pipeline.post.regulon_enrichment --results-dir results --fig-dir validation_figures
```

---

## Directory layout (short)

| Path | Content |
|------|---------|
| `run_all.py`, `run_config.yaml` | Main entry and stage toggles |
| `scripts/pipeline/` | Stages: subsets, RNA, LDA, Snakemake, filter, plots, `post/` |
| `scripts/utils/` | Genome helpers, Leiden sweep, barcode export |
| `data_inputs/` | RNA `.h5ad`, genome annotation, region sets, motif DBs, QC YAML |
| `scplus_pipeline/Snakemake/` | Snakemake workflow, `config/config.yaml`, outputs (e.g. `.h5mu`) |
| `results/`, `validation_figures/` | Tables and figures |
| `notebooks/scenicplus.ipynb` | Exploratory / ad hoc runs |

Snakemake metadata: **`scplus_pipeline/Snakemake/.snakemake/`** (listed in `.gitignore`).

---

## Helper CLI scripts

Run from **`code/05_scenicplus/`** (examples):

```bash
python scripts/utils/fetch_genome_annotation_biomart.py
python scripts/utils/leiden_param_sweep_scvi.py   # see module docstring for args
```

See **[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)** for inputs, stage order, and output file names.

---

## Dependencies

Use the project **`requirements.txt`** and a conda env that includes **scanpy**, **scvi-tools**, **pycisTopic**, **snakemake**, **scenicplus** (or install the vendored package from `code/dependencies/scenicplus`). Exact versions depend on your machine; align with **`docs/PIPELINE_GUIDE.md`** and Snakemake README if builds fail.

## Runtime and resources (rough)

- Small subset checks: tens of minutes.
- Full microglia runs: commonly hours depending on CPU cores, memory, and motif database size.
- Ensure temporary storage has enough space for Snakemake intermediates.
