# Third-year project: microglia multi-omics and GRN analysis

This repository contains pipelines and tooling for **single-cell RNA/ATAC** work on **microglia**, including **GRN inference** (baseline GRN, pySCENIC), **multi-omic integration** (scGLUE), **SCENIC+** (eRegulons from paired RNA+ATAC), and a **QC schema agent** that generates validated YAML QC plans from an `.h5ad` file.

---

## Repository layout

| Path | Purpose |
|------|---------|
| **`code/`** | Analysis code by stage — see [code/README.md](code/README.md) for an index. |
| **`data/`** | Shared inputs and references — [data/README.md](data/README.md). |
| **`qc_agent/`** | LangGraph QC YAML generator — [qc_agent/README.md](qc_agent/README.md). |
| **`documents/`** | Presentations and reports. |
| **`requirements.txt`** | Python dependencies (project root). |

Vendored libraries used by pipelines live under **`code/dependencies/`** (scGLUE, SCENIC+). Prefer linking to upstream docs for those packages; do not treat nested READMEs there as project documentation.

---

## Tool documentation (how to run)

Each component has its own README with setup and run instructions:

| Tool / component | README |
|------------------|--------|
| **QC agent** (LLM YAML QC schema) | [qc_agent/README.md](qc_agent/README.md) |
| **SCENIC+ pipeline** (microglia eRegulons, Snakemake, validation) | [code/05_scenicplus/README.md](code/05_scenicplus/README.md) |
| **pySCENIC** (GRN from RNA) | [code/04_pyscenic/README.md](code/04_pyscenic/README.md) |
| **scGLUE** (RNA + ATAC integration) | [code/03_scglue/README.md](code/03_scglue/README.md) |
| **Baseline GRN** (GRNBoost2) | [code/02_baseline_grn/README.md](code/02_baseline_grn/README.md) |
| **Preprocessing** (SEA-AD download / RNA+ATAC prep) | [code/01_preprocessing/README.md](code/01_preprocessing/README.md) |

---

## Paths and environments

- Notebooks and configs may use absolute paths from this project root. After cloning elsewhere, **search-and-replace** your base path or use environment variables where you add them.
- Create a virtual environment from the root `requirements.txt` when running Python tools; the QC agent can use a dedicated venv (see [qc_agent/README.md](qc_agent/README.md)).

---

## Licence and citation

Add your institution’s licence and citation requirements here when publishing.
