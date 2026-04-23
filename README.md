# Third-year project: microglia multi-omics and GRN analysis

This repository contains pipelines and tooling for **single-cell RNA/ATAC** work on **microglia**, including **GRN inference** (baseline GRN, pySCENIC), **multi-omic integration** (scGLUE), **SCENIC+** (eRegulons from paired RNA+ATAC), and a **QC schema agent** that generates validated YAML QC plans from an `.h5ad` file.

---

## Quickstart

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the two main entry points:

```bash
# SCENIC+ end-to-end orchestration
cd code/05_scenicplus
python run_all.py

# QC schema generator
cd ../../qc_agent
python qc_agent.py
```

Expected outputs:

- SCENIC+ tables and figures under `code/05_scenicplus/results/` and `code/05_scenicplus/validation_figures/`
- QC schema YAML at the `paths.output_yaml` location in `qc_agent/qc_agent_config.yaml`

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

## Path portability policy

Configs tracked in this repo use project-relative paths where possible. To keep notebooks portable, set a local project root variable at the top of each notebook and build paths from it (for example with `Path.cwd()` / `Path("..")` patterns) instead of hardcoding machine-specific absolute paths.

---

## Environments and dependencies

- **Root env (`requirements.txt`)** is the default for shared scientific tooling and analysis scripts.
- **Tool-specific extras** may still be needed for pipeline-specific workflows (see each tool README).
- **QC agent** can run in a separate virtual environment via `qc_agent/requirements.txt` if you want isolation.

Use `python3 --version` and pin a consistent interpreter per machine (Python 3.11+ recommended).

---

## Generated outputs policy

- **Tracked intentionally:** curated figures and summary tables used for interpretation/reporting.
- **Not tracked:** large intermediates, caches, heavy model artifacts, and re-creatable pipeline scratch outputs (configured in `.gitignore`).
- If outputs are regenerated, keep only final curated artifacts and avoid committing transient run products.

---

## Contribution boundaries

Project-authored analysis code lives primarily under:

- `code/01_preprocessing/` to `code/05_scenicplus/`
- `qc_agent/`

Vendored third-party code under `code/dependencies/` is included for reproducibility/convenience and should be treated as upstream software rather than original project implementation.

---

## Runtime expectations (rough)

- Preprocessing and SCENIC+/scGLUE stages can require substantial RAM/CPU and long runtimes (hours on larger inputs).
- QC agent runtime depends on local Ollama model size and context window.
- Prefer stage-wise runs first before full end-to-end execution on large data.
