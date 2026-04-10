# Data directory

Shared **inputs** and **reference** files for the pipelines. The repository [README](../README.md) describes how this folder fits into the whole project; **tool-specific run instructions** live next to each pipeline under `code/*/README.md`.

**Git:** This repo tracks **`README.md`** and **`human_tfs.csv`** here. Large assets are **not** committed: the **Cell Ranger reference** (`refdata-gex-GRCh38-2020-A/`) and **SEA-AD `.h5ad`** files under `sea_ad/` must be downloaded or copied locally (see preprocessing notebooks and project notes).

## Structure

| Path | Description |
|------|-------------|
| **sea_ad/** | SEA-AD (Seattle Alzheimer’s Disease) Atlas datasets: RNA and ATAC `.h5ad` files (full and subsets). |
| **refdata-gex-GRCh38-2020-A/** | Cell Ranger / STAR reference genome (GRCh38). |
| **human_tfs.csv** | Human transcription factor list (SCENIC, baseline GRN, etc.). |

## Notes

- Processed outputs from pipelines (SCENIC, scGLUE, SCENIC+, …) usually live under the corresponding **`code/*/`** directories (e.g. `code/04_pyscenic/outputs/`).
- Raw or downloaded inputs are stored here; **large files** are often excluded from Git — use `.gitignore` / Git LFS as appropriate.
