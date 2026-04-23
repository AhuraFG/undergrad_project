# pySCENIC

**pySCENIC** infers **regulons** (transcription factor → target gene programs) from single-cell RNA using **motif enrichment** and **AUCell** scoring. In this project it is driven mainly from the **`pyscenic.ipynb`** notebook.

---

## How to run

1. **Environment**  
   Install **pySCENIC** and dependencies (arboreto, cytoolz, etc.) in your conda/venv. The project root **`requirements.txt`** may not list every pySCENIC extra; follow [pySCENIC installation](https://github.com/aertslab/pySCENIC) if imports fail.

2. **Configuration**  
   Edit **`config/config.yaml`**:

   - **`Input.ref_tss`**: cisTarget rankings feather (rankings database).
   - **`Input.tfs_path`**, **`Input.motif_path`**: TF list and motif annotations.
   - **`Output.*`**: directories for loom intermediates and regulon/adjacency outputs.

   Config paths are project-relative; run from **`code/04_pyscenic/`** (or adjust paths if running elsewhere). The rankings feather file must exist locally — download the appropriate **cisTarget** resource for your genome build if missing.

3. **Notebook**  
   Open **`pyscenic.ipynb`** in Jupyter / VS Code, set the **RNA `.h5ad` path** (often under `data/sea_ad/` or a processed object), and run cells in order. Outputs go to **`outputs/`** (e.g. regulon adjacencies, AUCell scores, figures).

4. **Command-line alternative**  
   pySCENIC can be run fully from the CLI with `pyscenic grn` / `pyscenic aucell` using the same databases; this repo does not ship a separate shell wrapper — use the notebook or the official CLI if you prefer.

---

## Layout

| Path | Role |
|------|------|
| `pyscenic.ipynb` | Main workflow |
| `config/config.yaml` | Input/output paths for databases and results |
| `data_inputs/` | TF list, motif table, feather DBs (add large files as needed) |
| `outputs/` | Adjacencies, AUCell, figures, exported h5ad |

---

## See also

- **SCENIC+** (RNA+ATAC, eRegulons): [../05_scenicplus/README.md](../05_scenicplus/README.md)  
- **Shared TF list / data**: [../../data/README.md](../../data/README.md)

## Runtime and resources (rough)

- GRN and AUCell steps on moderate datasets can take from tens of minutes to multiple hours.
- Disk usage increases with loom intermediates and exported regulon tables.
