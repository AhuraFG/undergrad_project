# Baseline GRN (GRNBoost2)

A **baseline gene regulatory network** from single-cell RNA using **GRNBoost2** (via **arboreto** or the notebook’s chosen API). Used as a comparison point before or alongside **pySCENIC** / **SCENIC+** regulons.

---

## How to run

1. **Environment**  
   Install **scanpy**, **arboreto** (or `grnboost2` bindings), **pandas**, **numpy**, matching versions compatible with your AnnData.

2. **Notebook**  
   Open **`baseline.ipynb`**. Set:

   - Path to the **RNA `.h5ad`** (e.g. microglia object under `data/` or `code/05_scenicplus/data_inputs/`).
   - Path to **human TFs** (e.g. `data/human_tfs.csv`).

   Run cells in order. Outputs are written to **`grn_df/`** (edge lists / tables as defined in the notebook).

3. **Outputs**  
   Inspect CSVs in **`grn_df/`** for TF–target edges and scores. Use the same paths in downstream comparisons or figures.

---

## Layout

| Path | Role |
|------|------|
| `baseline.ipynb` | GRNBoost2 workflow |
| `grn_df/` | Exported GRN tables |

---

## See also

- **pySCENIC**: [../04_pyscenic/README.md](../04_pyscenic/README.md)  
- **TF list**: [../../data/README.md](../../data/README.md)
