# Preprocessing (SEA-AD and multi-omic prep)

Notebooks for **downloading** and **preprocessing** SEA-AD (or similar) **RNA** and **ATAC** data into analysis-ready **AnnData** objects stored under **`data/`** or paths consumed by **scGLUE** / **SCENIC+**.

---

## How to run

1. **Environment**  
   Use a conda env with **scanpy**, **anndata**, **numpy**, **pandas**, and any atlas-specific deps (see notebook imports). Large downloads may require stable network access.

2. **Notebooks**

   | Notebook | Purpose |
   |----------|---------|
   | **`get_data.ipynb`** | Fetch or stage raw / meta data into `data/`. |
   | **`sea-ad_rna_preprocessing.ipynb`** | RNA QC, normalization, subsetting → `.h5ad`. |
   | **`sea-ad_atac_preprocessing.ipynb`** | ATAC QC and feature matrix → `.h5ad`. |

3. **Paths**  
   Edit hardcoded paths at the top of each notebook to match your machine (project root, `data/sea_ad/`, etc.).

4. **Order**  
   Typically: **get_data** (if needed) → **RNA** and **ATAC** prep → export objects referenced by **`code/03_scglue/`** and **`code/05_scenicplus/`**.

---

## See also

- **Data layout**: [../../data/README.md](../../data/README.md)  
- **scGLUE**: [../03_scglue/README.md](../03_scglue/README.md)  
- **SCENIC+**: [../05_scenicplus/README.md](../05_scenicplus/README.md)
