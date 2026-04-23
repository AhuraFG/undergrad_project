# scGLUE

**scGLUE** integrates **scRNA-seq** and **scATAC-seq** (or similar paired modalities) into a shared latent space and alignment graph. This project uses the **GLUE** model from `code/dependencies/GLUE` (or a pip-installed `scglue`).

---

## How to run

1. **Environment**  
   Install **PyTorch**, **scglue**, **scanpy**, **networkx**, etc. (see the [GLUE documentation](https://scglue.readthedocs.io/) and the vendored `code/dependencies/GLUE` for editable install).

2. **Data**  
   Preprocessed RNA and ATAC **AnnData** objects and the **guidance graph** should live under **`scglue_model_data/`** (e.g. `rna-pp.h5ad`, `atac-pp.h5ad`, `guidance.graphml`). Keep notebook paths project-relative so the workflow remains portable.

3. **Notebooks**  
   - **`scglue_train.ipynb`** — train the GLUE model, save checkpoints under **`glue/`** (e.g. `fine-tune/`, `pretrain/`).  
   - If present, preprocessing notebooks (e.g. `scglue_pp.ipynb`) prepare inputs from raw or subset objects.

4. **Run order**  
   Typically: **preprocess RNA/ATAC** → **build guidance** → **pretrain** → **fine-tune** → use aligned embeddings downstream (e.g. SCENIC+ or visualization). Follow cell order inside each notebook.

---

## Layout

| Path | Role |
|------|------|
| `scglue_train.ipynb` | Main training notebook |
| `scglue_model_data/` | Processed h5ad + graphml |
| `glue/` | Saved model weights / checkpoints |

---

## See also

- **Preprocessing**: [../01_preprocessing/README.md](../01_preprocessing/README.md)  
- **Vendored package**: `code/dependencies/GLUE/README.md`  
- **SCENIC+** (downstream): [../05_scenicplus/README.md](../05_scenicplus/README.md)

## Runtime and resources (rough)

- Training generally ranges from tens of minutes (small subsets) to hours (larger inputs).
- GPU acceleration is recommended when available; CPU-only runs may be significantly slower.
