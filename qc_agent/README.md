# QC agent

Automated **single-cell QC schema generator**: inspects an AnnData `.h5ad`, fetches QC and tool documentation, runs a **local LLM** (Ollama), and writes a **validated YAML** QC plan (including optional `data_preparation` blocks for downstream tools).

For a deeper **architecture diagram** and per-node behaviour, see [QC_AGENT_PIPELINE.md](QC_AGENT_PIPELINE.md).

---

## Prerequisites

- **Python 3.11+** (or the version you use with this repo).
- **Ollama** installed locally, with the models named in `qc_agent_config.yaml` (e.g. `gemma3:12b`, `qwen3:14b`).
- Dependencies: install from the **project root**:

  ```bash
  cd /path/to/3rd_year_project
  pip install -r requirements.txt
  ```

  Use a virtual environment (e.g. `.venv_qc_agent`) if you prefer isolation.

---

## How to run

1. Edit **`qc_agent_config.yaml`** (next to `qc_agent.py`):

   - **`paths.h5ad`**: input AnnData (project-relative by default).
   - **`paths.output_yaml`**: where to write the QC schema YAML.
   - **`model.*`**: Ollama model names and context size.
   - **`target_tools`**: downstream tools (e.g. `scenic+`, `scvi-tools`) for `data_preparation` sections.
   - Optional: **`tool_references.yaml`** for local fallback text when GitHub doc fetch fails.

2. Run the agent from the **`qc_agent/`** directory (so relative paths resolve):

   ```bash
   cd qc_agent
   python qc_agent.py
   ```

3. **CLI overrides** (optional):

   ```bash
   python qc_agent.py --config qc_agent_config.yaml \
     --h5ad /path/to/file.h5ad \
     --output /path/to/out.yaml \
     --model gemma3:12b
   ```

The agent writes **`paths.output_yaml`** and may create **`qc_agent/.fetch_cache/`** for fetched pages (safe to delete; regenerates on next run).

## Runtime notes

- Runtime is model-dependent: larger Ollama models and longer context windows substantially increase wall-clock time.
- First runs can be slower due to model startup and initial documentation fetching.

---

## Evaluation (test scripts)

From **`qc_agent/test_scripts/`**:

```bash
cd qc_agent/test_scripts
python generate_test_inputs.py    # synthetic .h5ad variants under ../test_inputs/
python run_eval.py --fast         # inspection-only checks (no LLM)
python run_eval.py --full         # end-to-end agent + LLM (slow)
```

Evaluation config: **`eval_config.yaml`**. Results land under **`test_results/`**.

---

## Configuration reference (short)

| Key area | Role |
|----------|------|
| `model.generation` / `model.tool_use` | Ollama models for generation vs tool/doc selection |
| `paths.h5ad`, `paths.output_yaml`, `paths.tool_references` | Input/output and local tool blurbs |
| `defaults.*` | Assay / tissue / organism fallbacks |
| `target_tools` | Tool names + optional GitHub `repo` for doc fetch |
| `reference_urls` | Static URLs for general QC best-practice text |
| `fetch.max_chars_*` | Truncation limits for fetched content |

---

## Files in this directory

| File | Purpose |
|------|---------|
| `qc_agent.py` | Main LangGraph application |
| `qc_agent_config.yaml` | Default configuration |
| `tool_references.yaml` | Local summaries per tool |
| `QC_AGENT_PIPELINE.md` | Detailed pipeline / architecture notes |
| `test_scripts/` | `generate_test_inputs.py`, `run_eval.py`, `eval_config.yaml` |
