# QC Agent Pipeline

**Quick start and how to run:** [README.md](README.md).

Automated single-cell quality control schema generator. Inspects an AnnData `.h5ad` file, fetches relevant documentation from the web, queries a local LLM (via Ollama), and produces a validated YAML QC plan tailored to the dataset and target downstream tools.

## Architecture Overview

```
┌──────────────────┐
│  Configuration   │  qc_agent_config.yaml + CLI overrides
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 1: Inspect │  Load .h5ad → extract metadata, QC metrics,
│    AnnData       │  processing state, thresholds, mandatory rules
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 2: Fetch   │  Web search (DuckDuckGo) + page scraping
│  Documentation   │  for QC best practices & tool-specific docs
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 3: Build   │  Assemble structured prompt with docs,
│    Prompt        │  inspection data, instructions, YAML template
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 4: Query   │  Send prompt to Ollama (local LLM)
│    Ollama        │  
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 5: Parse,  │  YAML parsing → 12 consistency checks →
│  Validate, Save  │  self-correction loop (up to 5 attempts)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Output YAML    │  qc_schema_scenic_rna.yaml
└──────────────────┘
```

---

## Configuration

All tuneable parameters are externalised in `qc_agent_config.yaml`. The script contains no hardcoded paths, URLs, or model settings.

| Parameter | Description |
|---|---|
| `model.name` | Ollama model to use (e.g. `gemma3:12b`) |
| `model.num_ctx` | Context window in tokens — must fit the full prompt |
| `model.temperature` | Sampling temperature (0.0 = deterministic) |
| `paths.h5ad` | Input AnnData file path |
| `paths.output_yaml` | Output YAML file path |
| `paths.tool_references` | Local fallback reference file (relative to script) |
| `defaults.assay/tissue/organism` | Fallback values if not detectable from the data |
| `target_tools` | List of downstream tools to generate preparation plans for |
| `reference_urls` | Static URLs fetched for general QC best-practice content |
| `fetch.max_chars_per_page` | Max characters fetched per general reference page |
| `fetch.max_chars_per_tool_page` | Max characters fetched per tool documentation page |

Any config value can be overridden via CLI:

```bash
python qc_agent.py --config custom.yaml --h5ad /path/to/data.h5ad --model gemma3:12b
```

---

## Step 1 — Inspect the AnnData

**Function:** `inspect_anndata()`

Loads the `.h5ad` file and extracts a comprehensive data profile without modifying the object. This profile becomes the ground truth that the LLM must respect.

### What is extracted

| Category | Details |
|---|---|
| **Embedded schema** | Reads `adata.uns` for schema_version, title, batch_condition, etc. |
| **Shape** | `n_obs` (cells) and `n_vars` (genes) |
| **X processing state** | Classifies `adata.X` as one of: `likely_raw_counts`, `likely_log_normalised`, `likely_scaled_or_z_scored`, `likely_raw_or_low_depth_counts`, `ambiguous_or_mixed` — by sampling 50K values and checking integer-likeness, range, and sign |
| **Raw counts accessibility** | Checks `adata.raw.X`, `adata.layers['counts']`, and similar layers |
| **QC metrics** | Fuzzy-matches obs column names to canonical metrics (mitochondrial %, genes per cell, total counts, doublet scores, ATAC metrics) and computes distribution statistics (min, max, mean, median, p01, p05, p99) |
| **Suggested thresholds** | `min_genes` (p05), `max_genes` (p99), `max_pct_mito` (p99) — derived from the distributions |
| **Processing status inference** | Infers whether cell filtering, mito filtering, doublet detection, normalisation, and HVG selection have already been performed |
| **Metadata** | Assay, tissue, organism — scanned from `adata.uns` and `adata.obs` columns |
| **Embeddings** | Detects PCA, UMAP, scVI, Harmony, etc. in `adata.obsm` |
| **Clusters** | Detects Leiden, Louvain, or Seurat cluster columns in `adata.obs` |
| **Cell-type annotations** | Detects annotation columns (cell_type, Class, Subclass, etc.) and summarises unique values |
| **Ecosystem** | Infers Python/scanpy vs R/Seurat from presence of scVI embeddings, Seurat metadata, etc. |
| **Obs column inventory** | Lists all obs columns with dtype and unique count |
| **Mandatory rules** | Generates a list of non-negotiable status rules the LLM must follow (e.g. "normalisation MUST be already_done") |

### X state classification logic

The classifier samples up to 50,000 values from `adata.X` and examines:

- **Negative values** → `likely_scaled_or_z_scored`
- **>85% integer-like and max > 20** → `likely_raw_counts`
- **<50% integer-like, non-negative, max ≤ 20** → `likely_log_normalised`
- **>70% integer-like** → `likely_raw_or_low_depth_counts`
- Otherwise → `ambiguous_or_mixed`

---

## Step 2 — Fetch Documentation

### General QC documentation

**Function:** `fetch_documentation()`

Performs web searches via DuckDuckGo for QC best practices relevant to the specific assay, tissue, and organism:

- CELLxGENE schema specifications (if schema_version detected)
- Assay-specific QC thresholds (e.g. "10x multiome single-cell RNA-seq quality control thresholds")
- Tissue-specific preprocessing (e.g. "middle temporal gyrus single-cell preprocessing")
- Doublet detection methods for the organism
- Normalisation methods
- Brain-specific QC queries (triggered if tissue contains brain-related terms)

Also fetches static reference URLs defined in the config (e.g. sc-best-practices, scDblFinder vignette).

### Tool-specific documentation

**Function:** `fetch_tool_docs()`

For each target tool in `target_tools`, runs multiple search queries using the tool name and configured aliases:

- Input data format requirements
- `setup_anndata` / data preparation guides
- GitHub README / getting started
- Assay-specific preprocessing requirements
- Documentation / quickstart tutorials

**Search aliases** handle tools with special characters (e.g. `scenic+` also searches for `scenicplus`, `SCENIC+`, `"scenic plus" single-cell`).

**URL exclusion** filters out irrelevant results (e.g. pySCENIC results are excluded when searching for SCENIC+).

Up to 4 documentation pages are fetched per tool and their text content extracted (HTML stripped of nav/header/footer/scripts). A local `tool_references.yaml` file serves as a fallback when web search returns nothing.

---

## Step 3 — Build Prompt

**Function:** `build_prompt()`

Assembles the full prompt in this order (chosen so the ground-truth data inspection appears closest to where the model generates, reducing the chance of it being truncated from context):

```
Section 1 — Retrieved documentation
  General QC documentation (search snippets + fetched pages)

Section 1b — Target tool documentation
  Current data state summary
  Per-tool fetched documentation with sources

Section 2 — Data inspection summary (GROUND TRUTH)
  Full formatted inspection output
  Mandatory YAML status rules
  Suggested thresholds

Section 3 — Instructions
  10 critical rules
  YAML structure template with per-tool data_preparation entries
```

### Key prompt rules

1. **Mandatory statuses**: The LLM must obey the inferred processing statuses (e.g. if normalisation is detected as done, it must be marked `already_done`)
2. **Exact thresholds**: QC filter thresholds must match the computed p05/p99 values
3. **Raw counts accuracy**: Must not claim raw counts are missing if they are accessible
4. **Tool-aware data preparation**: The LLM must read each tool's documentation to understand WHERE it expects data (e.g. some tools want normalised in `.X` with raw in `.raw`; others want raw counts in `.X`). If the data is already in the correct format, it should say so.
5. **No contradictions**: `data_preparation.current_gaps` must not list steps that are already marked `already_done` in the QC sections
6. **Rationale required**: Every status must include a 1-2 sentence rationale citing specific data facts
7. **Tool differentiation**: Each tool's `data_preparation` section must be distinct

---

## Step 4 — Query Ollama

**Function:** `query_ollama()`

Sends the assembled prompt to the local Ollama instance as a single-turn chat message with the configured model, temperature, and context window (`num_ctx`).

---

## Step 5 — Parse, Validate, Save

### YAML Parsing

**Function:** `_try_parse_yaml()`

Attempts multiple parsing strategies to extract valid YAML from the model's response:

1. Direct YAML parse of the raw response
2. Extract from markdown code fences (` ```yaml ... ``` `)
3. Find content starting from `metadata:` key
4. Progressive truncation from the first top-level YAML key
5. Tab-to-space conversion with prose stripping

### Consistency Validation

**Function:** `_schema_consistency_issues()`

Runs 12 automated checks against the parsed YAML, comparing it to the ground-truth inspection data:

| Check | What it catches |
|---|---|
| 1. Raw counts contradiction | YAML claims raw counts are missing when they are accessible |
| 2. Normalisation status | Status doesn't match detected X processing state |
| 3. HVG status | Marked as pending when `highly_variable` annotation exists |
| 4. Cell filtering status | Filtering steps marked pending when min genes > 500 |
| 5. Missing rationale | Any section without a rationale field |
| 6. Misplaced sections | Normalisation/HVG/embedding placed inside `qc_filters` instead of their own sections |
| 7. Raw counts in .X | `data_preparation` claims raw counts are in .X when X is log-normalised |
| 8. QC contradiction in gaps | `current_gaps` lists HVG/normalisation as missing when they are `already_done` |
| 9. Mito max vs threshold | `already_done` claimed for mito but actual max exceeds threshold without proper rationale |
| 10. Threshold mismatch | YAML thresholds don't match computed p05/p99 values |
| 11. Tool section similarity | `data_preparation` sections for different tools are near-identical (>75% token overlap) |
| 12. Annotation contradiction | Claims cell-type annotations are missing when annotation columns exist |

### Self-Correction Loop

If consistency issues are found, the agent enters a correction loop (up to a few attempts per stage):

1. **Invalid YAML** → `_build_core_yaml_repair_prompt` (core QC) or `_build_tool_dp_repair_prompt` (per-tool `data_preparation`) with a skeleton filled from inspection
2. **Consistency issues** → sends a `_build_correction_prompt` that lists:
   - All already-done steps (so the model knows what NOT to list as gaps)
   - The specific errors with the actual problematic text quoted
   - The mandatory rules
   - The previous YAML for targeted fixing

The correction prompt is designed to help the model self-correct rather than hardcoding fixes.

### Output

**Function:** `save_output()`

If validation passes, writes the YAML via `yaml.safe_dump()` (preserving key order, allowing unicode). Prints a summary of pending vs done QC steps and the overall assessment.

If validation fails after all retries, the agent aborts with an error listing the remaining contradictions.

---

## Output YAML Schema

```yaml
metadata:
  assay: "10x multiome"
  tissue: "middle temporal gyrus"
  organism: "NCBITaxon:9606"
  n_cells: 3129
  n_genes: 15737

data_integrity_warnings: []

qc_filters:
  - step: min_genes
    status: already_done | pending
    rationale: "..."
    threshold: 1383
    distribution_basis: p05
  - step: max_genes
    ...
  - step: max_pct_mito
    ...

doublet_detection:
  method_recommendation: scDblFinder
  status: pending | already_done
  rationale: "..."

normalisation:
  recommendation: log1p
  status: already_done | pending
  rationale: "..."

feature_selection:
  hvg: true | false
  status: already_done | pending
  rationale: "..."

embedding_and_integration:
  methods: [X_scVI, X_umap]
  status: pending | already_done
  rationale: "..."

data_preparation:
  scenic+:
    description: "..."
    required_data_format: "..."
    current_gaps: "..."
    preparation_steps: "..."
    sources: [...]
  scvi-tools:
    description: "..."
    required_data_format: "..."
    current_gaps: "..."
    preparation_steps: "..."
    sources: [...]

overall_assessment: "..."
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `anndata` | Loading and inspecting `.h5ad` files |
| `numpy`, `scipy`, `pandas` | Statistical analysis of expression matrices |
| `ollama` | Python client for local Ollama LLM server |
| `pyyaml` | YAML parsing and serialisation |
| `duckduckgo_search` (or `ddgs`) | Web search for documentation |
| `requests`, `beautifulsoup4` | Fetching and parsing web pages |

---

## File Structure

```
qc_agent/
├── qc_agent.py              # Main pipeline script
├── qc_agent_config.yaml      # All tuneable parameters
├── tool_references.yaml      # Local fallback documentation for tools
└── QC_AGENT_PIPELINE.md      # This document
```
