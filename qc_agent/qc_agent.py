#!/usr/bin/env python3
"""Single-cell QC Schema Generator — LangGraph agent.

Nodes: inspect → fetch_docs → generate_qc → generate_tool_prep → validate → repair/save
"""
from __future__ import annotations
import os
if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass
import argparse, hashlib, json, logging, math, operator, re, sys, textwrap, time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

import yaml

log = logging.getLogger("qc_agent")

# ---------------------------------------------------------------------------
# Configuration — loaded from qc_agent_config.yaml (override with --config)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _SCRIPT_DIR / "qc_agent_config.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-cell QC schema generator")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                   help="Path to YAML config file (default: qc_agent_config.yaml next to script)")
    p.add_argument("--h5ad", type=Path, help="Override paths.h5ad from config")
    p.add_argument("--output", type=Path, help="Override paths.output_yaml from config")
    p.add_argument("--model", type=str, help="Override model.name from config")
    return p.parse_args()


CFG: Dict[str, Any] = {}


def _init_config() -> None:
    global CFG
    args = _parse_args()
    CFG = _load_config(args.config)
    if args.h5ad:
        CFG.setdefault("paths", {})["h5ad"] = str(args.h5ad)
    if args.output:
        CFG.setdefault("paths", {})["output_yaml"] = str(args.output)
    if args.model:
        CFG.setdefault("model", {})["name"] = args.model


def _cfg_model(role: str = "generation") -> str:
    m = CFG.get("model", {})
    return m.get(role, m.get("generation", m.get("name", "gemma3:12b")))

def _cfg_model_ctx() -> int:
    return int(CFG.get("model", {}).get("num_ctx", 131072))

def _cfg_model_temp() -> float:
    return float(CFG.get("model", {}).get("temperature", 0.0))

def _cfg_h5ad() -> str:
    return CFG["paths"]["h5ad"]

def _cfg_output() -> str:
    return CFG["paths"]["output_yaml"]

def _cfg_defaults() -> tuple:
    d = CFG.get("defaults", {})
    return d.get("assay", "scRNA-seq"), d.get("tissue", "unknown"), d.get("organism", "Homo sapiens")

def _iter_target_tools() -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """(tool name, spec dict or None)."""
    out: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for t in CFG.get("target_tools", []):
        if isinstance(t, dict):
            out.append((str(t["name"]), t))
        else:
            out.append((str(t), None))
    return out


def _cfg_target_tools() -> List[str]:
    return [name for name, _ in _iter_target_tools()]


def _cfg_tool_repo(tool_name: str) -> Optional[str]:
    """Return 'owner/repo' for a tool if configured, else None."""
    for name, spec in _iter_target_tools():
        if name.lower() == tool_name.lower():
            return spec.get("repo") if isinstance(spec, dict) else None
    return None

def _cfg_reference_urls() -> List[str]:
    return CFG.get("reference_urls", [])

def _cfg_max_chars_page() -> int:
    return int(CFG.get("fetch", {}).get("max_chars_per_page", 8000))

def _cfg_max_chars_tool() -> int:
    return int(CFG.get("fetch", {}).get("max_chars_per_tool_page", 12000))



def _cfg_tool_refs_path() -> Path:
    rel = CFG.get("paths", {}).get("tool_references", "tool_references.yaml")
    return _SCRIPT_DIR / rel
EMBEDDED_SCHEMA_PRIORITY_KEYS = ("schema_version", "schema_reference", "X_normalization",
                                  "default_embedding", "title", "batch_condition")
EMBEDDED_SCHEMA_KEY_SUBSTRINGS = ("schema", "processing", "pipeline", "method", "description")
CLUSTER_RE = re.compile(r"(leiden|louvain|cluster|seurat_clusters|clusters)", re.I)
EMBEDDING_RE = re.compile(r"(pca|umap|harmony|scvi|latent|tsne|x_umap|scanorama)", re.I)
ANNOTATION_RE = re.compile(
    r"(cell_?type|class|subclass|supertype|lineage|label|annot|compartment|"
    r"cell_?ontology|broad_type|fine_type|author_cell)", re.I)
ECOSYSTEM_PYTHON_HINTS = re.compile(r"(scvi|scanpy|scanorama|harmony|bbknn|scglue|X_scVI)", re.I)
ECOSYSTEM_R_HINTS = re.compile(r"(seurat|sctransform|SCTransform|monocle|Seurat)", re.I)

# Column matcher rules: canonical_metric -> lambda on lowered column name
_COL_RULES: List[Tuple[str, Any]] = [
    ("mitochondrial_percent", lambda c: "mito" in c or c == "mt" or "_mt" in c or c.endswith("mt") or "percent_mt" in c),
    ("ribosomal_percent", lambda c: "ribo" in c),
    ("doublet_score", lambda c: any(k in c for k in ("doublet", "scrublet", "pann", "p_doublet"))),
    ("genes_detected_per_cell", lambda c: ("gene" in c or "feature" in c) and "mito" not in c and "ribo" not in c),
    ("total_counts_per_cell", lambda c: not (("gene" in c or "feature" in c) and "mito" not in c and "ribo" not in c)
     and not any(k in c for k in ("mito", "ribo", "doublet", "scrublet"))
     and any(k in c for k in ("umi", "total", "count", "ncount", "sum"))),
    ("atac_tss_enrichment", lambda c: "tss" in c),
    ("atac_nucleosome_signal", lambda c: "nucleosome" in c or "nucleosomal" in c),
    ("atac_fragment_count", lambda c: "fragment" in c and "nucleosome" not in c),
]

# For data_preparation.description similarity: ignore boilerplate so shared AnnData vocabulary
# does not false-trigger when the model still differentiates tool roles.
_DP_SIM_STOP = frozenset("""
a an the and or for to of in on at by as is are was were be been being
it this that these those with from into than then not no yes if so such
any all each every both few more most other some such only same own than
too very can will just also than about over after again against before
between through during under while above below between
dataset data cell cells gene genes expression matrix matrices need needs
required use using used provide provides available present section must
should would could annadata h5ad obs var layers layer counts raw normalised
normalized normalization log embeddings embedding obsm uns
""".split())


def _data_prep_content_tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z][a-z0-9+.-]*", text.lower())
        if len(w) >= 3 and w not in _DP_SIM_STOP
    }


def _print_stage(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def _safe_import_ollama():
    try:
        import ollama
        return ollama
    except ImportError:
        print("Error: pip install ollama", file=sys.stderr); sys.exit(1)


def _ensure_h5ad_exists(path: Path) -> None:
    if not path.is_file():
        print(f"Error: h5ad not found: {path}", file=sys.stderr); sys.exit(1)


def _ensure_model_available(ollama_mod, model_name: str) -> None:
    try:
        models = ollama_mod.list()
    except Exception as e:
        print(f"Warning: could not list models ({e})", file=sys.stderr); return
    names = []
    try:
        items = getattr(models, "models", None) or (models.get("models", []) if isinstance(models, dict) else [])
        for m in items:
            n = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("name") or m.get("model", "") if isinstance(m, dict) else "")
            if n: names.append(str(n))
    except Exception:
        pass
    if not names: return
    base = model_name.split(":")[0]
    if not any(n == model_name or n.startswith(model_name + ":") or n.split(":")[0] == base for n in names):
        print(f"Error: model '{model_name}' not available. Run: ollama pull {model_name}", file=sys.stderr); sys.exit(1)


def _serialize_uns(val, depth=6):
    if depth <= 0: return "<max_depth>"
    if val is None or isinstance(val, (bool, str, int, float)): return val
    try:
        import numpy as np
        if isinstance(val, np.generic): return val.item()
        if isinstance(val, np.ndarray):
            return f"<ndarray shape={val.shape} dtype={val.dtype}>" if val.size > 64 else val.tolist()
    except Exception: pass
    if isinstance(val, dict):
        return {str(k): _serialize_uns(v, depth-1) for k, v in list(val.items())[:200]}
    if isinstance(val, (list, tuple)):
        out = [_serialize_uns(x, depth-1) for x in val[:128]]
        if len(val) > 128: out.append(f"... ({len(val)-128} more)")
        return out
    return str(val)[:2000]


def _extract_embedded_schema(uns) -> Tuple[Dict[str, Any], Optional[str]]:
    if uns is None or not hasattr(uns, "keys"): return {}, None
    try: keys = list(uns.keys())
    except Exception: return {}, None
    summary = {}
    for pk in EMBEDDED_SCHEMA_PRIORITY_KEYS:
        if pk in keys: summary[pk] = _serialize_uns(uns[pk])
    for k in keys:
        kl = str(k).lower()
        if any(s in kl for s in EMBEDDED_SCHEMA_KEY_SUBSTRINGS) and k not in summary:
            summary[k] = _serialize_uns(uns[k])
    sv = None
    if "schema_version" in keys:
        ser = _serialize_uns(uns["schema_version"])
        sv = json.dumps(ser) if isinstance(ser, (dict, list)) else str(ser)
    return summary, sv


def _stats_series(series) -> Dict[str, Any]:
    import numpy as np, pandas as pd
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return {}
    v = s.to_numpy(dtype=np.float64)
    return {"min": float(np.min(v)), "max": float(np.max(v)), "mean": float(np.mean(v)),
            "median": float(np.median(v)), "p01": float(np.percentile(v, 1)),
            "p05": float(np.percentile(v, 5)), "p99": float(np.percentile(v, 99))}


def _fuzzy_qc_metrics(obs) -> List[Dict[str, Any]]:
    cols, used, out = [str(c) for c in obs.columns], set(), []
    for canonical, matcher in _COL_RULES:
        for c in cols:
            if c in used: continue
            try: ok = matcher(c.lower())
            except Exception: ok = False
            if not ok: continue
            st = _stats_series(obs[c])
            if not st: continue
            used.add(c)
            out.append({"canonical_metric": canonical, "column_used": c, "stats": st})
            break
    return out


def _compute_thresholds(qc_metrics) -> Dict[str, Any]:
    sug = {"suggested_min_genes": None, "suggested_max_genes": None, "suggested_max_pct_mito": None}
    for m in qc_metrics:
        can, st = m.get("canonical_metric"), m.get("stats") or {}
        if can == "genes_detected_per_cell":
            p5, p99 = st.get("p05"), st.get("p99")
            if p5 is not None and not (isinstance(p5, float) and math.isnan(p5)):
                sug["suggested_min_genes"] = int(math.floor(float(p5)))
            if p99 is not None and not (isinstance(p99, float) and math.isnan(p99)):
                sug["suggested_max_genes"] = int(math.ceil(float(p99)))
        if can == "mitochondrial_percent":
            p99 = st.get("p99")
            if p99 is not None and not (isinstance(p99, float) and math.isnan(p99)):
                sug["suggested_max_pct_mito"] = float(math.ceil(float(p99) * 1000) / 1000)
    return sug


def _infer_distribution_flags(qc_metrics, x_state, x_details, hvg) -> Dict[str, Any]:
    flags = {"cell_filtering_likely_already_done": False, "cell_filtering_basis": None,
             "mito_filtering_likely_already_done": False, "mito_filtering_basis": None,
             "mito_max_exceeds_threshold": False,
             "doublet_detection_likely_already_done": False, "doublet_detection_basis": None,
             "normalisation_likely_already_done": False, "normalisation_basis": None,
             "hvg_selection_likely_already_done": bool(hvg),
             "hvg_basis": "adata.var['highly_variable'] is present and active" if hvg else None}
    gene_stats = next((m.get("stats") or {} for m in qc_metrics if m.get("canonical_metric") == "genes_detected_per_cell"), None)
    if gene_stats and "min" in gene_stats and float(gene_stats["min"]) > 500:
        flags["cell_filtering_likely_already_done"] = True
        flags["cell_filtering_basis"] = f"minimum genes per cell ({float(gene_stats['min']):.1f}) is above 500"

    mito_stats = next((m.get("stats") or {} for m in qc_metrics if m.get("canonical_metric") == "mitochondrial_percent"), None)
    if mito_stats:
        mito_max = mito_stats.get("max")
        mito_p99 = mito_stats.get("p99")
        if mito_max is not None and mito_p99 is not None:
            threshold = float(math.ceil(float(mito_p99) * 1000) / 1000)
            if float(mito_max) <= threshold:
                flags["mito_filtering_likely_already_done"] = True
                flags["mito_filtering_basis"] = (
                    f"max mito fraction ({float(mito_max):.4f}) is at or below "
                    f"the p99-based threshold ({threshold})")
            else:
                flags["mito_max_exceeds_threshold"] = True
                flags["mito_filtering_basis"] = (
                    f"max mito fraction ({float(mito_max):.4f}) EXCEEDS "
                    f"the p99-based threshold ({threshold}) — cells above threshold still present")

    doublet_stats = next((m for m in qc_metrics if m.get("canonical_metric") == "doublet_score"), None)
    if doublet_stats:
        flags["doublet_detection_likely_already_done"] = True
        flags["doublet_detection_basis"] = f"doublet score column '{doublet_stats.get('column_used', '')}' present in obs"

    if x_state == "likely_log_normalised":
        flags["normalisation_likely_already_done"] = True
        flags["normalisation_basis"] = "adata.X appears log-normalised (non-integer values, typical value range)"
    elif x_details.get("integer_like_fraction", 1.0) < 0.5 and float(x_details.get("max", 99)) < 15:
        flags["normalisation_likely_already_done"] = True
        flags["normalisation_basis"] = "adata.X is largely non-integer with max < 15, consistent with log-scale normalisation"
    return flags


def _infer_x_state(adata) -> Tuple[str, Dict[str, Any]]:
    import numpy as np
    from scipy import sparse
    details, X = {}, adata.X
    if sparse.issparse(X):
        if X.nnz == 0: return "empty_or_unknown", details
        idx = np.random.choice(X.nnz, size=min(50000, X.nnz), replace=False)
        flat = X.data[idx].astype(np.float64)
    else:
        Xd = np.asarray(X)
        if Xd.size == 0: return "empty_or_unknown", details
        flat = Xd.ravel()
        if flat.size > 50000: flat = np.random.choice(flat, size=50000, replace=False)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0: return "empty_or_unknown", details
    neg_frac, min_v, max_v = float(np.mean(flat < 0)), float(np.min(flat)), float(np.max(flat))
    details.update({"sample_size": int(flat.size), "min": min_v, "max": max_v, "neg_fraction": neg_frac})
    if neg_frac > 0.001 or min_v < -1e-6: return "likely_scaled_or_z_scored", details
    int_like_frac = float(np.mean(np.abs(flat - np.round(flat)) < 1e-5))
    details["integer_like_fraction"] = int_like_frac
    if int_like_frac > 0.85 and max_v > 20: return "likely_raw_counts", details
    median_v = float(np.median(flat))
    if int_like_frac < 0.5 and min_v >= -1e-6 and max_v <= 20 and median_v <= 12:
        return "likely_log_normalised", details
    if int_like_frac > 0.7: return "likely_raw_or_low_depth_counts", details
    return "ambiguous_or_mixed", details


def _adata_raw_has_x(adata) -> bool:
    raw = getattr(adata, "raw", None)
    if raw is None: return False
    try:
        if raw.X is None: return False
        shape = getattr(raw, "shape", None)
        return shape is None or (len(shape) == 2 and int(shape[0]) > 0 and int(shape[1]) > 0)
    except Exception: return False


def _raw_counts_accessible(adata) -> bool:
    if _adata_raw_has_x(adata): return True
    layers = getattr(adata, "layers", None) or {}
    return any(k.lower() in ("counts", "raw", "counts_raw") for k in layers.keys())


def _hvg_done(adata) -> bool:
    if "highly_variable" not in adata.var.columns: return False
    try: return bool(adata.var["highly_variable"].any())
    except Exception: return True


def _neighbor_graph_present(adata) -> bool:
    if "neighbors" in adata.uns: return True
    if hasattr(adata, "obsp") and adata.obsp is not None:
        return any("connectivities" in str(k).lower() or "distances" in str(k).lower() for k in adata.obsp.keys())
    return False


def _infer_assay_tissue_organism(adata) -> Tuple[str, str, str]:
    d_assay, d_tissue, d_organism = _cfg_defaults()
    assay, tissue, organism = d_assay, d_tissue, d_organism
    def scan(d):
        nonlocal assay, tissue, organism
        if not isinstance(d, dict): return
        for key, val in d.items():
            kl, vs = str(key).lower(), str(val).lower() if val is not None else ""
            if not vs or vs == "nan": continue
            if "assay" in kl or kl in ("modality", "omics"): assay = str(val)
            if "tissue" in kl or "organ" in kl: tissue = str(val)
            if "organism" in kl or "species" in kl: organism = str(val)
    if adata.uns: scan(dict(adata.uns))
    for col in adata.obs.columns:
        if any(x in col.lower() for x in ("assay", "tissue", "organism", "species")):
            try:
                v = adata.obs[col].dropna()
                if len(v): scan({col: str(v.iloc[0])})
            except Exception: pass
    return assay, tissue, organism


def _build_processing_lists(x_state, hvg, emb_keys, cluster_cols, neighbors, raw_ok, qc_n) -> Tuple[List[str], List[str]]:
    done, needed = [], []
    (done if x_state.startswith("likely_raw") or raw_ok else needed).append(
        "Raw counts available (X or layers/raw)" if (x_state.startswith("likely_raw") or raw_ok) else
        "Ensure raw counts are preserved (layers['counts'] or adata.raw)")
    (done if qc_n > 0 else needed).append(
        "Per-cell QC metrics present in obs" if qc_n > 0 else "Compute per-cell QC metrics")
    (done if hvg else needed).append(
        "Highly variable feature annotation (var)" if hvg else "Highly variable gene selection")
    if emb_keys: done.append(f"Embeddings in obsm: {', '.join(emb_keys[:8])}")
    else: needed.append("Dimensionality reduction / embedding (PCA, scVI, UMAP)")
    if cluster_cols: done.append(f"Clustering columns in obs: {', '.join(cluster_cols[:6])}")
    else: needed.append("Clustering (Leiden/Louvain)")
    (done if neighbors else needed).append(
        "Neighbor graph (uns/obsp)" if neighbors else "Neighbor graph construction")
    if x_state == "likely_log_normalised": done.append("Expression matrix appears log-normalised")
    if x_state == "likely_scaled_or_z_scored":
        done.append("Expression matrix appears scaled (z-scored)")
        if not raw_ok: needed.append("Scaled X without accessible raw counts — verify pipeline compatibility")
    return done, needed


def _build_mandatory_rules(qc_metrics, x_state, raw_ok, dist_flags, annotation_cols=None) -> List[str]:
    rules = []
    gene_min = next((m.get("stats", {}).get("min") for m in qc_metrics if m.get("canonical_metric") == "genes_detected_per_cell"), None)
    if gene_min is not None and float(gene_min) > 500:
        rules.append("min_genes step status MUST be 'already_done' (observed min gene count > 500).")
    if dist_flags.get("cell_filtering_likely_already_done"):
        rules.append(f"Cell filtering MUST be 'already_done' — {dist_flags.get('cell_filtering_basis', 'distribution evidence')}.")

    if dist_flags.get("mito_filtering_likely_already_done"):
        rules.append(f"max_pct_mito step status MUST be 'already_done' — {dist_flags.get('mito_filtering_basis')}.")
    elif dist_flags.get("mito_max_exceeds_threshold"):
        rules.append(
            f"max_pct_mito step: {dist_flags.get('mito_filtering_basis')}. "
            f"Rationale MUST cite the actual max (not p99) and note cells above threshold remain.")

    if dist_flags.get("doublet_detection_likely_already_done"):
        rules.append(f"doublet_detection status MUST be 'already_done' — {dist_flags.get('doublet_detection_basis')}.")

    if dist_flags.get("normalisation_likely_already_done") or x_state == "likely_log_normalised":
        rules.append("Normalisation status MUST be 'already_done' (adata.X consistent with log-normalisation).")
    if x_state in ("likely_raw_counts", "likely_raw_or_low_depth_counts"):
        rules.append("Normalisation status MUST be 'pending' (adata.X contains raw counts).")
    if x_state == "likely_scaled_or_z_scored":
        rules.append(
            "adata.X is scaled/z-scored (negative values present). "
            "MUST add a data_integrity_warnings entry about scaled/z-scored X.")
    if raw_ok:
        rules.append(
            "Raw counts ARE accessible (adata.raw.X or a counts layer exists). "
            "MUST NOT include any warning about missing or inaccessible raw counts "
            "in data_integrity_warnings or anywhere else in the YAML.")
    else:
        if x_state == "likely_log_normalised":
            rules.append(
                "MUST add data_integrity_warnings entry: raw counts not recoverable "
                "(X is log-normalised, no adata.raw and no counts layer).")
        elif x_state == "likely_scaled_or_z_scored":
            rules.append(
                "MUST add data_integrity_warnings entry: raw counts not accessible "
                "(X is scaled/z-scored, no adata.raw and no counts layer).")
        else:
            rules.append("Raw counts NOT accessible. MUST flag in data_integrity_warnings.")
    if dist_flags.get("hvg_selection_likely_already_done"):
        rules.append("Feature_selection/HVG status MUST be 'already_done'.")

    if annotation_cols:
        col_names = [a["column"] for a in annotation_cols]
        rules.append(
            f"Cell-type annotations ARE present in obs columns: {col_names}. "
            f"MUST NOT claim cell-type annotations are missing.")

    rules.append("Do not override mandatory statuses above; use data inspection facts only.")
    return rules


def _build_warnings(x_state, raw_ok, qc_n, layers_info) -> List[str]:
    w = []
    if x_state == "likely_log_normalised" and not raw_ok:
        w.append("adata.X looks log-normalised but no raw counts found — may be unrecoverable.")
    if x_state == "likely_scaled_or_z_scored" and not raw_ok:
        w.append("adata.X appears scaled/z-scored and raw counts not accessible.")
    if qc_n == 0:
        w.append("No standard QC metric columns detected in adata.obs.")
    if not layers_info.get("has_raw") and x_state not in ("likely_raw_counts", "likely_raw_or_low_depth_counts"):
        w.append("Consider storing raw counts in adata.layers['counts'] or adata.raw.")
    return w


def _infer_batch_keys(adata, embedded_schema: Dict[str, Any]) -> List[str]:
    """Obs column names useful as batch keys for scvi-tools (from uns + heuristics)."""
    out: List[str] = []
    if isinstance(embedded_schema, dict):
        bc = embedded_schema.get("batch_condition")
        if isinstance(bc, list):
            out.extend(str(x) for x in bc if x is not None)
        elif bc is not None:
            out.append(str(bc))
    if hasattr(adata, "obs"):
        hints = ("batch", "sample", "donor", "specimen", "library", "patient", "lane", "pool")
        for c in adata.obs.columns:
            cs = str(c)
            if cs in out:
                continue
            cl = cs.lower()
            if any(h in cl for h in hints):
                out.append(cs)
    return out[:16]


def _detect_annotation_cols(obs) -> List[Dict[str, Any]]:
    """Find cell-type / annotation columns and summarise their unique values."""
    out = []
    for c in obs.columns:
        if not ANNOTATION_RE.search(c):
            continue
        try:
            nuniq = int(obs[c].nunique())
            examples = [str(v) for v in obs[c].dropna().unique()[:10]]
            out.append({"column": c, "n_unique": nuniq, "examples": examples})
        except Exception:
            out.append({"column": c, "n_unique": None, "examples": []})
    return out


def _detect_ecosystem(emb_keys, uns_keys, layers_keys) -> str:
    """Guess Python-scanpy vs R-Seurat ecosystem from object contents."""
    all_keys = " ".join(emb_keys + uns_keys + layers_keys)
    py = bool(ECOSYSTEM_PYTHON_HINTS.search(all_keys))
    r = bool(ECOSYSTEM_R_HINTS.search(all_keys))
    if py and not r:
        return "python/scanpy"
    if r and not py:
        return "R/Seurat"
    if py and r:
        return "mixed"
    return "unknown"


def _obs_column_summary(obs, max_cols=40) -> List[str]:
    """Return a compact summary of all obs columns (name + dtype + nunique)."""
    cols = list(obs.columns)
    out = []
    for c in cols[:max_cols]:
        try:
            nuniq = int(obs[c].nunique())
            out.append(f"{c} ({obs[c].dtype}, {nuniq} unique)")
        except Exception:
            out.append(f"{c} ({obs[c].dtype})")
    if len(cols) > max_cols:
        out.append(f"... and {len(cols) - max_cols} more columns")
    return out


def inspect_anndata(h5ad_path: Path) -> Dict[str, Any]:
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path)
    embedded_schema, schema_version = _extract_embedded_schema(getattr(adata, "uns", None))
    x_state, x_details = _infer_x_state(adata)
    layers_keys = list(adata.layers.keys()) if adata.layers is not None else []
    raw_x_ok = _adata_raw_has_x(adata)
    layers_info = {"layer_keys": layers_keys, "has_adata_raw": getattr(adata, "raw", None) is not None,
                   "has_adata_raw_x": raw_x_ok,
                   "has_raw_like_layer": any(k.lower() in ("counts", "raw", "counts_raw") for k in layers_keys),
                   "has_raw": raw_x_ok or any(k.lower() in ("counts", "raw", "counts_raw") for k in layers_keys)}
    qc_metrics = _fuzzy_qc_metrics(adata.obs)
    hvg = _hvg_done(adata)
    dist_flags = _infer_distribution_flags(qc_metrics, x_state, x_details, hvg)
    emb_keys = [str(k) for k in (adata.obsm.keys() if hasattr(adata.obsm, "keys") else []) if EMBEDDING_RE.search(str(k))]
    cluster_cols = [c for c in adata.obs.columns if CLUSTER_RE.search(c.lower())]
    annotation_cols = _detect_annotation_cols(adata.obs)
    neighbors = _neighbor_graph_present(adata)
    raw_ok = _raw_counts_accessible(adata)
    assay, tissue, organism = _infer_assay_tissue_organism(adata)
    uns_keys = list(adata.uns.keys()) if adata.uns is not None else []
    ecosystem = _detect_ecosystem(emb_keys, uns_keys, layers_keys)
    obs_summary = _obs_column_summary(adata.obs)
    done, needed = _build_processing_lists(x_state, hvg, emb_keys, cluster_cols, neighbors, raw_ok, len(qc_metrics))
    batch_keys = _infer_batch_keys(adata, embedded_schema)
    mandatory = _build_mandatory_rules(qc_metrics, x_state, raw_ok, dist_flags, annotation_cols)
    if batch_keys and any("scvi" in str(t).lower() for t in _cfg_target_tools()):
        mandatory.append(
            "scvi-tools / SCVI: Batch effects are modeled DURING training — do NOT describe the RNA AnnData as "
            "defective because 'batch correction has not been applied' to the expression matrix. "
            f"Use batch key(s) in obs when calling setup_anndata (e.g. {batch_keys[:6]}). "
            "If obsm contains X_scVI or similar, note that a batch-aware latent representation may already exist."
        )
    return {"h5ad_path": str(h5ad_path), "shape": tuple(adata.shape),
            "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
            "embedded_schema_uns": embedded_schema, "schema_version": schema_version,
            "batch_keys_obs": batch_keys,
            "x_processing_state": x_state, "x_details": x_details, "layers_info": layers_info,
            "qc_metrics": qc_metrics, "suggested_thresholds": _compute_thresholds(qc_metrics),
            "distribution_inference": dist_flags,
            "mandatory_status_rules": mandatory,
            "highly_variable_annotation": hvg, "embedding_keys_obsm": emb_keys,
            "cluster_columns_obs": cluster_cols, "annotation_columns_obs": annotation_cols,
            "neighbor_graph_present": neighbors,
            "inferred_assay": assay, "inferred_tissue": tissue, "inferred_organism": organism,
            "inferred_ecosystem": ecosystem, "obs_column_summary": obs_summary,
            "processing_stages_completed": done, "processing_stages_needed": needed,
            "warnings": _build_warnings(x_state, raw_ok, len(qc_metrics), layers_info)}


def _reduced_inspection_dict() -> Dict[str, Any]:
    d_assay, d_tissue, d_organism = _cfg_defaults()
    return {"h5ad_path": None, "shape": None, "n_obs": None, "n_vars": None,
            "embedded_schema_uns": {}, "schema_version": None,
            "x_processing_state": "skipped_anndata_unavailable", "x_details": {}, "layers_info": {},
            "qc_metrics": [], "suggested_thresholds": {}, "distribution_inference": {},
            "mandatory_status_rules": [], "highly_variable_annotation": False,
            "embedding_keys_obsm": [], "cluster_columns_obs": [], "annotation_columns_obs": [],
            "neighbor_graph_present": False,
            "inferred_assay": d_assay, "inferred_tissue": d_tissue, "inferred_organism": d_organism,
            "inferred_ecosystem": "unknown", "obs_column_summary": [],
            "processing_stages_completed": [],
            "processing_stages_needed": ["Full AnnData inspection unavailable — install anndata and re-run."],
            "warnings": ["anndata not installed; data inspection skipped."],
            "batch_keys_obs": []}


def _format_inspection(insp: Dict[str, Any]) -> str:
    li = insp.get("layers_info", {})
    sth = insp.get("suggested_thresholds") or {}
    lines = []
    if insp.get("h5ad_path"): lines.append(f"Inspected: {insp['h5ad_path']}")
    lines.append("--- Embedded metadata from adata.uns ---")
    es = insp.get("embedded_schema_uns") or {}
    lines.append(json.dumps(es, indent=2, default=str) if es else "  (none)")
    if insp.get("schema_version"): lines.append(f"schema_version: {insp['schema_version']}")
    lines += [f"Shape: {insp['shape']}", f"Assay: {insp['inferred_assay']}",
              f"Tissue: {insp['inferred_tissue']}", f"Organism: {insp['inferred_organism']}",
              f"X state: {insp['x_processing_state']}"]
    if insp.get("x_details"): lines.append(f"X details: {json.dumps(insp['x_details'], indent=2)}")
    lines.append("--- Raw counts accessibility ---")
    if li.get("has_adata_raw_x"): lines.append("  adata.raw.X EXISTS and is accessible.")
    elif li.get("has_adata_raw"): lines.append("  adata.raw EXISTS but .X not confirmed.")
    else: lines.append("  adata.raw NOT present.")
    if li.get("has_raw_like_layer"): lines.append("  Raw-count layer (counts/raw/counts_raw) EXISTS.")
    if li.get("layer_keys"): lines.append(f"  Layer keys: {li['layer_keys']}")
    lines.append(f"  CONCLUSION: Raw counts {'ARE' if li.get('has_raw') else 'are NOT'} accessible."
                 + (" Do NOT claim they are missing." if li.get('has_raw') else ""))
    lines.append("--- QC metrics ---")
    for m in insp.get("qc_metrics", []):
        lines.append(f"  - {m['canonical_metric']} [{m['column_used']}]: {m['stats']}")
    if not insp.get("qc_metrics"): lines.append("  (none)")
    lines.append("--- REQUIRED thresholds (use these exact values) ---")
    for key, label in [("suggested_min_genes", "min_genes (5th pct)"),
                       ("suggested_max_genes", "max_genes (99th pct)"),
                       ("suggested_max_pct_mito", "max_pct_mito (99th pct)")]:
        if sth.get(key) is not None: lines.append(f"  * {label} = {sth[key]}")
    if not any(v is not None for v in sth.values()): lines.append("  (none computed)")
    lines.append("Only deviate if Section 2 gives a specific contradicting recommendation.")
    lines.append("--- Inferred processing status ---")
    lines.append(json.dumps(insp.get("distribution_inference") or {}, indent=2))
    lines.append("--- Mandatory YAML status rules ---")
    for r in insp.get("mandatory_status_rules", []): lines.append(f"  - {r}")
    lines += [f"HVG present: {insp.get('highly_variable_annotation')}",
              f"Embeddings: {insp.get('embedding_keys_obsm')}",
              f"Clusters: {insp.get('cluster_columns_obs')}",
              f"Neighbor graph: {insp.get('neighbor_graph_present')}",
              f"Ecosystem: {insp.get('inferred_ecosystem', 'unknown')}"]
    bk = insp.get("batch_keys_obs") or []
    lines.append("--- Batch / sample keys (for scvi-tools; from uns + obs name heuristics) ---")
    lines.append("  " + (", ".join(bk) if bk else "(none inferred — inspect obs for batch/sample/donor columns)"))
    # Cell-type / annotation columns
    ann = insp.get("annotation_columns_obs", [])
    lines.append("--- Cell-type / annotation columns ---")
    if ann:
        for a in ann:
            lines.append(f"  - {a['column']} ({a['n_unique']} unique): {a['examples']}")
        lines.append("  CONCLUSION: Cell-type annotations ARE present. Do NOT claim they are missing.")
    else:
        lines.append("  (none detected)")
    # Full obs column inventory
    obs_sum = insp.get("obs_column_summary", [])
    if obs_sum:
        lines.append("--- All obs columns ---")
        for s in obs_sum:
            lines.append(f"  - {s}")
    lines.append("--- Processing stages ---")
    lines.append("Completed:")
    lines += [f"  - {s}" for s in insp.get("processing_stages_completed", [])]
    lines.append("Needed:")
    lines += [f"  - {s}" for s in insp.get("processing_stages_needed", [])]
    lines.append("Warnings:")
    lines += [f"  - {w}" for w in insp.get("warnings", [])]
    return "\n".join(lines)


_CACHE_DIR: Optional[Path] = None


def _get_cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        from datetime import date
        _CACHE_DIR = Path(__file__).resolve().parent / ".fetch_cache" / date.today().isoformat()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_key(prefix: str, value: str) -> str:
    return prefix + "_" + hashlib.sha256(value.encode()).hexdigest()[:16]


def _cache_read(key: str) -> Optional[str]:
    p = _get_cache_dir() / key
    if p.exists():
        log.debug("Cache hit: %s", key)
        return p.read_text(encoding="utf-8")
    return None


def _cache_write(key: str, data: str) -> None:
    p = _get_cache_dir() / key
    p.write_text(data, encoding="utf-8")


def _web_search(query: str, max_results=5) -> List[Dict[str, str]]:
    cache_k = _cache_key("ddg", f"{query}|{max_results}")
    cached = _cache_read(cache_k)
    if cached is not None:
        try:
            return json.loads(cached)
        except Exception:
            pass
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            rows = ddgs.text(query, max_results=max_results) or []
        results = [{"title": str(r.get("title", "")),
                    "url": str(r.get("href", r.get("url", ""))),
                    "snippet": str(r.get("body", r.get("snippet", "")))} for r in rows]
        _cache_write(cache_k, json.dumps(results))
        return results
    except Exception as e:
        log.warning("DuckDuckGo search failed for %r: %s", query, e)
        return []


def _fetch_url_text(url: str, max_chars: int) -> str:
    cache_k = _cache_key("url", url)
    cached = _cache_read(cache_k)
    if cached is not None:
        return cached[:max_chars]
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=25, headers={"User-Agent": "qc-agent/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]): tag.decompose()
        full_text = "\n".join(ln for ln in (l.strip() for l in soup.get_text(separator="\n").splitlines()) if ln)
        _cache_write(cache_k, full_text)
        return full_text[:max_chars]
    except Exception as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return ""


def fetch_documentation(assay, tissue, organism, schema_version=None) -> str:
    blocks = []
    queries = []
    if schema_version:
        queries += [f"CELLxGENE schema {schema_version} column specifications",
                    f"scverse anndata schema {schema_version} metadata",
                    f"single-cell h5ad embedded schema version {schema_version}"]
    queries += [f"{assay} single-cell RNA-seq quality control thresholds",
                f"{tissue} single-cell preprocessing best practices QC",
                f"doublet detection single-nucleus RNA-seq {organism}",
                f"{assay} normalization scran sctransform QC metrics"]
    if any(x in tissue.lower() for x in ("brain", "cortex", "hippocampus", "cerebell", "snrna", "snatac", "neuron")):
        queries.append("brain cortex snRNA-seq QC thresholds mitochondrial")
    for qi, q in enumerate(queries):
        if qi > 0:
            time.sleep(1)
        results = _web_search(q, 5)
        blocks.append(f"--- Query: {q} ---")
        if not results:
            log.info("DDG query returned 0 results: %s", q[:80])
            blocks.append("(no results)")
            continue
        log.info("DDG query returned %d results: %s", len(results), q[:80])
        for i, r in enumerate(results, 1):
            blocks.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   Snippet: {r['snippet']}")
    blocks.append("--- Reference pages ---")
    for url in _cfg_reference_urls():
        txt = _fetch_url_text(url, _cfg_max_chars_page())
        if txt:
            log.info("Fetched %d chars from %s", len(txt), url)
        else:
            log.warning("Fetch returned no content: %s", url)
        blocks.append(f"URL: {url}\n---\n{txt or '(fetch failed)'}")
    return "\n\n".join(blocks)


def _load_local_tool_refs() -> Dict[str, Any]:
    """Load tool_references.yaml from path set in config."""
    ref_path = _cfg_tool_refs_path()
    if not ref_path.is_file(): return {}
    try:
        with open(ref_path) as f: return yaml.safe_load(f) or {}
    except Exception: return {}


# File relevance scoring for GitHub repo discovery
_GITHUB_FILE_PRIORITY = {
    "readme": 10, "tutorial": 9, "quickstart": 9, "getting_started": 8,
    "data_loading": 8, "input": 7, "format": 7, "prepare": 6, "setup": 6, "usage": 5,
}
_GITHUB_EXPLORE_DIRS = frozenset(("docs", "tutorials", "tutorial", "notebooks", "examples", "vignettes", "doc"))
_GITHUB_FILE_EXTS = (".md", ".rst", ".ipynb")


def _github_score_file(name: str) -> int:
    nl = name.lower()
    score = 0
    for kw, val in _GITHUB_FILE_PRIORITY.items():
        if kw in nl:
            score = max(score, val)
    return score


def _github_list_dir(owner_repo: str, path: str = "") -> List[Dict[str, Any]]:
    """List GitHub repo directory contents via API. Returns [] on failure."""
    import urllib.request
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    cache_k = _cache_key("gh", url)
    cached = _cache_read(cache_k)
    if cached:
        try: return json.loads(cached)
        except Exception: pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qc-agent/1.0", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        if isinstance(data, list):
            _cache_write(cache_k, json.dumps(data))
            return data
    except Exception as e:
        log.warning("GitHub API failed for %s/%s: %s", owner_repo, path, e)
    return []


def _github_fetch_raw(owner_repo: str, file_path: str, max_chars: int) -> str:
    """Fetch raw file content from GitHub."""
    import urllib.request
    url = f"https://raw.githubusercontent.com/{owner_repo}/main/{file_path}"
    cache_k = _cache_key("ghraw", f"{owner_repo}/{file_path}")
    cached = _cache_read(cache_k)
    if cached: return cached[:max_chars]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qc-agent/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode("utf-8", errors="replace")
        # For notebooks, extract markdown + code source cells only
        if file_path.endswith(".ipynb"):
            try:
                nb = json.loads(text)
                text = "\n\n".join("".join(c["source"]) for c in nb.get("cells", [])
                                   if c.get("cell_type") in ("markdown", "code"))
            except Exception:
                pass
        _cache_write(cache_k, text)
        return text[:max_chars]
    except Exception as e:
        log.warning("GitHub raw fetch failed %s/%s: %s", owner_repo, file_path, e)
        return ""


def _github_collect_candidates(owner_repo: str) -> List[str]:
    """Walk top-level + known doc dirs and return all doc file paths."""
    candidates: List[str] = []
    for item in _github_list_dir(owner_repo):
        name, itype = item.get("name", ""), item.get("type", "")
        if itype == "file" and any(name.lower().endswith(ext) for ext in _GITHUB_FILE_EXTS):
            candidates.append(name)
        elif itype == "dir" and name.lower() in _GITHUB_EXPLORE_DIRS:
            for sub in _github_list_dir(owner_repo, name):
                sname, stype = sub.get("name", ""), sub.get("type", "")
                if stype == "file" and any(sname.lower().endswith(ext) for ext in _GITHUB_FILE_EXTS):
                    candidates.append(f"{name}/{sname}")
    return candidates


def _llm_select_files(ollama_mod, tool: str, owner_repo: str, candidates: List[str]) -> List[str]:
    """Ask qwen3:14b to pick the 3 most relevant files for scRNA-seq input requirements.
    Falls back to keyword scoring if LLM selection fails."""
    if not candidates:
        return []

    file_list = "\n".join(f"  - {p}" for p in candidates)
    prompt = f"""You are helping select documentation files from the GitHub repo '{owner_repo}' for the tool '{tool}'.

Available files:
{file_list}

Task: identify the 3 files most relevant for understanding what scRNA-seq input data this tool requires — specifically: required preprocessing steps, AnnData format requirements, obs/var columns needed, whether raw counts or normalised data is expected, and any setup_anndata or prepare_data functions.

Rules:
- Prefer scRNA-specific files over ATAC, spatial, or multi-omic files
- Prefer preprocessing/tutorial notebooks over API reference docs
- README is useful if it covers input format
- Output ONLY a JSON array of the selected file paths, exactly as listed above. No explanation.

Example output: ["docs/tutorial.ipynb", "README.md", "docs/input_format.rst"]"""

    try:
        response = query_ollama(ollama_mod, prompt, role="tool_use")
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            selected = json.loads(match.group())
            # Validate paths exist in candidates
            valid = [p for p in selected if isinstance(p, str) and p in candidates]
            if valid:
                log.info("LLM selected %d files for '%s': %s", len(valid), tool, valid)
                return valid[:3]
    except Exception as e:
        log.warning("LLM file selection failed for '%s': %s — falling back to keyword scoring", tool, e)

    # Fallback: keyword scoring
    scored = sorted(((s, p) for p in candidates if (s := _github_score_file(p.split("/")[-1])) > 0), key=lambda x: -x[0])
    fallback = [p for _, p in scored[:3]]
    log.info("Keyword fallback selected: %s", fallback)
    return fallback


def _github_explore_repo(ollama_mod, tool: str, owner_repo: str, max_chars: int) -> Tuple[str, List[str]]:
    """LLM-driven: list repo files, ask qwen3:14b which to fetch, return content.
    Returns (combined_text, [source_urls])."""
    candidates = _github_collect_candidates(owner_repo)
    if not candidates:
        log.warning("No doc files found in %s", owner_repo)
        return "", []

    log.info("GitHub repo %s — %d candidate files found", owner_repo, len(candidates))
    selected = _llm_select_files(ollama_mod, tool, owner_repo, candidates)

    blocks, sources = [], []
    per_file = max_chars // max(len(selected), 1)
    for fpath in selected:
        text = _github_fetch_raw(owner_repo, fpath, per_file)
        if text:
            blocks.append(f"--- {fpath} ---\n{text}")
            sources.append(f"https://github.com/{owner_repo}/blob/main/{fpath}")
            log.info("Fetched %d chars from %s/%s", len(text), owner_repo, fpath)

    return "\n\n".join(blocks), sources


def fetch_tool_docs(ollama_mod, tools: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch input-format documentation for each target tool.
    Uses qwen3:14b to select the most relevant files from the GitHub repo.
    Falls back to local tool_references.yaml if no repo configured or fetch fails.
    Returns {tool: {"text": str, "sources": [str]}}."""
    local_refs = _load_local_tool_refs()
    tool_docs = {}
    max_chars = _cfg_max_chars_tool()

    for tool in tools:
        repo = _cfg_tool_repo(tool)
        text, sources = "", []

        if repo:
            log.info("Fetching docs for '%s' from GitHub repo: %s", tool, repo)
            text, sources = _github_explore_repo(ollama_mod, tool, repo, max_chars)
            if text:
                log.info("GitHub fetch: %d chars for '%s'", len(text), tool)

        if not text:
            # Fallback: local tool_references.yaml
            tl = tool.lower().replace("-", "").replace("_", "").replace(" ", "")
            for ref_name, ref_data in local_refs.items():
                rn = str(ref_name).lower().replace("-", "").replace("_", "").replace(" ", "")
                if tl == rn or tl in rn or rn in tl:
                    if isinstance(ref_data, dict):
                        log.warning("No GitHub docs for '%s'; using local tool_references.yaml", tool)
                        parts = [f"[Local reference for {tool}]"]
                        if ref_data.get("summary"): parts.append(f"Summary: {ref_data['summary']}")
                        for req in (ref_data.get("input_requirements") or []):
                            parts.append(f"- {req}")
                        if ref_data.get("data_format_notes"): parts.append(f"Notes: {ref_data['data_format_notes']}")
                        text = "\n".join(parts)
                        sources = [f"local: tool_references.yaml ({ref_name})"]
                    break

        tool_docs[tool] = {
            "text": text or "(no documentation found)",
            "sources": sources,
        }
    return tool_docs


def _format_data_state_summary(inspection: Dict[str, Any]) -> str:
    li = inspection.get("layers_info", {})
    x_state = inspection.get("x_processing_state", "unknown")
    ann_cols = inspection.get("annotation_columns_obs", [])
    ann_summary = ", ".join(f"{a['column']}({a['n_unique']})" for a in ann_cols) if ann_cols else "NONE"
    ecosystem = inspection.get("inferred_ecosystem", "unknown")
    bk = inspection.get("batch_keys_obs") or []
    bk_line = ", ".join(bk) if bk else "(none inferred)"
    return f"""CURRENT DATA STATE (from data inspection — use when writing data_preparation):
- adata.X contains: {x_state} ({"NOT raw counts" if "log" in x_state or "scaled" in x_state else "raw counts"})
- adata.raw.X exists: {li.get("has_adata_raw_x", False)}
- Raw count layers: {li.get("layer_keys", [])}
- Raw counts accessible: {li.get("has_raw", False)}{" (via adata.raw.X)" if li.get("has_adata_raw_x") else " (via layers)" if li.get("has_raw_like_layer") else ""}
- Embeddings: {inspection.get("embedding_keys_obsm", [])}
- Batch / sample keys for scvi-tools: {bk_line}
- HVG done: {inspection.get("highly_variable_annotation", False)}
- Clusters: {inspection.get("cluster_columns_obs", [])}
- Cell-type annotations: {ann_summary}
- Ecosystem: {ecosystem} (use methods from THIS ecosystem for recommendations, e.g. scanpy methods for python/scanpy)
- This inspection is of the RNA AnnData on disk; tools may need additional modalities or files — put those under other_data_required, not in rna_input_checklist."""


def _tool_data_prep_entry_template(tool: str, td: Dict[str, Any]) -> str:
    sources = td.get("sources", []) if isinstance(td, dict) else []
    src_yaml = "\n".join(f"      - \"{s}\"" for s in sources) if sources else "      - \"no sources available\""
    return f"""  {tool}:
    description: |
      4-10 sentences: the full workflow and inputs {tool} expects per Section 1b (RNA, multi-omic, GRN,
      region sets, etc.). Do NOT limit this block to RNA — cover whatever the tool documentation says the
      pipeline needs end-to-end. Cross-check the RNA AnnData against Section 2 where relevant. For scvi-tools:
      note that batch effects are usually modeled during training via obs batch keys, not as a separate
      "fix .X first" step.
    required_data_format: |
      Overall input picture: expression AnnData layout (.X vs layers vs .raw, obs/var), plus any other object
      types or files the docs require (e.g. peak matrices, fragment files, motif databases). Name actual fields
      from Section 2 for the RNA side. List modalities or resources not in this h5ad here or under
      other_data_required — do not pretend the file contains data it does not.
    rna_input_checklist:
      ONLY preparation of the RNA expression AnnData (slots, layers, raw vs normalised .X, obs columns such
      as batch keys and annotations, gene IDs). At least 5 items; each:
      - requirement: "<short RNA-side check>"
        status: done | not_done | not_applicable
        notes: "<one line citing Section 2>"
      Do NOT put ATAC, motifs, cistromes, or other modalities here — those belong in other_data_required.
    other_data_required: |
      External databases, reference files, downloads, and non-RNA modalities (e.g. ATAC, multiome fragments,
      TF motif libraries) the tool needs that are NOT covered by the RNA checklist. If the inspected h5ad
      supplies everything RNA-related and the docs require nothing else for your use case: say so explicitly
      (e.g. no extra DBs/modalities). Do NOT duplicate RNA AnnData checks here.
      Do NOT restate normalisation/HVG/batch-on-.X as "missing databases" — those are rna_input_checklist / core QC.
    preparation_steps: |
      Numbered plain-English steps for the full pipeline: RNA-side items (consistent with rna_input_checklist)
      plus obtaining or preparing any other_data_required. For scvi-tools: reference setup_anndata with batch
      keys from obs; do NOT frame batch as "expression matrix defective until batch-corrected" — SCVI learns
      batch during training.
    sources:
{src_yaml}"""


def _yaml_map_key(tool: str) -> str:
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tool):
        return tool
    return json.dumps(tool)


def build_core_prompt(inspection: Dict[str, Any], doc_text: str) -> str:
    return f"""## Section 1 — Retrieved documentation (general QC only — no per-tool pages)

{doc_text}

## Section 2 — Data inspection summary (GROUND TRUTH — always overrides Section 1)

{_format_inspection(inspection)}

## Section 3 — Instructions

You are a bioinformatics single-cell QC expert. Using ONLY Sections 1-2, produce the CORE of a YAML QC plan.

CRITICAL: Do NOT include a `data_preparation` key (per-tool blocks are generated separately with smaller context).

CRITICAL RULES:
1. MANDATORY STATUS RULES: Obey every bullet under "Mandatory YAML status rules" in Section 2.
2. THRESHOLDS: Copy exact "REQUIRED thresholds" values into qc_filters.
3. RAW COUNTS: Follow the CONCLUSION line in Section 2's raw counts section exactly.
4. DISTRIBUTION INFERENCE: If cell_filtering/normalisation/hvg flags are true, those statuses MUST be 'already_done'.
5. Use n_cells/n_genes from Section 2 shape. Do not invent numbers.
6. Address every Section 2 warning in the YAML.
7. Output ONLY valid YAML. No markdown fences. No text before or after.
8. RATIONALE IS MANDATORY: Every qc_filters step, doublet_detection, normalisation, feature_selection,
   embedding_and_integration MUST include a 'rationale' field citing Section 2.
9. STRUCTURE: qc_filters should ONLY contain filtering steps. No normalisation/HVG inside qc_filters.
10. OMIT `data_preparation` entirely.

YAML structure (no data_preparation):
metadata:
  assay: <Section 2>
  tissue: <Section 2>
  organism: <Section 2>
  n_cells: <n_obs>
  n_genes: <n_vars>
data_integrity_warnings:
  - <only if supported by Section 2>
qc_filters:
  - step: <name>
    status: pending | already_done
    rationale: "1-2 sentences citing Section 2 (REQUIRED)"
    threshold: <REQUIRED threshold>
    distribution_basis: <percentile>
doublet_detection:
  method_recommendation: <method>
  status: pending | already_done
  rationale: "REQUIRED"
normalisation:
  recommendation: <method>
  status: pending | already_done
  rationale: "REQUIRED"
feature_selection:
  hvg: true | false
  status: pending | already_done
  rationale: "REQUIRED"
embedding_and_integration:
  methods: <list>
  status: pending | already_done
  rationale: "REQUIRED"
overall_assessment: |
  Plain English summary.

REMEMBER: No data_preparation key. Obey Mandatory YAML status rules."""


def build_tool_data_prep_prompt(tool: str, inspection: Dict[str, Any], td: Dict[str, Any]) -> str:
    text = td.get("text", "") if isinstance(td, dict) else str(td)
    sources = td.get("sources", []) if isinstance(td, dict) else []
    src_line = f"\n  Sources used: {', '.join(sources)}" if sources else ""
    ykey = _yaml_map_key(tool)
    template = _tool_data_prep_entry_template(tool, td if isinstance(td, dict) else {})
    return f"""## Section 1b — Target tool documentation (ONLY "{tool}")

{_format_data_state_summary(inspection)}

### {tool}{src_line}
{text}

## Section 2 — Data inspection summary (GROUND TRUTH)

{_format_inspection(inspection)}

## Section 3 — Instructions

Produce ONLY a `data_preparation` mapping with exactly ONE tool entry. The tool YAML key MUST be {ykey} (same spelling as in config).

RULES:
- Cross-reference Section 2 for x_processing_state, layers, raw counts, batch keys, HVG, embeddings (RNA object).
- description / required_data_format / preparation_steps: follow Section 1b for the FULL tool workflow (not RNA-only).
- rna_input_checklist: ONLY RNA AnnData preparation checks; at least 5 items; status (done|not_done|not_applicable) and notes from Section 2.
- other_data_required: databases, files, and modalities outside that RNA checklist (ATAC, motifs, etc.); do not duplicate RNA slot checks.
- scvi-tools: batch effects are modeled during training via obs batch keys; do NOT claim batch correction is "missing" on .X.
- Fill `sources` with URLs from Section 1b where applicable.

Output ONLY valid YAML. Root key must be `data_preparation`. Example shape:

data_preparation:
  {ykey}:
    description: |
      ...
    required_data_format: |
      ...
    rna_input_checklist:
      - requirement: "..."
        status: done
        notes: "..."
    other_data_required: |
      ...
    preparation_steps: |
      ...
    sources:
      - "..."

Use this field structure (replace template content with real text):

{template}

No markdown fences. No text before or after the YAML."""


def _extract_tool_dp_block(parsed: Optional[dict], tool: str) -> Optional[dict]:
    if not isinstance(parsed, dict):
        return None
    dp = parsed.get("data_preparation")
    if isinstance(dp, dict):
        if tool in dp and isinstance(dp[tool], dict):
            return dp[tool]
        for k, v in dp.items():
            if str(k).strip() == str(tool).strip() and isinstance(v, dict):
                return v
    if tool in parsed and isinstance(parsed[tool], dict) and len(parsed) <= 2:
        return parsed[tool]
    return None


def query_ollama(ollama_mod, prompt: str, role: str = "generation") -> str:
    model = _cfg_model(role)
    response = ollama_mod.chat(model=model, messages=[{"role": "user", "content": prompt}],
                               options={"temperature": _cfg_model_temp(), "num_ctx": _cfg_model_ctx()},
                               keep_alive=0 if role == "tool_use" else "5m")
    if isinstance(response, dict):
        msg = response.get("message") or {}
        content = (msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")) or str(response.get("response", ""))
    else:
        msg = getattr(response, "message", None)
        content = (getattr(msg, "content", "") or "") if msg else ""
    return str(content)


def _try_parse_yaml(text: str) -> Tuple[Optional[dict], str]:
    text = text.strip()

    def _try_one(s: str) -> Optional[Tuple[dict, str]]:
        s = s.strip()
        if not s:
            return None
        try:
            r = yaml.safe_load(s)
            if isinstance(r, dict):
                return (r, s)
        except Exception:
            pass
        return None

    # 1. Raw
    result = _try_one(text)
    if result is not None:
        return result
    # 2. Markdown fences
    m = re.search(r"```(?:yaml|yml)?\s*([\s\S]*?)```", text, re.I)
    if m:
        result = _try_one(m.group(1))
        if result is not None:
            return result
    # 3. From metadata: key
    for line in text.splitlines():
        if line.strip().startswith("metadata:"):
            idx = text.find(line)
            if idx >= 0:
                result = _try_one(text[idx:])
                if result is not None:
                    return result
    # 4. Progressive truncation from first top-level key
    lines = text.splitlines()
    yaml_start = next((i for i, l in enumerate(lines) if re.match(r'^[a-z_]+:', l) and not l.startswith(' ')), None)
    if yaml_start is not None:
        for end in range(len(lines), yaml_start, -1):
            result = _try_one("\n".join(lines[yaml_start:end]))
            if result is not None:
                return result
    # 5. Tab fix + prose stripping
    cleaned, in_yaml = [], False
    for line in text.replace("\t", "    ").splitlines():
        if re.match(r'^[a-z_]+:', line) and not line.startswith(' '):
            in_yaml = True
        if in_yaml:
            cleaned.append(line)
    if cleaned:
        result = _try_one("\n".join(cleaned))
        if result is not None:
            return result
    return (None, text)


def _data_prep_text_blobs(tool_val: Dict[str, Any]) -> List[str]:
    """Flatten string fields + rna_input_checklist entries for consistency scans."""
    texts: List[str] = []
    for f in ("description", "required_data_format", "other_data_required", "preparation_steps", "current_gaps"):
        v = tool_val.get(f)
        if v is None:
            continue
        texts.append(str(v))
    cl = tool_val.get("rna_input_checklist")
    if isinstance(cl, list):
        for item in cl:
            if isinstance(item, dict):
                texts.append(" ".join(str(item.get(k, "")) for k in ("requirement", "status", "notes")))
            else:
                texts.append(str(item))
    return texts


def _summarise_schema(parsed) -> Tuple[int, int, str]:
    if not isinstance(parsed, dict): return 0, 0, ""
    pending = done = 0
    for item in (parsed.get("qc_filters") or []):
        if not isinstance(item, dict): continue
        if "done" in str(item.get("status", "")).lower(): done += 1
        else: pending += 1
    oa = parsed.get("overall_assessment", "")
    return pending, done, (oa.strip() if isinstance(oa, str) else "")


def _schema_consistency_issues(parsed, inspection, *, check_data_preparation: bool = True) -> List[str]:
    issues = []
    if not isinstance(parsed, dict): return ["Output is not a YAML mapping."]
    li = inspection.get("layers_info") or {}
    has_raw = bool(li.get("has_raw"))
    x_state = inspection.get("x_processing_state", "")
    dist = inspection.get("distribution_inference") or {}

    # Check 1: raw counts contradiction — scan entire YAML including data_integrity_warnings
    bad_raw_phrases = (
        "raw counts not recoverable", "raw counts are not recoverable", "no adata.raw",
        "missing adata.raw", "count layers are absent", "count layers missing",
        "raw counts may be unrecoverable", "no raw count", "raw counts are not available",
        "raw counts unavailable", "without accessible raw", "raw counts not accessible",
        "raw counts are missing", "no raw counts",
    )
    if has_raw:
        def _scan(obj, path=""):
            if isinstance(obj, str):
                ol = obj.lower()
                for p in bad_raw_phrases:
                    if p in ol:
                        issues.append(
                            f"Contradiction at '{path}': raw counts ARE accessible "
                            f"but YAML claims: \"{obj[:200]}\"")
                        return
            elif isinstance(obj, dict):
                for k, v in obj.items(): _scan(v, f"{path}.{k}" if path else str(k))
            elif isinstance(obj, list):
                for i, v in enumerate(obj): _scan(v, f"{path}[{i}]")
        _scan(parsed)

    # Check 1b: missing required warnings
    warn_text = " ".join(str(w) for w in (parsed.get("data_integrity_warnings") or [])).lower()
    if not has_raw and x_state in ("likely_log_normalised", "likely_scaled_or_z_scored"):
        if not any(p in warn_text for p in ("raw counts", "raw count", "not recoverable", "not accessible")):
            issues.append(
                "data_integrity_warnings is missing a required entry about raw counts not being "
                "accessible. Add a warning since adata.raw is None and no counts layer exists.")
    if x_state == "likely_scaled_or_z_scored":
        if not any(p in warn_text for p in ("scaled", "z-score", "zscore", "z score")):
            issues.append(
                "data_integrity_warnings is missing a required entry about adata.X being "
                "scaled/z-scored. Add a warning since negative values were detected in X.")

    # Check 2: normalisation
    norm = parsed.get("normalisation") or parsed.get("normalization") or {}
    if isinstance(norm, dict):
        ns = str(norm.get("status", "")).lower()
        if (dist.get("normalisation_likely_already_done") or x_state == "likely_log_normalised") and "done" not in ns:
            issues.append(f"Normalisation should be 'already_done' but is '{ns}'.")
        elif x_state in ("likely_raw_counts", "likely_raw_or_low_depth_counts") and "pending" not in ns:
            issues.append(f"X is raw counts, normalisation should be 'pending' but is '{ns}'.")

    # Check 3: HVG
    fs = parsed.get("feature_selection") or {}
    if isinstance(fs, dict) and dist.get("hvg_selection_likely_already_done") and "done" not in str(fs.get("status", "")).lower():
        issues.append(f"HVG should be 'already_done' but is '{fs.get('status')}'.")

    # Check 4: cell filtering
    if dist.get("cell_filtering_likely_already_done"):
        for item in (parsed.get("qc_filters") or []):
            if not isinstance(item, dict): continue
            step = str(item.get("step", "")).lower()
            status = str(item.get("status", "")).lower()
            if (("min" in step and "gene" in step) or ("filter" in step and ("cell" in step or "gene" in step))) and "done" not in status:
                issues.append(f"Cell filtering step '{item.get('step')}' should be 'already_done' but is '{status}'.")

    # Check 5: missing rationale fields
    for item in (parsed.get("qc_filters") or []):
        if isinstance(item, dict) and not item.get("rationale"):
            issues.append(f"qc_filters step '{item.get('step', '?')}' is missing a 'rationale' field. Add 1-2 sentences explaining why this status was chosen.")
    for section in ("doublet_detection", "normalisation", "normalization", "feature_selection", "embedding_and_integration"):
        sec = parsed.get(section)
        if isinstance(sec, dict) and "status" in sec and not sec.get("rationale"):
            issues.append(f"'{section}' is missing a 'rationale' field. Add 1-2 sentences explaining why this status was chosen.")

    # Check 6: normalisation/feature_selection misplaced inside qc_filters
    for item in (parsed.get("qc_filters") or []):
        if isinstance(item, dict):
            step = str(item.get("step", "")).lower()
            if any(k in step for k in ("normali", "feature_select", "hvg", "embedding")):
                issues.append(f"qc_filters contains '{item.get('step')}' which belongs in its own dedicated section, not under qc_filters. Remove it from qc_filters.")

    dp = parsed.get("data_preparation")
    batch_keys = inspection.get("batch_keys_obs") or []
    # Checks 7–7c, 11–12: data_preparation (skipped for core YAML before per-tool merge)
    if check_data_preparation:
        # Check 7: data_preparation should not claim raw counts are in .X when X is log-normalised
        if isinstance(dp, dict) and x_state in ("likely_log_normalised", "likely_scaled_or_z_scored"):
            for tool_name, tool_val in dp.items():
                if not isinstance(tool_val, dict):
                    continue
                for i, blob in enumerate(_data_prep_text_blobs(tool_val)):
                    text = blob.lower()
                    if "raw counts" in text and ("in .x" in text or "in the .x" in text or "stored in .x" in text or "are in .x" in text):
                        issues.append(
                            f"data_preparation.{tool_name} (block {i}) says raw counts are in .X, but x_processing_state={x_state} — "
                            f"adata.X is {x_state}, NOT raw counts. Raw counts are in adata.raw.X or a counts layer. Fix this.")

        # Check 8: data_preparation should not contradict QC section statuses
        if isinstance(dp, dict):
            fs = parsed.get("feature_selection") or {}
            hvg_done = isinstance(fs, dict) and "done" in str(fs.get("status", "")).lower()
            norm = parsed.get("normalisation") or parsed.get("normalization") or {}
            norm_done = isinstance(norm, dict) and "done" in str(norm.get("status", "")).lower()
            for tool_name, tool_val in dp.items():
                if not isinstance(tool_val, dict):
                    continue
                chunks: List[Tuple[str, str]] = []
                for field in ("other_data_required", "preparation_steps", "current_gaps", "description", "required_data_format"):
                    chunks.append((field, str(tool_val.get(field, ""))))
                cl = tool_val.get("rna_input_checklist")
                if isinstance(cl, list):
                    for i, item in enumerate(cl):
                        if isinstance(item, dict):
                            joined = " ".join(str(item.get(k, "")) for k in ("requirement", "status", "notes"))
                            chunks.append((f"rna_input_checklist[{i}]", joined))
                for field, actual_text in chunks:
                    text = actual_text.lower()
                    if hvg_done and ("hvg" in text or "highly variable" in text) and (
                            "missing" in text or "not done" in text or "not already" in text or "selection is missing" in text
                            or "needed" in text or "need to" in text or "required" in text):
                        issues.append(
                            f"data_preparation.{tool_name}.{field} claims HVG is missing/needed, but feature_selection "
                            f"status is 'already_done'. Remove any mention of HVG being missing. "
                            f"Current text: \"{actual_text[:200]}\"")
                    if norm_done and "normali" in text and (
                            "not done" in text or "missing" in text or "is needed" in text or "not yet" in text
                            or "need to" in text or "required" in text):
                        issues.append(
                            f"data_preparation.{tool_name}.{field} claims normalisation is missing/needed, but normalisation "
                            f"status is 'already_done'. Remove any mention of normalisation being missing. "
                            f"Current text: \"{actual_text[:200]}\"")

        # Check 7b: scvi-tools — batch narrative
        if isinstance(dp, dict) and batch_keys:
            bad_scvi = (
                "batch correction has not been",
                "batch correction has not",
                "batch correction not performed",
                "no batch correction",
                "without batch correction",
                "batch correction missing",
            )
            for tool_name, tool_val in dp.items():
                if "scvi" not in str(tool_name).lower():
                    continue
                if not isinstance(tool_val, dict):
                    continue
                for blob in _data_prep_text_blobs(tool_val):
                    tl = blob.lower()
                    if not any(b in tl for b in bad_scvi):
                        continue
                    if any(ok in tl for ok in ("during training", "setup_anndata", "learns batch", "models batch", "batch key")):
                        continue
                    issues.append(
                        f"data_preparation.{tool_name}: Reword batch handling — scvi-tools models batch effects during "
                        f"training using obs keys like {batch_keys[:4]}. Do not imply the RNA matrix is defective because "
                        f"batch correction was not pre-applied to .X.")
                    break

        # Check 7c: rna_input_checklist present and structured
        if isinstance(dp, dict):
            for tool_name, tool_val in dp.items():
                if not isinstance(tool_val, dict):
                    continue
                cl = tool_val.get("rna_input_checklist")
                if not isinstance(cl, list) or len(cl) < 5:
                    issues.append(
                        f"data_preparation.{tool_name}.rna_input_checklist must be a YAML list with at least 5 items "
                        f"(requirement, status, notes per item).")
                else:
                    for i, item in enumerate(cl):
                        if not isinstance(item, dict) or not str(item.get("requirement", "")).strip():
                            issues.append(
                                f"data_preparation.{tool_name}.rna_input_checklist[{i}] must be a mapping with "
                                f"non-empty 'requirement', 'status', and 'notes'.")

    # Check 9: mito max vs threshold sanity
    sth = inspection.get("suggested_thresholds") or {}
    mito_threshold = sth.get("suggested_max_pct_mito")
    if mito_threshold is not None:
        mito_stats = next((m.get("stats", {}) for m in (inspection.get("qc_metrics") or [])
                           if m.get("canonical_metric") == "mitochondrial_percent"), None)
        if mito_stats:
            mito_max = mito_stats.get("max")
            if mito_max is not None:
                for item in (parsed.get("qc_filters") or []):
                    if not isinstance(item, dict):
                        continue
                    step = str(item.get("step", "")).lower()
                    status = str(item.get("status", "")).lower()
                    if "mito" in step and "done" in status and float(mito_max) > float(mito_threshold):
                        rationale = str(item.get("rationale", "")).lower()
                        if str(round(mito_max, 4)) not in rationale and f"{mito_max:.4f}" not in rationale:
                            issues.append(
                                f"max_pct_mito is 'already_done' but actual max ({mito_max:.4f}) exceeds "
                                f"threshold ({mito_threshold}). Rationale must cite the real max, not p99.")

    # Check 10: threshold values match computed suggestions
    for item in (parsed.get("qc_filters") or []):
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "")).lower()
        yaml_thresh = item.get("threshold")
        if yaml_thresh is None:
            continue
        expected = None
        if "min" in step and "gene" in step:
            expected = sth.get("suggested_min_genes")
        elif "max" in step and "gene" in step:
            expected = sth.get("suggested_max_genes")
        elif "mito" in step:
            expected = sth.get("suggested_max_pct_mito")
        if expected is not None and abs(float(yaml_thresh) - float(expected)) > 1:
            issues.append(
                f"qc_filters step '{item.get('step')}' threshold ({yaml_thresh}) does not match "
                f"computed suggestion ({expected}). Use the computed value.")

    if check_data_preparation:
        # Check 11: data_preparation sections should not be near-identical across tools
        if isinstance(dp, dict) and len(dp) >= 2:
            tool_tokens = {}
            for tool_name, tool_val in dp.items():
                if not isinstance(tool_val, dict):
                    continue
                combined = " ".join(_data_prep_text_blobs(tool_val))
                tool_tokens[tool_name] = _data_prep_content_tokens(combined)
            names = list(tool_tokens.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = tool_tokens[names[i]], tool_tokens[names[j]]
                    if a and b:
                        overlap = len(a & b) / max(len(a | b), 1)
                        if overlap > 0.75:
                            issues.append(
                                f"data_preparation sections for '{names[i]}' and '{names[j]}' are too similar "
                                f"(token overlap {overlap:.0%}). Each tool has different requirements — "
                                f"differentiate them.")

        # Check 12: data_preparation should not claim cell-type annotations are missing when they exist
        ann_cols = inspection.get("annotation_columns_obs", [])
        if ann_cols and isinstance(dp, dict):
            ann_names = [a["column"] for a in ann_cols]
            for tool_name, tool_val in dp.items():
                if not isinstance(tool_val, dict):
                    continue
                for i, blob in enumerate(_data_prep_text_blobs(tool_val)):
                    text = blob.lower()
                    if ("lacks" in text or "missing" in text or "no cell" in text or "absent" in text) and ("annotation" in text or "cell type" in text or "cell_type" in text):
                        issues.append(
                            f"data_preparation.{tool_name} (block {i}) claims cell-type annotations are missing, "
                            f"but columns {ann_names} ARE present in obs. Remove that claim. "
                            f"Current text: \"{blob[:200]}\"")

    return issues


def _build_core_yaml_repair_prompt(insp: Dict[str, Any]) -> str:
    sth = insp.get("suggested_thresholds", {})
    dist = insp.get("distribution_inference", {})
    x_state = insp.get("x_processing_state", "?")
    has_raw = (insp.get("layers_info") or {}).get("has_raw", False)
    emb = insp.get("embedding_keys_obsm", [])
    ann_cols = insp.get("annotation_columns_obs", [])
    ecosystem = insp.get("inferred_ecosystem", "unknown")
    norm_st = "already_done" if (dist.get("normalisation_likely_already_done") or x_state == "likely_log_normalised") else "pending"
    doublet_st = "already_done" if dist.get("doublet_detection_likely_already_done") else "pending"
    hvg_st = "already_done" if dist.get("hvg_selection_likely_already_done") else "pending"
    emb_st = "already_done" if emb else "pending"
    emb_str = ", ".join(emb[:6]) if emb else "pca, umap"
    raw_note = "Do NOT add warnings about missing raw counts - they ARE accessible." if has_raw else "Raw counts NOT accessible - warn about this."
    ann_note = (f"Cell-type annotations present: {[a['column'] for a in ann_cols]}. "
                "Do NOT claim annotations are missing.") if ann_cols else ""
    eco_note = f"Ecosystem is {ecosystem} — use appropriate method names." if ecosystem != "unknown" else ""
    return f"""Your previous response was not valid YAML. Fill in this skeleton. Output ONLY the YAML, nothing else.
Do NOT include a data_preparation key.

metadata:
  assay: {insp.get("inferred_assay", "?")}
  tissue: {insp.get("inferred_tissue", "?")}
  organism: {insp.get("inferred_organism", "?")}
  n_cells: {insp.get("n_obs", "?")}
  n_genes: {insp.get("n_vars", "?")}
data_integrity_warnings:
  - "Fill in warnings based on data state, or remove if none"
qc_filters:
  - step: min_genes_filter
    status: pending
    rationale: "REPLACE"
    threshold: {sth.get("suggested_min_genes", "N/A")}
    distribution_basis: "5th percentile"
  - step: max_genes_filter
    status: pending
    rationale: "REPLACE"
    threshold: {sth.get("suggested_max_genes", "N/A")}
    distribution_basis: "99th percentile"
  - step: max_pct_mito_filter
    status: pending
    rationale: "REPLACE"
    threshold: {sth.get("suggested_max_pct_mito", "N/A")}
    distribution_basis: "99th percentile"
doublet_detection:
  method_recommendation: scDblFinder
  status: {doublet_st}
  rationale: "REPLACE"
normalisation:
  recommendation: "fill in (use methods from {ecosystem} ecosystem)"
  status: {norm_st}
  rationale: "REPLACE"
feature_selection:
  hvg: {"true" if hvg_st == "already_done" else "false"}
  status: {hvg_st}
  rationale: "REPLACE"
embedding_and_integration:
  methods: [{emb_str}]
  status: {emb_st}
  rationale: "REPLACE"
overall_assessment: |
  Fill in summary.

Replace ALL "REPLACE:" placeholders. {raw_note} {ann_note} {eco_note}
Use 2-space indent. No markdown fences. No text before or after."""


def _build_correction_prompt(yaml_text: str, issues: List[str], insp: Dict[str, Any], *, core_only: bool = False) -> str:
    li = insp.get("layers_info", {})
    dist = insp.get("distribution_inference", {})
    ann_cols = insp.get("annotation_columns_obs", [])
    rules = "\n".join(f"- {r}" for r in insp.get("mandatory_status_rules", []))
    errs = "\n".join(f"- {x}" for x in issues)
    already_done_steps = []
    if dist.get("normalisation_likely_already_done"):
        already_done_steps.append("normalisation")
    if dist.get("hvg_selection_likely_already_done"):
        already_done_steps.append("HVG/feature selection")
    if dist.get("cell_filtering_likely_already_done"):
        already_done_steps.append("cell filtering")
    if ann_cols:
        already_done_steps.append(f"cell-type annotations (columns: {[a['column'] for a in ann_cols]})")
    if li.get("has_raw"):
        already_done_steps.append("raw counts accessibility")
    done_summary = ", ".join(already_done_steps) if already_done_steps else "none"
    dp_rules = ""
    if not core_only:
        dp_rules = """
data_preparation.*.description / required_data_format / preparation_steps may describe the full workflow (multi-omic, etc.), not RNA-only.
rna_input_checklist: ONLY RNA AnnData preparation items; statuses must match Section 2.
other_data_required: external databases, files, modalities outside the RNA checklist — NOT normalisation/HVG/batch-on-.X
(those stay in rna_input_checklist or core QC). For scvi-tools: do NOT claim batch correction is "missing" on .X;
SCVI uses obs batch keys during training.
"""
    core_guard = ""
    if core_only:
        core_guard = (
            "OUTPUT MUST NOT contain a `data_preparation` key — remove it if present. "
            "Only fix metadata, qc_filters, doublet_detection, normalisation, feature_selection, "
            "embedding_and_integration, overall_assessment, data_integrity_warnings.\n\n"
        )
    preamble = f"""Your YAML has consistency errors. Fix ONLY the errors listed below — keep everything else unchanged.

ALREADY-DONE steps (do NOT list these as missing/needed/gaps anywhere): {done_summary}

ERRORS TO FIX:
{errs}

RULES:
{rules}

{core_guard}{dp_rules}
Previous YAML:
"""
    suffix = "\n\nOutput the corrected YAML. No markdown fences, no extra text."
    max_ctx = _cfg_model_ctx()
    chars_per_token = 3.5
    budget_chars = int(max_ctx * chars_per_token) - len(preamble) - len(suffix) - 4000
    if len(yaml_text) > budget_chars > 0:
        log.warning("Truncating previous YAML in correction prompt from %d to %d chars to fit context",
                    len(yaml_text), budget_chars)
        yaml_text = yaml_text[:budget_chars] + "\n... (truncated)"

    return preamble + yaml_text + suffix


def _build_tool_dp_repair_prompt(tool: str, broken_response: str, inspection: Dict[str, Any], td: Dict[str, Any]) -> str:
    ykey = _yaml_map_key(tool)
    tmpl = _tool_data_prep_entry_template(tool, td if isinstance(td, dict) else {})
    return f"""Your previous response was not valid YAML or was missing data_preparation.{tool}.
Output ONLY valid YAML. Root structure:

data_preparation:
  {ykey}:
    description: |
      REPLACE
    required_data_format: |
      REPLACE
    rna_input_checklist:
      - requirement: "REPLACE"
        status: done
        notes: "REPLACE"
    other_data_required: |
      REPLACE
    preparation_steps: |
      REPLACE
    sources:
      - "REPLACE"

Fill every field. At least 5 checklist items. Match Section 2 facts from this inspection context:

{_format_data_state_summary(inspection)}

Template reminder:
{tmpl}

Broken response (fix it):
{broken_response[:12000]}

Output ONLY the YAML. No markdown fences."""


def _ensure_core_without_data_prep(parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(parsed)
    out.pop("data_preparation", None)
    return out


# ---------------------------------------------------------------------------
# LangGraph agent — state, nodes, graph
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State flowing through the LangGraph agent."""
    inspection: Dict[str, Any]
    doc_text: str
    tool_docs: Dict[str, Any]
    qc_schema: Optional[Dict[str, Any]]
    tool_prep: Dict[str, Any]
    merged_schema: Optional[Dict[str, Any]]
    merged_yaml_text: str
    validation_issues: List[str]
    repair_attempts: Annotated[int, operator.add]


def _node_inspect(state: AgentState) -> dict:
    """Pure Python: read and inspect the h5ad file."""
    _print_stage("Node: inspect_anndata")
    h5ad = Path(_cfg_h5ad())
    _ensure_h5ad_exists(h5ad)
    try:
        import anndata  # noqa: F401
        inspection = inspect_anndata(h5ad)
        txt = _format_inspection(inspection)
        print(txt[:4000])
        if len(txt) > 4000:
            print("... (truncated)")
    except ImportError:
        print("Warning: anndata not installed, using reduced inspection.")
        inspection = _reduced_inspection_dict()
    return {"inspection": inspection}


def _node_fetch_docs(state: AgentState) -> dict:
    """Fetch QC reference docs and GitHub-sourced tool documentation."""
    _print_stage("Node: fetch_docs")
    inspection = state["inspection"]
    d_assay, d_tissue, d_organism = _cfg_defaults()
    assay = str(inspection.get("inferred_assay", d_assay))
    tissue = str(inspection.get("inferred_tissue", d_tissue))
    organism = str(inspection.get("inferred_organism", d_organism))
    sv = inspection.get("schema_version")

    doc_text = fetch_documentation(
        assay, tissue, organism,
        sv if isinstance(sv, str) and sv.strip() else None,
    )
    print(f"  QC documentation: {len(doc_text)} chars")

    tools = _cfg_target_tools()
    tool_docs: Dict[str, Any] = {}
    if tools:
        ollama_mod = _safe_import_ollama()
        print(f"  Selecting docs with: {_cfg_model('tool_use')}")
        tool_docs = fetch_tool_docs(ollama_mod, tools)
        for t, d in tool_docs.items():
            text = d["text"] if isinstance(d, dict) else d
            sources = d.get("sources", []) if isinstance(d, dict) else []
            print(f"  {t}: {len(text)} chars, {len(sources)} sources: {sources}")

    return {"doc_text": doc_text, "tool_docs": tool_docs}


def _node_generate_qc(state: AgentState) -> dict:
    """LLM: generate core QC schema (no data_preparation)."""
    _print_stage("Node: generate_qc_schema")
    inspection = state["inspection"]
    doc_text = state["doc_text"]
    ollama_mod = _safe_import_ollama()

    prompt = build_core_prompt(inspection, doc_text)
    print(f"  Prompt: {len(prompt)} chars")
    response = query_ollama(ollama_mod, prompt)
    print("  Received response.")

    parsed, parsed_text = _try_parse_yaml(response)
    for attempt in range(1, 4):
        if parsed is None:
            print(f"  YAML parse failed, repair attempt {attempt}/3")
            response = query_ollama(
                ollama_mod,
                _build_core_yaml_repair_prompt(inspection),
            )
            parsed, parsed_text = _try_parse_yaml(response)
            continue
        parsed = _ensure_core_without_data_prep(parsed)
        issues = _schema_consistency_issues(
            parsed, inspection, check_data_preparation=False,
        )
        if not issues:
            print("  Core QC schema validated.")
            break
        print(f"  {len(issues)} consistency issues, repair attempt {attempt}/3")
        for iss in issues:
            print(f"    - {iss}")
        response = query_ollama(
            ollama_mod,
            _build_correction_prompt(
                parsed_text, issues, inspection, core_only=True,
            ),
        )
        parsed, parsed_text = _try_parse_yaml(response)

    if parsed is not None:
        parsed = _ensure_core_without_data_prep(parsed)

    return {"qc_schema": parsed}


def _node_generate_tool_prep(state: AgentState) -> dict:
    """LLM: generate per-tool data_preparation blocks, then merge with QC schema."""
    _print_stage("Node: generate_tool_prep")
    inspection = state["inspection"]
    tool_docs = state["tool_docs"]
    qc_schema = state["qc_schema"]
    ollama_mod = _safe_import_ollama()

    tool_prep: Dict[str, Any] = {}
    for tool in _cfg_target_tools():
        td = tool_docs.get(tool, {"text": "", "sources": []})
        if not isinstance(td, dict):
            td = {"text": str(td), "sources": []}

        print(f"  Generating data_preparation for {tool!r}")
        prompt = build_tool_data_prep_prompt(tool, inspection, td)
        response = query_ollama(ollama_mod, prompt)

        block = None
        for attempt in range(1, 4):
            parsed_dp, _ = _try_parse_yaml(response)
            block = _extract_tool_dp_block(parsed_dp, tool)
            if block is not None:
                print(f"    {tool}: parsed successfully.")
                break
            print(f"    {tool}: parse failed, repair attempt {attempt}/3")
            response = query_ollama(
                ollama_mod,
                _build_tool_dp_repair_prompt(tool, response, inspection, td),
            )

        if block is not None:
            tool_prep[tool] = block
        else:
            print(f"    Warning: could not generate data_preparation for {tool!r}")

    # Merge core QC schema + tool prep blocks
    merged = dict(qc_schema) if qc_schema else {}
    if tool_prep:
        merged["data_preparation"] = tool_prep
    merged_text = yaml.safe_dump(
        merged, sort_keys=False, allow_unicode=True, default_flow_style=False,
    )

    return {
        "tool_prep": tool_prep,
        "merged_schema": merged,
        "merged_yaml_text": merged_text,
    }


def _node_validate(state: AgentState) -> dict:
    """Pure Python: run consistency checks on merged schema."""
    _print_stage("Node: validate_schema")
    merged = state.get("merged_schema")
    inspection = state["inspection"]

    if merged is None:
        return {"validation_issues": ["No schema to validate — generation failed."]}

    issues = _schema_consistency_issues(
        merged, inspection, check_data_preparation=True,
    )
    if issues:
        print(f"  {len(issues)} issues found:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  Validation passed.")

    return {"validation_issues": issues}


def _node_repair(state: AgentState) -> dict:
    """LLM: fix validation errors in the merged schema."""
    n = state.get("repair_attempts", 0) + 1
    _print_stage(f"Node: repair_schema (attempt {n}/3)")
    merged_text = state["merged_yaml_text"]
    issues = state["validation_issues"]
    inspection = state["inspection"]
    ollama_mod = _safe_import_ollama()

    prompt = _build_correction_prompt(
        merged_text, issues, inspection, core_only=False,
    )
    response = query_ollama(ollama_mod, prompt)
    parsed, parsed_text = _try_parse_yaml(response)

    if parsed is not None:
        new_text = yaml.safe_dump(
            parsed, sort_keys=False, allow_unicode=True, default_flow_style=False,
        )
        return {
            "merged_schema": parsed,
            "merged_yaml_text": new_text,
            "repair_attempts": 1,
        }

    print("  Repair produced invalid YAML; keeping previous version.")
    return {"repair_attempts": 1}


def _node_save(state: AgentState) -> dict:
    """Write final YAML to disk."""
    _print_stage("Node: save_schema")
    merged = state.get("merged_schema")
    issues = state.get("validation_issues", [])
    out_path = Path(_cfg_output())

    if issues and merged is not None:
        merged["validation_warnings"] = issues
        print(f"  Saving with {len(issues)} unresolved validation warnings.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if merged is not None:
        with open(out_path, "w") as f:
            yaml.safe_dump(
                merged, f, sort_keys=False, allow_unicode=True,
                default_flow_style=False,
            )
        print(f"  Wrote YAML to {out_path}")
    else:
        raw = state.get("merged_yaml_text", "")
        with open(out_path, "w") as f:
            f.write(raw)
        print(f"  Warning: no valid schema — saved raw text to {out_path}")

    pending, done, overall = _summarise_schema(merged or {})
    print(f"\n{'=' * 72}")
    print(f"Summary — Pending: {pending} | Done: {done}")
    if overall:
        print(textwrap.fill(overall.strip(), width=88))

    return {}


def _route_after_validation(state: AgentState) -> str:
    """Decide whether to save or repair based on validation results."""
    if not state.get("validation_issues"):
        return "save"
    if state.get("repair_attempts", 0) >= 3:
        return "save"
    return "repair"


def build_graph() -> Any:
    """Construct the LangGraph state machine."""
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(AgentState)

    builder.add_node("inspect", _node_inspect)
    builder.add_node("fetch_docs", _node_fetch_docs)
    builder.add_node("generate_qc", _node_generate_qc)
    builder.add_node("generate_tool_prep", _node_generate_tool_prep)
    builder.add_node("validate", _node_validate)
    builder.add_node("repair", _node_repair)
    builder.add_node("save", _node_save)

    # inspect → fetch_docs → generate_qc → generate_tool_prep → validate
    builder.add_edge(START, "inspect")
    builder.add_edge("inspect", "fetch_docs")
    builder.add_edge("fetch_docs", "generate_qc")
    builder.add_edge("generate_qc", "generate_tool_prep")
    builder.add_edge("generate_tool_prep", "validate")

    # validate → save (pass) or repair (fail, up to 3×)
    builder.add_conditional_edges(
        "validate",
        _route_after_validation,
        {"save": "save", "repair": "repair"},
    )
    builder.add_edge("repair", "validate")
    builder.add_edge("save", END)

    return builder.compile()


def main() -> None:
    _init_config()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ollama_mod = _safe_import_ollama()
    gen_model, tool_model = _cfg_model("generation"), _cfg_model("tool_use")
    _ensure_model_available(ollama_mod, gen_model)
    if tool_model != gen_model:
        _ensure_model_available(ollama_mod, tool_model)

    print(f"Generation model: {gen_model} | Tool-use model: {tool_model} | Context: {_cfg_model_ctx()}")
    print(f"Input:  {_cfg_h5ad()}")
    print(f"Output: {_cfg_output()}")
    print(f"Tools:  {_cfg_target_tools()}")

    graph = build_graph()

    initial_state: AgentState = {
        "inspection": {},
        "doc_text": "",
        "tool_docs": {},
        "qc_schema": None,
        "tool_prep": {},
        "merged_schema": None,
        "merged_yaml_text": "",
        "validation_issues": [],
        "repair_attempts": 0,
    }

    graph.invoke(initial_state)
    print("\nDone.")


if __name__ == "__main__":
    main()
