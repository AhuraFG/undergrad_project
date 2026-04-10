#!/usr/bin/env python3
"""
Quantitative evaluation of the QC agent on synthetic test cases.

Modes
-----
  --fast   Inspection-only (no LLM). Tests that inspect_anndata returns
           the correct flags for each synthetic variant. Runs in seconds.

  --full   Runs the complete LangGraph agent on each test case (LLM calls
           included). Tests that the final YAML output matches ground truth.
           Takes ~5 min per case.

Usage
-----
    # Generate inputs first (once):
    python generate_test_inputs.py

    # Fast inspection-accuracy eval:
    python run_eval.py --fast

    # Full LLM eval:
    python run_eval.py --full

    # Both:
    python run_eval.py --fast --full
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AGENT_DIR.parent
VENV_PYTHON = PROJECT_ROOT / ".venv_qc_agent" / "bin" / "python3"
EVAL_CONFIG = SCRIPT_DIR / "eval_config.yaml"
TEST_INPUTS = AGENT_DIR / "test_inputs"
TEST_RESULTS = AGENT_DIR / "test_results"

sys.path.insert(0, str(AGENT_DIR))

# ---------------------------------------------------------------------------
# Ground truth definitions
# ---------------------------------------------------------------------------
# Each entry defines what the agent MUST report for that test case.
# Keys:
#   normalisation_status      — "already_done" | "pending"
#   feature_selection_status  — "already_done" | "pending"
#   x_state_contains          — substring expected in x_processing_state
#   raw_accessible            — bool: should inspect report raw counts accessible
#   raw_warning_expected      — bool: should YAML data_integrity_warnings mention raw counts
#   scaled_warning_expected   — bool: should YAML warn about scaled/z-scored data
#   min_genes_status          — "already_done" | "pending" | None (skip check)

GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "base": {
        "normalisation_status": "already_done",
        "feature_selection_status": "already_done",
        "x_state_contains": "log_normalised",
        "raw_accessible": True,
        "raw_warning_expected": False,
        "scaled_warning_expected": False,
        "min_genes_status": "already_done",
    },
    "raw_counts_in_X": {
        "normalisation_status": "pending",
        "feature_selection_status": "already_done",
        "x_state_contains": "raw",
        "raw_accessible": True,   # raw.X still present
        "raw_warning_expected": False,
        "scaled_warning_expected": False,
        "min_genes_status": None,  # min genes may differ when X is raw
    },
    "no_hvg": {
        "normalisation_status": "already_done",
        "feature_selection_status": "pending",
        "x_state_contains": "log_normalised",
        "raw_accessible": True,
        "raw_warning_expected": False,
        "scaled_warning_expected": False,
        "min_genes_status": "already_done",
    },
    "no_raw_counts": {
        "normalisation_status": "already_done",
        "feature_selection_status": "already_done",
        "x_state_contains": "log_normalised",
        "raw_accessible": False,
        "raw_warning_expected": True,
        "scaled_warning_expected": False,
        "min_genes_status": "already_done",
    },
    "scaled_x": {
        "normalisation_status": "pending",
        "feature_selection_status": "already_done",
        "x_state_contains": "scaled",
        "raw_accessible": False,
        "raw_warning_expected": True,
        "scaled_warning_expected": True,
        "min_genes_status": None,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_status(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).lower().strip()
    return "already_done" if "done" in s else ("pending" if "pending" in s else s)


def _warnings_text(yaml_dict: dict) -> str:
    """Flatten data_integrity_warnings to a single string."""
    w = yaml_dict.get("data_integrity_warnings") or []
    if isinstance(w, list):
        return " ".join(str(x) for x in w).lower()
    return str(w).lower()


def _qc_filter_status(yaml_dict: dict, keyword: str) -> Optional[str]:
    """Find qc_filters step whose name contains keyword and return its status."""
    for item in (yaml_dict.get("qc_filters") or []):
        if not isinstance(item, dict):
            continue
        if keyword in str(item.get("step", "")).lower():
            return _normalise_status(item.get("status"))
    return None

# ---------------------------------------------------------------------------
# Fast eval — inspection accuracy (no LLM)
# ---------------------------------------------------------------------------

def run_fast_eval() -> Dict[str, Dict]:
    """Test inspect_anndata on each variant and compare to ground truth."""
    from qc_agent import inspect_anndata  # type: ignore

    print("\n" + "=" * 70)
    print("FAST EVAL — Inspection accuracy (no LLM)")
    print("=" * 70)

    results = {}
    for case_name, gt in GROUND_TRUTH.items():
        h5ad = TEST_INPUTS / f"{case_name}.h5ad"
        if not h5ad.exists():
            print(f"\n  [{case_name}] SKIP — file not found: {h5ad}")
            results[case_name] = {"skipped": True, "reason": "file not found"}
            continue

        print(f"\n  [{case_name}]")
        try:
            insp = inspect_anndata(h5ad)
        except Exception as e:
            print(f"    ERROR during inspection: {e}")
            results[case_name] = {"error": str(e)}
            continue

        checks: List[Tuple[str, bool, str, str]] = []  # (field, pass, expected, got)

        # x_state
        x_state = insp.get("x_processing_state", "")
        exp_x = gt.get("x_state_contains", "")
        checks.append((
            "x_processing_state",
            exp_x in x_state,
            f"contains '{exp_x}'",
            x_state,
        ))

        # raw accessible
        raw_ok = insp.get("layers_info", {}).get("has_raw", False)
        exp_raw = gt.get("raw_accessible")
        if exp_raw is not None:
            checks.append(("raw_accessible", raw_ok == exp_raw, str(exp_raw), str(raw_ok)))

        # dist_flags
        dist = insp.get("distribution_inference", {})

        # normalisation flag
        exp_norm = gt.get("normalisation_status")
        if exp_norm is not None:
            norm_done = dist.get("normalisation_likely_already_done", False)
            got_norm = "already_done" if norm_done else "pending"
            checks.append(("normalisation_flag", got_norm == exp_norm, exp_norm, got_norm))

        # HVG flag
        exp_hvg = gt.get("feature_selection_status")
        if exp_hvg is not None:
            hvg_done = dist.get("hvg_selection_likely_already_done", False)
            got_hvg = "already_done" if hvg_done else "pending"
            checks.append(("hvg_flag", got_hvg == exp_hvg, exp_hvg, got_hvg))

        passed = sum(1 for _, ok, _, _ in checks if ok)
        total = len(checks)

        for field, ok, exp, got in checks:
            mark = "PASS" if ok else "FAIL"
            print(f"    [{mark}] {field}: expected={exp!r}  got={got!r}")

        results[case_name] = {
            "passed": passed,
            "total": total,
            "accuracy": passed / total if total else 0.0,
            "checks": [
                {"field": f, "pass": ok, "expected": e, "got": g}
                for f, ok, e, g in checks
            ],
        }
        print(f"    Score: {passed}/{total} ({100*passed/total:.0f}%)")

    return results

# ---------------------------------------------------------------------------
# Full eval — run the agent, compare YAML output
# ---------------------------------------------------------------------------

def _run_agent(h5ad: Path, output_yaml: Path) -> Tuple[bool, float, int]:
    """
    Run qc_agent.py on h5ad, write to output_yaml.
    Returns (success, elapsed_seconds, returncode).
    """
    cmd = [
        str(VENV_PYTHON),
        str(AGENT_DIR / "qc_agent.py"),
        "--config", str(EVAL_CONFIG),
        "--h5ad", str(h5ad),
        "--output", str(output_yaml),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed, result.returncode


def _score_yaml(yaml_dict: dict, gt: dict, case_name: str) -> Dict[str, Any]:
    """Compare agent YAML output against ground truth. Returns per-field results."""
    checks: List[Tuple[str, bool, str, str]] = []

    # 1. normalisation status
    exp_norm = gt.get("normalisation_status")
    if exp_norm is not None:
        norm_sec = yaml_dict.get("normalisation") or yaml_dict.get("normalization") or {}
        got = _normalise_status(norm_sec.get("status"))
        checks.append(("normalisation.status", got == exp_norm, exp_norm, str(got)))

    # 2. feature_selection status
    exp_hvg = gt.get("feature_selection_status")
    if exp_hvg is not None:
        fs_sec = yaml_dict.get("feature_selection") or {}
        got = _normalise_status(fs_sec.get("status"))
        checks.append(("feature_selection.status", got == exp_hvg, exp_hvg, str(got)))

    # 3. min_genes qc_filter status
    exp_mg = gt.get("min_genes_status")
    if exp_mg is not None:
        got = _qc_filter_status(yaml_dict, "min_gene") or _qc_filter_status(yaml_dict, "min_genes")
        checks.append(("qc_filters.min_genes.status", got == exp_mg, exp_mg, str(got)))

    # 4. raw warning presence
    exp_raw_warn = gt.get("raw_warning_expected")
    if exp_raw_warn is not None:
        warn_text = _warnings_text(yaml_dict)
        raw_phrases = ["raw counts not", "raw counts may", "no raw count",
                       "missing raw", "without accessible raw", "not recoverable"]
        got_raw_warn = any(p in warn_text for p in raw_phrases)
        checks.append((
            "data_integrity_warnings[raw]",
            got_raw_warn == exp_raw_warn,
            str(exp_raw_warn),
            str(got_raw_warn),
        ))

    # 5. scaled warning presence (for scaled_x case)
    exp_scaled_warn = gt.get("scaled_warning_expected")
    if exp_scaled_warn is not None:
        warn_text = _warnings_text(yaml_dict)
        got_scaled_warn = any(p in warn_text for p in ["scaled", "z-score", "zscore"])
        checks.append((
            "data_integrity_warnings[scaled]",
            got_scaled_warn == exp_scaled_warn,
            str(exp_scaled_warn),
            str(got_scaled_warn),
        ))

    # 6. Hallucination count (run consistency checker)
    from qc_agent import inspect_anndata, _schema_consistency_issues  # type: ignore
    h5ad = TEST_INPUTS / f"{case_name}.h5ad"
    hallucinations = []
    if h5ad.exists():
        try:
            insp = inspect_anndata(h5ad)
            hallucinations = _schema_consistency_issues(
                yaml_dict, insp, check_data_preparation=False,
            )
        except Exception as e:
            hallucinations = [f"check error: {e}"]

    # 7. Rationale coverage
    sections_with_status = []
    for item in (yaml_dict.get("qc_filters") or []):
        if isinstance(item, dict) and "status" in item:
            sections_with_status.append(("qc_filters." + item.get("step", "?"), item))
    for sec in ("normalisation", "feature_selection", "doublet_detection", "embedding_and_integration"):
        block = yaml_dict.get(sec)
        if isinstance(block, dict) and "status" in block:
            sections_with_status.append((sec, block))

    missing_rationale = [name for name, block in sections_with_status
                         if not str(block.get("rationale", "")).strip()]

    # 8. Validation warnings from agent's own output
    agent_warnings = yaml_dict.get("validation_warnings") or []

    passed = sum(1 for _, ok, _, _ in checks if ok)
    total = len(checks)

    return {
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total else 0.0,
        "checks": [{"field": f, "pass": ok, "expected": e, "got": g}
                   for f, ok, e, g in checks],
        "hallucinations": len(hallucinations),
        "hallucination_details": hallucinations,
        "missing_rationale": missing_rationale,
        "agent_validation_warnings": len(agent_warnings),
    }


def run_full_eval() -> Tuple[Dict[str, Any], Path]:
    """Run the full LangGraph agent on each variant and score the YAML output."""
    print("\n" + "=" * 70)
    print("FULL EVAL — End-to-end agent accuracy (LLM calls included)")
    print("=" * 70)

    TEST_RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = TEST_RESULTS / timestamp
    run_dir.mkdir()

    results = {}
    for case_name, gt in GROUND_TRUTH.items():
        h5ad = TEST_INPUTS / f"{case_name}.h5ad"
        out_yaml = run_dir / f"{case_name}.yaml"

        if not h5ad.exists():
            print(f"\n  [{case_name}] SKIP — file not found: {h5ad}")
            results[case_name] = {"skipped": True}
            continue

        print(f"\n  [{case_name}] running agent...", flush=True)
        success, elapsed, rc = _run_agent(h5ad, out_yaml)
        print(f"    Finished in {elapsed:.0f}s (exit code {rc})")

        if not success or not out_yaml.exists():
            print(f"    ERROR: agent failed (exit code {rc})")
            results[case_name] = {"error": f"agent exit code {rc}", "elapsed": elapsed}
            continue

        with open(out_yaml) as f:
            yaml_dict = yaml.safe_load(f) or {}

        scores = _score_yaml(yaml_dict, gt, case_name)
        scores["elapsed_s"] = round(elapsed, 1)
        results[case_name] = scores

        for ch in scores["checks"]:
            mark = "PASS" if ch["pass"] else "FAIL"
            print(f"    [{mark}] {ch['field']}: expected={ch['expected']!r}  got={ch['got']!r}")
        print(f"    Status accuracy:    {scores['passed']}/{scores['total']} "
              f"({100*scores['accuracy']:.0f}%)")
        print(f"    Hallucinations:     {scores['hallucinations']}")
        print(f"    Missing rationale:  {scores['missing_rationale']}")
        print(f"    Agent warnings:     {scores['agent_validation_warnings']}")

    return results, run_dir

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(fast_results: Optional[dict], full_results: Optional[dict]):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    cases = list(GROUND_TRUTH.keys())
    header = f"{'Case':<22} "

    if fast_results:
        header += f"{'Fast':>10}"
    if full_results:
        header += f"{'Full':>10}  {'Halluc':>7}  {'Rationale':>9}  {'Time(s)':>7}"

    print(header)
    print("-" * 70)

    fast_accs, full_accs, halluc_counts = [], [], []

    for case in cases:
        row = f"{case:<22} "

        if fast_results:
            r = fast_results.get(case, {})
            if r.get("skipped"):
                row += f"{'SKIP':>10}"
            elif r.get("error"):
                row += f"{'ERROR':>10}"
            else:
                acc = r.get("accuracy", 0)
                fast_accs.append(acc)
                row += f"{acc*100:>9.0f}%"

        if full_results:
            r = full_results.get(case, {})
            if r.get("skipped"):
                row += f"{'SKIP':>10}"
            elif r.get("error"):
                row += f"{'ERROR':>10}"
            else:
                acc = r.get("accuracy", 0)
                full_accs.append(acc)
                halluc = r.get("hallucinations", 0)
                halluc_counts.append(halluc)
                miss = len(r.get("missing_rationale", []))
                row += (f"{acc*100:>9.0f}%  {halluc:>7}  {miss:>9}  "
                        f"{r.get('elapsed_s', '?'):>7}")

        print(row)

    print("-" * 70)
    totals = "OVERALL" + " " * 15
    if fast_results and fast_accs:
        totals += f"{sum(fast_accs)/len(fast_accs)*100:>9.0f}%"
    if full_results and full_accs:
        totals += (f"{sum(full_accs)/len(full_accs)*100:>9.0f}%  "
                   f"{sum(halluc_counts):>7}")
    print(totals)


def save_report(fast_results, full_results, run_dir: Optional[Path]):
    """Save JSON report for later comparison."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "fast_eval": fast_results,
        "full_eval": full_results,
    }
    out = (run_dir or TEST_RESULTS) / "eval_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {out}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate QC agent accuracy")
    parser.add_argument("--fast", action="store_true",
                        help="Run fast inspection-only eval (no LLM)")
    parser.add_argument("--full", action="store_true",
                        help="Run full end-to-end agent eval (LLM calls)")
    args = parser.parse_args()

    if not args.fast and not args.full:
        parser.error("Specify at least one of --fast or --full")

    fast_results = None
    full_results = None
    run_dir = None

    if args.fast:
        fast_results = run_fast_eval()

    if args.full:
        full_results, run_dir = run_full_eval()

    print_summary(fast_results, full_results)
    save_report(fast_results, full_results, run_dir)


if __name__ == "__main__":
    main()
