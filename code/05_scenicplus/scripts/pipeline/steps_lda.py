"""Step 3: cisTopic LDA and region set export."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

from .paths import DATA_INPUTS, OUT_DIR

DEFAULT_TOPIC_GRID = [5, 10, 15, 20, 25, 30]


def step_lda_and_copy(args):
    import matplotlib

    matplotlib.use("Agg")
    from pycisTopic.lda_models import run_cgs_models, evaluate_models
    from pycisTopic.topic_binarization import binarize_topics
    from pycisTopic.utils import region_names_to_coordinates

    raw_path = os.path.join(OUT_DIR, "cistopic_obj_raw.pkl")
    if not os.path.exists(raw_path):
        print("[Step 3] Missing", raw_path, "- create once via scenicplus.ipynb (ATAC + cisTopic object).")
        sys.exit(1)
    with open(raw_path, "rb") as f:
        cistopic_obj = pickle.load(f)

    n_topics_list = getattr(args, "topic_grid", None) or DEFAULT_TOPIC_GRID
    if isinstance(n_topics_list, int):
        n_topics_list = [n_topics_list]
    n_topics_list = sorted(set(int(x) for x in n_topics_list))
    if not n_topics_list:
        n_topics_list = DEFAULT_TOPIC_GRID.copy()

    print("[Step 3] Running cisTopic LDA topic grid:", n_topics_list)
    models = run_cgs_models(
        cistopic_obj=cistopic_obj,
        n_topics=n_topics_list,
        n_cpu=args.cores,
        n_iter=getattr(args, "lda_n_iter", 500),
        random_state=555,
        alpha=50,
        alpha_by_topic=True,
        eta=0.1,
        eta_by_topic=False,
        save_path=None,
    )

    best_idx = 0
    selection_metric = "first_model"
    metrics_path = os.path.join(OUT_DIR, "lda_topic_grid_metrics.csv")
    if len(models) > 1:
        print("  Evaluating models to select best...")
        metrics_rows = []
        for m in models:
            row = {"n_topics": int(getattr(m, "n_topic", np.nan))}
            if hasattr(m, "metrics") and hasattr(m.metrics, "columns") and "Metric" in getattr(m.metrics, "index", []):
                for col in m.metrics.columns:
                    row[str(col)] = m.metrics.loc["Metric", col]
            metrics_rows.append(row)
        metrics = pd.DataFrame(metrics_rows).sort_values("n_topics").reset_index(drop=True)
        metrics.to_csv(metrics_path, index=False)

        eval_err = None
        selected_model = None
        eval_attempts = (
            lambda: evaluate_models(models, select_model=None, return_model=True, plot=False, plot_metrics=False),
            lambda: evaluate_models(models, select_model=None, return_model=True, plot=False),
            lambda: evaluate_models(models, select_model=None),
        )
        for fn in eval_attempts:
            try:
                out = fn()
                if out is not None and hasattr(out, "n_topic"):
                    selected_model = out
                    break
            except Exception as e:
                eval_err = e
                continue

        if selected_model is not None:
            selected_topics = int(getattr(selected_model, "n_topic"))
            if selected_topics in n_topics_list:
                best_idx = n_topics_list.index(selected_topics)
                selection_metric = "evaluate_models_combined"
        else:
            if eval_err is not None:
                print(f"  evaluate_models failed ({eval_err}); falling back to metric ranking.")
            for col, minimize in [("Log-likelihood", False), ("loglikelihood", False), ("Cao_Juan_2009", True), ("Arun_2010", True)]:
                if col in metrics.columns:
                    s = metrics[col].values
                    ranked_idx = int(np.argmin(s) if minimize else np.argmax(s))
                    ranked_topic = int(metrics.loc[ranked_idx, "n_topics"])
                    if ranked_topic in n_topics_list:
                        best_idx = n_topics_list.index(ranked_topic)
                        selection_metric = col
                        break
        best_idx = min(max(0, best_idx), len(models) - 1)
        print(f"  Best model: {n_topics_list[best_idx]} topics (index {best_idx}). Metrics saved to lda_topic_grid_metrics.csv")
    best_model = models[best_idx]
    n_selected = n_topics_list[best_idx]
    selection_summary = {
        "tested_topic_counts": n_topics_list,
        "n_models": len(models),
        "selected_model_index": int(best_idx),
        "selected_topic_count": int(n_selected),
        "selection_metric": selection_metric,
        "metrics_csv": metrics_path if len(models) > 1 else None,
    }
    selection_path = os.path.join(OUT_DIR, "lda_topic_grid_selection.json")
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2)
    print(f"  Saved topic-grid selection summary to {selection_path}")

    cistopic_obj.add_LDA_model(best_model)
    with open(os.path.join(OUT_DIR, "cistopic_obj.pkl"), "wb") as f:
        pickle.dump(cistopic_obj, f)

    region_sets_root = os.path.join(OUT_DIR, "region_sets")
    for d in ["Topics_top_3k", "Topics_otsu"]:
        os.makedirs(os.path.join(region_sets_root, d), exist_ok=True)

    binarize_ntop = getattr(args, "binarize_ntop", 3000)
    for method, ntop in [("ntop", binarize_ntop), ("otsu", None)]:
        kw = {"method": method, "plot": False}
        if ntop is not None:
            kw["ntop"] = ntop
        region_bin = binarize_topics(cistopic_obj, **kw)
        out_sub = os.path.join(region_sets_root, "Topics_top_3k" if ntop else "Topics_otsu")
        for topic_name, regions in region_bin.items():
            bed_df = region_names_to_coordinates(regions.index).sort_values(["Chromosome", "Start", "End"])
            bed_df.to_csv(os.path.join(out_sub, f"{topic_name}.bed"), sep="\t", header=False, index=False)

    os.makedirs(DATA_INPUTS, exist_ok=True)
    shutil.copy(os.path.join(OUT_DIR, "cistopic_obj.pkl"), os.path.join(DATA_INPUTS, "cistopic_obj.pkl"))
    if os.path.exists(os.path.join(DATA_INPUTS, "region_sets")):
        shutil.rmtree(os.path.join(DATA_INPUTS, "region_sets"))
    shutil.copytree(region_sets_root, os.path.join(DATA_INPUTS, "region_sets"))
    print(f"  Selected {n_selected} topics; binarised (ntop={binarize_ntop}, otsu). Copied to data_inputs/")
