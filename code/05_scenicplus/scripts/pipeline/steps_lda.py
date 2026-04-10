"""Step 3: cisTopic LDA and region set export."""
from __future__ import annotations

import os
import pickle
import shutil
import sys

import numpy as np

from .paths import DATA_INPUTS, OUT_DIR


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

    n_topics_list = getattr(args, "topic_grid", None) or [20]
    if isinstance(n_topics_list, int):
        n_topics_list = [n_topics_list]
    n_topics_list = sorted(set(int(x) for x in n_topics_list))
    if not n_topics_list:
        n_topics_list = [20]

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
    if len(models) > 1:
        print("  Evaluating models to select best...")
        metrics = None
        try:
            out = evaluate_models(cistopic_obj, models, select_model=None, return_metrics=True)
            if out is not None:
                metrics = out[0] if isinstance(out, (tuple, list)) and len(out) else out
        except TypeError:
            try:
                metrics = evaluate_models(cistopic_obj, models, return_metrics=True)
            except Exception:
                metrics = evaluate_models(cistopic_obj, models)
                metrics = getattr(metrics, "__getitem__", lambda i: None)(0) if isinstance(metrics, (tuple, list)) else metrics
        except Exception as e:
            print(f"  evaluate_models failed ({e}); using first model.")
        if metrics is not None and hasattr(metrics, "columns") and hasattr(metrics, "index") and len(metrics) == len(models):
            for col, minimize in [("Log-likelihood", False), ("loglikelihood", False), ("Cao_Juan_2009", True), ("Arun_2010", True)]:
                if col in metrics.columns:
                    s = metrics[col].values
                    best_idx = int(np.argmin(s) if minimize else np.argmax(s))
                    break
            else:
                best_idx = 0
            best_idx = min(max(0, best_idx), len(models) - 1)
            metrics_path = os.path.join(OUT_DIR, "lda_topic_grid_metrics.csv")
            metrics.to_csv(metrics_path)
            print(f"  Best model: {n_topics_list[best_idx]} topics (index {best_idx}). Metrics saved to lda_topic_grid_metrics.csv")
    best_model = models[best_idx]
    n_selected = n_topics_list[best_idx]

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
