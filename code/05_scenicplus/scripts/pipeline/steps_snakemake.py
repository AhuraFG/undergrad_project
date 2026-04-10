"""Step 4: Snakemake SCENIC+ workflow."""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd

from .paths import PIPELINE_ROOT, SNAKE_DIR
from .snakemake_config import write_snakemake_config


def step_snakemake(args):
    config_path = os.path.join(SNAKE_DIR, "config", "config.yaml")
    if not os.path.isfile(config_path):
        write_snakemake_config(args)
    else:
        print("[Step 4] Using existing Snakemake config (config/config.yaml).")
    print("[Step 4] Running Snakemake...")

    tmp_dir = os.path.join(PIPELINE_ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    env = os.environ.copy()
    bin_dir = os.path.dirname(sys.executable)
    env["PATH"] = os.pathsep.join([bin_dir, env.get("PATH", "")])

    snake_cmd = [
        sys.executable,
        "-m",
        "snakemake",
        "-c",
        str(args.cores),
        "--rerun-incomplete",
    ]
    subprocess.run([sys.executable, "-m", "snakemake", "--unlock"], cwd=SNAKE_DIR, env=env, check=False)
    rc = subprocess.run(snake_cmd, cwd=SNAKE_DIR, env=env)

    if rc.returncode != 0:
        log_dir = os.path.join(SNAKE_DIR, ".snakemake", "log")
        if os.path.isdir(log_dir):
            logs = sorted(os.listdir(log_dir))
            if logs:
                print(f"  Snakemake log: {os.path.join(log_dir, logs[-1])}")

        cist = os.path.join(SNAKE_DIR, "cistromes_direct.h5ad")
        rtg = os.path.join(SNAKE_DIR, "region_to_gene_adj.tsv")
        if os.path.isfile(cist) and os.path.isfile(rtg):
            print("  Snakemake failed. Filtering region_to_gene and retrying...")
            import scanpy as sc

            cist_adata = sc.read_h5ad(cist)
            valid = set(cist_adata.obs_names)
            df = pd.read_csv(rtg, sep="\t")
            region_col = "region" if "region" in df.columns else "Region"
            if region_col in df.columns:
                df = df[df[region_col].isin(valid)]
                df.to_csv(rtg, sep="\t", index=False)
            subprocess.run([sys.executable, "-m", "snakemake", "--unlock"], cwd=SNAKE_DIR, env=env, check=False)
            rc = subprocess.run(snake_cmd, cwd=SNAKE_DIR, env=env)

    if rc.returncode != 0:
        sys.exit(rc.returncode)
    print("  Snakemake done.")
