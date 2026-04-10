"""Paths and constants for the SCENIC+ microglia pipeline (05_scenicplus root)."""
from __future__ import annotations

import os

# 05_scenicplus directory (parent of scripts/)
PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(PIPELINE_ROOT, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sea_ad")
DATA_INPUTS = os.path.join(PIPELINE_ROOT, "data_inputs")
SNAKE_DIR = os.path.join(PIPELINE_ROOT, "scplus_pipeline", "Snakemake")
OUT_DIR = os.path.join(PIPELINE_ROOT, "outs")
TF_LIST_PATH = os.path.join(PROJECT_ROOT, "data", "human_tfs.csv")
GEX_PATH = os.path.join(DATA_INPUTS, "SEAAD_MTG_microglia_rna.h5ad")
MDATA_PATH = os.path.join(SNAKE_DIR, "scplusmdata.h5mu")
PLOTS_DIR = os.path.join(PIPELINE_ROOT, "plots")
RESULTS_DIR = os.path.join(PIPELINE_ROOT, "results")

CONTAMINANT_MARKERS = {"T_NK": ["TRAC", "CD3D", "CD3E", "NKG7", "GNLY"], "B": ["CD79A", "MS4A1"]}
MICROGLIA_PATTERNS = ["microglia", "Microglia", "Microglia-PVM", "PVM"]
MICROGLIA_TFS = {"SPI1", "IRF8", "MAFB", "CEBPB", "CEBPD", "STAT1", "STAT3", "RELA", "JUN", "FOS", "NFKB1", "RUNX1"}
LYMPHOID_TFS = {"TBX21", "IKZF1", "IKZF3", "BCL6", "GATA3", "FOXP3"}
