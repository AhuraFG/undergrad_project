"""
Fetch SCENIC+ genome annotation via pybiomart (Option 1 — recommended).
Writes to data_inputs/genome_annotation.tsv and scplus_pipeline/Snakemake/genome_annotation.tsv.
"""
from pathlib import Path

import pandas as pd
import pybiomart

BASE = Path(__file__).resolve().parent.parent.parent
DATA_INPUTS = BASE / "data_inputs"
SNAKE_DIR = BASE / "scplus_pipeline" / "Snakemake"

server = pybiomart.Server(host="http://www.ensembl.org")
mart = server["ENSEMBL_MART_ENSEMBL"]
dataset = mart["hsapiens_gene_ensembl"]

annot = dataset.query(
    attributes=[
        "chromosome_name",
        "start_position",
        "end_position",
        "strand",
        "external_gene_name",
        "transcription_start_site",
        "transcript_biotype",
    ]
)

annot.columns = [
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "Gene",
    "Transcription_Start_Site",
    "Transcript_type",
]
annot["Chromosome"] = "chr" + annot["Chromosome"].astype(str)
annot = annot[annot["Transcript_type"] == "protein_coding"].copy()
annot["Strand"] = annot["Strand"].map({1: "+", -1: "-"})

# Keep only main chromosomes
main_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
annot = annot[annot["Chromosome"].isin(main_chroms)]

annot = annot.dropna()

for out_path in [DATA_INPUTS / "genome_annotation.tsv", SNAKE_DIR / "genome_annotation.tsv"]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annot.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(annot)} rows -> {out_path}")
