"""Fetch genome annotation and chromsizes for SCENIC+ Snakemake (run once)."""
import io
import requests
import shutil
from pathlib import Path

import pandas as pd
import pyranges as pr

BASE = Path(__file__).resolve().parent.parent.parent
DATA_INPUTS = BASE / "data_inputs"
SNAKE_DIR = BASE / "scplus_pipeline" / "Snakemake"
DATA_INPUTS.mkdir(parents=True, exist_ok=True)
SNAKE_DIR.mkdir(parents=True, exist_ok=True)

# Genome annotation: Ensembl GTF release 108 GRCh38
gtf_path = DATA_INPUTS / "Homo_sapiens.GRCh38.108.gtf.gz"
if not gtf_path.exists():
    gtf_url = "https://ftp.ensembl.org/pub/release-108/gtf/homo_sapiens/Homo_sapiens.GRCh38.108.gtf.gz"
    r = requests.get(gtf_url, stream=True)
    r.raise_for_status()
    gtf_path.write_bytes(r.content)
annot = pr.read_gtf(str(gtf_path))
df = annot.df

# SCENIC+ requires: Chromosome, Start, End, Strand, Gene, Transcription_Start_Site
# Use transcript rows and collapse to one row per gene
if "Feature" in df.columns:
    df = df[df["Feature"] == "transcript"].copy()
gene_col = "gene_name" if "gene_name" in df.columns else "Gene"
if gene_col not in df.columns:
    gene_col = [c for c in df.columns if "gene" in c.lower()][0]
df["Gene"] = df[gene_col]
df["Transcription_Start_Site"] = df.apply(
    lambda r: r["Start"] if r["Strand"] == "+" else r["End"], axis=1
)
ga = df.groupby("gene_id", as_index=False).agg({
    "Chromosome": "first",
    "Start": "min",
    "End": "max",
    "Strand": "first",
    "Gene": "first",
})
# TSS = 5' end of gene: Start for + strand, End for - strand
ga["Transcription_Start_Site"] = ga.apply(
    lambda r: r["Start"] if r["Strand"] == "+" else r["End"], axis=1
)
ga = ga[["Chromosome", "Start", "End", "Strand", "Gene", "Transcription_Start_Site"]]
out_annot = DATA_INPUTS / "genome_annotation.tsv"
ga.to_csv(out_annot, sep="\t", index=False)
shutil.copy(out_annot, SNAKE_DIR / "genome_annotation.tsv")
print("Wrote genome_annotation.tsv -> data_inputs and Snakemake/")

# Chromsizes from UCSC → SCENIC+ format: Chromosome, Start, End
r = requests.get(
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"
)
r.raise_for_status()
chrom_df = pd.read_csv(
    io.StringIO(r.text),
    sep="\t",
    header=None,
    names=["Chromosome", "End"],
)
chrom_df["Start"] = 0
chrom_df = chrom_df[["Chromosome", "Start", "End"]]
chromsizes_path = SNAKE_DIR / "chromsizes.tsv"
chrom_df.to_csv(chromsizes_path, sep="\t", index=False)
print("Wrote chromsizes.tsv -> Snakemake/")
print("Done.")
