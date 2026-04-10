"""Convert GTF or existing annotation to SCENIC+ genome_annotation with standard columns.

Output format (to match expected gene annotations):
  Chromosome, Start, End, Strand, gene_id, gene_name, gene_biotype, Gene, Transcription_Start_Site

SCENIC+ requires Gene and Transcription_Start_Site; Gene = gene_name, TSS = 5' end of gene.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA_INPUTS = BASE / "data_inputs"
SNAKE_DIR = BASE / "scplus_pipeline" / "Snakemake"

# Prefer GTF to get gene_id, gene_name, gene_biotype; else use existing TSV
gtf_path = DATA_INPUTS / "Homo_sapiens.GRCh38.108.gtf.gz"
inp_tsv = SNAKE_DIR / "genome_annotation.tsv"
if not inp_tsv.exists():
    inp_tsv = DATA_INPUTS / "genome_annotation.tsv"

if gtf_path.exists():
    # Read GTF in chunks to get gene rows (gene_id, gene_name, gene_biotype)
    chunks = []
    for chunk in pd.read_csv(gtf_path, sep="\t", comment="#", header=None, low_memory=False,
                             names=["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attributes"],
                             chunksize=100_000):
        chunk = chunk[chunk["Feature"] == "gene"]
        if chunk.empty:
            continue
        def get_attr(s, key):
            if pd.isna(s):
                return ""
            for part in str(s).strip(";").split(";"):
                part = part.strip()
                if part.startswith(key + " "):
                    return part[len(key) + 2:].strip('"')
            return ""
        chunk["gene_id"] = chunk["Attributes"].apply(lambda s: get_attr(s, "gene_id"))
        chunk["gene_name"] = chunk["Attributes"].apply(lambda s: get_attr(s, "gene_name"))
        chunk["gene_biotype"] = chunk["Attributes"].apply(lambda s: get_attr(s, "gene_biotype"))
        chunks.append(chunk[["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name", "gene_biotype"]])
    df = pd.concat(chunks, ignore_index=True)
    df["Gene"] = df["gene_name"]
    df["Transcription_Start_Site"] = df.apply(lambda r: r["Start"] if r["Strand"] == "+" else r["End"], axis=1)
else:
    df = pd.read_csv(inp_tsv, sep="\t", low_memory=False)
    if "Feature" in df.columns:
        df = df[df["Feature"] == "gene"].copy()
        df["gene_id"] = df.get("gene_id", "")
        df["gene_name"] = df.get("gene_name", df.get("Gene", ""))
        df["Gene"] = df["gene_name"]
        df["gene_biotype"] = df.get("gene_biotype", "")
        df["Transcription_Start_Site"] = df.apply(lambda r: r["Start"] if r["Strand"] == "+" else r["End"], axis=1)
    else:
        df["gene_id"] = df.get("gene_id", "")
        df["gene_name"] = df.get("gene_name", df.get("Gene", ""))
        df["Gene"] = df["gene_name"]
        df["gene_biotype"] = df.get("gene_biotype", "")
        if "Transcription_Start_Site" not in df.columns:
            df["Transcription_Start_Site"] = df.apply(lambda r: r["Start"] if r["Strand"] == "+" else r["End"], axis=1)

# Normalize chromosome to chr-prefix
df["Chromosome"] = df["Chromosome"].astype(str).apply(lambda c: c if c.startswith("chr") else f"chr{c}")

ga = df[["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name", "gene_biotype", "Gene", "Transcription_Start_Site"]].copy()

for out in [DATA_INPUTS / "genome_annotation.tsv", SNAKE_DIR / "genome_annotation.tsv"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    ga.to_csv(out, sep="\t", index=False)
    print("Wrote", out, "rows:", len(ga))

print("Columns:", list(ga.columns))
