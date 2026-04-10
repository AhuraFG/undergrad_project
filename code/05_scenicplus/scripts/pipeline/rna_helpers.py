"""RNA count detection and TF helpers."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .paths import TF_LIST_PATH


def get_tf_symbols():
    if not os.path.isfile(TF_LIST_PATH):
        return set()
    df = pd.read_csv(TF_LIST_PATH)
    col = [c for c in df.columns if "symbol" in c.lower()]
    if not col:
        return set()
    return set(df[col[0]].dropna().astype(str).str.strip().str.upper().unique())


def safe_get_vec(adata, gene):
    if gene not in adata.var_names:
        return None
    j = list(adata.var_names).index(gene)
    x = adata.X[:, j]
    return np.asarray(x.toarray()).ravel() if issparse(x) else np.asarray(x).ravel()


def matrix_sample_looks_like_counts(X, n_obs: int, n_vars: int) -> bool:
    if n_obs == 0 or n_vars == 0:
        return False
    size = n_obs * n_vars
    if issparse(X):
        if X.nnz == 0:
            return False
        n_sample = min(400, n_obs)
        rng = np.random.default_rng(42)
        rows = rng.choice(n_obs, size=n_sample, replace=False)
        chunks = []
        for i in rows:
            row = np.asarray(X[i, :].toarray()).ravel()
            chunks.append(row)
        arr = np.concatenate(chunks)
    else:
        if size > 1e6:
            rng = np.random.default_rng(42)
            rows = rng.choice(n_obs, size=min(500, n_obs), replace=False)
            cols = rng.choice(n_vars, size=min(2000, n_vars), replace=False)
            arr = np.asarray(X[np.ix_(rows, cols)]).astype(float).ravel()
        else:
            arr = np.asarray(X).astype(float).ravel()
    if arr.size == 0:
        return False
    if np.any(arr < 0):
        return False
    mx = float(np.max(arr))
    if mx < 20 or mx > 1e7:
        return False
    pos = arr[arr > 0]
    if pos.size == 0:
        return False
    if np.mean(np.abs(pos - np.round(pos)) < 0.01) < 0.9:
        return False
    return True


def has_real_counts(rna):
    if "counts" in getattr(rna, "layers", {}):
        c = rna.layers["counts"]
        n_obs, n_vars = c.shape[0], c.shape[1]
        return matrix_sample_looks_like_counts(c, n_obs, n_vars)
    if rna.raw is not None:
        if matrix_sample_looks_like_counts(rna.raw.X, rna.n_obs, rna.raw.n_vars):
            return True
        raw = rna.raw.to_adata()
        if list(raw.var_names) == list(rna.var_names):
            return True
        common = rna.var_names[rna.var_names.isin(raw.var_names)]
        if len(common) == rna.n_vars:
            return True
    return False
