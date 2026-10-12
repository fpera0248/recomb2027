"""Shared IO for the Python wrappers. Input: pool h5ad (raw counts in X, obs['label']); output: parquet with cell_id, score, call."""
import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_pool(path):
    a = ad.read_h5ad(path)
    d = a.X.data if sp.issparse(a.X) else np.asarray(a.X).ravel()
    assert np.allclose(d, np.round(d)), "raw integer counts expected"
    return a


def opaque_order(pid, obs_names):
    """Order by sha1(pid NUL obs_name) and its inverse."""
    dg = np.array([hashlib.sha1(f"{pid}\x00{o}".encode()).hexdigest() for o in obs_names])
    o = np.argsort(dg, kind="stable")
    return o, np.argsort(o)


def write_out(cell_id, score, call, path):
    pd.DataFrame({"cell_id": np.asarray(cell_id).astype(str), "score": np.asarray(score, float), "call": np.asarray(call, bool)}).to_parquet(path, index=False)
