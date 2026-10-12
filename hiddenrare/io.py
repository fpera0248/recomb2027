"""Pool loading and the cell-id convention shared by all stages.

A pool is a raw-count h5ad plus a table with obs_name, label and donor per cell. Methods only
see counts, the visible label and the donor; cells are reordered and renamed by a hash of the
pool id and the barcode so row order cannot encode the truth. Evaluation maps scores back to
the truth through the same hash.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import scanpy as sc


def opaque_order(pool_id, obs_names):
    """Sort order and 16-hex cell ids: sha1(pool_id NUL obs_name), sorted."""
    digests = [hashlib.sha1(f"{pool_id}\x00{o}".encode()).hexdigest() for o in obs_names]
    order = np.argsort(digests)
    ids = np.array([digests[i][:16] for i in order])
    assert len(set(ids)) == len(ids), f"{pool_id}: cell id collision"
    return order, ids


def load_pool(h5ad, table, pool_id=None):
    """AnnData with raw counts in X and obs[label, donor], indexed by the opaque cell ids."""
    pool_id = pool_id or os.path.splitext(os.path.basename(table))[0]
    df = pd.read_parquet(table)
    a = sc.read_h5ad(h5ad, backed="r")
    pos = a.obs_names.get_indexer(df["obs_name"].values)
    if (pos < 0).any():
        a.file.close()
        raise KeyError(f"{pool_id}: {int((pos < 0).sum())} barcodes missing from {h5ad}")
    order = np.argsort(pos)          # sorted positions read faster from the backed file
    sub = a[pos[order]].to_memory()
    if sub.raw is not None:          # some CELLxGENE files keep the raw counts in .raw
        sub = sc.AnnData(X=sub.raw.X.copy(), obs=sub.obs, var=pd.DataFrame(index=sub.raw.var_names))
    a.file.close()
    sub = sub[np.argsort(order)].copy()
    sub.obsm.clear(); sub.varm.clear(); sub.obsp.clear(); sub.uns.clear()
    for k in list(sub.layers.keys()):
        if k is not None:
            del sub.layers[k]
    sub.var = pd.DataFrame(index=sub.var_names)
    order, ids = opaque_order(pool_id, df["obs_name"].values)
    sub = sub[order].copy()
    sub.obs_names = ids
    sub.obs = pd.DataFrame({"label": df["label"].values[order], "donor": df["donor"].values[order]}, index=ids)
    return sub


def rprec(score, is_rare):
    """Precision in the top k = number of positives; ties at the k-th score are split at their expected value."""
    k = int(is_rare.sum())
    if k == 0:
        return np.nan
    t = np.sort(score)[::-1][k - 1]
    above, tie = score > t, score == t
    return (is_rare[above].sum() + (k - above.sum()) * is_rare[tie].mean()) / k


def _check_rprec():
    r = np.array([1, 1, 0, 0, 0], bool)
    assert rprec(np.array([.9, .8, .1, .2, .3]), r) == 1.0
    assert rprec(np.array([.1, .9, .8, .2, .3]), r) == 0.5
    assert rprec(np.ones(5), r) == 0.4
    assert np.isnan(rprec(np.ones(5), np.zeros(5, bool)))


if __name__ == "__main__":
    _check_rprec()
