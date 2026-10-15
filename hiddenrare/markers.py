"""Marker specificity of each candidate group against the rest of its label.

A real cell type carries genes detected in most of its cells and in few cells of the rest of
the label; an over-split fragment of one population does not. Detection means count > 0, on
genes detected in at least 1% of the pool. delta_g = fraction detected in the group minus
fraction detected in the rest of the label.

    python -m hiddenrare.markers POOL.h5ad PILOTS.parquet OUT_DIR
writes OUT_DIR/<pool>.markers.parquet, row-aligned with <pool>.groups.parquet
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .io import load_pool


def detection_matrix(a):
    X = sp.csr_matrix(a.X, copy=True)
    X.data = (X.data > 0).astype(np.float32)
    X.eliminate_zeros()
    return X[:, np.asarray(X.mean(0)).ravel() >= 0.01].tocsr()


def markers(a, G):
    X = detection_matrix(a)
    labels = a.obs["label"].astype(str).values
    n = a.n_obs
    lab_ix = {l: np.where(labels == l)[0] for l in pd.unique(labels)}
    lab_sum = {l: np.asarray(X[ix].sum(0)).ravel() for l, ix in lab_ix.items()}
    rows, cols = np.concatenate(G.members.values), np.repeat(
        np.arange(len(G)), G["size"].values
    )
    S = (
        sp.csr_matrix((np.ones(len(rows), np.float32), (cols, rows)), shape=(len(G), n))
        @ X
    ).toarray()
    out = []
    for j, (l, m) in enumerate(zip(G.label.values, G["size"].values)):
        p_in = S[j] / m
        p_out = (lab_sum[l] - S[j]) / max(len(lab_ix[l]) - m, 1)
        d = np.sort(p_in - p_out)[::-1]
        lr = np.sort(np.log((p_in + 0.01) / (p_out + 0.01)))[::-1]
        out.append(
            dict(
                mk_top1=d[0],
                mk_top5=d[:5].mean(),
                mk_top20=d[:20].mean(),
                mk_n40=int((d >= 0.4).sum()),
                mk_n25=int((d >= 0.25).sum()),
                mk_lr5=lr[:5].mean(),
                mk_neg5=-np.sort(p_in - p_out)[:5].mean(),
            )
        )
    return pd.DataFrame(out)


def run(pool, table, out_dir):
    pid = os.path.splitext(os.path.basename(table))[0]
    out = f"{out_dir}/{pid}.markers.parquet"
    if os.path.exists(out):
        return
    a = load_pool(pool, table, pid)
    G = pd.read_parquet(
        f"{out_dir}/{pid}.groups.parquet", columns=["label", "size", "members"]
    )
    markers(a, G).to_parquet(out, index=False)
    print(pid, len(G), flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:4])
