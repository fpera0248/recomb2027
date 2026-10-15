"""Halo contrast of each candidate group.

The halo is as many nearest same-label cells outside the group (to the group centroid in the
pool PCA) as the group has members. A fragment of an abundant population shares its markers
with its halo; a whole rare type does not.

    python -m hiddenrare.halo POOL.h5ad PILOTS.parquet OUT_DIR
writes OUT_DIR/<pool>.halo.parquet, row-aligned with <pool>.groups.parquet
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .candidates import preprocess_pca
from .io import load_pool
from .markers import detection_matrix


def halos(Z, labels, G):
    """Halo index arrays and the halo/member distance ratio per group."""
    lab_ix = {l: np.where(labels == l)[0] for l in pd.unique(labels)}
    H, ratio = [], []
    for l, m in zip(G.label.values, G.members.values):
        pi = lab_ix[l]
        out = pi[~np.isin(pi, m)]
        c = Z[m].mean(0)
        d = np.linalg.norm(Z[out] - c, axis=1)
        k = min(len(m), len(out))
        j = np.argpartition(d, k - 1)[:k]
        H.append(out[j])
        ratio.append(
            np.median(d[j]) / (np.median(np.linalg.norm(Z[m] - c, axis=1)) + 1e-9)
        )
    return H, np.array(ratio)


def features(X, labels, G, H, ratio):
    n = X.shape[0]
    lab_sum = {l: np.asarray(X[labels == l].sum(0)).ravel() for l in pd.unique(labels)}

    def indicator(sets):
        lens = list(map(len, sets))
        return sp.csr_matrix(
            (
                np.ones(sum(lens), np.float32),
                (np.repeat(np.arange(len(sets)), lens), np.concatenate(sets)),
            ),
            shape=(len(sets), n),
        )

    S = (indicator(list(G.members.values)) @ X).toarray()
    Sh = (indicator(H) @ X).toarray()
    out = []
    for j, (l, m, h) in enumerate(zip(G.label.values, G["size"].values, map(len, H))):
        p_in = S[j] / m
        p_halo = Sh[j] / max(h, 1)
        p_out = (lab_sum[l] - S[j]) / max((labels == l).sum() - m, 1)
        dh = np.sort(p_in - p_halo)[::-1]
        top = np.argsort(p_in - p_out)[::-1][:5]
        out.append(
            dict(
                hl_top5=dh[:5].mean(),
                hl_top20=dh[:20].mean(),
                hl_n25=int((dh >= 0.25).sum()),
                hl_mk5=(p_in - p_halo)[top].mean(),
                hl_neg5=-dh[::-1][:5].mean(),
                hl_ratio=ratio[j],
            )
        )
    return pd.DataFrame(out)


def run(pool, table, out_dir):
    pid = os.path.splitext(os.path.basename(table))[0]
    out = f"{out_dir}/{pid}.halo.parquet"
    if os.path.exists(out):
        return
    a = load_pool(pool, table, pid)
    G = pd.read_parquet(
        f"{out_dir}/{pid}.groups.parquet", columns=["label", "size", "members"]
    )
    labels = a.obs["label"].astype(str).values
    H, ratio = halos(preprocess_pca(a)[0], labels, G)
    features(detection_matrix(a), labels, G, H, ratio).to_parquet(out, index=False)
    print(pid, len(G), flush=True)


def _check():
    # cells 0-9 are a fragment of a 40-cell population sharing genes 0-4; cells 40-49 are a distinct type on genes 5-9
    rng = np.random.RandomState(0)
    X = np.zeros((100, 20), np.float32)
    X[:40, :5] = 1
    X[40:50, 5:10] = 1
    X[:, 10:] = rng.rand(100, 10) < 0.5
    Z = np.r_[rng.randn(40, 2), rng.randn(10, 2) + [6, 0], rng.randn(50, 2) + [0, 6]]
    labels = np.array(["L"] * 100)
    G = pd.DataFrame(
        {
            "label": ["L", "L"],
            "size": [10, 10],
            "members": [np.arange(10), np.arange(40, 50)],
        }
    )
    H, r = halos(Z, labels, G)
    F = features(sp.csr_matrix(X), labels, G, H, r)
    assert F.hl_mk5[0] < 0.2 < 0.8 < F.hl_mk5[1] and r[1] > r[0], F


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(*sys.argv[1:4])
    else:
        _check()
