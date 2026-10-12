"""In-house baselines, same input and output as the main method: Leiden subclustering and an
isolation forest on the PCA of each label.

    python -m hiddenrare.baselines POOL.h5ad PILOTS.parquet OUT_ROOT [leiden_sub iforest_pca]
writes OUT_ROOT/<method>/<pool>.parquet with cell_id, score, call and group (-1 = no group)
"""
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from .candidates import N_NEIGHBORS, RNG, preprocess_pca
from .io import load_pool

N_JOBS = 2
N_ESTIMATORS = 100
RES_GRID = (0.3, 0.5, 0.8, 1.2)
MIN_SHARE = 0.10          # a subcluster under 10% of its label is a candidate
MIN_LABEL_N = 20


def leiden_labels(X, resolution):
    a = sc.AnnData(X)
    a.obsm["X_rep"] = X
    sc.pp.neighbors(a, n_neighbors=min(N_NEIGHBORS, a.n_obs - 1), use_rep="X_rep", random_state=RNG)
    sc.tl.leiden(a, resolution=resolution, random_state=RNG, flavor="igraph", n_iterations=2, directed=False)
    return a.obs["leiden"].values.astype(int)


def groups_from_call(adata, X, call, labels, resolution=0.5, min_size=5):
    """Leiden-cluster the called cells within each label; every subcluster of at least min_size is a group."""
    obs_names = adata.obs_names.values
    call = np.asarray(call)
    groups = []
    for lab in pd.unique(labels):
        idx = np.where((labels == lab) & call)[0]
        if len(idx) < min_size:
            continue
        cl = leiden_labels(X[idx], resolution)
        for c in np.unique(cl):
            sel = idx[cl == c]
            if len(sel) >= min_size:
                groups.append(list(obs_names[sel]))
    return groups


def leiden_sub(adata):
    """One resolution per label (best mean silhouette on the grid); subclusters under MIN_SHARE of the label score 1 - share."""
    X, _ = preprocess_pca(adata)
    obs_names = adata.obs_names.values
    labels = adata.obs["label"].astype(str).values
    score = pd.Series(0.0, index=obs_names)
    call = pd.Series(False, index=obs_names)
    groups = []
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < MIN_LABEL_N:
            continue
        Xs = X[idx]
        best_sil, best_cl = -2.0, None
        for res in RES_GRID:
            cl = leiden_labels(Xs, res)
            if len(set(cl)) < 2:
                continue
            sil = silhouette_score(Xs, cl, sample_size=min(2000, len(cl)), random_state=RNG)
            if sil > best_sil:
                best_sil, best_cl = sil, cl
        if best_cl is None:
            continue
        for c, cnt in pd.Series(best_cl).value_counts().items():
            share = cnt / len(idx)
            if share < MIN_SHARE:
                ids = obs_names[idx[best_cl == c]]
                score.loc[ids] = 1.0 - share
                call.loc[ids] = True
                groups.append(list(ids))
    return dict(score=score, call=call, groups=groups)


def iforest_pca(adata):
    """Isolation forest per label on the pool PCA; call = the forest's own outlier decision."""
    X, _ = preprocess_pca(adata)
    labels = adata.obs["label"].astype(str).values
    obs_names = adata.obs_names.values
    score = pd.Series(0.0, index=obs_names)
    call = pd.Series(False, index=obs_names)
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 10:
            continue
        clf = IsolationForest(n_estimators=N_ESTIMATORS, contamination="auto", random_state=RNG, n_jobs=N_JOBS).fit(X[idx])
        score.iloc[idx] = -clf.score_samples(X[idx])
        call.iloc[idx] = clf.predict(X[idx]) == -1
    return dict(score=score, call=call, groups=groups_from_call(adata, X, call.values, labels))


METHODS = {"leiden_sub": leiden_sub, "iforest_pca": iforest_pca}


def run(pool, table, out_root, methods):
    pid = os.path.splitext(os.path.basename(table))[0]
    todo = [m for m in methods if not os.path.exists(f"{out_root}/{m}/{pid}.parquet")]
    if not todo:
        return
    a = load_pool(pool, table, pid)
    for m in todo:
        p = METHODS[m](a.copy())
        gid = pd.Series(-1, index=a.obs_names)
        for i, g in enumerate(p["groups"]):
            gid.loc[g] = i
        os.makedirs(f"{out_root}/{m}", exist_ok=True)
        pd.DataFrame({"cell_id": a.obs_names.values, "score": p["score"].reindex(a.obs_names).values.astype(float),
                      "call": p["call"].reindex(a.obs_names).values.astype(bool), "group": gid.values}
                     ).to_parquet(f"{out_root}/{m}/{pid}.parquet", index=False)
        print(pid, m, a.n_obs, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or list(METHODS))
