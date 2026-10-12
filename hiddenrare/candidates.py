"""Candidate groups: Leiden subclusters of every visible label, at many resolutions, in two spaces.

For each label with at least MIN_PARENT cells we cluster (g) in the PCA of the whole pool and
(l) in a PCA computed on the label's own cells. Every cluster with at least MIN_GROUP cells and
under MAX_SHARE of the label is kept (identical sets are merged) with its geometry features.

    python -m hiddenrare.candidates POOL.h5ad PILOTS.parquet OUT_DIR
writes OUT_DIR/<pool>.groups.parquet and <pool>.cells.parquet
"""
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from .io import load_pool

RNG = 0
N_HVG, N_PCS, N_NEIGHBORS = 2000, 50, 15
K = 15                                           # neighbours used for cohesion and the isolation rank
RESOLUTIONS = (0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 8.0)
MIN_PARENT, MIN_GROUP, MAX_SHARE = 50, 10, 0.10


def hvg_genes(adata, n_hvg=N_HVG):
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_hvg, a.n_vars - 1))
    return a.var_names[a.var.highly_variable].tolist()


def preprocess_pca(adata, n_pcs=N_PCS, hvg=None):
    """log1p CP10K, 2000 HVGs, scale, PCA. Returns the embedding and the HVG list."""
    hvg = hvg or hvg_genes(adata)
    a = adata[:, hvg].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    sc.pp.pca(a, n_comps=min(n_pcs, a.n_obs - 1, len(hvg) - 1), svd_solver="arpack", random_state=RNG)
    return a.obsm["X_pca"], hvg


def leiden_multi(Z, resolutions):
    a = sc.AnnData(np.zeros((len(Z), 1), np.float32))
    a.obsm["X_rep"] = Z
    sc.pp.neighbors(a, n_neighbors=min(N_NEIGHBORS, len(Z) - 1), use_rep="X_rep", random_state=RNG)
    out = []
    for r in resolutions:
        sc.tl.leiden(a, resolution=r, random_state=RNG, flavor="igraph", n_iterations=2, directed=False, key_added="l")
        out.append(a.obs["l"].values.astype(int))
    return out


def dprime(Z, idx):
    """Separation of the group from the rest along the axis between their centroids."""
    m = np.zeros(len(Z), bool)
    m[idx] = True
    u = Z[m].mean(0) - Z[~m].mean(0)
    nu = np.linalg.norm(u)
    if nu == 0:
        return 0.0
    u /= nu
    p_in, p_out = Z[m] @ u, Z[~m] @ u
    return float(nu / (np.sqrt(0.5 * (p_in.var() + p_out.var())) + 1e-9))


def candidate_groups(a):
    """-> (groups table, per-cell isolation rank). Members are positions in a."""
    labels = a.obs["label"].astype(str).values
    n = a.n_obs
    donor = a.obs["donor"].astype(str).values if "donor" in a.obs else np.array(["d"] * n)
    Zg = preprocess_pca(a)[0]
    rows, isolation = [], np.zeros(n)
    for lab in pd.unique(labels):
        pi = np.where(labels == lab)[0]
        if len(pi) < MIN_PARENT:
            continue
        for space in "gl":
            Zl = Zg[pi] if space == "g" else preprocess_pca(a[pi].copy())[0]
            dist, nb = NearestNeighbors(n_neighbors=K + 1).fit(Zl).kneighbors(Zl)
            nb = nb[:, 1:]
            if space == "g":
                dc = np.linalg.norm(Zl - np.median(Zl, 0), axis=1)
                isolation[pi] = pd.Series(np.log(dc / dist[:, -1] + 1e-9)).rank(pct=True).values
            seen = set()
            for res, cl in zip(RESOLUTIONS, leiden_multi(Zl, RESOLUTIONS)):
                for c, cnt in pd.Series(cl).value_counts().items():
                    if cnt < MIN_GROUP or cnt / len(pi) >= MAX_SHARE:
                        continue
                    gi = np.where(cl == c)[0]
                    key = gi.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    inside = np.zeros(len(pi), bool)
                    inside[gi] = True
                    dv = pd.Series(donor[pi[gi]]).value_counts(normalize=True)
                    rows.append(dict(space=space, label=lab, res=res, size=int(cnt), share=cnt / len(pi),
                                     dprime=dprime(Zl, gi), cohesion=float(inside[nb[gi]].mean()),
                                     n_donor=int(len(dv)), top_donor=float(dv.iloc[0]),
                                     parent_top_donor=float(pd.Series(donor[pi]).value_counts(normalize=True).iloc[0]),
                                     members=pi[gi].astype(np.int32)))
    return pd.DataFrame(rows), isolation


def run(pool, table, out_dir):
    pid = os.path.splitext(os.path.basename(table))[0]
    out = f"{out_dir}/{pid}.groups.parquet"
    if os.path.exists(out):
        return
    a = load_pool(pool, table, pid)
    G, iso = candidate_groups(a)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"cell_id": a.obs_names.values, "tb_g": iso}).to_parquet(f"{out_dir}/{pid}.cells.parquet", index=False)
    G.to_parquet(out, index=False)
    print(pid, a.n_obs, len(G), flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:4])
