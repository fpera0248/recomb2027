"""m10 v2 split-island blind hidden-rare detector (2026-10-06, see PREREG_v2.md and DECISIONS.md).
adata (raw integer counts, obs label/donor) -> dict(score, call, groups, ablations); same contract as methods.py.
Per thinning t (N_THIN of them, seeds RNG+t):
 1. Binomial thinning of the full counts into halves A, B (eps 0.5). HVGs chosen on A ONLY.
 2. Per visible label ("parent", n >= MIN_PARENT): PCA per half on those HVGs, kNN graphs, Leiden islands on A
    (RES_GRID), size in [MIN_ISLAND, MAX_SHARE*n], deduped. Donor filter drops donor-concentrated islands.
 3. Island statistic S = d' of island vs rest-of-parent on half B along the island-rest centroid axis (non-saturating).
    Ablation A-only: the same d' on half A (no replication). Cohesion Tn (kNN, ceiling-corrected) feeds the m1-rerank ablation.
 4. Cell score = max S over surviving islands containing it (-1 if none) + 1e-3 * isolation rank (tie-break only).
Final score = mean over thinnings. call = score >= TAU (one global constant, calibrated on held-out D1null pilots, never per pilot).
Removed vs v1: the simulated one-population null (it could not absorb native subtypes), full-count HVGs, saturating T.
No truth is read; nothing is tuned per pilot."""
import os, sys
sys.path.insert(0, os.environ.get("SI_H") or os.path.join(os.environ.get("R5") or os.path.expanduser("~/work/r5/data"), "hmod"))
import numpy as np, pandas as pd, scipy.sparse as sp, igraph as ig, random
import anndata as ad
from sklearn.neighbors import NearestNeighbors
import methods as M

RNG, K, N_THIN = 0, 15, 5
MIN_PARENT, MIN_ISLAND, MAX_SHARE = 50, 10, 0.10
RES_GRID = tuple(M.RES_GRID) + (2.0, 4.0)
MIN_EFF_DONORS, DON_MARGIN = 2.0, 0.20     # donor filter (skipped when the parent itself has < MIN_EFF_DONORS)
NONE_SCORE = -1.0
TAU = 4.0   # FROZEN 2026-10-06: smallest grid value (0.5..6 step .25) with median calls <= 40 and any_group <= 0.6 on the 60 calibration D1null pilots (results/calibration.txt)


def _knn(Z, k=K):
    k = min(k, len(Z) - 1)
    return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]


def _carve(nbA, n):
    edges = list({(min(i, j), max(i, j)) for i in range(n) for j in nbA[i]})
    g = ig.Graph(n=n, edges=edges); seen, out = set(), []
    for res in RES_GRID:
        ig.set_random_number_generator(random.Random(RNG))
        mem = np.array(g.community_leiden("modularity", resolution=res, n_iterations=2).membership)
        for c in np.unique(mem):
            idx = np.where(mem == c)[0]
            k = frozenset(idx.tolist())
            if MIN_ISLAND <= len(idx) <= MAX_SHARE * n and k not in seen:
                seen.add(k); out.append(idx)
    return out


def _donor_ok(idx, donors, par_eff, par_top):
    if par_eff < MIN_EFF_DONORS:
        return True
    p = pd.Series(donors[idx]).value_counts(normalize=True).values
    return (1 / (p ** 2).sum()) >= MIN_EFF_DONORS and p.max() <= par_top + DON_MARGIN


def _dprime(Z, idx):
    m = np.zeros(len(Z), bool); m[idx] = True
    u = Z[m].mean(0) - Z[~m].mean(0); nu = np.linalg.norm(u)
    if nu == 0: return 0.0
    u /= nu; pi, po = Z[m] @ u, Z[~m] @ u
    sd = np.sqrt(0.5 * (pi.var() + po.var())) + 1e-9
    return float(nu / sd)


def _cohesion(idx, nbB, n):
    """kNN cohesion on B, chance- and ceiling-corrected: (c - base) / (cmax - base), cmax = min(K, m-1)/K."""
    m = len(idx); inside = np.zeros(n, bool); inside[idx] = True
    c = inside[nbB[idx]].mean(); base = (m - 1) / (n - 1); cmax = min(nbB.shape[1], m - 1) / nbB.shape[1]
    return float((c - base) / max(cmax - base, 1e-9))


def _pca(C, hvg):
    return M.preprocess_pca(ad.AnnData(C, var=pd.DataFrame(index=hvg)), hvg=list(hvg))[0]


def m10_split_island(adata, n_thin=N_THIN, ablations=True):
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
    X = X.astype(np.float32)
    assert (X.data >= 0).all() and np.allclose(X.data, np.round(X.data)), "raw integer counts required"
    labels = adata.obs["label"].astype(str).values; donors = adata.obs["donor"].astype(str).values
    obs = adata.obs_names.values; n = len(obs); var = adata.var_names.values
    parents = [l for l in pd.unique(labels) if (labels == l).sum() >= MIN_PARENT]
    pidx = {l: np.where(labels == l)[0] for l in parents}
    pdon = {}
    for l, pi in pidx.items():
        p = pd.Series(donors[pi]).value_counts(normalize=True).values; pdon[l] = (1 / (p ** 2).sum(), p.max())
    S = np.full((n_thin, n), NONE_SCORE); SA = S.copy(); TB = np.zeros(n); grp = np.full((n_thin, n), -1)
    m1g = M.m1_leiden_sub(adata)["groups"] if ablations else []
    pos = pd.Series(np.arange(n), index=obs); m1S = np.full(n, NONE_SCORE)
    for t in range(n_thin):
        rs = np.random.RandomState(RNG + t)
        A = X.copy(); A.data = rs.binomial(X.data.astype(np.int64), 0.5).astype(np.float32); A.eliminate_zeros()
        B = (X - A).tocsr(); B.eliminate_zeros()
        hvg = M._hvg_genes(ad.AnnData(A, var=pd.DataFrame(index=var)))          # HVGs on half A only
        hix = pd.Index(var).get_indexer(hvg); gid = 0
        for l in parents:
            pi = pidx[l]
            ZA, ZB = _pca(A[pi][:, hix].toarray(), hvg), _pca(B[pi][:, hix].toarray(), hvg)
            nbA, nbB = _knn(ZA), _knn(ZB); np_ = len(pi)
            if t == 0:
                dc = np.linalg.norm(ZA - np.median(ZA, 0), axis=1)
                dk = NearestNeighbors(n_neighbors=K + 1).fit(ZA).kneighbors(ZA)[0][:, -1]
                TB[pi] = pd.Series(np.log(dc / dk + 1e-9)).rank(pct=True).values
                for g in m1g:                                                     # ablation: m1 clusters, B-cohesion
                    gi = pos[g].values
                    if labels[gi[0]] == l:
                        m1S[gi] = _cohesion(np.searchsorted(pi, gi), nbB, np_)
            for idx in _carve(nbA, np_):
                if not _donor_ok(idx, donors[pi], *pdon[l]):
                    continue
                s, sa = _dprime(ZB, idx), _dprime(ZA, idx); gl = pi[idx]
                upd = s > S[t, gl]; S[t, gl[upd]] = s; grp[t, gl[upd]] = gid
                SA[t, gl] = np.maximum(SA[t, gl], sa); gid += 1
    sc_ = S.mean(0); fin = lambda v: pd.Series(v + 1e-3 * TB, index=obs, dtype=float)
    score = fin(sc_)
    call = pd.Series(False if TAU is None else (sc_ >= TAU), index=obs)
    best = S.argmax(0); g_of = grp[best, np.arange(n)]
    gids = pd.Series(np.where(call.values, best * 100000 + g_of, -1), index=obs)
    groups = [list(obs[(gids == k).values]) for k in gids[gids >= 0].unique()]
    return dict(score=score, call=call, groups=groups, abl_aonly=fin(SA.mean(0)), abl_m1rerank=fin(m1S))
