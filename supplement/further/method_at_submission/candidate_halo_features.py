"""Halo-contrast marker features for every sr3 candidate group (dev; motivated by the HLCA diagnosis: 60% of the groups outranking small
HLCA targets were pure FRAGMENTS of abundant subtypes, which rest-of-label markers (sr3m) cannot tell from a whole rare type).
halo = the `size` cells of the same label, outside the group, nearest the group's centroid in global PCA-50. A fragment shares its markers
with its halo (the rest of its own population); a whole rare type does not. Blind: counts + visible labels only.
  python sr3h.py PILOT_ID -> $SI/cands/<pid>.halo.parquet (row-aligned with <pid>.groups.parquet)"""
import os, sys, time
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): os.environ.setdefault(v, "2")
SI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SI); R5 = os.environ.get("R5") or HR_WORK + "/r5/data"
import numpy as np, pandas as pd, scipy.sparse as sp


def halos(Z, labels, G):
    """member and halo index arrays per group; halo = nearest same-label non-members to the group centroid (as many as members)."""
    lab_ix = {l: np.where(labels == l)[0] for l in pd.unique(labels)}; H, ratio = [], []
    for l, m in zip(G.label.values, G.members.values):
        pi = lab_ix[l]; out = pi[~np.isin(pi, m)]; c = Z[m].mean(0)
        d = np.linalg.norm(Z[out] - c, axis=1); k = min(len(m), len(out)); j = np.argpartition(d, k - 1)[:k]
        H.append(out[j]); ratio.append(np.median(d[j]) / (np.median(np.linalg.norm(Z[m] - c, axis=1)) + 1e-9))
    return H, np.array(ratio)


def features(X, labels, G, H, ratio):
    n = X.shape[0]; lab_sum = {l: np.asarray(X[labels == l].sum(0)).ravel() for l in pd.unique(labels)}
    ind = lambda sets: sp.csr_matrix((np.ones(sum(map(len, sets)), np.float32), (np.repeat(np.arange(len(sets)), list(map(len, sets))), np.concatenate(sets))),
                                     shape=(len(sets), n))
    S = (ind(list(G.members.values)) @ X).toarray(); Sh = (ind(H) @ X).toarray(); out = []
    for j, (l, m, h) in enumerate(zip(G.label.values, G["size"].values, map(len, H))):
        pin = S[j] / m; ph = Sh[j] / max(h, 1); pout = (lab_sum[l] - S[j]) / max((labels == l).sum() - m, 1)
        dh = np.sort(pin - ph)[::-1]; top = np.argsort(pin - pout)[::-1][:5]
        out.append(dict(hl_top5=dh[:5].mean(), hl_top20=dh[:20].mean(), hl_n25=int((dh >= 0.25).sum()), hl_mk5=(pin - ph)[top].mean(),
                        hl_neg5=-dh[::-1][:5].mean(), hl_ratio=ratio[j]))
    return pd.DataFrame(out)


def run(pid):
    import loader; from sr2 import M
    d = os.environ.get("SI_CANDS", f"{R5}/cands"); out = f"{d}/{pid}.halo.parquet"
    if os.path.exists(out): return
    t0 = time.time(); a, _true = loader.load_pilot(pid)          # truth is never passed to the method
    G = pd.read_parquet(f"{d}/{pid}.groups.parquet", columns=["label", "size", "members"]); labels = a.obs["label"].astype(str).values
    X = sp.csr_matrix(a.X, copy=True); X.data = (X.data > 0).astype(np.float32); X.eliminate_zeros(); X = X[:, np.asarray(X.mean(0)).ravel() >= 0.01].tocsr()
    H, ratio = halos(M.preprocess_pca(a)[0], labels, G); F = features(X, labels, G, H, ratio)
    F.to_parquet(out + ".tmp", index=False); os.rename(out + ".tmp", out); print(pid, len(G), round(time.time() - t0, 1), flush=True)


def demo():
    """fragment (cells 0-9 of a 40-cell population sharing genes 0-4) vs a whole distinct type (cells 40-49, genes 5-9): halo keeps only the latter."""
    rng = np.random.RandomState(0); X = np.zeros((100, 20), np.float32); X[:40, :5] = 1; X[40:50, 5:10] = 1; X[:, 10:] = rng.rand(100, 10) < 0.5
    Z = np.r_[rng.randn(40, 2), rng.randn(10, 2) + [6, 0], rng.randn(50, 2) + [0, 6]]; labels = np.array(["L"] * 100)
    G = pd.DataFrame({"label": ["L", "L"], "size": [10, 10], "members": [np.arange(10), np.arange(40, 50)]})
    H, r = halos(Z, labels, G); F = features(sp.csr_matrix(X), labels, G, H, r)
    assert F.hl_mk5[0] < 0.2 < 0.8 < F.hl_mk5[1] and r[1] > r[0], F


if __name__ == "__main__":
    demo(); run(sys.argv[1]) if len(sys.argv) > 1 else print("demo ok")
