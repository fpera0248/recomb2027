"""Dev stage 2: dump every candidate group once, so ranking rules can be compared offline without re-clustering.
Why: sr2 dev sweep showed more candidates alone buy nothing (dev mean l_multi 0.231 vs m11 0.230); the open question is how to ORDER them
(10-30-cell groups lose every top slot to larger high-d' groups; HLCA targets have median 31 cells).
Per visible label (>= MIN_PARENT cells) and space (g = global PCA-50, l = the label's own HVG + PCA-50):
  candidates = Leiden subclusters at every RES_ALL value with MIN_GROUP_SIZE <= size and share < MIN_SHARE (deduplicated per space);
  group features: size, share, d', kNN cohesion (share of members' K-NN inside the group), donor count and top-donor share;
  cell features: within-label isolation rank (m11's tie-break) and LOF (n_neighbors 20) in each space.
  python sr3.py PILOT_ID -> $SI/cands/<pid>.groups.parquet + <pid>.cells.parquet   (blind: truth is never loaded into the method)"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): os.environ.setdefault(v, "2")
SI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SI); R5 = os.environ.get("R5") or os.path.expanduser("~/work/r5/data")
import numpy as np, pandas as pd
import loader, split_island_v2 as S2
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sr2 import _leiden_multi, RES_ALL, M


def dump(a):
    labels = a.obs["label"].astype(str).values; n = a.n_obs
    donor = a.obs["donor"].astype(str).values if "donor" in a.obs else np.array(["d"] * n)
    Zg = M.preprocess_pca(a)[0]; G = []; C = {f"{k}_{s}": np.zeros(n) for k in ("tb", "lof") for s in "gl"}
    for l in pd.unique(labels):
        pi = np.where(labels == l)[0]
        if len(pi) < S2.MIN_PARENT: continue
        for s in "gl":
            Zl = Zg[pi] if s == "g" else M.preprocess_pca(a[pi].copy())[0]
            dist, nb = NearestNeighbors(n_neighbors=S2.K + 1).fit(Zl).kneighbors(Zl); nb = nb[:, 1:]
            dc = np.linalg.norm(Zl - np.median(Zl, 0), axis=1)
            C[f"tb_{s}"][pi] = pd.Series(np.log(dc / dist[:, -1] + 1e-9)).rank(pct=True).values
            C[f"lof_{s}"][pi] = -LocalOutlierFactor(n_neighbors=min(20, len(pi) - 1)).fit(Zl).negative_outlier_factor_
            seen = set()
            for r, cl in zip(RES_ALL, _leiden_multi(Zl, RES_ALL)):
                for c, cnt in pd.Series(cl).value_counts().items():
                    if cnt < M.MIN_GROUP_SIZE or cnt / len(pi) >= M.MIN_SHARE: continue
                    gi = np.where(cl == c)[0]; key = gi.tobytes()
                    if key in seen: continue
                    seen.add(key); ins = np.zeros(len(pi), bool); ins[gi] = True
                    dv = pd.Series(donor[pi[gi]]).value_counts(normalize=True)
                    G.append(dict(space=s, label=l, res=r, size=int(cnt), share=cnt / len(pi), dprime=S2._dprime(Zl, gi),
                                  cohesion=float(ins[nb[gi]].mean()), n_donor=int(len(dv)), top_donor=float(dv.iloc[0]),
                                  parent_top_donor=float(pd.Series(donor[pi]).value_counts(normalize=True).iloc[0]),
                                  members=pi[gi].astype(np.int32)))
    return pd.DataFrame(G), pd.DataFrame(C)


def run(pid):
    d = os.environ.get("SI_CANDS", f"{R5}/cands"); out = f"{d}/{pid}.groups.parquet"
    if os.path.exists(out): return
    t0 = time.time(); a, _true = loader.load_pilot(pid)          # truth is never passed to the method
    G, C = dump(a); C.insert(0, "cell_id", a.obs_names.values)
    os.makedirs(d, exist_ok=True); C.to_parquet(f"{d}/{pid}.cells.parquet", index=False)
    G.to_parquet(out + ".tmp", index=False); os.rename(out + ".tmp", out)
    print(pid, a.n_obs, round(time.time() - t0, 1), len(G), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
