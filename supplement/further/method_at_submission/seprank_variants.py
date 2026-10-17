"""Dev variants of seprank. Developed on the already-seen D1/D2 pilots; HLCA is the held-out test.
Diagnosis of the seprank baseline: on D2 only 25% of rare cells sit in ANY Leiden-subcluster candidate group, so carving, not ranking, is the bottleneck.
Every variant: per visible label, candidate groups = Leiden subclusters with share < MAX_SHARE and >= MIN_GROUP_SIZE cells;
cell score = max d' over the candidate groups holding it (-1 if none) + 1e-3 * within-label isolation rank (same tie-break as the seprank baseline).
  g_sil   : global PCA-50, the Leiden baseline's silhouette-picked resolution   (= seprank re-implemented; sanity row)
  g_multi : global PCA-50, every resolution in RES_ALL (no silhouette pick)
  l_sil   : per-label HVG + PCA-50 (standard subclustering), silhouette-picked resolution
  l_multi : per-label HVG + PCA-50, every resolution in RES_ALL
  python sr2.py VARIANT PILOT_ID -> $SI_PREDS/<VARIANT>/<pid>.parquet   (call column left False; TAU is calibrated after the freeze)"""
import os, sys, time
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(v, "2")
SI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SI); R5 = os.environ.get("R5") or HR_WORK + "/r5/data"
import numpy as np, pandas as pd, scanpy as sc
import loader, split_island_v2 as S2
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
M = S2.M if hasattr(S2, "M") else __import__("methods")
RES_ALL = (0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 8.0)


def _leiden_multi(Z, res_list):
    a = sc.AnnData(np.zeros((len(Z), 1), np.float32)); a.obsm["X_rep"] = Z
    sc.pp.neighbors(a, n_neighbors=min(M.N_NEIGHBORS, len(Z) - 1), use_rep="X_rep", random_state=M.RNG)
    out = []
    for r in res_list:
        sc.tl.leiden(a, resolution=r, random_state=M.RNG, flavor="igraph", n_iterations=2, directed=False, key_added="l")
        out.append(a.obs["l"].values.astype(int))
    return out


def _sil_pick(Z, cls):
    best, bc = -2.0, None
    for cl in cls:
        if len(set(cl)) < 2: continue
        s = silhouette_score(Z, cl, sample_size=min(2000, len(cl)), random_state=M.RNG)
        if s > best: best, bc = s, cl
    return [] if bc is None else [bc]


def score(a, variant):
    local, multi = variant.startswith("l_"), variant.endswith("_multi")
    labels = a.obs["label"].astype(str).values; n = a.n_obs
    Zg = None if local else M.preprocess_pca(a)[0]
    best = np.full(n, -1.0); TB = np.zeros(n); ncand = 0
    for l in pd.unique(labels):
        pi = np.where(labels == l)[0]
        if len(pi) < S2.MIN_PARENT: continue
        Zl = M.preprocess_pca(a[pi].copy())[0] if local else Zg[pi]
        dc = np.linalg.norm(Zl - np.median(Zl, 0), axis=1)
        dk = NearestNeighbors(n_neighbors=S2.K + 1).fit(Zl).kneighbors(Zl)[0][:, -1]
        TB[pi] = pd.Series(np.log(dc / dk + 1e-9)).rank(pct=True).values
        cls = _leiden_multi(Zl, RES_ALL if multi else M.RES_GRID)
        if not multi: cls = _sil_pick(Zl, cls)
        seen = set()
        for cl in cls:
            for c, cnt in pd.Series(cl).value_counts().items():
                if cnt < M.MIN_GROUP_SIZE or cnt / len(pi) >= M.MIN_SHARE: continue
                gi = np.where(cl == c)[0]; key = gi.tobytes()
                if key in seen: continue
                seen.add(key); ncand += 1
                best[pi[gi]] = np.maximum(best[pi[gi]], S2._dprime(Zl, gi))
    return best, best + 1e-3 * TB, ncand


def run(variant, pid):
    d = f"{os.environ.get('SI_PREDS', R5 + '/preds_dev')}/{variant}"; out = f"{d}/{pid}.parquet"
    if os.path.exists(out): return
    t0 = time.time(); a, _true = loader.load_pilot(pid)          # truth is never passed to the method
    dp, s, ncand = score(a, variant)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"cell_id": a.obs_names.values, "score": s, "call": False, "dprime": dp}).to_parquet(out + ".tmp", index=False)
    os.rename(out + ".tmp", out)
    print(variant, pid, a.n_obs, round(time.time() - t0, 1), ncand, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
