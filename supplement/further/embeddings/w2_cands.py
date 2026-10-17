"""python w2_cands.py SPACE VARIANT [pid ...]  (default: all phase-1 pools)
Locked pipeline (sr3 -> sr3m -> sr3h candidate groups and 41-feature inputs, plus m1 Leiden subclustering) run in representation SPACE.
SPACE: pca | pca30 | scvi | scgpt | gf | noise (planted random embedding); VARIANT: F | H | P (ignored for pca / pca30, use F).
Outputs (arm = SPACE_VARIANT): $W2_OUT/arms/<arm>/cands/<pid>.{groups,cells,markers,halo}.parquet and preds_base/m1_leiden_sub/<pid>.parquet. Resumable.
Truth is never loaded here. Diff vs locked scripts: one run loads the pool once instead of three times; sr3m/sr3h bodies copied from their run()."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import w2lib as W; W.env()
import numpy as np, pandas as pd, scipy.sparse as sp
from threadpoolctl import threadpool_limits
import numba
FEAT_T = int(os.environ.get("W2_FEAT_THREADS", 2))         # locked set-B features (sr3, sr3m, sr3h) ran with 2 threads, m1 with 3 (worker_sr3*.sh, env.sh)
M1_T = int(os.environ.get("W2_M1_THREADS", 3))


def threads(n): numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS)); return threadpool_limits(n)


def arm_name(space, variant): return space if space in ("pca", "pca30") else f"{space}_{variant}"


def run(space, variant, pid):
    arm = arm_name(space, variant); d = f"{W.arm_dir(arm)}/cands"; mdir = f"{W.arm_dir(arm)}/preds_base/m1_leiden_sub"; os.makedirs(d, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    if all(os.path.exists(f) for f in (f"{d}/{pid}.halo.parquet", f"{mdir}/{pid}.parquet")): return
    if space not in ("pca", "pca30") and not os.path.exists(W.emb_path(space, pid)): print("no embedding yet", space, pid, flush=True); return
    mod, patch, dexp = W.install(space, variant, pid); sr3, sr3m, sr3h, M, loader = mod["sr3"], mod["sr3m"], mod["sr3h"], mod["M"], mod["loader"]
    a, _t = loader.load_pilot(pid); del _t            # truth never passed
    if patch is not None and space not in ("pca30",): assert list(a.obs_names) == list(patch.ids), "embedding rows do not match loader order"
    assert not sp.issparse(a.X) or not (a.X.data == 0).any(), "explicit zeros in X: sr3m.markers (eliminate_zeros on a shared buffer) would corrupt later steps"
    tl = threads(FEAT_T); t0 = time.time(); G, C = sr3.dump(a); C.insert(0, "cell_id", a.obs_names.values); t1 = time.time()
    C.to_parquet(f"{d}/{pid}.cells.parquet", index=False); G.to_parquet(f"{d}/{pid}.groups.parquet", index=False)
    Gm = G[["label", "size", "members"]]; sr3m.markers(a, Gm).to_parquet(f"{d}/{pid}.markers.parquet", index=False)
    labels = a.obs["label"].astype(str).values
    X = sp.csr_matrix(a.X, copy=True); X.data = (X.data > 0).astype(np.float32); X.eliminate_zeros(); X = X[:, np.asarray(X.mean(0)).ravel() >= 0.01].tocsr()
    H, ratio = sr3h.halos(M.preprocess_pca(a)[0], labels, Gm); sr3h.features(X, labels, Gm, H, ratio).to_parquet(f"{d}/{pid}.halo.parquet", index=False); t2 = time.time()
    tl.unregister(); tl = threads(M1_T); p = M.m1_leiden_sub(a.copy()); al = lambda v: v.reindex(a.obs_names).values if isinstance(v, pd.Series) else np.asarray(v)
    pd.DataFrame({"cell_id": a.obs_names.values, "score": al(p["score"]).astype(float), "call": al(p["call"]).astype(bool)}).to_parquet(f"{mdir}/{pid}.parquet", index=False)
    t3 = time.time(); tl.unregister()
    if patch is not None: patch.check(dexp)
    # candidate-set diagnostic inputs are in groups.parquet; n groups and the space split are logged here
    W.log_time("feat", arm, pid, t2 - t0, groups=len(G), g=int((G.space == "g").sum()), l=int((G.space == "l").sum()), dim_g=(patch.dim_g if patch else 50))
    W.log_time("m1", arm, pid, t3 - t2)
    print(arm, pid, round(t2 - t0, 1), round(t3 - t2, 1), len(G), flush=True)
    # Known limit: files are written non-atomically per file; halo.parquet + m1 are written last and gate the skip test above


if __name__ == "__main__":
    sp_, var = sys.argv[1], sys.argv[2]; ids = sys.argv[3:] or W.pools()
    for pid in ids: run(sp_, var, pid)
