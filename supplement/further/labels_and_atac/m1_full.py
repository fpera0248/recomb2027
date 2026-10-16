"""python m1_full.py PID (env R5/SI_H/SLIM_DIR = variant root). Same logic as methods.m1_leiden_sub (locked, imported pieces) but also saves the FULL silhouette-chosen partition
(preds_base/m1_full) and a kNN-disagreement score (share of 15 global-PCA neighbours with another visible label). Writes preds_base/m1_leiden_sub in the locked format.
One-line diff vs locked code: extra outputs only; scoring path is the same calls in the same order."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.environ["SI_H"]); sys.path.insert(0, os.path.expanduser("~/work/r5/code"))
import numpy as np, pandas as pd, loader, methods as M
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
pid = sys.argv[1]; R5 = os.environ["R5"]; a, _ = loader.load_pilot(pid)
X, _h = M.preprocess_pca(a); ids = a.obs_names.values; labels = a.obs["label"].astype(str).values
score = pd.Series(0.0, index=ids); call = pd.Series(False, index=ids); gid = pd.Series(-1, index=ids); cluster = pd.Series([f"{l}|0" for l in labels], index=ids); ng = 0
for lab in pd.unique(labels):
    idx = np.where(labels == lab)[0]
    if len(idx) < M.MIN_LABEL_N: continue
    Xs = X[idx]; best_sil, best_cl = -2.0, None
    for res in M.RES_GRID:
        cl = M._leiden_labels(Xs, res)
        if len(set(cl)) < 2: continue
        sil = silhouette_score(Xs, cl, sample_size=min(2000, len(cl)), random_state=M.RNG)
        if sil > best_sil: best_sil, best_cl = sil, cl
    if best_cl is None: continue
    cluster.iloc[idx] = [f"{lab}|{c}" for c in best_cl]
    for c, cnt in pd.Series(best_cl).value_counts().items():
        if cnt / len(idx) < M.MIN_SHARE:
            sel = ids[idx[best_cl == c]]; score.loc[sel] = 1.0 - cnt / len(idx); call.loc[sel] = True; gid.loc[sel] = ng; ng += 1
d = f"{R5}/preds_base/m1_leiden_sub"; os.makedirs(d, exist_ok=True)
pd.DataFrame({"cell_id": ids, "score": score.values, "call": call.values, "group": gid.values}).to_parquet(f"{d}/{pid}.parquet.tmp", index=False); os.rename(f"{d}/{pid}.parquet.tmp", f"{d}/{pid}.parquet")
_, ix = NearestNeighbors(n_neighbors=16).fit(X).kneighbors(X); lc = pd.factorize(labels)[0]
d = f"{R5}/preds_base/m1_full"; os.makedirs(d, exist_ok=True)
pd.DataFrame({"cell_id": ids, "cluster": cluster.values, "knn_dis": (lc[ix[:, 1:]] != lc[:, None]).mean(1)}).to_parquet(f"{d}/{pid}.parquet.tmp", index=False); os.rename(f"{d}/{pid}.parquet.tmp", f"{d}/{pid}.parquet")
print(pid, "m1_full done", flush=True)
