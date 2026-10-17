"""python embedding_qa.py SPACE [pid ...] : blind embedding QA -> $W2_OUT/qa.csv. 15-NN agreement on the VISIBLE broad labels (an input, not hidden truth):
mean over cells of the share of the 15 nearest neighbours (cosine on E; Euclidean on PCA-50) that carry the cell's own label; PCA-50 of the same pool is the reference.
Also: row-order check against the loader, finiteness, per-dimension SD range (a collapsed or constant embedding shows up here)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import embedding_lib as W; W.env()
import numpy as np, pandas as pd, loader, methods as M
from sklearn.neighbors import NearestNeighbors


def agree(Z, lab, metric):
    nb = NearestNeighbors(n_neighbors=16, metric=metric).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]; return float((lab[nb] == lab[:, None]).mean())


f = f"{W.OUT}/qa.csv"; have = set(map(tuple, pd.read_csv(f)[["space", "pid"]].values)) if os.path.exists(f) else set()
for pid in (sys.argv[2:] or W.pools()):
    if (sys.argv[1], pid) in have: continue
    E, ids = W.load_emb(sys.argv[1], pid); a, _ = loader.load_pilot(pid); assert list(a.obs_names) == ids, "row misalignment"
    lab = np.asarray(a.obs["label"].astype(str).tolist()); Z = M.preprocess_pca(a)[0]; sd = E.std(0)    # not .values: pandas 3 gives an Arrow string array that cannot be fancy-indexed
    r = dict(space=sys.argv[1], pid=pid, d=E.shape[1], agree_emb=agree(E, lab, "cosine" if sys.argv[1] != "scvi" else "euclidean"), agree_pca50=agree(Z, lab, "euclidean"),
             sd_min=float(sd.min()), sd_max=float(sd.max()))
    f = f"{W.OUT}/qa.csv"; pd.DataFrame([r]).to_csv(f, mode="a", header=not os.path.exists(f), index=False); print(r, flush=True)
