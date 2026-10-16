"""python fit_worker.py TAG KIND NPILOTS SEED : fit one model on a pilot subset of train_xy.npz in a fresh process; writes models/TAG.pkl + models/TAG.json (fit time, peak RSS, n_iter, size, predict time per 100k groups)."""
import os, sys, time, json, pickle, resource, numpy as np
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["OPENBLAS_NUM_THREADS"] = "4"
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
tag, kind, npil, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); OUT = os.path.expanduser("~/work/jiaqi1011/out/w1/models"); os.makedirs(OUT, exist_ok=True)
D = os.path.expanduser("~/work/jiaqi1011/out/w1/npy"); X, y, pid = np.load(D+"/X.npy", mmap_mode="r"), np.load(D+"/y.npy"), np.load(D+"/pid.npy"); u = np.unique(pid)  # npz decompressed once to .npy so peak RSS reflects only the rows used
if npil < len(u): keep = np.random.RandomState(seed).choice(u, npil, replace=False); m = np.isin(pid, keep); X, y = X[m], y[m]
else: X = np.array(X)
if kind == "gbr200k": i = np.sort(np.random.RandomState(seed).choice(len(y), 200000, replace=False)); X, y = X[i], y[i]
nrow = len(y)
M = {"hgb": lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=seed),
     "rf": lambda: RandomForestRegressor(300, min_samples_leaf=20, n_jobs=4, random_state=seed),
     "et": lambda: ExtraTreesRegressor(300, min_samples_leaf=20, n_jobs=4, random_state=seed),
     "gbr200k": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=seed),
     "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
     "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor((64, 64), early_stopping=True, max_iter=100, random_state=seed))}[kind]()
t = time.time(); M.fit(X, y); ft = time.time() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # MB (macOS ru_maxrss is bytes)
Xp = np.array(np.load(D+"/X.npy", mmap_mode="r")[:100000]); t = time.time(); M.predict(Xp); pt = time.time() - t
f = f"{OUT}/{tag}.pkl"; pickle.dump(M, open(f, "wb"))
r = dict(tag=tag, kind=kind, n_pilots=min(npil, len(u)), seed=seed, n_rows=nrow, fit_s=ft, peak_rss_mb=rss, predict_s_per_100k=pt, size_mb=os.path.getsize(f) / 1e6, n_iter=int(getattr(M, "n_iter_", -1)) if kind == "hgb" else None)
json.dump(r, open(f"{OUT}/{tag}.json", "w")); print(r)
