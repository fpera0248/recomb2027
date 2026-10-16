"""python make_variants.py PID [PID ...] : write corrupted pilot parquets for every variant under W/<variant>/hmod/pilots (ORIGINAL pid and row order kept, so cell ids/order match the clean run).
Also W/relabel/<variant>/<pid>.parquet (obs_name, new_label, changed) and W/transfer_log.csv. Resumable (skips existing parquets). Test: python make_variants.py --selftest"""
import os, sys, shutil, hashlib, numpy as np, pandas as pd
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from w6common import *
K = 15

def knn_votes(X, y_codes, nlab, k=K, Xq=None):
    """vote matrix (n,nlab) of k nearest reference neighbours (cosine); leave-one-out if Xq is None. Returns votes, mean dist."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + (Xq is None), metric="cosine").fit(X)
    d, ix = nn.kneighbors(X if Xq is None else Xq)
    if Xq is None: d, ix = d[:, 1:], ix[:, 1:]
    V = np.zeros((len(ix), nlab)); rows = np.repeat(np.arange(len(ix)), k); np.add.at(V, (rows, y_codes[ix].ravel()), 1)
    return V, d.mean(1)

def random_relabel(labels, r, seed):
    rng = np.random.default_rng(seed); labs, inv = np.unique(labels, return_inverse=True); n = np.bincount(inv); new = labels.copy()
    for i, l in enumerate(labs):
        idx = np.where(inv == i)[0]; mv = rng.choice(idx, int(round(r * len(idx))), replace=False)
        p = n.astype(float); p[i] = 0; p /= p.sum(); new[mv] = labs[rng.choice(len(labs), len(mv), p=p)]
    return new

def boundary_relabel(labels, X, r):
    labs, inv = np.unique(labels, return_inverse=True); V, md = knn_votes(X, inv, len(labs))
    own = V[np.arange(len(V)), inv] / K; pick = np.lexsort((-md, own))[:int(round(r * len(labels)))]; new = labels.copy()
    cen = np.stack([X[inv == i].mean(0) for i in range(len(labs))]); cn = cen / np.linalg.norm(cen, axis=1, keepdims=True)
    for j in pick:
        v = V[j].copy(); v[inv[j]] = -1
        if v.max() <= 0:
            s = cn @ (X[j] / np.linalg.norm(X[j])); s[inv[j]] = -9; v = s
        new[j] = labs[int(np.argmax(v))]
    return new

def merge_labels(labels, Xlog):
    """merge the two labels (>=50 cells) with highest centroid Pearson correlation. Xlog: cells x HVG log1p-CPM."""
    vc = pd.Series(labels).value_counts(); labs = [l for l, c in vc.items() if c >= 50]
    if len(labs) < 2: labs = [l for l, c in vc.items() if c >= 10]      # pool with a single label >= 50 cells: fall back to labels >= 10 cells
    if len(labs) < 2: return labels.copy(), (None, None, float("nan"))   # nothing to merge: MERGE = identity, flagged in the log
    C = np.corrcoef(np.stack([Xlog[labels == l].mean(0) for l in labs])); np.fill_diagonal(C, -2); i, j = np.unravel_index(C.argmax(), C.shape)
    new = np.where(np.isin(labels, [labs[i], labs[j]]), f"{labs[i]}+{labs[j]}", labels); return new, (labs[i], labs[j], float(C[i, j]))

def _load(pid):
    sys.path.insert(0, f"{HOME}/work/jiaqi1011/code/w1"); import harness  # sets env, sys.path to locked code
    import loader, methods as M
    os.environ["SLIM_DIR"] = f"{CLEAN}/slim_atlas"
    a, tt = loader.load_pilot(pid); return a, M

def make(pid, force=False):
    sys.path.insert(0, f"{HOME}/work/jiaqi1011/code/w1"); import harness, hlib
    df = pd.read_parquet(f"{CLEAN}/hmod/pilots/{pid}.parquet")
    si, oid = hlib._opaque_order(pid, df.obs_name.values)
    todo = [v for v in VARIANTS if force or not os.path.exists(f"{root(v)}/hmod/pilots/{pid}.parquet")]
    if not todo: return
    a, M = _load(pid); lab = df.label.astype(str).values[si]; assert (a.obs.label.values == lab).all()
    X, hvg = M.preprocess_pca(a); log = []; res = {}
    res["ID"] = lab.copy()
    for r, f in RATES.items():
        for s in (0, 1): res[f"R{r}s{s}"] = random_relabel(lab, f, int(hashlib.sha1(f"{pid}{r}{s}".encode()).hexdigest()[:8], 16))
        res[f"B{r}"] = boundary_relabel(lab, X, f)
    import scanpy as sc
    b = a[:, hvg].copy(); sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b); Xl = b.X.toarray()
    res["MERGE"], pair = merge_labels(lab, Xl); log.append((pid, "MERGE", np.nan, (res["MERGE"] != lab).mean(), "+".join(map(str, pair[:2])) + f" r={pair[2]:.3f}"))
    # cross-pool transfer: reference = next pool of the same atlas (donor-disjoint), HVG/scale/PCA fit on reference
    P = pools(); sib = P[(P.atlas == P[P.pilot_id == pid].atlas.iloc[0])].pilot_id.tolist(); ref = sib[(sib.index(pid) + 1) % len(sib)]
    ra, _ = _load(ref); assert (ra.var_names == a.var_names).all()
    rh = M._hvg_genes(ra); rb = ra[:, rh].copy(); sc.pp.normalize_total(rb, target_sum=1e4); sc.pp.log1p(rb); Rx = rb.X.toarray(); mu, sd = Rx.mean(0), Rx.std(0) + 1e-8
    from sklearn.decomposition import PCA
    pca = PCA(50, random_state=0).fit(np.clip((Rx - mu) / sd, None, 10)); q = a[:, rh].copy(); sc.pp.normalize_total(q, target_sum=1e4); sc.pp.log1p(q)
    Z = pca.transform(np.clip((q.X.toarray() - mu) / sd, None, 10)); Zr = pca.transform(np.clip((Rx - mu) / sd, None, 10))
    rl, rinv = np.unique(ra.obs.label.astype(str).values, return_inverse=True); V, _ = knn_votes(Zr, rinv, len(rl), Xq=Z)
    res["KNNX"] = rl[V.argmax(1)]; res["KNNXu"] = np.where(V.max(1) / K >= 0.6, res["KNNX"], "unassigned")
    for v in ("KNNX", "KNNXu"): log.append((pid, v, (res[v] == lab).mean(), (res[v] != lab).mean(), f"ref={ref}"))
    for pk, name in ((0.10, "KNNT10"), (0.02, "KNNT2")):
        rng = np.random.default_rng(int(hashlib.sha1(f"{pid}{name}".encode()).hexdigest()[:8], 16)); t = a.copy(); t.X = t.X.tocsr().copy()
        t.X.data = rng.binomial(t.X.data.astype(np.int64), pk).astype(np.float32); t.X.eliminate_zeros()
        Xt, _ = M.preprocess_pca(t); labs, inv = np.unique(lab, return_inverse=True); V, _ = knn_votes(Xt, inv, len(labs)); res[name] = labs[V.argmax(1)]
        log.append((pid, name, (res[name] == lab).mean(), (res[name] != lab).mean(), ""))
    for v in VARIANTS:
        if v not in todo: continue
        new = np.empty(len(df), dtype=object); new[si] = res[v]; d2 = df.copy(); d2["label"] = new
        os.makedirs(f"{root(v)}/hmod/pilots", exist_ok=True); d2.to_parquet(f"{root(v)}/hmod/pilots/{pid}.parquet", index=False)
        os.makedirs(f"{W}/relabel/{v}", exist_ok=True)
        pd.DataFrame({"obs_name": df.obs_name.values, "new_label": new, "changed": new != df.label.astype(str).values}).to_parquet(f"{W}/relabel/{v}/{pid}.parquet", index=False)
        if v.startswith(("R", "B")) and v != "ID": assert abs((res[v] != lab).mean() - int(v[1:3]) / 100) < 0.005, (pid, v, (res[v] != lab).mean())   # planted fraction check
        if v.startswith(("R", "B")) and v != "ID": log.append((pid, v, np.nan, (res[v] != lab).mean(), ""))
    pd.DataFrame(log, columns=["pid", "variant", "acc", "frac_changed", "note"]).to_csv(f"{W}/transfer_log_{pid}.csv", index=False)   # per-pid file: no concurrent appends from xargs -P

def setup_roots():
    for v in VARIANTS:
        r = root(v); os.makedirs(f"{r}/hmod/pilots", exist_ok=True)
        for f in ("hlib.py", "methods.py", "rawutil.py", "rprec.py"): shutil.copy(f"{CLEAN}/hmod/{f}", f"{r}/hmod/{f}") if os.path.exists(f"{CLEAN}/hmod/{f}") else shutil.copy(f"{CODE_R5}/hmod/{f}", f"{r}/hmod/{f}")
        if not os.path.lexists(f"{r}/slim_atlas"): os.symlink(f"{CLEAN}/slim_atlas", f"{r}/slim_atlas")

if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        lab = np.array(list("aaaaaaaabbbbbbbbcc") * 20); n = random_relabel(lab, .1, 0); assert 0.08 < (n != lab).mean() < 0.12
        assert (random_relabel(lab, .1, 0) == n).all(); rng = np.random.default_rng(0); Xs = rng.normal(size=(len(lab), 5)) + (lab[:, None] == "b") * 4
        assert (boundary_relabel(lab, Xs, .1) != lab).sum() == 36; print("ok"); sys.exit()
    setup_roots()
    for pid in sys.argv[1:]: make(pid); print("made", pid, flush=True)
