"""One pool, one seed, one CSV row per setting (see DESIGN.md).
  python -m combination.run_one POOL SEED [--clf lr|rf|mlp] [--out CSV]
  python -m combination.run_one --plan     donor split and eligibility of every pool and seed, from metadata only
HR_DATA holds pilots/, manifests/, pools.csv and predictions/ (the paper-tier assets); HR_POOLS holds the pool h5ad files (default HR_DATA/pools).
Pseudo-labels use only the method's own calls on training-donor cells; true labels enter the true_* settings, the scores and the purity diagnostic."""
class IdMismatch(Exception): pass
import os, sys, time, glob, hashlib, argparse, resource, warnings
ROOT = os.environ.get("HR_DATA", "assets"); PILOTS_DIR = f"{ROOT}/pilots"; POOLS = os.environ.get("HR_POOLS") or f"{ROOT}/pools"
HERE = os.path.dirname(os.path.abspath(__file__)); PREDS = os.environ.get("COMBO_PREDS") or f"{ROOT}/predictions"
sys.dont_write_bytecode = True; sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
from hiddenrare.io import opaque_order, load_pool, rprec
from eval.targets import targets
from eval.score_atl import in_top
TARGET = 256     # each pseudo or hidden class is topped up to this many cells
CLFS = ("lr", "rf", "mlp")
MINC = 5          # a class needs this many training cells, else its cells keep the broad label
MIN_TEST_U = 20     # eligibility: true hidden cells in the test donors
MIN_TRAIN_U = 20    # and in the training donors, in classes of at least MINC cells
NHVG, NPC, SEEDS = 2000, 50, (0, 1, 2)
SETTINGS = ["broad", "leiden_dup", "leiden_nodup", "leiden_collapsed_dup", "ours_dup", "ours_nodup", "true_dup", "true_nodup"]
PAIR = {"leiden_nodup": "leiden_dup", "ours_nodup": "ours_dup", "true_nodup": "true_dup"}


def split(pid, donors, seed):
    """Donor-disjoint, deterministic. Donors sorted by sha1(pool|donor); the test block is max(1, round(D/3)) consecutive donors from (seed * n_test) mod D.
    None if D < 2, or D == 2 with seed >= 2 (it would repeat the seed 0 split)."""
    d = sorted(set(donors), key=lambda x: hashlib.sha1(f"{pid}|{x}".encode()).hexdigest()); D = len(d)
    if D < 2 or (D == 2 and seed >= 2): return None
    nt = max(1, int(round(D / 3))); s = (seed * nt) % D; te = {d[(s + i) % D] for i in range(nt)}
    return [x for x in d if x not in te], sorted(te)


def load_meta(pid):
    p = pd.read_parquet(f"{PILOTS_DIR}/{pid}.parquet", columns=["obs_name", "label", "true_type", "donor"]).astype({"label": str, "true_type": str, "donor": str})
    si, oid = opaque_order(pid, p.obs_name.values); return p.iloc[si].set_index(oid)


def true_classes(df, tg):
    """hidden-type key per cell (label||true type), empty if none."""
    k = np.full(len(df), "", object)
    for L, T, _ in tg: k[((df.label == L) & (df.true_type == T)).values] = f"{L}||{T}"
    return k


def eligibility(pid, seed, df=None):
    df = load_meta(pid) if df is None else df; sp = split(pid, df.donor.values, seed)
    if sp is None: return None, ("single donor" if df.donor.nunique() < 2 else "two donors: seed repeats the seed-0 split")
    tr = df.donor.isin(sp[0]).values; tg = targets(df); k = true_classes(df, tg); u = k != ""
    ut = int((u & ~tr).sum()); vc = pd.Series(k[u & tr]).value_counts(); ur = int(vc[vc >= MINC].sum())
    if ut < MIN_TEST_U: return None, f"test donors hold {ut} hidden cells (< {MIN_TEST_U})"
    if ur < MIN_TRAIN_U: return None, f"train donors hold {ur} hidden cells in classes >= {MINC} (< {MIN_TRAIN_U})"
    return (sp, tr, k, u), "ok"


def keep_classes(key, mask):
    """class per cell in mask, empty for classes with fewer than MINC training cells."""
    vc = pd.Series(key[mask]).value_counts(); ok = set(vc[vc >= MINC].index); out = np.full(len(key), "", object)
    for i in np.where(mask)[0]:
        if key[i] in ok: out[i] = key[i]
    return out


def pseudo_labels(df, tr, pid):
    """pseudo class per cell for each method, empty = broad label; training donors only."""
    lei = pd.read_parquet(f"{PREDS}/leiden_sub/{pid}.parquet").set_index("cell_id").reindex(df.index)
    ours = pd.read_parquet(f"{PREDS}/hiddenrare/{pid}.parquet").set_index("cell_id").reindex(df.index)
    ids = set(df.index)
    if not (len(lei) == len(df) and set(pd.read_parquet(f"{PREDS}/leiden_sub/{pid}.parquet", columns=["cell_id"]).cell_id) == ids and
            set(pd.read_parquet(f"{PREDS}/hiddenrare/{pid}.parquet", columns=["cell_id"]).cell_id) == ids) or lei.score.isna().any() or ours.score.isna().any():
        raise IdMismatch("cell ids differ")
    call = (lei.call.values.astype(bool)) & (lei.group.values >= 0) & tr; n = len(df); lab = df.label.values
    out = {}
    out["leiden_dup"] = keep_classes(np.array([f"{lab[i]}||g{g}" for i, g in enumerate(lei.group.values)], object), call)
    out["leiden_collapsed_dup"] = keep_classes(np.array([f"{l}||*" for l in lab], object), call)

    # ours has no call set: take as many top-scoring training cells as Leiden calls (stable order for ties)
    K = int(call.sum()); idx = np.where(tr)[0]; top = idx[np.argsort(-ours.score.values[idx], kind="stable")[:K]]
    om = np.zeros(n, bool); om[top] = True
    out["ours_dup"] = keep_classes(np.array([f"{l}||*" for l in lab], object), om)
    return out, dict(K_leiden_calls_train=K, ours_score_at_K=float(ours.score.values[top].min()) if K else np.nan), om, call


def features(a, tr, seed):
    """Part I features with gene filter and HVGs from training cells only: genes in >= 3 training cells, 2,000 seurat_v3 HVGs, log1p(1e4 * counts / library size), PCA-50 fitted on <= 30,000 training cells, all cells projected."""
    import scanpy as sc, scipy.sparse as sp
    from sklearn.decomposition import PCA
    X = sp.csr_matrix(a.X, dtype=np.float32); keep = np.asarray((X[tr] > 0).sum(0)).ravel() >= 3; X = X[:, keep]
    b = sc.AnnData(X[tr].copy()); sc.pp.highly_variable_genes(b, flavor="seurat_v3", n_top_genes=NHVG)
    X = X[:, np.where(b.var.highly_variable.values)[0]].toarray().astype(np.float32)
    X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4).astype(np.float32)
    itr = np.where(tr)[0]; fit = np.random.RandomState(seed).choice(itr, min(30000, len(itr)), replace=False)
    return PCA(NPC, random_state=seed).fit(X[fit]).transform(X)


def fit_predict(Z, y, tr, te, cls, seed, dup, clf="lr"):
    """Part I readouts on PCA-50. dup tops every class in cls up to TARGET cells by drawing with replacement from its training cells.
    Returns classes, test probabilities, number of added cells, n_iter."""
    rng = np.random.RandomState(seed); itr = np.where(tr)[0]; ytr = y[itr]; add = []
    if dup:
        for c in cls:
            m = itr[ytr == c]; k = TARGET - len(m)
            if k > 0: add.append(rng.choice(m, k, replace=True))
    ii = np.r_[itr, np.concatenate(add)] if add else itr
    if clf == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, random_state=seed).fit(Z[ii], y[ii]); n_it = int(np.max(m.n_iter_))
    elif clf == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=4).fit(Z[ii], y[ii]); n_it = np.nan
    else:
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(max_iter=500, random_state=seed).fit(Z[ii], y[ii]); n_it = int(m.n_iter_)
    return m.classes_, m.predict_proba(Z[te]), len(ii) - len(itr), n_it


def run(pid, seed, out, clf="lr"):
    t0 = time.time(); df = load_meta(pid); el, why = eligibility(pid, seed, df)
    if el is None:
        row = dict(pool=pid, seed=seed, clf=clf, setting="*", status="ineligible: " + why); pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False); return
    (dtr, dte), tr, tk, u = el; te = ~tr
    a = load_pool(f"{POOLS}/{pd.read_parquet(f'{PILOTS_DIR}/{pid}.parquet', columns=['source_path']).source_path.iloc[0]}", f"{PILOTS_DIR}/{pid}.parquet", pid)[df.index]; Z = features(a, tr, seed); del a
    try: ps, info, om, lcall = pseudo_labels(df, tr, pid)
    except IdMismatch:
        pd.DataFrame([dict(pool=pid, seed=seed, clf=clf, setting="*", status="dropped: prediction cell ids differ from pool ids")]).to_csv(out, mode="a", header=not os.path.exists(out), index=False); return
    lab = df.label.values
    ps["true_dup"] = keep_classes(tk, tr & u)
    ps["leiden_nodup"], ps["ours_nodup"], ps["true_nodup"], ps["broad"] = ps["leiden_dup"], ps["ours_dup"], ps["true_dup"], np.full(len(df), "", object)
    U = u & te; nU = int(U.sum()); rows, store = [], {}
    for st in SETTINGS:  # broad-only has no pseudo class, so its score is 1 - max probability and is never compared as a detector
        p = ps[st]; y = np.where(p != "", p, lab).astype(object); cls = sorted(set(p[p != ""]))
        dup = st.endswith("_dup")
        classes, P, n_add, n_it = fit_predict(Z, y, tr, te, cls, seed, dup, clf); parent = {c: c.split("||")[0] for c in classes}
        isps = np.array([c in set(cls) for c in classes]); pred = classes[P.argmax(1)]
        s = P[:, isps].sum(1) if isps.any() else 1 - P.max(1)
        ut = U[te]; call = np.array([c in set(cls) for c in pred]); tp = int((call & ut).sum())
        from sklearn.metrics import average_precision_score, f1_score
        ptrue = np.array([parent[c] for c in pred]); lt = lab[te]
        store[st] = (s, ut, int(call.sum()))
        trp = np.isin(y[tr], cls); pur = float(u[tr][np.isin(y[tr], cls)].mean()) if trp.any() else np.nan
        rows.append(dict(pool=pid, seed=seed, clf=clf, setting=st, status="ok", n_added=n_add, n_iter=n_it, hit_iter_cap=bool(clf == "lr" and n_it >= 2000), n_train_donors=len(dtr), n_test_donors=len(dte), n_train=int(tr.sum()), n_test=int(te.sum()),
                         n_hidden_test=nU, n_pseudo_classes=len(cls), n_pseudo_train_cells=int(trp.sum()), pseudo_train_purity_diag=pur,
                         ap=float(average_precision_score(ut, s)), rprec=float(rprec(s, ut)), n_calls=int(call.sum()),
                         recall_argmax=tp / nU, precision_argmax=(tp / call.sum()) if call.sum() else 0.0,
                         macro_f1_broad=float(f1_score(lt, ptrue, average="macro", labels=sorted(set(lt))))))
    for r in rows:  # precision and recall of the ranking at K = argmax calls of the duplication counterpart
        s, ut, nc = store[r["setting"]]; K = store[PAIR.get(r["setting"], r["setting"])][2]; r["k_ref"] = K
        if K: tp = in_top(s, K)[ut].sum(); r["prec_at_kref"], r["rec_at_kref"] = tp / K, tp / ut.sum()
        else: r["prec_at_kref"] = r["rec_at_kref"] = np.nan
        r.update(info); r["runtime_s_pool_seed"] = round(time.time() - t0, 1); r["maxrss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


def plan():
    man = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/manifests/atlas_manifest_*.csv"))]).set_index("pilot_id")
    kept = [(p, man.atlas[p]) for p in pd.read_csv(f"{ROOT}/pools.csv").pilot_id]; rows = []
    for pid, at in kept:
        df = load_meta(pid)
        for s in SEEDS:
            el, why = eligibility(pid, s, df); rows.append(dict(pool=pid, atlas=at, seed=s, eligible=el is not None, reason=why, n_donors=df.donor.nunique()))
    r = pd.DataFrame(rows); print(r.groupby("atlas").agg(pools=("pool", "nunique"), seeds_ok=("eligible", "sum")).to_string())
    pp = r.groupby("pool").eligible.sum(); print("pools with >=1 eligible seed:", int((pp > 0).sum()), "; with 3:", int((pp == 3).sum())); return r


def selftest():
    d = ["a", "b", "c", "d"] * 3; s = [split("p", d, k) for k in SEEDS]; assert len({tuple(x[1]) for x in s}) == 3 and all(not set(x[0]) & set(x[1]) for x in s)
    assert split("p", ["a"], 0) is None and len(split("p", ["a", "b"], 0)[1]) == 1
    assert split("p", ["a", "b"], 2) is None
    k = keep_classes(np.array(["x"] * 6 + ["y"] * 2, object), np.ones(8, bool)); assert (k[:6] == "x").all() and (k[6:] == "").all()


if __name__ == "__main__":
    selftest(); ap = argparse.ArgumentParser(); ap.add_argument("pool", nargs="?"); ap.add_argument("seed", nargs="?", type=int); ap.add_argument("--out"); ap.add_argument("--clf", default="lr", choices=CLFS); ap.add_argument("--plan", action="store_true")
    g = ap.parse_args()
    if g.plan: ap.error("--plan needs --out") if not g.out else plan().to_csv(g.out, index=False)
    else: run(g.pool, g.seed, g.out or f"{HERE}/rows_{g.pool}_{g.seed}_{g.clf}.csv", g.clf)
