"""Learned scorer with natural-hierarchy training targets (truth(): union of score_hlca targets on HLCA / atlas pilots) and CV grouped by atlas.
Learned candidate-group scorer. Scale-free group features -> P(group is the hidden rare type) with a gradient-boosted
classifier; cell score = max over its groups of P**g / size**a (a = 0: rank by P; a = 1: Smith's rule for an unknown matched k).
Training label (offline truth, never a method input): purity >= 0.5 and recall >= 0.3 against the pilot's hidden type.
  python learn.py cv      DEV_IDS  OUT.csv           -> out-of-fold P (GroupKFold by dataset, 5 folds), rprec per (pilot, rule)
  python learn.py fit     DEV_IDS[,IDS2]  MODEL.pkl  -> fit on every group of the listed pilots
  python learn.py loao    BENCH_IDS NAT_IDS OUT.csv  -> leave-one-atlas-out E1_union on natural pools
  python learn.py apply   MODEL.pkl IDS OUT.csv      -> rprec per (pilot, rule) on new pilots (the model never saw them)
  python learn.py predict MODEL.pkl PID OUT.parquet RULE -> cell scores for one pilot without truth (the frozen method's output)"""
import os, sys, pickle, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import rescore as R
K = 15
RULES = {"P": (1, 0.0), "P_s05": (1, 0.5), "P_s1": (1, 1.0), "P2_s1": (2, 1.0), "P4_s1": (4, 1.0)}
HL = ["hl_top5", "hl_top20", "hl_n25", "hl_mk5", "hl_neg5", "hl_ratio"]
MK = ["mk_top1", "mk_top5", "mk_top20", "mk_n40", "mk_n25", "mk_lr5", "mk_neg5"]
F = ["lsize", "lshare", "dprime", "cohesion", "coh_corr", "stab", "lres", "space_l", "ldonor", "donor_excess", "dp_pct", "coh_pct", "dp_z", "coh_z"]


def featurize(G):
    X = pd.DataFrame(index=G.index); n_l = G["size"] / G.share
    X["lsize"] = np.log(G["size"]); X["lshare"] = np.log(G.share); X["dprime"] = G.dprime; X["cohesion"] = G.cohesion
    base = (G["size"] - 1) / (n_l - 1); cmax = np.minimum(K, G["size"] - 1) / K
    X["coh_corr"] = (G.cohesion - base) / np.maximum(cmax - base, 1e-9); X["stab"] = G.stab; X["lres"] = np.log(G.res)
    X["space_l"] = (G.space == "l").astype(int); X["ldonor"] = np.log(G.n_donor); X["donor_excess"] = G.top_donor - G.parent_top_donor
    for c, s in (("dprime", "dp"), ("cohesion", "coh")):
        X[f"{s}_pct"] = G[c].rank(pct=True)
        X[f"{s}_z"] = (G[c] - G[c].median()) / (G[c].quantile(0.75) - G[c].quantile(0.25) + 1e-9)
    cols = F
    if os.environ.get("FEATS", "mk") in ("mk", "mkh"):
        for c in MK: X[c] = G[c]; X[c + "_pct"] = G[c].rank(pct=True)
        cols = F + MK + [c + "_pct" for c in MK]
    if os.environ.get("FEATS", "mk") == "mkh":
        for c in HL: X[c] = G[c]; X[c + "_pct"] = G[c].rank(pct=True)
        X["hl_keep"] = G.hl_mk5 / np.maximum(G.mk_top5, 1e-3); cols = cols + HL + [c + "_pct" for c in HL] + ["hl_keep"]
    return X[cols].values


def truth(pid, C):
    """per-cell target index (-1 = none) and target sizes. Benchmark pilots: the hidden type (is_rare). Natural pilots (HLCA / atlases,
    is_rare all 0): every (label, finer type) target of score_hlca.targets (the pre-registered rule), so the training label is their union."""
    from hlib import _opaque_order, PILOTS_DIR
    from score_hlca import targets
    df = pd.read_parquet(f"{PILOTS_DIR}/{pid}.parquet"); si, oid = _opaque_order(pid, df.obs_name.values); df = df.iloc[si].set_index(oid).reindex(C.cell_id.values)
    if df.is_rare.astype(bool).any(): t = np.where(df.is_rare.astype(bool), 0, -1); return t, np.array([(t == 0).sum()])
    df = df.astype({"label": str, "true_type": str}); t = np.full(len(df), -1); T = targets(df)
    for i, (L, TT, n) in enumerate(T): t[((df.label == L) & (df.true_type == TT)).values] = i
    return t, np.array([n for _, _, n in T], int)


def load_all(ids):
    out = []
    for pid in ids:
        if not os.path.exists(f"{R.CD}/{pid}.groups.parquet"): continue
        G = pd.read_parquet(f"{R.CD}/{pid}.groups.parquet"); C = pd.read_parquet(f"{R.CD}/{pid}.cells.parquet"); G["stab"] = R.stability(G)
        G = extra(pid, G)
        if G is None: continue
        t, nt = truth(pid, C); r = t >= 0
        if not len(nt): continue
        y = np.zeros(len(G), int); pur = np.zeros(len(G))
        for j, m in enumerate(G.members.values):
            h = np.bincount(t[m][t[m] >= 0], minlength=len(nt)); y[j] = int(((h / len(m) >= 0.5) & (h / nt >= 0.3)).any()); pur[j] = h.sum() / len(m)
        out.append((pid, G, C, r, featurize(G), pur if os.environ.get("TARGET") == "pur" else y))
    return out


def dataset_of(pid, man):
    """grouping unit for CV: benchmark pilots -> manifest dataset; HLCA -> 'hlca'; atl_<name>_p<k> -> <name>."""
    if pid.startswith("hlca_"): return "hlca"
    if pid.startswith("atl_"): return pid.split("_")[1]
    return man.loc[pid, "dataset"]


def extra(pid, G):
    """append the per-group feature files FEATS needs (None if one is missing)."""
    f = os.environ.get("FEATS", "mk"); parts = {"mk": ["markers"], "mkh": ["markers", "halo"]}.get(f, [])
    if not all(os.path.exists(f"{R.CD}/{pid}.{p}.parquet") for p in parts): return None
    return pd.concat([G] + [pd.read_parquet(f"{R.CD}/{pid}.{p}.parquet") for p in parts], axis=1)


def model():
    """TARGET=pur: regress the group's union purity (= P(a member is a target cell), what matched-k precision rewards); default: classifier."""
    kw = dict(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0)
    return HistGradientBoostingRegressor(**kw) if os.environ.get("TARGET") == "pur" else HistGradientBoostingClassifier(**kw)


def prob(m, X): return np.clip(m.predict(X), 0, 1) if isinstance(m, HistGradientBoostingRegressor) else m.predict_proba(X)[:, 1]


def score_rows(pid, G, C, r, P):
    return [(pid, name, int(r.sum()), R.rprec(R.cell_score(G, P ** g / G["size"].values ** a, C, "tb_g"), r)) for name, (g, a) in RULES.items()]


def cv(ids, out):
    D = load_all(ids); man = pd.read_csv(f"{R.R5}/manifest.csv").set_index("pilot_id"); ds = np.array([dataset_of(d[0], man) for d in D])
    rows = []
    for tr, te in GroupKFold(min(5, len(set(ds)))).split(D, groups=ds):
        m = model().fit(np.vstack([D[i][4] for i in tr]), np.concatenate([D[i][5] for i in tr]))
        for i in te:
            pid, G, C, r, X, y = D[i]; rows += score_rows(pid, G, C, r, prob(m, X))
    pd.DataFrame(rows, columns=["pilot_id", "method", "n_rare", "rprec"]).to_csv(out, index=False)


def loao(bench, nat, out):
    """leave-one-atlas-out over natural pools: fit on bench + the other atlases' pools, score the held-out atlas (E1_union = union of targets)."""
    B = load_all(bench); N = load_all(nat); man = pd.read_csv(f"{R.R5}/manifest.csv").set_index("pilot_id")
    at = np.array([dataset_of(d[0], man) for d in N]); rows = []
    for a in sorted(set(at)):
        tr = B + [N[i] for i in np.where(at != a)[0]]
        m = model().fit(np.vstack([d[4] for d in tr]), np.concatenate([d[5] for d in tr]))
        for i in np.where(at == a)[0]: pid, G, C, r, X, y = N[i]; rows += score_rows(pid, G, C, r, prob(m, X))
        print("held out", a, flush=True)
    pd.DataFrame(rows, columns=["pilot_id", "method", "n_rare", "rprec"]).to_csv(out, index=False)


def fit(ids, path):
    D = load_all(ids); m = model().fit(np.vstack([d[4] for d in D]), np.concatenate([d[5] for d in D]))
    pickle.dump(m, open(path, "wb")); print("fit on", len(D), "pilots,", sum(len(d[5]) for d in D), "groups,", sum(d[5].sum() for d in D), "positives")


def apply(path, ids, out):
    m = pickle.load(open(path, "rb")); rows = []
    for pid, G, C, r, X, y in load_all(ids): rows += score_rows(pid, G, C, r, prob(m, X))
    pd.DataFrame(rows, columns=["pilot_id", "method", "n_rare", "rprec"]).to_csv(out, index=False)


def predict(path, pid, out, rule):
    """the frozen method's output for one pilot: no truth is read (HLCA / any new data)."""
    m = pickle.load(open(path, "rb")); g, a = RULES[rule]
    G = pd.read_parquet(f"{R.CD}/{pid}.groups.parquet"); C = pd.read_parquet(f"{R.CD}/{pid}.cells.parquet"); G["stab"] = R.stability(G)
    G = extra(pid, G); P = prob(m, featurize(G)); s = R.cell_score(G, P ** g / G["size"].values ** a, C, "tb_g")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"cell_id": C.cell_id.values, "score": s, "call": False}).to_parquet(out, index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]; ids = lambda f: [l.split()[0] for l in open(f) if l.strip()]
    if cmd == "cv": cv(ids(sys.argv[2]), sys.argv[3])
    elif cmd == "fit": fit(sum([ids(f) for f in sys.argv[2].split(",")], []), sys.argv[3])
    elif cmd == "loao": loao(ids(sys.argv[2]), ids(sys.argv[3]), sys.argv[4])
    elif cmd == "apply": apply(sys.argv[2], ids(sys.argv[3]), sys.argv[4])
    elif cmd == "predict": predict(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
