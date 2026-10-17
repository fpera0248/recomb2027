"""Scoring harness: rebuild scorer predictions (learned_scorer.predict, FEATS=mkh, P4_s1) + blend (rank_combination_sweep, 0.75/0.25 with m1_leiden_sub) + union AP / E1 (score_atl) for any regressor.
Read-only use of ~/work/r5/code (locked). Set B = r5/data (57 test pools, manifest split==test5), set A = ablation_donor/data (69 pools)."""
import os, sys, glob, pickle, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
os.environ.update(FEATS="mkh", TARGET="pur", PYTHONDONTWRITEBYTECODE="1", R5=HR_WORK + "/r5/data", SI_H=HR_WORK + "/r5/data/hmod")
sys.dont_write_bytecode = True
sys.path.insert(0, HR_WORK + "/r5/code"); H = HR_WORK
import learn14 as L, score_atl as S, hlib
from combo_r2 import combo
from score_hlca import targets
FEAT_COLS = list(range(41)); G_, A_ = L.RULES["P4_s1"]
ROOT = {"B": f"{H}/r5/data", "A": f"{H}/ablation_donor/data"}

def pools(setname):
    if setname == "A": return [l.split()[0] for l in open(f"{H}/ablation_donor/out/pools_ok_all.txt") if l.strip()]
    M = pd.concat([pd.read_csv(f) for f in glob.glob(f"{H}/r5/data/atlas_manifest_*.csv")]); return list(M[M.split == "test5"].pilot_id)

_cache = {}
def load(pid, setname):
    """features X, groups G, cells C, m1 series, truth df and targets for one pool (cached; no model involved)."""
    k = (pid, setname)
    if k in _cache: return _cache[k]
    R = ROOT[setname]; L.R.CD = f"{R}/cands"; hlib.PILOTS_DIR = f"{R}/hmod/pilots"
    G = pd.read_parquet(f"{L.R.CD}/{pid}.groups.parquet"); C = pd.read_parquet(f"{L.R.CD}/{pid}.cells.parquet"); G["stab"] = L.R.stability(G)
    G = L.extra(pid, G); assert G is not None, pid
    m1 = pd.read_parquet(f"{R}/preds_base/m1_leiden_sub/{pid}.parquet").set_index("cell_id").score
    df = pd.read_parquet(f"{hlib.PILOTS_DIR}/{pid}.parquet", columns=["obs_name", "label", "true_type"]).astype({"label": str, "true_type": str})
    si, oid = hlib._opaque_order(pid, df.obs_name.values); df = df.iloc[si].set_index(oid)
    d = dict(X=L.featurize(G), G=G, C=C, m1=m1, df=df, tg=targets(df)); _cache[k] = d; return d

def cell_scores(P, d):
    """P4_s1 cell score, as learned_scorer.predict (index = cell_id of C order)."""
    return pd.Series(L.R.cell_score(d["G"], P ** G_ / d["G"]["size"].values ** A_, d["C"], "tb_g"), index=d["C"].cell_id.values)

def evaluate(predict, pid, setname):
    """predict: X(n,41) -> P in [0,1]. Returns (rows dict, sc series, bl series). rows: union_AP and E1_union for scorer alone ('sc') and blend ('bl')."""
    d = load(pid, setname); s = cell_scores(np.clip(predict(d["X"]), 0, 1), d); b = combo(s, d["m1"], 0.75)
    out = dict(pid=pid)
    for nm, v in (("sc", s), ("bl", b)):
        r = S.score(d["df"], d["tg"], v.reindex(d["df"].index).fillna(0).values, pid, nm)
        for row in r:
            if row[2] in ("union_AP", "E1_union"): out[f"{nm}_{row[2]}"] = row[6]
    return out, s, b

def evaluate_set(predict, setname):
    return pd.DataFrame([evaluate(predict, p, setname)[0] for p in pools(setname)])
