"""Offline ranking rules over sr3 candidate dumps (DEV only: the truth columns here are for evaluation/training, never a method input).
  python rescore.py feats IDS OUT.parquet   -> one row per candidate group: scale-free features + stability + (offline) recall/purity vs the hidden type
  python rescore.py eval  IDS RULES OUT.csv -> matched-k rprec per (pilot, rule); RULES is a python file defining RULES = {name: f(G, C) -> cell score}
Cell score from group scores: max over the groups holding the cell (groups overlap across resolutions); cells in no group get -1 + 1e-3 * tie-break."""
import os, sys, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
SI = os.path.dirname(os.path.abspath(__file__)); R5 = os.environ.get("R5") or HR_WORK + "/r5/data"; sys.path.insert(0, os.environ.get("SI_H", R5 + "/hmod"))
from hlib import _opaque_order, PILOTS_DIR
from rprec import rprec
CD = os.environ.get("SI_CANDS", R5 + "/cands")


def load(pid):
    G = pd.read_parquet(f"{CD}/{pid}.groups.parquet"); C = pd.read_parquet(f"{CD}/{pid}.cells.parquet")
    df = pd.read_parquet(f"{PILOTS_DIR}/{pid}.parquet", columns=["obs_name", "is_rare"]); si, oid = _opaque_order(pid, df.obs_name.values)
    r = pd.Series(df.is_rare.values[si].astype(bool), index=oid).reindex(C.cell_id.values).values
    return G, C, r


def stability(G):
    """number of OTHER resolutions (same space and label) holding a group with Jaccard >= 0.5 to this one."""
    st = np.zeros(len(G), int)
    for _, g in G.groupby(["space", "label"]):
        sets = [set(m) for m in g.members]; res = g.res.values; ix = g.index.values
        for a in range(len(ix)):
            hit = {res[b] for b in range(len(ix)) if res[b] != res[a] and len(sets[a] & sets[b]) >= 0.5 * len(sets[a] | sets[b])}
            st[ix[a]] = len(hit)
    return st


def cell_score(G, gs, C, tb="tb_g"):
    gs = np.asarray(gs, float); keep = np.isfinite(gs)                 # a rule drops a group by giving it NaN/inf
    gs = pd.Series(gs[keep]).rank(method="dense").values / max(keep.sum(), 1)   # rank scale, so the tie-break never reorders groups
    s = np.full(len(C), -np.inf)
    for m, v in zip(G.members.values[keep], gs): np.maximum.at(s, m, v)
    out = np.where(np.isfinite(s), s, -1.0)
    return out + 1e-6 * C[tb].values      # tie-break strictly below any group-score difference used here


def feats(ids, out):
    rows = []
    for pid in ids:
        if not os.path.exists(f"{CD}/{pid}.groups.parquet"): continue
        G, C, r = load(pid); k = r.sum(); G["stab"] = stability(G)
        G["hit"] = [r[m].sum() for m in G.members]; G["recall"] = G.hit / max(k, 1); G["purity"] = G.hit / G["size"]
        G["pilot_id"] = pid; G["n_rare"] = int(k); rows.append(G.drop(columns="members"))
    pd.concat(rows).to_parquet(out, index=False)


def evaluate(ids, rules_file, out):
    ns = {}; exec(open(rules_file).read(), ns); RULES = ns["RULES"]; rows = []
    for pid in ids:
        if not os.path.exists(f"{CD}/{pid}.groups.parquet"): continue
        G, C, r = load(pid); G["stab"] = stability(G)
        for name, f in RULES.items():
            gs, tb = f(G, C); rows.append((pid, name, int(r.sum()), rprec(cell_score(G, gs, C, tb), r)))
    pd.DataFrame(rows, columns=["pilot_id", "method", "n_rare", "rprec"]).to_csv(out, index=False)


def demo():
    G = pd.DataFrame({"members": [np.array([0, 1]), np.array([1, 2, 3])]}); C = pd.DataFrame({"tb_g": [0.1, 0.2, 0.3, 0.4, 0.5]})
    s = cell_score(G, np.array([2.0, 1.0]), C)
    assert s[1] > s[2] > 0 > s[4] and s[0] > s[3] and abs(s[2] - s[3]) < 1e-5


if __name__ == "__main__":
    demo(); ids = [l.split()[0] for l in open(sys.argv[2]) if l.strip()]
    if sys.argv[1] == "feats": feats(ids, sys.argv[3])
    else: evaluate(ids, sys.argv[3], sys.argv[4])
