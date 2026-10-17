"""python w2_score.py OUT.csv DIAG.csv ARM [ARM ...]   (arms = directories under $W2_OUT/arms; first ARM is only a pool-set anchor if --pools is not given)
Scores three arms of the experiment for every ARM on the SAME pools (intersection of finished pools unless --pools FILE):
  leiden   : m1 Leiden subclustering in that space alone
  frozen   : the shipped scorer (learn14 predict, FEATS=mkh, P4_s1) applied as is; 'sc' = scorer alone, 'bl' = blend 0.75/0.25 with m1 in the same space
  retrain  : scorer retrained leave-one-atlas-out inside set B (fit on the groups of the other atlases' pools in this space, predict the held-out atlas); same blend
Metrics: union_AP (primary) and E1_union per pool. Diagnostics (DIAG.csv, truth used only here at evaluation): number of candidate groups, g/l split,
best Jaccard of any candidate group with each target, l-groups duplicating a g group. Uses the scoring harness for features, predictions, blend and metrics (copy: harness.py)."""
import os, sys, glob, argparse, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import w2lib as W
import harness as Hn
from harness import L, S, hlib

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("diag"); ap.add_argument("arms", nargs="+"); ap.add_argument("--pools"); ap.add_argument("--model")
o = ap.parse_args()
MODEL = o.model or next(p for p in (os.environ.get("R5_MODEL", ""), f"{W.HR_WORK}/r5/refit_dev_nat.pkl",
        f"{W.HR_WORK}/cfgen_natrare/j1004/h/si_v3/vm3_m12/r4_out/verify/v4_integ/refit_dev_nat.pkl") if p and os.path.exists(p))
frozen = pickle.load(open(MODEL, "rb")); print("model", MODEL, flush=True)
atlas = lambda pid: pid.split("_")[1]


def done(arm, pid):
    d = f"{W.arm_dir(arm)}"; return os.path.exists(f"{d}/cands/{pid}.halo.parquet") and os.path.exists(f"{d}/preds_base/m1_leiden_sub/{pid}.parquet")


for arm in o.arms:
    Hn.ROOT[arm] = W.arm_dir(arm); lk = f"{W.arm_dir(arm)}/hmod"
    if not os.path.lexists(lk): os.symlink(f"{W.ensure_root()}/hmod", lk)
pools = [l.split()[0] for l in open(o.pools)] if o.pools else sorted(p for p in {os.path.basename(f)[:-len('.halo.parquet')] for f in glob.glob(f"{W.arm_dir(o.arms[0])}/cands/*.halo.parquet")} if all(done(a, p) for a in o.arms))   # every finished pool, phase 1 and 2 alike
print(len(pools), "pools common to", o.arms, flush=True)
rows, diag = [], []


def put(arm, pid, stage, r):
    rows.append(dict(arm=arm, pid=pid, atlas=atlas(pid), stage=stage, **{k: v for k, v in r.items() if k != "pid"}))


def pur_of(pid, d):
    t, nt = L.truth(pid, d["C"]); out = np.zeros(len(d["G"]))
    for j, m in enumerate(d["G"].members.values): h = np.bincount(t[m][t[m] >= 0], minlength=len(nt)); out[j] = h.sum() / len(m)
    return out


def jacc(d):
    """per-target best Jaccard of any candidate group of the target's label (positions via cell ids), plus dup stats."""
    df, G, C = d["df"], d["G"], d["C"]; pos = pd.Series(np.arange(len(C)), index=C.cell_id.values); out = []
    for (Lb, T, n) in d["tg"]:
        tm = np.zeros(len(C), bool); tm[pos.loc[df.index[((df.label == Lb) & (df.true_type == T)).values]].values] = True; best = 0.0
        for m, sz in zip(G.members[G.label == Lb].values, G["size"][G.label == Lb].values):
            i = tm[m].sum(); best = max(best, i / (sz + n - i))
        out.append(best)
    return np.array(out)


for arm in o.arms:
    D = {pid: Hn.load(pid, arm) for pid in pools}
    for pid, d in D.items():                                                               # arm 1 + diagnostics
        s = d["m1"].reindex(d["df"].index).fillna(0).values
        put(arm, pid, "leiden", {r[2]: r[6] for r in S.score(d["df"], d["tg"], s, pid, "m1") if r[2] in ("union_AP", "E1_union")})
        G = d["G"]; gs = {m.tobytes() for m in G.members[G.space == "g"]}; ls = [m.tobytes() for m in G.members[G.space == "l"]]
        j = jacc(d); diag.append(dict(arm=arm, pid=pid, atlas=atlas(pid), n_groups=len(G), n_g=int((G.space == "g").sum()), n_l=len(ls), l_dup_g=int(sum(x in gs for x in ls)),
                                      n_targets=len(j), mean_best_jaccard=j.mean(), frac_j50=(j >= 0.5).mean()))
    for pid in pools:                                                                      # arm 2
        r, _, _ = Hn.evaluate(lambda X: frozen.predict(X), pid, arm); r = {k: v for k, v in r.items()}
        put(arm, pid, "frozen", r)
    ats = sorted({atlas(p) for p in pools})                                                 # arm 3
    if len(ats) < 2: continue
    Xs = {pid: D[pid]["X"] for pid in pools}; Ys = {pid: pur_of(pid, D[pid]) for pid in pools}
    for a in ats:
        tr = [p for p in pools if atlas(p) != a]; m = L.model().fit(np.vstack([Xs[p] for p in tr]), np.concatenate([Ys[p] for p in tr]))
        for pid in [p for p in pools if atlas(p) == a]:
            r, _, _ = Hn.evaluate(lambda X: m.predict(X), pid, arm); put(arm, pid, "retrain", r)
    print(arm, "scored", flush=True)
res = pd.DataFrame(rows); res.to_csv(o.out, index=False); pd.DataFrame(diag).to_csv(o.diag, index=False); print(len(res), "rows ->", o.out)
