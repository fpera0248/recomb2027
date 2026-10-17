"""P2: can the annotation be refined? python p2_refine.py calibrate   -> threshold t on SET A (macro best-match F1 of arm A1), written to OUT/p2_t_cal.json BEFORE any set B F1 is computed
                                python p2_refine.py apply [pidmax]  -> arms A0..A4 on clean set-B pools (ID variant outputs of run_unit), OUT/p2_pool.csv
Arms: A0 broad label | A1 broad + ours (groups with P>=t, greedy by P desc then size desc, overlapping cells keep first assignment, >=10 cells left)
A2s broad + Leiden small subclusters (<10% of label, as stored) | A2f broad + full silhouette partition | A3 A1 + Leiden small subclusters on the residual | A4 size-matched random control of A1.
Scores vs fine truth (true_type): ARI, NMI (whole pool), target-pair F1 (best match and one-to-one Hungarian), all-type F1 secondary."""
import os, sys, json, glob, pickle, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.dont_write_bytecode = True; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from w6common import *
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
GRID = [round(x, 2) for x in np.arange(0.10, 0.81, 0.05)]; MING = 10

def greedy(G, P, t, cells):
    """class array: label|gK for accepted groups. G needs label,size,members; cells = cell_id array aligned with members."""
    cls = pd.Series(None, index=cells, dtype=object); k = 0
    for i in np.lexsort((-G["size"].values, -P)):
        if P[i] < t: break
        m = np.asarray(G["members"].iloc[i]); m = m[cls.isna().values[m]]
        if len(m) >= MING: cls.iloc[m] = f"{G['label'].iloc[i]}|g{k}"; k += 1
    return cls, k

def f1_tables(df, cls, tg):
    """returns best-match F1 and one-to-one F1 per target pair, plus all-type best-match macro."""
    lab = df.label.values; tt = df.true_type.values; c = cls.values; bm, oo, allt = [], [], []
    for L in np.unique(lab):
        m = lab == L; cl = pd.unique(c[m]); types = [T for T, n in pd.Series(tt[m]).value_counts().items() if n >= MING]
        if not types: continue
        F = np.zeros((len(types), len(cl))); ci = pd.Series(c[m]); 
        for a, T in enumerate(types):
            tm = tt[m] == T; nT = tm.sum(); inter = ci[tm].value_counts()
            for b, k in enumerate(cl): F[a, b] = 2 * inter.get(k, 0) / (nT + (ci == k).sum())
        r, q = linear_sum_assignment(-F); one = dict(zip(r, F[r, q]))
        for a, T in enumerate(types):
            allt.append(F[a].max())
            if any(L == L2 and T == T2 for L2, T2, _ in tg): bm.append(F[a].max()); oo.append(one.get(a, 0.0))
    return np.array(bm), np.array(oo), np.array(allt)

def arm_metrics(df, cls, tg):
    bm, oo, allt = f1_tables(df, cls, tg); joint = df.label.values + "||" + cls.values.astype(str)
    return dict(F1_bm=bm.mean() if len(bm) else np.nan, F1_oo=oo.mean() if len(oo) else np.nan, F1_alltypes=allt.mean(), ARI=ARI(df.true_type.values, cls.values.astype(str)),
                NMI=NMI(df.true_type.values, cls.values.astype(str)), n_classes=int(pd.Series(cls.values).nunique()), n_targets=len(tg))

def random_control(df, cls, seed):
    """same label, same group sizes as the accepted A1 groups, random disjoint cells."""
    rng = np.random.default_rng(seed); out = pd.Series([f"{l}|rest" for l in df.label], index=df.index, dtype=object)
    for k, n in cls[cls.str.contains(r"\|g\d+$", na=False)].value_counts().items():
        L = k.rsplit("|g", 1)[0]; pool = out.index[(df.label.values == L) & (out.values == f"{L}|rest")]; out.loc[rng.choice(pool, n, replace=False)] = k
    return out

def wrap(cls, df): return cls.where(cls.notna(), pd.Series([f"{l}|rest" for l in df.label], index=df.index))

def calibrate(grid=None):
    ext = grid is not None; grid = grid or GRID
    import harness as h
    M = pickle.load(open(h.MP if hasattr(h, "MP") else (f"{h.H}/r5/refit_dev_nat.pkl" if os.path.exists(f"{h.H}/r5/refit_dev_nat.pkl") else f"{h.H}/cfgen_natrare/j1004/h/si_v3/vm3_m12/r4_out/verify/v4_integ/refit_dev_nat.pkl"), "rb"))
    rows = []
    for pid in h.pools("A"):
        d = h.load(pid, "A"); P = np.clip(M.predict(d["X"]), 0, 1); df, tg = d["df"], d["tg"]; cells = d["C"].cell_id.values
        if not tg: continue
        df = df.reindex(cells)
        if df.true_type.isna().any(): continue      # set-A pools whose cell ids do not match their candidate dump (cereb): skipped
        for t in grid:
            cls, k = greedy(d["G"], P, t, cells); r = arm_metrics(df, wrap(cls, df), tg); rows.append(dict(pid=pid, atlas=pid.rsplit("_p", 1)[0], t=t, n_groups=k, **r))
    R = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); R.to_csv(f"{OUT}/p2_calibration_{'ext_' if ext else ''}setA.csv", index=False)
    if ext: print(R.groupby("t")[["F1_bm", "F1_oo", "ARI", "n_groups"]].mean().round(4)); return   # descriptive extension below the grid edge; t_cal stays declared at 0.10
    S = R.groupby(["t", "atlas"]).F1_bm.mean().groupby("t").mean(); best = float(S[S >= S.max() - 1e-12].index.max())
    json.dump(dict(t_cal=best, curve=S.round(4).to_dict(), rule="max atlas-mean macro best-match target F1 of A1 on set A; ties -> larger t"), open(f"{OUT}/p2_t_cal.json", "w"), indent=1); print(S.round(4).to_dict(), "t_cal", best)

def apply(maxp=2):
    ts = json.load(open(f"{OUT}/p2_t_cal.json"))["t_cal"]; rows = []; sys.path.insert(0, f"{HR_WORK}/jiaqi1011/code/w1"); import harness as h
    from score_hlca import targets
    for pid in pools().query("pool <= @maxp").pilot_id:
        g = f"{OUT}/groupP/ID__{pid}.parquet"; u = f"{OUT}/units/ID__{pid}.csv"
        if not (os.path.exists(g) and os.path.exists(u)): continue
        G = pd.read_parquet(g); R = root("ID"); C = pd.read_parquet(f"{R}/cands/{pid}.cells.parquet"); cells = C.cell_id.values
        d = pd.read_parquet(f"{CLEAN}/hmod/pilots/{pid}.parquet", columns=["obs_name", "label", "true_type"]).astype({"label": str, "true_type": str}); si, oid = h.hlib._opaque_order(pid, d.obs_name.values)
        df = d.iloc[si].set_index(oid).reindex(cells); tg = targets(df)
        i0 = int(np.argmax(G["size"].values)); assert (df.label.values[np.asarray(G["members"].iloc[i0])] == G["label"].iloc[i0]).all(), "members must index cells.parquet rows"
        m1 = pd.read_parquet(f"{R}/preds_base/m1_leiden_sub/{pid}.parquet").set_index("cell_id").reindex(cells); mf = pd.read_parquet(f"{R}/preds_base/m1_full/{pid}.parquet").set_index("cell_id").reindex(cells)
        small = pd.Series([f"{l}|m{g}" if g >= 0 else f"{l}|rest" for l, g in zip(df.label, m1.group)], index=cells)
        P = G["P"].values
        for tname, t in (("tcal", ts), ("t0.3", 0.3), ("t0.5", 0.5), ("t0.7", 0.7)):
            a1, k = greedy(G, P, t, cells); a1 = wrap(a1, df)
            a3 = a1.where(a1.str.contains(r"\|g\d+$"), small); vc = a3.value_counts(); a3 = a3.where(~a3.str.contains(r"\|m\d+$") | a3.map(vc).ge(MING), pd.Series([f"{l}|rest" for l in df.label], index=cells))
            arms = {"A0": pd.Series(df.label.values, index=cells), "A1": a1, "A2s": small, "A2f": mf.cluster, "A3": a3}
            for s in range(5): arms[f"A4_{s}"] = random_control(df, a1, s)
            for an, cls in arms.items(): rows.append(dict(pid=pid, atlas=pid.split("_")[1], thr=tname, t=t, arm=an, n_groups_A1=k, n_pairs_P_ge_t=int((P >= t).sum()), **arm_metrics(df, cls, tg)))
    R = pd.DataFrame(rows); R["arm_"] = R.arm.str.replace(r"_\d$", "", regex=True); R.to_csv(f"{OUT}/p2_pool.csv", index=False); print(R.groupby(["thr", "arm_"])[["F1_bm", "F1_oo", "ARI", "NMI", "n_classes"]].mean().round(3))

if __name__ == "__main__":
    if sys.argv[1] == "calibrate_ext": sys.path.insert(0, f"{HR_WORK}/jiaqi1011/code/w1"); calibrate([0.0, 0.02, 0.05, 0.075])
    elif sys.argv[1] == "calibrate": sys.path.insert(0, f"{HR_WORK}/jiaqi1011/code/w1"); calibrate()
    else: apply(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
