"""python run_unit.py VARIANT PID : features (candidate_dump, candidate_marker_features, candidate_halo_features) + Leiden subclustering (leiden_sub_full) + frozen scorer + blend on the variant labels, scored against CLEAN-label targets.
Writes OUT/units/<variant>__<pid>.csv (long metric table) and OUT/groupP/<variant>__<pid>.parquet. Resumable (unit csv = done). Targets are asserted equal to the clean target list."""
import os, sys, subprocess, time, pickle, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.dont_write_bytecode = True; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from label_corruption_common import *
v, pid = sys.argv[1], sys.argv[2]
CLEAN = os.environ.get("W6_CLEAN", CLEAN); OUT = os.environ.get("W6_OUT", OUT); os.makedirs(f"{OUT}/units", exist_ok=True); os.makedirs(f"{OUT}/groupP", exist_ok=True)
fout = f"{OUT}/units/{v}__{pid}.csv"
if os.path.exists(fout): sys.exit(0)
R = os.environ.get("W6_ROOT") or root(v); assert os.path.exists(f"{R}/hmod/pilots/{pid}.parquet"), "run make_variants first"
env = dict(os.environ, R5=R, SI_H=f"{R}/hmod", SLIM_DIR=f"{R}/slim_atlas", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2", PYTHONWARNINGS="ignore", PYTHONDONTWRITEBYTECODE="1")
t0 = time.time(); run = lambda *a: subprocess.run([sys.executable, "-W", "ignore", *a], env=env, check=True, cwd=R)
for nm, ext in (("sr3", "groups"), ("sr3m", "markers"), ("sr3h", "halo")):
    if not os.path.exists(f"{R}/cands/{pid}.{ext}.parquet"): run(f"{CODE_R5}/{nm}.py", pid)
if not os.path.exists(f"{R}/preds_base/m1_full/{pid}.parquet"): run(f"{HERE}/leiden_sub_full.py", pid)
t1 = time.time() - t0
sys.path.insert(0, HR_WORK + "/jiaqi1011/code/w1"); import harness as h                      # locked code, clean R5 env in this process
from harness import L, S, hlib, G_, A_
from combo_r2 import combo
from score_hlca import targets
from sklearn.metrics import average_precision_score as AP
MP = f"{h.H}/r5/refit_dev_nat.pkl"; MP = MP if os.path.exists(MP) else f"{h.H}/cfgen_natrare/j1004/h/si_v3/vm3_m12/r4_out/verify/v4_integ/refit_dev_nat.pkl"; M = pickle.load(open(MP, "rb"))
L.R.CD = f"{R}/cands"; G = pd.read_parquet(f"{L.R.CD}/{pid}.groups.parquet"); C = pd.read_parquet(f"{L.R.CD}/{pid}.cells.parquet"); G["stab"] = L.R.stability(G); G = L.extra(pid, G)
P = np.clip(M.predict(L.featurize(G)), 0, 1); sc = pd.Series(L.R.cell_score(G, P ** G_ / G["size"].values ** A_, C, "tb_g"), index=C.cell_id.values)
m1 = pd.read_parquet(f"{R}/preds_base/m1_leiden_sub/{pid}.parquet").set_index("cell_id").score; bl = combo(sc, m1, 0.75)
G.assign(P=P).to_parquet(f"{OUT}/groupP/{v}__{pid}.parquet")
def clean_df(label_dir):
    d = pd.read_parquet(f"{label_dir}/{pid}.parquet", columns=["obs_name", "label", "true_type"]).astype({"label": str, "true_type": str}); si, oid = hlib._opaque_order(pid, d.obs_name.values)
    return d.iloc[si].set_index(oid), d.iloc[si].obs_name.values
df, onames = clean_df(f"{CLEAN}/hmod/pilots"); tg = targets(df)
dv, _ = clean_df(f"{R}/hmod/pilots"); assert (dv.index == df.index).all() and (dv.true_type == df.true_type).all()
rel = pd.read_parquet(f"{W}/relabel/{v}/{pid}.parquet").set_index("obs_name").changed.reindex(onames).values if (v not in ("ID", "MERGE") and not os.environ.get("W6_ROOT")) else np.zeros(len(df), bool)
pos = np.zeros(len(df), bool)
for L_, T_, _n in tg: pos |= ((df.label == L_) & (df.true_type == T_)).values
K = int(pos.sum()); knn = pd.read_parquet(f"{R}/preds_base/m1_full/{pid}.parquet").set_index("cell_id").knn_dis.reindex(df.index).values
rows = []; add = lambda k, m, x: rows.append((v, pid, k, m, float(x)))
sv = {"bl": bl, "sc": sc, "m1": m1}
for k, s in sv.items():
    x = s.reindex(df.index).fillna(0).values
    for r in S.score(df, tg, x, pid, k):
        if r[2] in ("union_AP", "E1_union"): add(k, r[2], r[6])
    keep = ~(rel & pos); add(k, "AP_nonrelabelled_pos", AP(pos[keep], x[keep]))   # relabelled positives removed from the universe
    if k == "bl": add(k, "AP_permuted_scores", AP(pos, np.random.default_rng(0).permutation(x))); add(k, "prevalence", pos.mean())   # chance reference (permuted scores give ~prevalence); not a pipeline test
    if k == "bl" and v == "ID":   # planted known-signal test through the same score path: pos + noise must give union AP near 1 (asserted in analyze_label_corruption.py)
        for r in S.score(df, tg, pos + np.random.default_rng(1).normal(0, 0.3, len(pos)), pid, "planted"):
            if r[2] == "union_AP": add("planted", "union_AP_planted", r[6])
    top = np.argsort(-x, kind="stable")[:K]; add(k, "decoy_capture", (rel[top] & ~pos[top]).mean()); add(k, "topK_pos_share", pos[top].mean())
    if rel.any(): add(k, "mislabel_AP", AP(rel, x)); add(k, "mislabel_prev", rel.mean())
if rel.any(): add("knn_dis", "mislabel_AP", AP(rel, knn))
if v == "MERGE":   # targets recomputed on the merged labels (the relevant answer for coarsened annotations)
    tg2 = targets(dv); add("meta", "n_targets_clean", len(tg)); add("meta", "n_targets_merged", len(tg2))
    for k, s in sv.items():
        for r in S.score(dv, tg2, s.reindex(df.index).fillna(0).values, pid, k):
            if r[2] in ("union_AP", "E1_union"): add(k, r[2] + "_recomputed", r[6])
if os.environ.get("W6_OOD"):   # share of group-feature values outside the training 1st-99th percentile (zero-shot out-of-distribution check)
    tr = np.load(f"{h.H}/ablation_donor/out/train_xy.npz")["X"]; lo, hi = np.percentile(tr, [1, 99], axis=0); Xg = np.asarray(L.featurize(G)); out = (Xg < lo) | (Xg > hi)
    add("meta", "ood_value_share", out.mean()); add("meta", "ood_group_share_any", out.any(1).mean()); add("meta", "ood_group_share_gt25pct_features", (out.mean(1) > 0.25).mean())
add("meta", "n_targets", len(tg)); add("meta", "K", K); add("meta", "n_groups", len(G)); add("meta", "sec_features", t1); add("meta", "sec_total", time.time() - t0)
pd.DataFrame(rows, columns=["variant", "pid", "arm", "metric", "value"]).to_csv(fout + ".tmp", index=False); os.rename(fout + ".tmp", fout); print(v, pid, f"{time.time()-t0:.0f}s", flush=True)
