import os, sys, pickle, json, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h, numpy as np, pandas as pd
os.environ["OMP_NUM_THREADS"] = "4"
M = pickle.load(open(f"{h.H}/r5/refit_dev_nat.pkl", "rb")) if os.path.exists(f"{h.H}/r5/refit_dev_nat.pkl") else pickle.load(open(f"{h.H}/cfgen_natrare/j1004/h/si_v3/vm3_m12/r4_out/verify/v4_integ/refit_dev_nat.pkl", "rb"))
R = h.ROOT["B"]; ref = pd.read_csv(f"{h.H}/r5/code/scores_preds_frozen4.csv"); ref = ref[(ref.metric == "union_AP")].set_index("pilot_id").rprec
rows = []; dm = db = du = 0
for p in h.pools("B"):
    r, s, b = h.evaluate(M.predict, p, "B")
    a = pd.read_parquet(f"{R}/preds_m14p/m14p_frozen/{p}.parquet").set_index("cell_id").score; f = pd.read_parquet(f"{R}/preds_frozen4/frozen4/{p}.parquet").set_index("cell_id").score
    dm = max(dm, (s.reindex(a.index) - a).abs().max()); db = max(db, (b.reindex(f.index) - f).abs().max()); du = max(du, abs(r["bl_union_AP"] - ref[p]))
    rows.append(r)
res = dict(n_pools=len(rows), max_abs_diff_m14p=float(dm), max_abs_diff_frozen4=float(db), max_abs_diff_unionAP_vs_scores_preds_frozen4=float(du), mean_bl_unionAP=float(np.mean([r["bl_union_AP"] for r in rows])), mean_sc_unionAP=float(np.mean([r["sc_union_AP"] for r in rows])))
print(res); json.dump(res, open(f"{h.H}/jiaqi1011/out/w1/parity.json", "w"), indent=1); pd.DataFrame(rows).to_csv(f"{h.H}/jiaqi1011/out/w1/deployed_setB.csv", index=False)
# set A check against the donor-ablation scores
SA = pd.read_csv(f"{h.H}/ablation_donor/out/scores_all.csv"); SA = SA[(SA.metric == "union_AP")]
rA = pd.DataFrame([h.evaluate(M.predict, p, "A")[0] for p in h.pools("A")]); rA.to_csv(f"{h.H}/jiaqi1011/out/w1/deployed_setA.csv", index=False)
for tag, col in (("bl_Cint", "bl_union_AP"), ("sc_Cint", "sc_union_AP")):
    q = SA[SA.method == tag].set_index("pilot_id").rprec; print(tag, "setA max abs diff", float((rA.set_index("pid")[col] - q.reindex(rA.pid)).abs().max()), "mean", rA[col].mean())
