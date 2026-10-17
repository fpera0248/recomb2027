"""E1 (N18a/b): per-type adaptive augmentation policy vs duplication, from saved same-processing 31-type results.

README: python3 run.py [--smoke]   (seconds; reads probe31_runs.csv, writes out/*.csv + out/report.txt; no training)
Input: part1_results/probe31_runs.csv (part1/probe31.py schema; 'upsample'=duplication, 'real'=no augmentation).
Unit rows = (seed, type, n): 248 probes (each type probed once per rung n=1..128). Metric: rec200 (recall in top 200 eval cells,
matched operating point), per classifier. Donor-disjoint splits are those of the saved runs.
PRE-DECLARED (before running): policy = leave-one-type-out (LOTO): for held-out type t, per n-bin {<=4, 8-16, >=32}, pick the arm
with highest mean rec200 over the OTHER 30 types in that bin (uses only train-side n of t plus other types' outcomes). Baselines: dup;
fixed = best single arm over other types (LOTO); oracle = best arm per probe on test (optimistic upper bound, not a method).
Test: unit = type (mean over its 8 probes), part1.signflip.signflip_p two-sided, BH over the 4 classifiers.
SUCCESS iff policy-minus-dup d>=+0.01 AND q<0.05 in >=2 of 4 classifiers AND policy beats LOTO fixed (q<0.05) in the same classifiers.
Else report 'not shown'. Deep generators are not in this table (only dup/no-aug/jitter/jitternz/smote/li2025/jmo).
"""
import os, sys, argparse
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
R = os.environ.get("R8", HR_WORK + "/recomb2027-r8i3wt")
sys.path.insert(0, R)
from part1.signflip import signflip_p
from statsmodels.stats.multitest import multipletests

ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--csv", default=HR_WORK + "/recomb27/v2/p1/part1_results/probe31_runs.csv")
ap.add_argument("--out", default=None)
ap.add_argument("--exclude", default="", help="comma list of arms to drop (secondary run: own diagnostics)")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
a.out = a.out or os.path.join(here, "out_smoke" if a.smoke else "out")
os.makedirs(a.out, exist_ok=True)
D = pd.read_csv(a.csv); D = D[D.probe]
binf = lambda n: 0 if n <= 4 else (1 if n <= 16 else 2)
rows, per = [], []
for k in (["lr"] if a.smoke else ["lr", "knn", "rf", "mlp"]):
    w = D[D.clf == k].pivot_table(index=["name_", "seed", "n"], columns="arm", values="rec200").reset_index()
    arms = [c for c in w.columns if c not in ("name_", "seed", "n") and c not in a.exclude.split(",")]
    w["bin"] = w.n.map(binf)
    types = sorted(w.name_.unique())[: (6 if a.smoke else None)]
    w = w[w.name_.isin(types)].copy()
    N = len(w); pol, fix, pls, fls, prg = (np.empty(N) for _ in range(5)); pa = [None] * N
    pick = lambda o, col, v: (o.groupby(col)[arms].mean().idxmax(axis=1).get(v) or o[arms].mean().idxmax())  # fallback: empty bin -> overall best
    for t in types:
        m = (w.name_ == t).values; o = w[~m]
        best = o.groupby("bin")[arms].mean().idxmax(axis=1)
        fb = o[arms].mean().idxmax()
        for i in np.where(m)[0]:
            ch = best.get(w["bin"].iat[i]) or fb; pol[i] = w[ch].iat[i]; fix[i] = w[fb].iat[i]; pa[i] = ch
            ol = o[o.seed != w.seed.iat[i]]  # sensitivity: also leave out the probe's seed
            pls[i] = w[pick(ol, "bin", w["bin"].iat[i])].iat[i]; fls[i] = w[ol[arms].mean().idxmax()].iat[i]
            prg[i] = w[pick(o, "n", w.n.iat[i])].iat[i]  # sensitivity: bin = rung n, no free edges
    w["policy"], w["fixed"], w["oracle"] = pol, fix, w[arms].max(axis=1).values  # oracle = per-probe max (noise-inflated, loose bound)
    w["policy_ls"], w["fixed_ls"], w["policy_rung"] = pls, fls, prg
    tb = w.groupby("name_")[arms].transform("mean")  # per-type oracle: best arm per type chosen on test, applied to all its probes
    w["oracle_type"] = [w.at[i, tb.loc[i].idxmax()] for i in w.index]
    w["choice"] = pa
    w["clf"] = k; per.append(w)
    sp = lambda x, y: signflip_p((w[x] - w[y]).values, w.name_.values)
    r = dict(clf=k, dup=w.upsample.mean(), none=w["real"].mean(), policy=w.policy.mean(), fixed=w.fixed.mean(), oracle=w.oracle.mean())
    for nm, (x, y) in dict(pol_vs_dup=("policy", "upsample"), pol_vs_fixed=("policy", "fixed"), oracle_vs_dup=("oracle", "upsample"),
                                    oracletype_vs_dup=("oracle_type", "upsample"), oracletype_vs_fixed=("oracle_type", "fixed"),
                                    polls_vs_fixedls=("policy_ls", "fixed_ls"), polrung_vs_fixed=("policy_rung", "fixed")).items():
        p, dt = sp(x, y); r[nm + "_d"] = (w[x] - w[y]).mean(); r[nm + "_p"] = p; r[nm + "_types_pos"] = int((dt > 0).sum())
    rows.append(r)
T = pd.DataFrame(rows)
for c in ("pol_vs_dup", "pol_vs_fixed", "polls_vs_fixedls", "polrung_vs_fixed"):
    T[c + "_q"] = multipletests(T[c + "_p"], method="fdr_bh")[1]
ok = (T.pol_vs_dup_d >= 0.01) & (T.pol_vs_dup_q < 0.05) & (T.pol_vs_fixed_q < 0.05) & (T.pol_vs_fixed_d > 0)
T["success"] = ok
pd.concat(per).to_csv(os.path.join(a.out, "per_probe.csv"), index=False)
T.to_csv(os.path.join(a.out, "summary.csv"), index=False)
verdict = "SUCCESS" if ok.sum() >= 2 else "NOT SHOWN"
rep = ("Rule is nested across types (other types' held-out outcomes), NOT train-only; policy_ls also leaves out the probe's seed. "
       "Diagnostic is n-bin only (per size, not per type); candidates are the arms in the CSV only. oracle = per-probe max (loose); oracle_type = best arm per type on test.\n") + T.round(4).to_string() + f"\nVERDICT: {verdict} ({int(ok.sum())}/{len(T)} classifiers pass)\n"
# N18b: which arm wins where (mean rec200 minus dup, by n-bin)
al = pd.concat(per); arms = [c for c in al.columns if c in D.arm.unique()]
rep += "\nmean rec200 minus dup by clf and n-bin (0:<=4,1:8-16,2:>=32)\n" + al.groupby(["clf", "bin"])[arms].mean().sub(al.groupby(["clf", "bin"]).upsample.mean(), axis=0).round(3).to_string() + "\n"
open(os.path.join(a.out, "report.txt"), "w").write(rep); print(rep)
if a.smoke: assert len(T) == 1 and np.isfinite(T.policy.iat[0])
