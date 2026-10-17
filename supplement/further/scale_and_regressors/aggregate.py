import os, glob, json, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
import numpy as np, pandas as pd, matplotlib, matplotlib.ticker; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = HR_WORK + "/jiaqi1011"; O = f"{H}/out/w1"; F = f"{H}/figs/w1"; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
def save(fig, n): [fig.savefig(f"{F}/{n}.{e}") for e in ("pdf", "png")]; plt.close(fig)
meta = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(f"{O}/models/*.json"))])
def summ(tag):
    d = pd.read_csv(f"{O}/eval/{tag}.csv") if tag in meta.tag.values else None
    return d
def agg(d):
    out = {}
    for S in "AB":
        x = d[d.set == S].copy(); x["atlas"] = x.pid.str.split("_").str[1]
        for c in ("sc_union_AP", "bl_union_AP", "sc_E1_union", "bl_E1_union"):
            out[f"{S}_{c}_pool"] = x[c].mean(); out[f"{S}_{c}_atlas"] = x.groupby("atlas")[c].mean().mean()
    return out
R = []
for _, m in meta.iterrows():
    f = f"{O}/eval/{m.tag}.csv"
    if os.path.exists(f): R.append({**m.to_dict(), **agg(pd.read_csv(f))})
R = pd.DataFrame(R); R.to_csv(f"{O}/all_models_results.csv", index=False)
# deployed model
dep = pd.concat([pd.read_csv(f"{O}/deployed_setB.csv").assign(set="B"), pd.read_csv(f"{O}/deployed_setA.csv").assign(set="A")]); depa = agg(dep)
# m1 Leiden reference
import harness as h
m1 = []
for S in "AB":
    for p in h.pools(S):
        d = h.load(p, S); import score_atl as SA
        r = {x[2]: x[6] for x in SA.score(d["df"], d["tg"], d["m1"].reindex(d["df"].index).fillna(0).values, p, "m1") if x[2] in ("union_AP", "E1_union")}
        m1.append(dict(pid=p, set=S, bl_union_AP=r["union_AP"], bl_E1_union=r["E1_union"], sc_union_AP=r["union_AP"], sc_E1_union=r["E1_union"]))
m1 = pd.DataFrame(m1); m1a = agg(m1); json.dump(dict(deployed=depa, leiden=m1a), open(f"{O}/reference_means.json", "w"), indent=1)
# learning curve table (hgb)
hg = R[R.kind == "hgb"].groupby("n_pilots").agg(**{k: (k, "mean") for k in ["n_rows", "fit_s", "peak_rss_mb", "n_iter", "size_mb", "predict_s_per_100k", "B_bl_union_AP_pool", "B_sc_union_AP_pool", "A_bl_union_AP_pool", "A_sc_union_AP_pool", "B_bl_union_AP_atlas", "A_bl_union_AP_atlas"]},
    fit_sd=("fit_s", "std"), B_bl_sd=("B_bl_union_AP_pool", "std"), A_bl_sd=("A_bl_union_AP_pool", "std"), B_sc_sd=("B_sc_union_AP_pool", "std"), A_sc_sd=("A_sc_union_AP_pool", "std")).reset_index()
hg.to_csv(f"{O}/scaling_learning_curve.csv", index=False); print(hg.round(4).to_string())
# 364 refit vs deployed on set B (per pool paired)
rf = pd.concat([pd.read_csv(f"{O}/eval/hgb_n364_s{s}.csv").assign(seed=s) for s in range(3)])
dB = dep[dep.set == "B"].set_index("pid"); cmp = {}
for s in range(3):
    x = rf[(rf.seed == s) & (rf.set == "B")].set_index("pid")
    for c in ("sc_union_AP", "bl_union_AP"):
        dd = x[c] - dB[c].reindex(x.index); cmp[f"s{s}_{c}"] = dict(mean_diff=dd.mean(), sd_pool=dd.std(), max_abs=dd.abs().max(), n_pools_better=int((dd > 0).sum()), n_pools_worse=int((dd < 0).sum()))
json.dump(cmp, open(f"{O}/refit364_vs_deployed_setB.json", "w"), indent=1); print(json.dumps(cmp, indent=1))
# figures: python3 plot_w1.py (no refit needed)
