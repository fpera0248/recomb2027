"""python w2_analyze.py [OUTDIR=$W2_OUT] : tables + figures from the scoring outputs res_<arm>.csv / diag_<arm>.csv (each = PCA and ONE other arm on their common pools),
timing.csv and qa.csv. Works on partial outputs. Everything is POST HOC / exploratory (10-atlas test set, spent).
Primary: arm 3 (scorer retrained leave-one-atlas-out inside the set), full method, variant F, each space vs PCA retrained; union AP per atlas (mean over its pools),
exact two-sided sign-flip over atlases, Holm over the primary family. Everything else is descriptive with raw p."""
import os, sys, glob, itertools, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("W2_OUT", os.path.expanduser("~/work/jiaqi1011/out/w2_fm"))
FIG = os.environ.get("W2_FIG", os.path.expanduser("~/work/jiaqi1011/figs/w2_fm")); os.makedirs(FIG, exist_ok=True)
NAME = {"pca": "PCA", "pca30": "PCA (30 comp.)", "scgpt": "scGPT", "scvi": "scVI", "gf": "Geneformer V1-10M", "noise": "random noise"}
VAR = {"F": "global and within-label spaces swapped", "H": "global space swapped, within-label = PCA", "P": "both swapped, global cut to 50 dims"}
STAGE = [("leiden", "Leiden subclustering alone", "union_AP"), ("frozen", "Our method, trained scorer as is", "bl_union_AP"), ("retrain", "Our method, scorer retrained (leave-one-atlas-out)", "bl_union_AP")]
SECOND = [("frozen", "Scorer alone, as is", "sc_union_AP"), ("retrain", "Scorer alone, retrained", "sc_union_AP"), ("leiden", "Leiden alone, E1", "E1_union"), ("frozen", "Our method as is, E1", "bl_E1_union"), ("retrain", "Our method retrained, E1", "bl_E1_union")]
label = lambda arm: NAME.get(arm.split("_")[0], arm) + ("" if "_" not in arm else f" ({VAR.get(arm.split('_')[1], arm.split('_')[1])})")


def signflip(d):
    d = np.asarray(d, float); n = len(d)
    if n == 0: return np.nan
    obs = abs(d.mean()); S = np.array(list(itertools.product([-1, 1], repeat=n))); return float((np.abs((S * d).mean(1)) >= obs - 1e-12).mean())


def boot(d, B=10000, seed=0):
    r = np.random.RandomState(seed); d = np.asarray(d); i = r.randint(0, len(d), (B, len(d))); m = d[i].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty(len(p)); run = 0
    for k, i in enumerate(o): run = max(run, (len(p) - k) * p[i]); out[i] = min(run, 1.0)
    return out


R, D = [], []
for f in sorted(glob.glob(f"{OUT}/res_*.csv")):
    r = pd.read_csv(f); r["file"] = os.path.basename(f); R.append(r); g = f.replace("res_", "diag_")
    if os.path.exists(g): D.append(pd.read_csv(g))
if not R: sys.exit("no res_*.csv yet")
R = pd.concat(R); rows = []
R = R.drop_duplicates(["arm", "stage", "pid", "file"], keep="last")
for fn, grp in R.groupby("file"):
    arms = [a for a in grp.arm.unique() if a != "pca"]
    for arm in arms:
        for stage, sname, col in STAGE + SECOND:
            sub = grp[grp.stage == stage]
            if col not in sub: continue
            at = sub.pivot_table(index="atlas", columns="arm", values=col, aggfunc="mean")
            if "pca" not in at or arm not in at: continue
            at = at.dropna(subset=["pca", arm]); d = (at[arm] - at["pca"]).values
            if not len(d): continue
            lo, hi = boot(d); rows.append(dict(arm=arm, stage=stage, metric=col, what=sname, n_atlas=len(d), n_pools=int(sub[sub.arm == arm].pid.nunique()), pca=at["pca"].mean(), space=at[arm].mean(),
                                               diff=d.mean(), ci_lo=lo, ci_hi=hi, ahead=int((d > 0).sum()), p_signflip=signflip(d)))
T = pd.DataFrame(rows)
if len(T):
    T["primary"] = (T.stage == "retrain") & (T.metric == "bl_union_AP") & T.arm.str.endswith("_F") & ~T.arm.str.startswith(("noise", "pca30", "gf"))     # Geneformer: separate secondary contrast (laptop PCA reference), raw p, never in the Holm family
    T["p_holm"] = np.nan
    if T.primary.any(): T.loc[T.primary, "p_holm"] = holm(T.loc[T.primary, "p_signflip"])
    T.to_csv(f"{OUT}/paired_tests.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30); print(T[T.metric.isin(["union_AP", "bl_union_AP"])].round(3).to_string(index=False))

# decomposition: PCA frozen -> PCA retrained (in-domain training), PCA retrained -> S retrained (representation), shift effect = (S retrain - S frozen) - (PCA retrain - PCA frozen); retraining_gain_in_S is the uncorrected difference; all on atlas means
dec = []
for fn, grp in R.groupby("file"):
    for arm in [a for a in grp.arm.unique() if a != "pca"]:
        g = grp[grp.stage.isin(["frozen", "retrain"])].groupby(["arm", "stage", "atlas"]).bl_union_AP.mean().groupby(["arm", "stage"]).mean().unstack()      # atlas means, like every other table
        if {"frozen", "retrain"} <= set(g.columns) and "pca" in g.index and arm in g.index:
            dec.append(dict(arm=arm, pca_frozen=g.loc["pca", "frozen"], pca_retrain=g.loc["pca", "retrain"], s_frozen=g.loc[arm, "frozen"], s_retrain=g.loc[arm, "retrain"]))
if dec: dd = pd.DataFrame(dec); dd["training_effect_pca"] = dd.pca_retrain - dd.pca_frozen; dd["representation_effect"] = dd.s_retrain - dd.pca_retrain; dd["retraining_gain_in_S"] = dd.s_retrain - dd.s_frozen; dd["shift_effect"] = dd.retraining_gain_in_S - dd.training_effect_pca; dd.to_csv(f"{OUT}/decomposition.csv", index=False); print(dd.round(3).to_string(index=False))

# diagnostics
if D:
    DG = pd.concat(D).drop_duplicates(["arm", "pid"], keep="last"); DG.to_csv(f"{OUT}/diag_all.csv", index=False)
    s = DG.groupby("arm").agg(pools=("pid", "nunique"), groups=("n_groups", "mean"), g_groups=("n_g", "mean"), l_groups=("n_l", "mean"), l_dup_g=("l_dup_g", "mean"),
                              best_jaccard=("mean_best_jaccard", "mean"), frac_target_jaccard50=("frac_j50", "mean")).round(3); s.to_csv(f"{OUT}/diag_summary.csv"); print(s)

# timing
tm = None
if os.path.exists(f"{OUT}/timing.csv"):
    t = pd.read_csv(f"{OUT}/timing.csv", on_bad_lines="skip"); t = t[(t.stage != "stage") & ~t.space.str.endswith("_sub")]; t["wall_s"] = t.wall_s.astype(float); tm = t.groupby(["stage", "space"]).wall_s.agg(["count", "median", "max"]).round(1); tm.to_csv(f"{OUT}/timing_summary.csv"); print(tm)
if os.path.exists(f"{OUT}/qa.csv"): q = pd.read_csv(f"{OUT}/qa.csv").drop_duplicates(["space", "pid"], keep="last"); print(q.groupby("space")[["agree_emb", "agree_pca50"]].mean().round(3))

# ---- figures
if len(T):
    prim = T[T.metric.isin([c for _, _, c in STAGE]) & ~((T.stage == "frozen") & (T.metric != "bl_union_AP"))]
    arms = [a for a in T.arm.unique() if not a.startswith("noise")]
    if arms:
        fig, ax = plt.subplots(len(STAGE), len(arms), figsize=(2.6 * len(arms) + 1, 2.8 * len(STAGE)), squeeze=False, sharey="row")
        for i, (stage, sname, col) in enumerate(STAGE):
            for j, arm in enumerate(arms):
                a = ax[i, j]; fn = [f for f, g in R.groupby("file") if arm in set(g.arm)]
                if not fn: continue
                g = R[(R.file == fn[0]) & (R.stage == stage)]; at = g.pivot_table(index="atlas", columns="arm", values=col, aggfunc="mean").dropna(subset=["pca", arm]) if col in g else pd.DataFrame()
                for _, r in at.iterrows(): a.plot([0, 1], [r["pca"], r[arm]], "-o", color="#4c78a8" if r[arm] >= r["pca"] else "#e45756", lw=1, ms=3, alpha=.8)
                if len(at): a.plot([0, 1], [at["pca"].mean(), at[arm].mean()], "-s", color="k", lw=2.2, ms=5)
                a.set_xticks([0, 1]); a.set_xticklabels(["PCA", label(arm).split(" (")[0]], fontsize=8); a.set_xlim(-.3, 1.3)
                if j == 0: a.set_ylabel(sname.replace(" (", "\n("), fontsize=7)
                a.set_title(label(arm) if i == 0 else "", fontsize=8); a.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Union AP per atlas, representation swapped (post hoc, 10-atlas test set)", fontsize=9); fig.tight_layout()
        for e in ("pdf", "png"): fig.savefig(f"{FIG}/w2_paired_atlas.{e}", dpi=200)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 0.16 * len(arms) * 4 + 0.6))
        y = 0; ticks = []; tl = []
        for arm in arms:   # bold header row per representation, then one row per stage
            ax.text(-0.012, y, label(arm), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7, fontweight="bold"); y += 1
            for stage, sname, col in STAGE:
                r = T[(T.arm == arm) & (T.stage == stage) & (T.metric == col)]
                if len(r): r = r.iloc[0]; ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="o", ms=3, color="k", capsize=2); ticks.append(y); tl.append(sname)
                y += 1
            y += .4
        ax.axvline(0, color="grey", lw=.8); ax.set_yticks(ticks); ax.set_yticklabels(tl, fontsize=7); ax.invert_yaxis(); ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("difference in union AP vs PCA (atlas mean, 95% CI over atlases)", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
        for e in ("pdf", "png"): fig.savefig(f"{FIG}/w2_diff_forest.{e}", dpi=200, bbox_inches="tight")
        plt.close(fig)
if tm is not None and len(tm):
    fig, ax = plt.subplots(figsize=(6, 3)); x = tm["median"].reset_index(); x["k"] = x.stage + " / " + x.space; ax.barh(x.k, x["median"], color="#4c78a8"); ax.set_xlabel("median seconds per pool"); ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7); fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(f"{FIG}/w2_time_per_pool.{e}", dpi=200)
print("done")
