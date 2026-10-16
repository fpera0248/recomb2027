"""python analyze_w6.py : final tables + figures for W6 from whatever outputs exist (partial outputs are fine).
Reads OUT/units/*.csv (P1), OUT/p2_pool.csv + p2_t_cal.json (P2), OUT/p3/units/*.csv + p3_pools.json (P3). Writes OUT/*.csv and ~/work/jiaqi1011/figs/w6_annot_atac/*.{pdf,png}.
Everything here is post hoc on spent test atlases. Text uses 'our method' and 'Leiden subclustering'."""
import os, sys, json, glob, itertools, numpy as np, pandas as pd
sys.dont_write_bytecode = True; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from w6common import OUT, HOME
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG = f"{HOME}/work/jiaqi1011/figs/w6_annot_atac"; os.makedirs(FIG, exist_ok=True)
COL = {"ours": "#1f5fa8", "leiden": "#d9822b", "scorer": "#6aa6d8", "knn": "#7a7a7a", "ctrl": "#b9b9b9"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
def save(fig, name): fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=200); plt.close(fig)

def signflip(d, side="two"):
    """exact sign-flip test of mean(d) over atlases (n <= 12)."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3: return np.nan
    S = np.array(list(itertools.product([-1, 1], repeat=n))) @ d / n; o = d.mean()
    return float((S >= o - 1e-12).mean()) if side == "greater" else float((np.abs(S) >= abs(o) - 1e-12).mean())
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0
    for i, j in enumerate(o): run = max(run, (len(ps) - i) * ps[j]); out[j] = min(run, 1)
    return out
def selftest():
    assert abs(signflip([1] * 8) - 2 / 256) < 1e-12 and abs(signflip([-1] * 8) - 2 / 256) < 1e-12 and holm([0.01, 0.04, 0.03]).tolist() == [0.03, 0.06, 0.06]
selftest()

# ------------------------------------------------------------------ P1
def bootci(d, B=5000):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; r = np.random.default_rng(0)
    return tuple(np.percentile([r.choice(d, len(d)).mean() for _ in range(B)], [2.5, 97.5])) if len(d) else (np.nan, np.nan)
def p1():
    fs = glob.glob(f"{OUT}/units/*.csv")
    if not fs: return
    U = pd.concat([pd.read_csv(f) for f in fs]); U["atlas"] = U.pid.str.split("_").str[1]
    grp = lambda v: v[:3] if v.startswith("R") else v
    U["group"] = U.variant.map(grp)
    ap = U[U.metric == "union_AP"].copy(); name = {"bl": "ours", "m1": "leiden", "sc": "scorer"}; ap["method"] = ap.arm.map(name)
    pool = ap.groupby(["group", "method", "atlas", "pid"]).value.mean().reset_index()          # mean over seeds within pool
    clean = pool[pool.group == "ID"].set_index(["method", "pid"]).value
    pool["clean"] = [clean.get((m, p), np.nan) for m, p in zip(pool.method, pool.pid)]
    # only pools with all three methods present for ID and the group
    atl = pool.groupby(["group", "method", "atlas"])[["value", "clean"]].mean().reset_index(); atl["retained"] = atl.value / atl.clean; atl["drop"] = atl.clean - atl.value
    atl.to_csv(f"{OUT}/p1_atlas_union_AP.csv", index=False)
    PV = U[(U.metric == "prevalence") & (U.variant == "ID")].groupby("pid").value.mean(); PREV = PV.groupby(PV.index.str.split("_").str[1]).mean()   # atlas-mean ID prevalence
    pl = U[U.metric == "union_AP_planted"]
    if len(pl): assert pl.value.min() > float(os.environ.get("W6_PLANTED_MIN", 0.8)), ("planted known-signal test failed", pl.value.min())
    W = atl.pivot_table(index=["group", "atlas"], columns="method", values=["value", "clean", "retained", "drop"])
    rows = []
    for g in sorted(atl.group.unique()):
        if g == "ID" or g not in W.index.get_level_values(0): continue
        w = W.loc[g]; ok = w[("value", "ours")].notna() & w[("value", "leiden")].notna()
        if ok.sum() == 0: continue
        w = w[ok]; lvl = w[("value", "ours")] - w[("value", "leiden")]; dd = w[("drop", "leiden")] - w[("drop", "ours")]; rr = w[("retained", "ours")] - w[("retained", "leiden")]
        pv = PREV.reindex(w.index)   # chance-corrected retained (AP(r)-prev)/(AP(0)-prev); the floor is prevalence, not the other method's level
        ccr = lambda m: (w[("value", m)] - pv) / (w[("clean", m)] - pv); cc = ccr("ours") - ccr("leiden")
        rows.append(dict(group=g, n_atlas=int(ok.sum()), AP_clean_ours=w[("clean", "ours")].mean(), AP_clean_leiden=w[("clean", "leiden")].mean(), AP_ours=w[("value", "ours")].mean(), AP_leiden=w[("value", "leiden")].mean(), AP_scorer=w[("value", "scorer")].mean() if ("value", "scorer") in w else np.nan,
                         level_gap=lvl.mean(), p_level=signflip(lvl), drop_ours=w[("drop", "ours")].mean(), drop_leiden=w[("drop", "leiden")].mean(), D_drop_diff=dd.mean(), p_D=signflip(dd), retained_ours=w[("retained", "ours")].mean(), retained_leiden=w[("retained", "leiden")].mean(), retained_gap=rr.mean(), p_retained=signflip(rr),
                         n_atlas_D_pos=int((dd > 0).sum()), D_median=dd.median(), D_ci_lo=bootci(dd)[0], D_ci_hi=bootci(dd)[1], cc_retained_gap=cc.mean(), p_cc_retained=signflip(cc)))
    T = pd.DataFrame(rows)
    for fam, gs in (("random", ["R05", "R10", "R20"]), ("boundary", ["B05", "B10", "B20"])):
        m = T.group.isin(gs)
        for c in ("p_D", "p_level", "p_retained", "p_cc_retained"):
            if m.any(): T.loc[m, c + "_holm"] = holm(T.loc[m, c].fillna(1).values)   # Holm over the 3 rates within each family; HEADLINE = D(r) (pre-registered DESIGN 1.4), level gap co-reported
    T.to_csv(f"{OUT}/p1_summary.csv", index=False); print(T.round(3).to_string())
    # figure 1: union AP vs corruption rate
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for a, (fam, gs, ttl) in zip(ax[:2], (("random", ["ID", "R05", "R10", "R20"], "Random relabelling"), ("boundary", ["ID", "B05", "B10", "B20"], "Least-confident cells relabelled"))):
        x = [0, 5, 10, 20]
        for m, c, lab in (("ours", COL["ours"], "Our method"), ("leiden", COL["leiden"], "Leiden subclustering"), ("scorer", COL["scorer"], "Our scorer alone")):
            ys = [atl[(atl.group == g) & (atl.method == m)].value.mean() for g in gs]; a.plot(x, ys, "-o", color=c, label=lab, lw=2 if m != "scorer" else 1, ms=4)
            for at, d in atl[atl.method == m].groupby("atlas"):
                v = [d[d.group == g].value.mean() for g in gs]
                if m != "scorer": a.plot(x, v, "-", color=c, alpha=0.18, lw=0.8)
        a.set_title(ttl); a.set_xlabel("Labels corrupted (% of cells)"); a.set_xticks(x)
    ax[0].set_ylabel("Union AP vs clean-label targets"); ax[0].legend(frameon=False, fontsize=8)
    kk = ["ID", "KNNX", "KNNXu", "KNNT10", "KNNT2", "MERGE"]; lab = ["clean", "transfer\n(other pool)", "transfer,\nunassigned", "kNN 10%\nUMIs", "kNN 2%\nUMIs", "two labels\nmerged"]; wd = 0.38
    for i, (m, c) in enumerate((("ours", COL["ours"]), ("leiden", COL["leiden"]))):
        ax[2].bar(np.arange(len(kk)) + (i - .5) * wd, [atl[(atl.group == g) & (atl.method == m)].value.mean() if ((atl.group == g) & (atl.method == m)).any() else np.nan for g in kk], wd, color=c)
    ax[2].set_xticks(range(len(kk))); ax[2].set_xticklabels(lab, fontsize=7); ax[2].set_title("Label transfer and merged labels"); fig.tight_layout(); save(fig, "fig_w6_p1_union_ap_vs_corruption")
    # mislabel detection
    md = U[U.metric.isin(["mislabel_AP", "mislabel_prev"])].pivot_table(index=["variant", "pid"], columns=["arm", "metric"], values="value")
    if ("bl", "mislabel_AP") in md.columns:
        md = md.reset_index(); md["group"] = md.variant.map(grp); cols = [c for c in md.columns if c[1] == "mislabel_AP"]
        S = md.groupby("group")[[("bl", "mislabel_AP"), ("sc", "mislabel_AP"), ("m1", "mislabel_AP"), ("knn_dis", "mislabel_AP"), ("bl", "mislabel_prev")]].mean(); S.columns = ["AP_ours", "AP_scorer", "AP_leiden", "AP_knn_disagreement", "prevalence"]; S.to_csv(f"{OUT}/p1_mislabel_detection.csv"); print(S.round(3))
        fig, a = plt.subplots(figsize=(6.5, 3.2)); S2 = S.drop(index=["MERGE"], errors="ignore"); x = np.arange(len(S2)); w = 0.2
        for i, (c, k, l) in enumerate((("AP_ours", "ours", "Our method"), ("AP_leiden", "leiden", "Leiden subclustering"), ("AP_knn_disagreement", "knn", "Neighbour label disagreement"))): a.bar(x + (i - 1) * w, S2[c], w, color=COL[k], label=l)
        a.plot(x, S2.prevalence, "k_", ms=14, label="chance (prevalence)"); a.set_xticks(x); a.set_xticklabels(S2.index, rotation=45, fontsize=7); a.set_ylabel("AP for detecting relabelled cells"); a.legend(frameon=False, fontsize=7); save(fig, "fig_w6_p1_mislabel_detection")
    # extras: decoy capture, non-relabelled AP, MERGE both target sets, runtime
    X = U[U.metric.isin(["decoy_capture", "AP_nonrelabelled_pos", "union_AP_recomputed", "union_AP"]) & U.arm.isin(["bl", "m1", "sc"])].groupby(["group", "arm", "metric"]).value.mean().unstack("metric"); X.to_csv(f"{OUT}/p1_extras.csv")
    tls = glob.glob(os.path.expanduser("~/work/jiaqi1011/data/w6/transfer_log*.csv"))
    if tls: L = pd.concat([pd.read_csv(t) for t in tls]).drop_duplicates(["pid", "variant"]); L.groupby("variant")[["acc", "frac_changed"]].mean().to_csv(f"{OUT}/p1_label_transfer_accuracy.csv")
    # reproduction check ID rerun vs stored predictions
    st = f"{HOME}/work/r5/code/scores_preds_frozen4.csv"
    if os.path.exists(st):
        s = pd.read_csv(st); s = s[s.metric == "union_AP"].set_index("pilot_id").rprec; r = ap[(ap.variant == "ID") & (ap.arm == "bl")].set_index("pid").value
        pd.DataFrame(dict(rerun=r, stored=s.reindex(r.index))).assign(diff=lambda d: d.rerun - d.stored).to_csv(f"{OUT}/p1_repro_check_ID_vs_stored.csv"); print("ID rerun vs stored, mean abs diff", (r - s.reindex(r.index)).abs().mean())
    # permuted-score sanity (planted violation): permuted positives must give AP near prevalence
    # (done in p1_planted_check.py with real cands; see DESIGN)

# ------------------------------------------------------------------ P2
def p2():
    f = f"{OUT}/p2_pool.csv"
    if not os.path.exists(f): return
    R = pd.read_csv(f); R["atlas"] = R.pid.str.split("_").str[1]; cal = json.load(open(f"{OUT}/p2_t_cal.json"))
    A = R.groupby(["thr", "arm_", "atlas"])[["F1_bm", "F1_oo", "F1_alltypes", "ARI", "NMI", "n_classes", "n_groups_A1"]].mean().reset_index(); A.to_csv(f"{OUT}/p2_atlas.csv", index=False)
    S = A.groupby(["thr", "arm_"])[["F1_bm", "F1_oo", "F1_alltypes", "ARI", "NMI", "n_classes"]].mean().round(4); S.to_csv(f"{OUT}/p2_summary.csv"); print(S)
    rows = []
    for thr in A.thr.unique():
        for ref in ("A2s", "A2f", "A4", "A0"):
            for met in ("F1_bm", "F1_oo", "ARI", "NMI"):
                p = A[A.thr == thr].pivot(index="atlas", columns="arm_", values=met)
                if ref in p and "A1" in p: d = (p["A1"] - p[ref]); rows.append(dict(thr=thr, vs=ref, metric=met, mean_diff=d.mean(), n_atlas=int(d.notna().sum()), p=signflip(d)))
    T = pd.DataFrame(rows); 
    for thr in T.thr.unique():
        m = (T.thr == thr) & (T.vs != "A0") & T.metric.isin(["F1_bm", "F1_oo"]); T.loc[m, "p_holm"] = holm(T.loc[m, "p"].fillna(1).values)
    T.to_csv(f"{OUT}/p2_tests.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True); P = A[A.thr == "tcal"]; arms = ["A0", "A1", "A2s", "A2f", "A3", "A4"]; names = ["Broad label", "Broad + our groups", "Broad + Leiden small", "Broad + Leiden all", "Ours + Leiden small", "Random control"]   # 1:1 at print width
    cols = [COL["ctrl"], COL["ours"], COL["leiden"], COL["leiden"], "#5b8f5b", "#e0e0e0"]
    for a, met, ttl in zip(ax, ("F1_bm", "F1_oo", "ARI"), ("Best-match F1, hidden subtypes", "One-to-one F1, hidden subtypes", "ARI vs fine labels (pool)")):
        for i, ar in enumerate(arms):
            d = P[P.arm_ == ar]; y = len(arms) - 1 - i; a.barh(y, d[met].mean(), color=cols[i]); a.plot(d[met], np.full(len(d), y) + np.linspace(-.25, .25, len(d)), "k.", ms=2, alpha=.5)
        a.set_yticks(range(len(arms))[::-1]); a.set_yticklabels(names, fontsize=7); a.set_title(ttl, fontsize=7.5); a.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "fig_w6_p2_refinement")

# ------------------------------------------------------------------ P3
def p3():
    fs = glob.glob(f"{OUT}/p3/units/*.csv")
    if not fs: return
    U = pd.concat([pd.read_csv(f) for f in fs]); W = U.pivot_table(index=["variant", "pid"], columns=["arm", "metric"], values="value")
    keep = [c for c in W.columns if c[1] in ("union_AP", "E1_union", "n_targets", "K", "n_groups", "ood_value_share", "ood_group_share_any", "ood_group_share_gt25pct_features")]; T = W[keep]; T.to_csv(f"{OUT}/p3_table.csv"); print(T.round(3).to_string())
    M = U[(U.metric == "union_AP") & U.arm.isin(["bl", "m1", "sc"])].groupby(["variant", "arm"]).value.agg(["mean", "count"]).reset_index(); M.to_csv(f"{OUT}/p3_union_ap_by_arm.csv", index=False)
    lab = {"rna": "RNA", "ga": "Gene activity\n(ATAC)", "gak": "Gene activity,\nkNN labels"}; fig, a = plt.subplots(figsize=(3.3, 2.1)); w = .38   # 1:1 at column width
    for i, (arm, c, l) in enumerate((("bl", COL["ours"], "Our method"), ("m1", COL["leiden"], "Leiden subclustering"))):
        v = [M[(M.variant == k) & (M.arm == arm)]["mean"].mean() for k in lab]; a.bar(np.arange(3) + (i - .5) * w, v, w, color=c, label=l)
        for j, k in enumerate(lab):
            pts = U[(U.variant == k) & (U.arm == arm) & (U.metric == "union_AP")].value.values; a.plot(np.full(len(pts), j + (i - .5) * w), pts, "k.", ms=3)
    a.set_xticks(range(3)); a.set_xticklabels(list(lab.values()), fontsize=7); a.tick_params(labelsize=7); a.set_ylabel("Union AP (2 donor-disjoint pools)", fontsize=7); a.legend(frameon=False, fontsize=7); save(fig, "fig_w6_p3_atac_pilot")

if __name__ == "__main__":
    for f in (p1, p2, p3):
        try: f()
        except Exception as e: import traceback; traceback.print_exc(); print("FAILED", f.__name__)
