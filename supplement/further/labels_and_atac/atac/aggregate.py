"""python aggregate.py [--out OUT] : R1 tables and figure from OUT/<dataset>/arms/units/*.csv (run_unit.py output) and OUT/selection.csv (partial results are fine).
Methods: our method (blend, arm bl), scorer alone (sc), Leiden subclustering (m1), chance (pool prevalence of the target cells). Arms: rna, ga, gak (GA with coarse labels re-derived by kNN).
Per pool and per dataset (mean over pools) union AP; across datasets (units = datasets, <dataset>__pilot excluded): paired differences with exact two-sided sign-flip p; the kidney regression rerun (<dataset>__pilot) vs p3_union_ap_by_arm.csv.
Writes OUT/aggregate/{pool_union_ap,dataset_means,tests,kidney_regression}.csv and atac_all_union_ap.{pdf,png}. Descriptive, post hoc."""
import os, sys, glob, itertools, argparse, tempfile, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.dont_write_bytecode = True
HOME = os.path.expanduser("~"); NAME = {"bl": "our method", "sc": "scorer alone", "m1": "Leiden subclustering", "chance": "chance"}
COL = {"our method": "#1f5fa8", "scorer alone": "#6aa6d8", "Leiden subclustering": "#d9822b", "chance": "#b9b9b9"}

def signflip(d):
    """exact two-sided sign-flip test of mean(d) over datasets (n <= 20)."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3: return np.nan
    assert n <= 20; S = np.array(list(itertools.product([-1, 1], repeat=n))) @ d / n; return float((np.abs(S) >= abs(d.mean()) - 1e-12).mean())

def load(out):
    rows = []
    for f in glob.glob(f"{out}/*/arms/units/*.csv"):
        ds = f.split(os.sep)[-4]; u = pd.read_csv(f); u = u[((u.metric == "union_AP") & u.arm.isin(["bl", "sc", "m1"])) | ((u.metric == "prevalence") & (u.arm == "bl"))]
        for _, r in u.iterrows(): rows.append((ds, r.pid, r.variant, "chance" if r.metric == "prevalence" else NAME[r.arm], r.value))
    L = pd.DataFrame(rows, columns=["dataset", "pool", "arm", "method", "ap"]); has = L[L.method != "chance"][["dataset", "pool", "arm"]].drop_duplicates()
    return L.merge(has, on=["dataset", "pool", "arm"])    # drops chance rows of pools without target pairs (no union AP for any method)

def tests(dm):
    """dm: dataset x (arm, method) mean AP. Paired contrasts across datasets."""
    W = dm.pivot_table(index="dataset", columns=["arm", "method"], values="ap"); rows = []
    def add(name, a, b):
        if a not in W.columns or b not in W.columns: return
        d = (W[a] - W[b]).dropna()
        rows.append(dict(contrast=name, n_datasets=len(d), mean_diff=d.mean() if len(d) else np.nan, wins=int((d > 0).sum()), ties=int((d == 0).sum()), p_exact_signflip=signflip(d.values)))
    for ar in ("ga", "gak", "rna"):
        add(f"our method minus Leiden subclustering ({ar})", (ar, "our method"), (ar, "Leiden subclustering")); add(f"our method minus chance ({ar})", (ar, "our method"), (ar, "chance"))
        add(f"scorer alone minus Leiden subclustering ({ar})", (ar, "scorer alone"), (ar, "Leiden subclustering"))
    add("our method: RNA minus gene activity", ("rna", "our method"), ("ga", "our method")); add("Leiden subclustering: RNA minus gene activity", ("rna", "Leiden subclustering"), ("ga", "Leiden subclustering"))
    return pd.DataFrame(rows)

def figure(dm, T, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
    W = dm.pivot_table(index="dataset", columns=["arm", "method"], values="ap"); fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, ar, ttl in zip(ax[:2], ("ga", "rna"), ("Gene activity from chromatin accessibility", "RNA of the same cells")):
        if ar not in W.columns.get_level_values(0): a.set_visible(False); continue
        w = W[ar].dropna(subset=["our method", "Leiden subclustering"])
        for ds, r in w.iterrows(): a.plot([0, 1], [r["Leiden subclustering"], r["our method"]], "-o", color="#8a8a8a", lw=1, ms=4, alpha=.8)
        for x, m in enumerate(("Leiden subclustering", "our method")): a.plot([x - .18, x + .18], [w[m].mean()] * 2, color=COL[m], lw=3)
        a.scatter([0.5] * len(w), w["chance"], marker="_", s=120, color=COL["chance"], label="chance (per dataset)"); a.set_xticks([0, 1]); a.set_xticklabels(["Leiden\nsubclustering", "our method"]); a.set_ylabel("union AP"); a.set_title(ttl, fontsize=10)
        t = T[T.contrast == f"our method minus Leiden subclustering ({ar})"]
        if len(t): a.text(.02, .97, f"n = {int(t.n_datasets.iloc[0])} datasets, mean diff {t.mean_diff.iloc[0]:+.3f}\nexact sign-flip p = {t.p_exact_signflip.iloc[0]:.3f}", transform=a.transAxes, va="top", fontsize=8)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    if ("rna", "our method") in W.columns and ("ga", "our method") in W.columns:
        w = W[[("rna", "our method"), ("ga", "our method")]].dropna(); ax[2].scatter(w[("ga", "our method")], w[("rna", "our method")], color=COL["our method"]); m = max(w.max().max(), .05) * 1.05
        ax[2].plot([0, m], [0, m], color="#b9b9b9", lw=1); ax[2].set_xlabel("union AP, gene activity"); ax[2].set_ylabel("union AP, RNA"); ax[2].set_title("Our method, one point per multiome dataset", fontsize=10)
    else: ax[2].set_visible(False)
    fig.tight_layout(); fig.savefig(path + ".pdf", bbox_inches="tight"); fig.savefig(path + ".png", bbox_inches="tight", dpi=200); plt.close(fig)

def main(out):
    L = load(out); assert len(L), f"no unit csvs under {out}"; os.makedirs(f"{out}/aggregate", exist_ok=True); main_ = L[~L.dataset.str.endswith("__pilot")]
    pool = main_.pivot_table(index=["dataset", "pool", "arm"], columns="method", values="ap").reset_index(); pool.to_csv(f"{out}/aggregate/pool_union_ap.csv", index=False)
    dm = main_.groupby(["dataset", "arm", "method"]).ap.agg(ap="mean", n_pools="count").reset_index()
    if os.path.exists(f"{out}/selection.csv"): s = pd.read_csv(f"{out}/selection.csv")[["dataset_id", "coarse", "fine", "multiome", "zero_shot", "n_targets"]].rename(columns={"dataset_id": "dataset"}); dm = dm.merge(s, on="dataset", how="left")
    dm.to_csv(f"{out}/aggregate/dataset_means.csv", index=False); T = tests(dm).assign(subset="all datasets")
    if "zero_shot" in dm: z = dm[dm.zero_shot == "zero-shot"]; T = pd.concat([T, tests(z).assign(subset="zero-shot datasets only")]) if len(z) else T
    T.to_csv(f"{out}/aggregate/tests.csv", index=False); print(T.to_string(index=False)); figure(dm, T, f"{out}/aggregate/atac_all_union_ap")
    P = L[L.dataset.str.endswith("__pilot")]; ref = f"{HR_WORK}/jiaqi1011/out/w6_annot_atac/p3_union_ap_by_arm.csv"
    if len(P) and os.path.exists(ref):
        r = pd.read_csv(ref); r = r.assign(method=r.arm.map(NAME)).drop(columns="arm"); r = r.rename(columns={"variant": "arm", "mean": "published"}); m = P.groupby(["arm", "method"]).ap.mean().reset_index().rename(columns={"ap": "rerun"})
        K = m.merge(r[["arm", "method", "published"]], on=["arm", "method"], how="left"); K["diff"] = K.rerun - K.published; K.to_csv(f"{out}/aggregate/kidney_regression.csv", index=False); print(K.to_string(index=False))

def demo():
    assert abs(signflip([1] * 8) - 2 / 256) < 1e-12 and np.isnan(signflip([1, 2]))
    d = tempfile.mkdtemp(); rng = np.random.default_rng(0)
    for i in range(6):
        for k in range(2):
            for v in ("ga", "rna"):
                if v == "rna" and i == 5: continue
                os.makedirs(f"{d}/ds{i}/arms/units", exist_ok=True); b = .05 + .03 * (v == "rna") + rng.normal(0, .005)
                pd.DataFrame([(v, f"p{k}", "bl", "union_AP", b + .02), (v, f"p{k}", "sc", "union_AP", b), (v, f"p{k}", "m1", "union_AP", b), (v, f"p{k}", "bl", "prevalence", .01)], columns=["variant", "pid", "arm", "metric", "value"]).to_csv(f"{d}/ds{i}/arms/units/{v}__p{k}.csv", index=False)
    L = load(d); dm = L.groupby(["dataset", "arm", "method"]).ap.mean().reset_index(); T = tests(dm).set_index("contrast")
    c = T.loc["our method minus Leiden subclustering (ga)"]; assert c.n_datasets == 6 and abs(c.mean_diff - .02) < 1e-9 and c.wins == 6 and abs(c.p_exact_signflip - 2 / 64) < 1e-12
    assert T.loc["our method: RNA minus gene activity"].n_datasets == 5 and abs(T.loc["our method minus chance (ga)"].mean_diff - (.05 + .02 - .01)) < .01
    # pilot rerun path + pool without targets (chance dropped) + zero-shot subset
    os.makedirs(f"{d}/ds0__pilot/arms/units", exist_ok=True); pd.DataFrame([("ga", "p0", "bl", "union_AP", .07), ("ga", "p0", "sc", "union_AP", .06), ("ga", "p0", "m1", "union_AP", .05)], columns=["variant", "pid", "arm", "metric", "value"]).to_csv(f"{d}/ds0__pilot/arms/units/ga__p0.csv", index=False)
    pd.DataFrame([("ga", "p9", "bl", "prevalence", 0.0)], columns=["variant", "pid", "arm", "metric", "value"]).to_csv(f"{d}/ds1/arms/units/ga__p9.csv", index=False)
    assert "p9" not in set(load(d).pool); pd.DataFrame(dict(dataset_id=[f"ds{i}" for i in range(6)], coarse="c", fine="f", multiome=True, zero_shot=["zero-shot"] * 4 + ["not zero-shot on study"] * 2, n_targets=3)).to_csv(f"{d}/selection.csv", index=False)
    main(d); assert os.path.exists(f"{d}/aggregate/atac_all_union_ap.png") and os.path.exists(f"{d}/aggregate/kidney_regression.csv")
    assert (pd.read_csv(f"{d}/aggregate/tests.csv").subset == "zero-shot datasets only").any()

if __name__ == "__main__":
    if len(sys.argv) == 1: demo(); print("demo ok"); sys.exit(0)
    p = argparse.ArgumentParser(); p.add_argument("--out", default=f"{HR_WORK}/jiaqi1011/out/atac_all"); main(p.parse_args().out)
