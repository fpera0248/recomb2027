"""Panel of public datasets: does any published generator beat plain duplication of the rare class?

Input: PANEL_DIR/<dataset>/clf.csv (every augmentation and readout) and real.csv (no augmentation), plus eligible.csv (dataset, tissue).
Probe rows only. The unit is the dataset; inside a dataset the unit is the probe type (8 per dataset), averaged over the seeds.

Per dataset and readout, d = recall of an augmentation minus recall of duplication, per (seed, type); exact two-sided sign-flip over the
types; BH within family. P2: one-sided studentized max-T sign-flip over every published generator and readout, family-wise error 0.05 per
dataset. P1: per augmentation and readout over datasets: mean d, bootstrap interval, sign-flip p with BH over the rows. P3: duplication
minus no augmentation. Metric rec200 is recall in the top-200 evaluation cells (the matched operating point), rec_argmax the default threshold.
"""
import glob
import os

import numpy as np
import pandas as pd

from .stats import bh, max_t, signflip_p

DELTA = 0.02   # non-inferiority margin in recall
FAM = {"published": ["scvi", "scvi_cond", "synth_w1", "synthbal_w1", "sdiff", "nb", "sd3_nosingle"], "latent": ["lat_gauss", "lat_boot"],
       "spread": ["jitter_s1.0", "jitternz_s1.0", "smote_k5"], "noaug": ["real"]}
FAMOF = {a: f for f, v in FAM.items() for a in v}
DEEP = ["scvi", "scvi_cond", "synth_w1", "synthbal_w1", "sdiff"]
SUB = {"all": lambda X: X, "no_n1": lambda X: X[X.n > 1], "n_le8": lambda X: X[X.n <= 8], "n_ge16": lambda X: X[X.n >= 16]}


def load(panel_dir, eligible):
    E = pd.read_csv(eligible).set_index("dataset_id")
    S = []
    for f in sorted(glob.glob(f"{panel_dir}/*/clf.csv")):
        R = os.path.dirname(f)
        x = pd.concat([pd.read_csv(f), pd.read_csv(f"{R}/real.csv")]).rename(columns={"arm": "method"})
        x = x[x.probe]
        x["ds"] = os.path.basename(R)
        x["tissue"] = E.tissue.get(x.ds.iloc[0], "n/a")
        S.append(x)
    S = pd.concat(S)
    assert set(S.method) <= set(FAMOF) | {"upsample"}, set(S.method) - set(FAMOF)
    assert not S.duplicated(["ds", "seed", "name_", "clf", "method"]).any()
    return S


def within(S, metric, sub):
    rows, P2 = [], []
    for ds, X in S.groupby("ds"):
        X = SUB[sub](X)
        w = X.pivot_table(index=["seed", "name_"], columns=["clf", "method"], values=metric)
        R, TM = [], {}
        for (k, a) in w.columns:
            if a == "upsample" or (k, "upsample") not in w.columns:
                continue
            m = w[(k, a)].notna() & w[(k, "upsample")].notna()
            dd = (w[(k, a)] - w[(k, "upsample")])[m]
            p, dt = signflip_p(dd.values, dd.index.get_level_values("name_").values)
            if FAMOF[a] == "published":
                TM[(k, a)] = dd.groupby(level="name_").mean()
            R.append(dict(ds=ds, tissue=X.tissue.iloc[0], metric=metric, sub=sub, clf=k, method=a, fam=FAMOF[a], d=float(dt.mean()), pos=int((dt > 0).sum()),
                          types=len(dt), p=p, mde=1.92 * dt.std(ddof=1), units=int(m.sum()), seeds=dd.index.get_level_values("seed").nunique()))
        if not R:
            continue
        R = pd.DataFrame(R)
        R["q"] = R.groupby("fam").p.transform(lambda s: bh(s.values))
        rows.append(R)
        if TM:
            pa = max_t(pd.DataFrame(TM))
            r = pa.idxmin()
            P2.append(dict(ds=ds, tissue=X.tissue.iloc[0], metric=metric, sub=sub, p_maxT=pa.min(), tests=len(pa), clf=r[0], method=r[1], d=float(TM[r].mean()), pos=int((TM[r] > 0).sum())))
    return pd.concat(rows), pd.DataFrame(P2)


def across(T):
    out = []
    for (mt, sb, k, a), g in T.groupby(["metric", "sub", "clf", "method"]):
        d = g.d.values
        b = np.random.default_rng(0).choice(d, (5000, len(d))).mean(1)
        p = signflip_p(d, np.arange(len(d)))[0] if len(d) > 1 else np.nan
        out.append(dict(metric=mt, sub=sb, clf=k, method=a, fam=FAMOF[a], n_ds=len(d), mean_d=d.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), hi95=np.percentile(b, 95),
                        median_d=np.median(d), p_ds=p, ds_win=int(((g.q < .05) & (g.d > 0)).sum()), ds_loss=int(((g.q < .05) & (g.d < 0)).sum())))
    A = pd.DataFrame(out)
    A["q_ds"] = A.groupby(["metric", "sub"]).p_ds.transform(lambda s: bh(s.fillna(1).values))
    A["noninf"] = A.hi95 < DELTA
    return A


def run(panel_dir, eligible, metrics=("rec200",), subs=("all",)):
    """-> per-dataset table, max-T table (one row per dataset), across-dataset table."""
    S = load(panel_dir, eligible)
    W = [within(S, m, sb) for m in metrics for sb in subs]
    T = pd.concat([w[0] for w in W])
    return T, pd.concat([w[1] for w in W]), across(T)
