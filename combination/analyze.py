"""Paired test on run_one.py rows, one classifier per call.
  python -m combination.analyze 'ROWS_GLOB' OUT.csv plan.csv [lr|rf|mlp]
Seeds are averaged within a pool and pools are weighted so every atlas counts equally: weighted pool-level sign-flip (20,000 draws) and the exact sign-flip over atlas mean differences.
Completeness is checked against plan.csv first. Family: P primary, S secondary (Holm-adjusted atlas exact p), D descriptive, N non-inferiority
(one-sided 95% t lower bound of the atlas-mean differences, to be compared with -0.01). HR_DATA holds manifests/.
"""

import sys, glob, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy import stats
from eval.analyze import exact
from eval.paper import weights, pool_flip, holm as paper_holm

ROOT = os.environ.get("HR_DATA", "assets")
# (A, B, metric, family). Cross-method recall uses rprec; rec_at_kref only for duplication against none, where K is shared.
CON = [
    ("ours_dup", "leiden_dup", "ap", "P"),
    ("ours_dup", "leiden_dup", "rprec", "S"),
    ("ours_dup", "leiden_collapsed_dup", "ap", "S"),
    ("ours_dup", "ours_nodup", "ap", "S"),
    ("ours_dup", "ours_nodup", "rec_at_kref", "S"),
    ("ours_dup", "ours_nodup", "recall_argmax", "D"),
    ("leiden_dup", "leiden_nodup", "rec_at_kref", "D"),
    ("true_dup", "true_nodup", "rec_at_kref", "D"),
    ("true_dup", "ours_dup", "ap", "D"),
    ("true_dup", "ours_dup", "rprec", "D"),
    ("ours_dup", "leiden_dup", "macro_f1_broad", "D"),
    ("ours_dup", "broad", "macro_f1_broad", "N"),
    ("leiden_dup", "broad", "macro_f1_broad", "N"),
]


def holm(p):
    h = paper_holm(dict(enumerate(np.asarray(p, float))))
    return np.array([h[i] for i in range(len(h))])


def check(d, plan, clf):
    exp = plan[plan.eligible][["pool", "seed"]].drop_duplicates()
    got = d[(d.clf == clf)][["pool", "seed", "status"]].drop_duplicates()
    ok = got[got.status == "ok"][["pool", "seed"]]
    miss = exp.merge(ok, how="left", indicator=True).query("_merge=='left_only'")
    extra = ok.merge(exp, how="left", indicator=True).query("_merge=='left_only'")
    drop = got[got.status.str.startswith("dropped")]
    print(
        f"expected {len(exp)} eligible pool-seeds, ok {len(ok)}, missing {len(miss)}, unexpected {len(extra)}, dropped {len(drop)}"
    )
    if len(drop):
        print("dropped (cell ids differ):", sorted(set(drop.pool)))
    assert (
        len(miss) == 0 and len(extra) == 0
    ), f"incomplete run: missing {miss.values.tolist()[:10]}, unexpected {extra.values.tolist()[:10]}"
    if "hit_iter_cap" in d:
        print(
            "fits at the lr iteration cap:",
            int(d[d.clf == clf].hit_iter_cap.fillna(False).astype(bool).sum()),
        )


def compare(v, man):
    s = v.dropna()
    at = man.loc[s.index, "atlas"]
    wt = weights(at)
    dd = (s.ours - s.leiden).values
    A = (s.ours - s.leiden).groupby(at.values).mean()
    return dict(
        n_pools=len(s),
        n_atlases=len(A),
        A_mean=round(float(wt @ s.ours), 4),
        B_mean=round(float(wt @ s.leiden), 4),
        diff=round(float(wt @ dd), 4),
        pool_flip_p=round(pool_flip(dd, wt), 5),
        pool_weighted_diff=round(float(dd.mean()), 4),
        atlas_ahead=int((A > 0).sum()),
        atlas_exact_p=round(exact(A.values), 5),
    )


def main(pat, out, planf, clf="lr"):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))])
    d = d[d.clf == clf]
    check(d, pd.read_csv(planf), clf)
    d = d[d.status == "ok"]
    man = pd.concat(
        [
            pd.read_csv(f)
            for f in sorted(glob.glob(f"{ROOT}/manifests/atlas_manifest_*.csv"))
        ]
    ).set_index("pilot_id")
    res = []
    for A, B, met, fam in CON:
        w = (
            d[d.setting.isin([A, B])].groupby(["pool", "setting"])[met].mean().unstack()
        )  # seed mean within pool
        if A not in w or B not in w:
            continue
        w = w.dropna()
        if w.empty:
            continue
        w.index.name = "pilot_id"
        v = w.rename(columns={A: "ours", B: "leiden"})
        r = pd.DataFrame([compare(v, man)])
        am = (v.ours - v.leiden).groupby(man.loc[v.index, "atlas"].values).mean()
        n = len(am)
        r["lower95_one_sided"] = (
            am.mean() - stats.t.ppf(0.95, n - 1) * am.std(ddof=1) / np.sqrt(n)
            if n > 1
            else np.nan
        )
        r.insert(0, "clf", clf)
        r.insert(1, "A", A)
        r.insert(2, "B", B)
        r.insert(3, "metric", met)
        r["family"] = fam
        res.append(r)
    r = pd.concat(res, ignore_index=True)
    s = r.family == "S"
    r["atlas_exact_p_holm"] = np.nan
    r.loc[s, "atlas_exact_p_holm"] = holm(r.loc[s, "atlas_exact_p"].values)
    r.to_csv(out, index=False)
    print(
        r[
            [
                "family",
                "A",
                "B",
                "metric",
                "n_pools",
                "n_atlases",
                "diff",
                "pool_flip_p",
                "atlas_ahead",
                "atlas_exact_p",
                "atlas_exact_p_holm",
                "lower95_one_sided",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main(*sys.argv[1:5])
