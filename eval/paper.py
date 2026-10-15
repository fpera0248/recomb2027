"""Held-out test, published-tool comparison and literature metrics from saved cell scores.

Inputs are the release assets, extracted under one root: pilots/, manifests/, pools.csv, predictions/<method>/<pool>.parquet,
tool_out/<tool>/<pool>.parquet. Nothing is refit.
"""

import glob

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

from .analyze import compare, exact
from .score_atl import score, truth_table

OURS = "hiddenrare"
METRICS = ("union_AP", "E1_union")


def pools(root):
    """Returns pool ids and their manifest (atlas of each pool)."""
    pids = pd.read_csv(f"{root}/pools.csv").pilot_id.tolist()
    man = pd.concat(
        [
            pd.read_csv(f)
            for f in sorted(glob.glob(f"{root}/manifests/atlas_manifest_*.csv"))
        ]
    ).set_index("pilot_id")
    return pids, man.loc[pids]


def read_scores(path, df):
    p = pd.read_parquet(path).set_index("cell_id")
    assert set(p.index) == set(df.index), (path, "cell ids differ from the pool")
    return p.reindex(df.index)


def pool_metrics(root, methods, pids=None, call=False):
    """methods: {name: directory with <pool>.parquet}. Returns a long table (pilot_id, method, metric, value) of union AP and E1.
    call=True ranks cells by the method's declared call (calls first, then by score) instead of its saved score.
    """
    pids = pids or pools(root)[0]
    rows = []
    for pid in pids:
        df, tg = truth_table(f"{root}/pilots", pid)
        for name, d in methods.items():
            p = read_scores(f"{d}/{pid}.parquet", df)
            s = p.score.fillna(0).values
            if call:
                s = p.call.astype(float).values * (np.abs(s).max() * 2 + 1) + s
            rows += [
                (pid, name, r[2], r[6])
                for r in score(df, tg, s, pid, name)
                if r[3] == "*" and r[2] in METRICS
            ]
    return pd.DataFrame(rows, columns=["pilot_id", "method", "metric", "value"])


def wide(table, metric):
    return table[table.metric == metric].pivot(
        index="pilot_id", columns="method", values="value"
    )


def atlas_equal(w, man):
    """Mean over atlases of the per-atlas mean, per method."""
    return w.groupby(man.loc[w.index, "atlas"].values).mean().mean()


def weights(atlas):
    return (1 / atlas.map(atlas.value_counts()) / atlas.nunique()).values


def pool_flip(diff, w, n=20000):
    """Pool-level weighted sign-flip p, 20,000 draws, seed 0; it cannot fall below 1/(n+1)."""
    s = np.random.RandomState(0).choice([-1.0, 1.0], (n, len(diff)))
    return (
        int((np.abs((s * (w * diff)).sum(1)) >= abs((w * diff).sum()) - 1e-12).sum())
        + 1
    ) / (n + 1)


def two_level_boot(w, a, b, atlas, B=2000, seed=0):
    """Percentile 95% interval of the atlas-equal mean difference: resample atlases, then pools within each sampled atlas."""
    rng = np.random.RandomState(seed)
    g = [
        (w.loc[v.index, a] - w.loc[v.index, b]).values for _, v in atlas.groupby(atlas)
    ]
    out = [
        np.mean(
            [
                g[i][rng.randint(0, len(g[i]), len(g[i]))].mean()
                for i in rng.randint(0, len(g), len(g))
            ]
        )
        for _ in range(B)
    ]
    return np.percentile(out, [2.5, 97.5])


def holm(p):
    adj, run = {}, 0.0
    for i, k in enumerate(sorted(p, key=p.get)):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def versus(w, a, b, man):
    """Atlas-level exact sign-flip (all 2^A patterns) plus the pool-weighted flip, bootstrap interval and leave-one-atlas-out range."""
    atlas = man.loc[w.index, "atlas"]
    r = compare(w, a, b, atlas)
    pa = pd.Series(r["per_atlas_diff"])
    d = (w[a] - w[b]).values
    r["p_pool_flip"] = pool_flip(d, weights(atlas))
    r["ci"] = two_level_boot(w, a, b, atlas)
    r["loo"] = (
        min(pa.drop(k).mean() for k in pa.index),
        max(pa.drop(k).mean() for k in pa.index),
    )
    r["atlas_sd"] = float(pa.std())
    return r


def binary(y, c):
    tp, fp, fn, tn = (y & c).sum(), (~y & c).sum(), (y & ~c).sum(), (~y & ~c).sum()
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    ok = c.any() and (~c).any()
    return dict(
        P=P,
        R=R,
        F1=2 * P * R / max(P + R, 1e-12),
        Gmean_P_R=np.sqrt(P * R),
        MCC=matthews_corrcoef(y, c) if ok else 0.0,
        kappa=cohen_kappa_score(y, c) if ok else 0.0,
        ARI_bin=adjusted_rand_score(y, c),
        AMI_bin=adjusted_mutual_info_score(y, c),
    )


def literature_metrics(root, methods, pids=None):
    """Metrics of the tool papers on every pool. K_*: binary call of the top K cells (K = number of hidden cells; ties broken by a seeded
    draw), own_*: the tool's declared calls, part_*: partition against the hidden subtypes. Returns a long table (pilot_id, method, metric, value).
    """
    pids = pids or pools(root)[0]
    rows = []
    for pid in pids:
        df, tg = truth_table(f"{root}/pilots", pid)
        tm = [((df.label == lab) & (df.true_type == typ)).values for lab, typ, _ in tg]
        u = np.any(tm, 0)
        K = int(u.sum())
        if not K:
            continue
        for name, d in methods.items():
            p = read_scores(f"{d}/{pid}.parquet", df)
            s = p.score.fillna(0).values
            for r in score(df, tg, s, pid, name):
                if r[3] == "*" and r[2] in METRICS:
                    rows.append((pid, name, r[2], r[6]))
            c = np.zeros(len(s), bool)
            c[np.lexsort((np.random.RandomState(0).rand(len(s)), -s))[:K]] = True
            rows += [(pid, name, "K_" + k, v) for k, v in binary(u, c).items()]
            if "call" in p and p.call.notna().all():
                c2 = p.call.values.astype(bool)
                rows += [(pid, name, "own_" + k, v) for k, v in binary(u, c2).items()]
                rows.append((pid, name, "own_ncalled", c2.sum() / K))
            if "group" in p:
                tt, g = (
                    pd.factorize(df.label + "|" + df.true_type)[0],
                    pd.factorize(p.group.astype(str))[0],
                )
                rows += [
                    (pid, name, "part_ARI_vs_finetype", adjusted_rand_score(tt, g)),
                    (
                        pid,
                        name,
                        "part_AMI_vs_finetype",
                        adjusted_mutual_info_score(tt, g),
                    ),
                ]
    return pd.DataFrame(rows, columns=["pilot_id", "method", "metric", "value"])


def versus_table(table, man, a=OURS):
    """Atlas-level paired comparison of method a against every other method, for every metric of the table."""
    out = []
    for met in sorted(table.metric.unique()):
        w = wide(table, met)
        if a not in w:
            continue
        for b in w.columns:
            if b == a or len(w[[a, b]].dropna()) < 5:
                continue
            r = compare(w, a, b, man["atlas"])
            out.append(
                dict(
                    metric=met,
                    vs=b,
                    n_pools=r["n_pools"],
                    n_atlases=r["n_atlases"],
                    ours_atlas_equal=r["a_mean"],
                    other_atlas_equal=r["b_mean"],
                    diff=r["diff"],
                    atlas_ahead=r["atlas_ahead"],
                    atlas_exact_p=round(r["p"], 5),
                )
            )
    return pd.DataFrame(out)


if __name__ == "__main__":
    assert abs(exact([1, 1, 1]) - 0.25) < 1e-12 and holm({"a": 0.01, "b": 0.04}) == {
        "a": 0.02,
        "b": 0.04,
    }
    assert pool_flip(np.ones(12), np.ones(12) / 12) < 0.002
    a = pd.Series(list("aabbb"))
    assert abs(weights(a).sum() - 1) < 1e-12 and weights(a)[0] == 0.25
    c = binary(np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 0], bool))
    assert c["P"] == 1 and c["R"] == 0.5
