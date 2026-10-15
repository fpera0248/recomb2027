"""Metrics per (pool, method) from saved cell scores.

    python -m eval.score_atl PRED_ROOT OUT.csv PILOT_DIR MANIFEST.csv [MANIFEST.csv ...]

PRED_ROOT/<method>/<pool>.parquet holds cell_id and score. K is the number of target cells in
the pool (union of all targets); ties are split at their expected value.
Pool rows: E1_union (precision in the top K), union_AP, E1_dontcare (near-miss subtype cells
removed: T != L, not a target, under 20% of the label), E1_halfK, E1_2K.
Target rows: masked_AP and masked_rprec (the other targets' cells removed, each target scored
alone), det50 (at least half of the target's cells in the top K), recall_at_K, E2_within_label,
per_target.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from hiddenrare.io import opaque_order, rprec

from .targets import BAD, targets


def in_top(s, K):
    """Expected membership of each cell in the top K, ties split evenly."""
    t = np.sort(s)[::-1][K - 1]
    above, tie = s > t, s == t
    return above + tie * (K - above.sum()) / tie.sum()


def near_miss(df, tg):
    nl = df.label.value_counts()
    tset = {(lab, typ) for lab, typ, _ in tg}
    m = np.zeros(len(df), bool)
    for (lab, typ), n in df.groupby(["label", "true_type"]).size().items():
        if (
            typ != lab
            and typ not in BAD
            and (lab, typ) not in tset
            and n / nl[lab] < 0.20
        ):
            m |= ((df.label == lab) & (df.true_type == typ)).values
    return m


def score(df, tg, s, pid, meth):
    rows = []
    tm = [((df.label == lab) & (df.true_type == typ)).values for lab, typ, _ in tg]
    u = np.any(tm, 0)
    K = int(u.sum())
    if not K:
        return rows
    top = in_top(s, K)
    for (lab, typ, n), t in zip(tg, tm):
        m = (df.label == lab).values
        keep = ~(u & ~t)
        r = top[t].sum() / n
        for met, v in (
            ("E2_within_label", rprec(s[m], t[m])),
            ("per_target", rprec(s, t)),
            ("masked_rprec", rprec(s[keep], t[keep])),
            ("masked_AP", average_precision_score(t[keep], s[keep])),
            ("recall_at_K", r),
            ("det50", float(r >= 0.5)),
        ):
            rows.append((pid, meth, met, lab, typ, n, v))
    nm = ~near_miss(df, tg)
    for met, v in (
        ("E1_union", rprec(s, u)),
        ("union_AP", average_precision_score(u, s)),
        ("E1_dontcare", rprec(s[nm], u[nm])),
        ("E1_halfK", in_top(s, max(K // 2, 1))[u].sum() / max(K // 2, 1)),
        ("E1_2K", in_top(s, min(2 * K, len(s)))[u].sum() / K),
    ):
        rows.append((pid, meth, met, "*", "*", K, v))
    return rows


def truth_table(pilot_dir, pid):
    df = pd.read_parquet(
        f"{pilot_dir}/{pid}.parquet", columns=["obs_name", "label", "true_type"]
    ).astype({"label": str, "true_type": str})
    order, ids = opaque_order(pid, df.obs_name.values)
    df = df.iloc[order].set_index(ids)
    return df, targets(df)


def _check():
    df = pd.DataFrame(
        {
            "label": ["a"] * 200,
            "true_type": ["a"] * 170 + ["x"] * 15 + ["y"] * 10 + ["z"] * 5,
        }
    )
    tg = targets(df)
    assert sorted(t for _, t, _ in tg) == ["x", "y"]
    s = np.r_[np.zeros(170), np.full(15, 3.0), np.full(10, 2.0), np.full(5, 1.0)]
    r = {(m, t): v for _, _, m, _, t, _, v in score(df, tg, s, "p", "m")}
    assert (
        r[("E1_union", "*")] == 1
        and r[("masked_AP", "y")] == 1
        and r[("det50", "x")] == 1
        and r[("E1_dontcare", "*")] == 1
    )
    assert abs(in_top(np.array([1.0, 1, 1, 0]), 2).sum() - 2) < 1e-12


if __name__ == "__main__":
    _check()
    root, out, pilot_dir = sys.argv[1:4]
    pools = set(pd.concat([pd.read_csv(f) for f in sys.argv[4:]]).pilot_id)
    rows, cache = [], {}
    for f in sorted(glob.glob(f"{root}/*/atl_*.parquet")):
        meth, pid = f.split(os.sep)[-2], os.path.basename(f)[:-8]
        if pid not in pools:
            continue
        if pid not in cache:
            cache[pid] = truth_table(pilot_dir, pid)
        df, tg = cache[pid]
        s = (
            pd.read_parquet(f)
            .set_index("cell_id")
            .reindex(df.index)
            .score.fillna(0)
            .values
        )
        rows += score(df, tg, s, pid, meth)
    pd.DataFrame(
        rows,
        columns=["pilot_id", "method", "metric", "label", "target", "n_rare", "rprec"],
    ).to_csv(out, index=False)
    print(len(rows), "rows")
