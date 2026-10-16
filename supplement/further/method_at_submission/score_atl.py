"""python score_atl.py PRED_ROOT OUT.csv MANIFEST.csv [MANIFEST.csv ...] : metrics per (atlas pilot, method) (PREREG_ATLAS.md + Amendment 1).
Targets = score_hlca.targets (unchanged). K = union of all target cells in the pool. Ties are split at expected value (as rprec.py).
Pool rows:   E1_union (PRIMARY, matched-k precision on the union), union_AP, E1_dontcare (near-miss subtype cells removed: T != L, T not BAD,
             not a target, share < 0.20), E1_halfK (precision in the top K/2), E1_2K (union recall in the top 2K).
Target rows: masked_AP and masked_rprec (KEY SECONDARY: the other targets' cells removed, so each target is scored alone, ceiling 1, equal weight),
             det50 (1 if >= 50% of the target's cells are in the top K), recall_at_K, E2_within_label, per_target (shared ranking; ceiling < 1)."""
import os, sys, glob, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.environ.get("SI_H") or os.path.join(os.environ.get("R5") or os.path.expanduser("~/work/r5/data"), "hmod")); from hlib import _opaque_order, PILOTS_DIR
from rprec import rprec
from score_hlca import targets, BAD


def in_top(s, K):
    """expected membership of each cell in the top K (ties split evenly)."""
    t = np.sort(s)[::-1][K - 1]; above, tie = s > t, s == t
    return above + tie * (K - above.sum()) / tie.sum()


def near_miss(df, tg):
    nl = df.label.value_counts(); tset = {(L, T) for L, T, _ in tg}; m = np.zeros(len(df), bool)
    for (L, T), n in df.groupby(["label", "true_type"]).size().items():
        if T != L and T not in BAD and (L, T) not in tset and n / nl[L] < 0.20: m |= ((df.label == L) & (df.true_type == T)).values
    return m


def score(df, tg, s, pid, meth):
    rows = []; tm = [((df.label == L) & (df.true_type == T)).values for L, T, _ in tg]; u = np.any(tm, 0); K = int(u.sum())
    if not K: return rows
    top = in_top(s, K)
    for (L, T, n), t in zip(tg, tm):
        m = (df.label == L).values; keep = ~(u & ~t); r = top[t].sum() / n
        for met, v in (("E2_within_label", rprec(s[m], t[m])), ("per_target", rprec(s, t)), ("masked_rprec", rprec(s[keep], t[keep])),
                       ("masked_AP", average_precision_score(t[keep], s[keep])), ("recall_at_K", r), ("det50", float(r >= 0.5))):
            rows.append((pid, meth, met, L, T, n, v))
    nm = ~near_miss(df, tg)
    for met, v in (("E1_union", rprec(s, u)), ("union_AP", average_precision_score(u, s)), ("E1_dontcare", rprec(s[nm], u[nm])),
                   ("E1_halfK", in_top(s, max(K // 2, 1))[u].sum() / max(K // 2, 1)), ("E1_2K", in_top(s, min(2 * K, len(s)))[u].sum() / K)):
        rows.append((pid, meth, met, "*", "*", K, v))
    return rows


def demo():
    df = pd.DataFrame({"label": ["a"] * 200, "true_type": ["a"] * 170 + ["x"] * 15 + ["y"] * 10 + ["z"] * 5})
    tg = targets(df); assert sorted(t for _, t, _ in tg) == ["x", "y"]
    s = np.r_[np.zeros(170), np.full(15, 3.0), np.full(10, 2.0), np.full(5, 1.0)]
    r = {(m, t): v for _, _, m, _, t, _, v in score(df, tg, s, "p", "m")}
    assert r[("E1_union", "*")] == 1 and r[("masked_AP", "y")] == 1 and r[("det50", "x")] == 1 and r[("E1_dontcare", "*")] == 1
    assert abs(in_top(np.array([1., 1, 1, 0]), 2).sum() - 2) < 1e-12


if __name__ == "__main__":
    demo(); man = set(pd.concat([pd.read_csv(f) for f in sys.argv[3:]]).pilot_id); rows, cache = [], {}
    for f in sorted(glob.glob(f"{sys.argv[1]}/*/atl_*.parquet")):
        meth, pid = f.split(os.sep)[-2], os.path.basename(f)[:-8]
        if pid not in man: continue
        if pid not in cache:
            df = pd.read_parquet(f"{PILOTS_DIR}/{pid}.parquet", columns=["obs_name", "label", "true_type"]).astype({"label": str, "true_type": str})
            si, oid = _opaque_order(pid, df.obs_name.values); df = df.iloc[si].set_index(oid); cache[pid] = (df, targets(df))
        df, tg = cache[pid]; rows += score(df, tg, pd.read_parquet(f).set_index("cell_id").reindex(df.index).score.fillna(0).values, pid, meth)
    pd.DataFrame(rows, columns=["pilot_id", "method", "metric", "label", "target", "n_rare", "rprec"]).to_csv(sys.argv[2], index=False); print(len(rows), "rows")
