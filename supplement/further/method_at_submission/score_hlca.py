"""python score_hlca.py PRED_ROOT OUT.csv : matched-k rprec (rprec.rprec, unchanged) per (HLCA pilot, method, target).
Targets (fixed before any HLCA prediction existed): within a pilot, every (visible label L, finest type T) with
T != L, T not in {None, nan, Unknown}, n(L,T) >= 10 and n(L,T) / n(L) < 0.10. is_rare = the pool cells with label L and finest type T.
One method score vector per pilot is scored against every target of that pilot."""
import os, sys, glob, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.path.insert(0, os.environ.get("SI_H") or os.path.join(os.environ.get("R5") or HR_WORK + "/r5/data", "hmod")); from hlib import _opaque_order, PILOTS_DIR
from rprec import rprec
BAD = {"None", "nan", "Unknown"}


def targets(df):
    nl = df.label.value_counts(); out = []
    for (L, T), n in df.groupby(["label", "true_type"]).size().items():
        if T != L and T not in BAD and n >= 10 and n / nl[L] < 0.10: out.append((L, T, int(n)))
    return out


def demo():
    df = pd.DataFrame({"label": ["a"] * 200 + ["b"] * 50, "true_type": ["a"] * 185 + ["x"] * 15 + ["b"] * 40 + ["y"] * 10})
    assert targets(df) == [("a", "x", 15)]       # y is 20% of b -> not rare


def main(root, out):
    rows, cache = [], {}
    for f in sorted(glob.glob(f"{root}/*/hlca_*.parquet")):
        meth, pid = f.split(os.sep)[-2], os.path.basename(f)[:-8]
        if pid not in cache:
            df = pd.read_parquet(f"{PILOTS_DIR}/{pid}.parquet", columns=["obs_name", "label", "true_type"]).astype({"label": str, "true_type": str})
            si, oid = _opaque_order(pid, df.obs_name.values); df = df.iloc[si].set_index(oid); cache[pid] = (df, targets(df))
        df, tg = cache[pid]; p = pd.read_parquet(f).set_index("cell_id").reindex(df.index); assert not p.score.isna().all(), f
        s = p.score.fillna(0).values
        for L, T, n in tg:
            rows.append((pid, meth, L, T, n, rprec(s, ((df.label == L) & (df.true_type == T)).values)))
    pd.DataFrame(rows, columns=["pilot_id", "method", "label", "target", "n_rare", "rprec"]).to_csv(out, index=False); print(len(rows), "rows")


if __name__ == "__main__":
    demo(); main(sys.argv[1], sys.argv[2])
