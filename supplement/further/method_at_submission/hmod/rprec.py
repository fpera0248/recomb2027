"""Matched-operating-point recall for the hidden-rare pilots. For every saved
prediction preds/<method>/<pilot_id>.parquet, call exactly k = n_rare_true
cells (the top-k scores; ties at the k-th score are split at their expected
value) and report recall at k (= precision at k, R-precision). Null pilots
(no rare cells) get NaN. Truth comes from the pilot parquet via the same
opaque ids hlib.load_pilot assigns, so no h5ad is opened.
Two rankings per prediction: rprec ranks by the saved score; rprec_call ranks
by the method's own call set only (called cells first, ties split at expected
value), because the loop methods' saved score is a last-round uncertainty
against their own pseudo-labels, not their output.
Usage: python rprec.py OUT.csv"""
import os, sys, glob
import numpy as np
import pandas as pd
from hlib import _opaque_order, H


def rprec(score, is_rare):
    k = int(is_rare.sum())
    if k == 0:
        return np.nan
    t = np.sort(score)[::-1][k - 1]
    above, tie = score > t, score == t
    return (is_rare[above].sum() + (k - above.sum()) * is_rare[tie].mean()) / k


def demo():
    r = np.array([1, 1, 0, 0, 0], bool)
    assert rprec(np.array([.9, .8, .1, .2, .3]), r) == 1.0
    assert rprec(np.array([.1, .9, .8, .2, .3]), r) == 0.5
    assert rprec(np.ones(5), r) == 0.4                     # all tied -> base rate
    assert np.isnan(rprec(np.ones(5), np.zeros(5, bool)))


def main(out):
    truth, rows = {}, []
    for f in sorted(glob.glob(os.path.join(H, "preds", "*", "*.parquet"))):
        method, pid = f.split(os.sep)[-2], os.path.basename(f)[:-8]
        if pid not in truth:
            pp = os.path.join(H, "pilots", pid + ".parquet")
            if not os.path.exists(pp):
                continue
            df = pd.read_parquet(pp, columns=["obs_name", "is_rare"])
            sort_idx, oid = _opaque_order(pid, df["obs_name"].values)
            truth[pid] = pd.Series(df["is_rare"].values[sort_idx].astype(bool), index=oid)
        p = pd.read_parquet(f)
        ids = p[next(c for c in ("cell_id", "id", "obs_name") if c in p.columns)].values  # opaque id; older drivers called it id, scCAD obs_name
        is_rare = truth[pid].reindex(ids)
        assert not is_rare.isna().any(), f"{f}: cell ids do not match the pilot"
        r = is_rare.values.astype(bool)
        rows.append((pid, method, int(r.sum()), rprec(p["score"].fillna(0.0).values, r),
                     rprec(p["call"].fillna(False).values.astype(float), r)))
    pd.DataFrame(rows, columns=["pilot_id", "method", "n_rare_true_rp", "rprec", "rprec_call"]).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    demo()
    main(sys.argv[1])
