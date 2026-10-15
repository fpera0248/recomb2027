"""Null test: does a method flag a pool that hides nothing?

Each real pool has a twin with every hidden cell removed. A method's evidence for a pool is the mean score of its 200 top cells
(for the blended method, which has no native scale, the learned scorer's score on the blend's top 200). The threshold flags 95% of
the real pools; the false-call rate is the share of twins above it (about 0.95 for an uninformative score).

PRED_ROOT/<method>/<pool>.parquet holds real and twin pools; twins.csv has pilot_id, atlas and real_pool.
"""
import numpy as np
import pandas as pd

TOP = 200


def evidence(root, method, pid):
    d = pd.read_parquet(f"{root}/{method}/{pid}.parquet")
    if method == "hiddenrare":
        top = d.nlargest(TOP, "score").cell_id
        return pd.read_parquet(f"{root}/scorer/{pid}.parquet").set_index("cell_id").loc[top, "score"].mean()
    return d.score.nlargest(TOP).mean()


def quantile(v, w, q=0.05):
    o = np.argsort(v)
    return v[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), q - 1e-12)]


def false_call_rate(real, twin, atlas, by_atlas):
    """by_atlas: every atlas counts the same (threshold and rate); otherwise every pool does."""
    w = np.array([1 / (atlas == a).sum() for a in atlas]) if by_atlas else np.ones(len(real))
    hit = (twin >= quantile(real, w)).astype(float)
    return float(np.mean([hit[atlas == a].mean() for a in np.unique(atlas)])) if by_atlas else float(hit.mean())


def run(root, methods, twins):
    """-> {method: {atlas_equal, pool_weighted}} over the real/twin pairs of the twins table."""
    atlas = twins.atlas.values
    out = {}
    for m in methods:
        r = np.array([evidence(root, m, p) for p in twins.real_pool])
        t = np.array([evidence(root, m, p) for p in twins.pilot_id])
        out[m] = dict(atlas_equal=false_call_rate(r, t, atlas, True), pool_weighted=false_call_rate(r, t, atlas, False))
    return out


if __name__ == "__main__":
    a = np.array(list("aabb"))
    assert false_call_rate(np.array([1., 2, 3, 4]), np.array([0., 0, 0, 0]), a, True) == 0
    assert false_call_rate(np.array([1., 2, 3, 4]), np.array([9., 9, 9, 9]), a, False) == 1
