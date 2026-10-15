"""Rerun tier: score the held-out pools again from the frozen candidate tables.

    python -m eval.rerun ASSET_DIR OUT_DIR [POOL ...]

ASSET_DIR holds the extracted release assets. For each pool the learned scorer (models/refit_dev_nat.pkl) is applied to the frozen candidate
groups and features (cands/), blended with the saved Leiden subclustering scores, and scored against the hidden labels. OUT_DIR gets
preds/hiddenrare, preds/scorer and rerun.csv: per pool the union AP of the rerun and of the saved scores, and the Spearman correlation of
the per-cell scores. models/refit_dev_nat.pkl is a refit of the same recipe; the fit behind the saved scores was not kept, so the rerun matches them in ranking, not exactly.
"""
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

from hiddenrare.blend import blend
from hiddenrare.scorer import predict

from .paper import OURS, pool_metrics, pools

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "refit_dev_nat.pkl")


def run(root, out, pids):
    for p in pids:
        predict(MODEL, f"{root}/cands", p, f"{out}/preds/scorer/{p}.parquet")
        blend(f"{out}/preds/scorer/{p}.parquet", f"{root}/predictions/leiden_sub/{p}.parquet", f"{out}/preds/{OURS}/{p}.parquet")
    new = pool_metrics(root, {"rerun": f"{out}/preds/{OURS}"}, pids)
    old = pool_metrics(root, {"saved": f"{root}/predictions/{OURS}"}, pids)
    t = pd.concat([new, old]).pivot_table(index=["pilot_id", "metric"], columns="method", values="value").reset_index()
    t["spearman"] = [spearmanr(*[pd.read_parquet(f).set_index("cell_id").score.sort_index() for f in (f"{out}/preds/{OURS}/{pid}.parquet", f"{root}/predictions/{OURS}/{pid}.parquet")])[0]
                     for pid in t.pilot_id]
    t.to_csv(f"{out}/rerun.csv", index=False)
    return t


def summary(t, man):
    """Print the rerun against the saved scores. With all 69 pools, also check the tolerance: atlas-equal union AP within 0.01, mean Spearman above 0.8."""
    u = t[t.metric == "union_AP"].set_index("pilot_id")
    print(u[["rerun", "saved", "spearman"]].round(4).to_string())
    print(f"Spearman of per-cell scores: mean {u.spearman.mean():.3f}, min {u.spearman.min():.3f}; union AP difference per pool: mean abs {(u.rerun - u.saved).abs().mean():.3f}")
    if len(u) == len(man):
        ae = u[["rerun", "saved"]].groupby(man.loc[u.index, "atlas"].values).mean().mean()
        print(f"atlas-equal union AP: rerun {ae.rerun:.4f}, saved {ae.saved:.4f}")
        assert abs(ae.rerun - ae.saved) < 0.01 and u.spearman.mean() > 0.8, "rerun outside the stated tolerance"


if __name__ == "__main__":
    root, out = sys.argv[1:3]
    ids, man = pools(root)
    summary(run(root, out, sys.argv[3:] or ids), man)
