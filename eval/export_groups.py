"""Group files for the null gate.

    python -m eval.export_groups PRED_ROOT PILOT_DIR MANIFEST_DIR OUT_ROOT [POOL_ID ...]

The group of a cell is its Leiden subcluster within its visible label, taken from the
leiden_sub predictions and shared by all methods; group_share is the subcluster size over the
label size. Writes OUT_ROOT/<method>/<pool>.parquet (cell_id, group, group_share) for the methods
that run on the twin pools, and OUT_ROOT/twin_map.csv (twin_id, pilot_id) from the null manifests.
Only the visible label is read.
"""
import glob
import os
import sys

import pandas as pd

from hiddenrare.io import opaque_order

METHODS = ("hiddenrare", "leiden_sub", "iforest_pca")


def reference_groups(pred_root, pilot_dir, pid):
    pl = pd.read_parquet(f"{pilot_dir}/{pid}.parquet", columns=["obs_name", "label"])
    order, ids = opaque_order(pid, pl.obs_name.values)
    lab = pd.Series(pl.label.astype(str).values[order], index=ids)
    p = pd.read_parquet(f"{pred_root}/leiden_sub/{pid}.parquet", columns=["cell_id", "group"]).set_index("cell_id")
    assert p.index.isin(lab.index).all(), f"{pid}: cell ids differ from the pilot table"
    l = lab.reindex(p.index)
    key = l.values + "|" + p.group.astype(str).values
    size = pd.Series(key).map(pd.Series(key).value_counts()).values
    return pd.DataFrame({"cell_id": p.index.values, "group": key, "group_share": size / l.map(l.value_counts()).values})


if __name__ == "__main__":
    pred_root, pilot_dir, man_dir, out = sys.argv[1:5]
    pids = sys.argv[5:] or sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{pred_root}/leiden_sub/*.parquet"))
    for pid in pids:
        g = reference_groups(pred_root, pilot_dir, pid)
        for m in METHODS:
            os.makedirs(f"{out}/{m}", exist_ok=True)
            g.to_parquet(f"{out}/{m}/{pid}.parquet", index=False)
    tm = [pd.read_csv(f)[["pilot_id", "block"]].rename(columns={"pilot_id": "twin_id", "block": "pilot_id"})
          for f in sorted(glob.glob(f"{man_dir}/atlas_manifest_null5_*.csv"))]
    if tm:
        pd.concat(tm).to_csv(f"{out}/twin_map.csv", index=False)
    print(len(pids), "pools")
