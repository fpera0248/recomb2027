"""Per-label arm for a tool that returns clusters (RaceID, CHOIR, GiniClust3 with the README filters).
usage: python run_lab.py pool.h5ad in.parquet out.parquet
in.parquet: cell_id, group (as written by the R runners); the visible label comes from obs['label'] of the pool. Rule: common.label_arm.
"""

import sys

import numpy as np
import pandas as pd

from common import label_arm, read_pool, write_out

if __name__ == "__main__":
    p = read_pool(sys.argv[1])
    d = pd.read_parquet(sys.argv[2]).set_index("cell_id").reindex(p.obs_names)
    assert d.group.notna().all(), "tool output misses pool cells"
    s, call = label_arm(d.group.values, p.obs["label"].astype(str).values)
    write_out(p.obs_names, s, call, sys.argv[3])
    print("n", len(s), "called", int(call.sum()))
