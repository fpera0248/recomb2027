"""python gate_pca.py pid... : PCA arm recomputed under this protocol vs the stored locked outputs ($R5SRC/cands, preds_base/m1_leiden_sub). Expected identical on the mini
(pinned venv); on a different environment (e.g. the laptop) Leiden partitions differ, which is why every contrast is computed within one machine. Writes gate_pca.txt."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import embedding_lib as W
import numpy as np, pandas as pd
lines = []
for pid in sys.argv[1:]:
    A, B = f"{W.R5D}/cands", f"{W.arm_dir('pca')}/cands"; r = {}
    g0, g1 = pd.read_parquet(f"{A}/{pid}.groups.parquet"), pd.read_parquet(f"{B}/{pid}.groups.parquet"); r["n_groups"] = (len(g0), len(g1))
    r["members_equal"] = len(g0) == len(g1) and all(np.array_equal(a, b) for a, b in zip(g0.members, g1.members))
    for k in ("markers", "halo"):
        x, y = pd.read_parquet(f"{A}/{pid}.{k}.parquet"), pd.read_parquet(f"{B}/{pid}.{k}.parquet"); r[k + "_maxdiff"] = float(np.abs(x.values - y.values).max()) if x.shape == y.shape else "shape"
    m0 = pd.read_parquet(f"{W.R5D}/preds_base/m1_leiden_sub/{pid}.parquet").score.values; m1 = pd.read_parquet(f"{W.arm_dir('pca')}/preds_base/m1_leiden_sub/{pid}.parquet").score.values
    r["m1_maxdiff"] = float(np.abs(m0 - m1).max()); lines.append(f"{pid} {r}")
open(f"{W.OUT}/gate_pca.txt", "a").write("\n".join(lines) + "\n"); print("\n".join(lines))
