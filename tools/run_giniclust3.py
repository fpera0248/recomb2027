"""GiniClust3, https://github.com/rdong08/GiniClust3 commit a58d0c886cde690398c2a46e4320f1247f72cd4f, README call sequence without its two filters.
usage: python run_giniclust3.py pool.h5ad out.parquet
normalize_per_cell 1e4; calGini/clusterGini (defaults, neighbors 5, res 0.1); calFano/clusterFano (15, 0.1); consensus.
This is the earlier run the paper labels GiniClust3 (no filters, 5 neighbors). The GiniClust3 (swapped filters) rows (genes detected in at least 200 cells, no cell filter, and 3 neighbors on the 8-atlas set) came from a runner that is not included.
GiniClust3 returns only clusters, so: call = final consensus cluster with under 1% of the cells (the tool's own rarity cutoff), score = -ln(cluster share).
"""

import os
import sys
import warnings

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "4")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anndata
import giniclust3 as gc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from common import read_pool, write_out

if __name__ == "__main__":
    p = read_pool(sys.argv[1])
    X = p.X.toarray() if sp.issparse(p.X) else np.asarray(p.X)
    a = anndata.AnnData(X=np.asarray(X, dtype=np.float64))
    a.obs_names = p.obs_names
    a.var_names = p.var_names
    sc.pp.normalize_per_cell(a, counts_per_cell_after=1e4)
    gc.gini.calGini(a)
    gc.gini.clusterGini(a)
    gc.fano.calFano(a)
    gc.fano.clusterFano(a)
    cons = {
        "giniCluster": np.array(a.obs["rare"].values.tolist()),
        "fanoCluster": np.array(a.obs["fano"].values.tolist()),
    }
    gc.consensus.generateMtilde(cons)
    gc.consensus.clusterMtilde(cons)
    final = np.array(cons["finalCluster"])
    sz = pd.Series(final).value_counts()
    share = pd.Series(final).map(sz / len(final)).values
    write_out(a.obs_names, -np.log(share), share < 0.01, sys.argv[2])
    print(
        "n",
        len(final),
        "final_clusters",
        len(sz),
        "rare_called",
        int((share < 0.01).sum()),
    )
