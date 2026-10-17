"""GiniClust3, https://github.com/rdong08/GiniClust3 commit a58d0c886cde690398c2a46e4320f1247f72cd4f, README call sequence with its two filters.
usage: python run_giniclust3_readme.py pool.h5ad out.parquet; writes out.parquet (pool-wide) and out_lab.parquet (per visible label).
The README loads a genes x cells table, so its sc.pp.filter_cells(min_genes=3) removes genes seen in under 3 cells and sc.pp.filter_genes(min_cells=200) removes
cells with under 200 genes, in that order, before it transposes to cells x genes. Here the pool is already cells x genes, so the same two filters are
filter_genes(min_cells=3) then filter_cells(min_genes=200). Then normalize_per_cell 1e4, calGini, clusterGini(neighbors=3), calFano, clusterFano (defaults,
Fano neighbors 15, both resolutions 0.1), generateMtilde, clusterMtilde. run_giniclust3.py is the unfiltered run (5 Gini neighbors).
GiniClust3 returns only clusters, so: call = final consensus cluster with under 1% of the kept cells (the tool's own rarity cutoff), score = -ln(cluster share).
Cells the 200-gene filter removes are not scored: they get a score below every kept cell, no call, and the group "dropped".
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

from common import label_arm, read_pool

DROPPED = "dropped"


def adapter(final):
    """Cluster share -> score and call; the score of the largest cluster is 0, so dropped cells are placed at min - 1 by the caller."""
    s = pd.Series(final)
    share = s.map(s.value_counts() / len(s)).values
    return -np.log(share), share < 0.01


def demo():
    s, c = adapter(np.array(["a"] * 990 + ["b"] * 5))
    assert c[-5:].all() and not c[:990].any() and s[-1] > s[0]


if __name__ == "__main__":
    demo()
    full = read_pool(sys.argv[1])
    p = full.copy()
    sc.pp.filter_genes(p, min_cells=3)
    sc.pp.filter_cells(p, min_genes=200)
    kept = p.n_obs
    X = p.X.toarray() if sp.issparse(p.X) else np.asarray(p.X)
    a = anndata.AnnData(X=np.asarray(X, dtype=np.float64))
    a.obs_names = p.obs_names
    a.var_names = p.var_names
    sc.pp.normalize_per_cell(a, counts_per_cell_after=1e4)
    gc.gini.calGini(a)
    gc.gini.clusterGini(a, neighbors=3)
    gc.fano.calFano(a)
    gc.fano.clusterFano(a)
    cons = {
        "giniCluster": np.array(a.obs["rare"].values.tolist()),
        "fanoCluster": np.array(a.obs["fano"].values.tolist()),
    }
    gc.consensus.generateMtilde(cons)
    gc.consensus.clusterMtilde(cons)
    final = np.array(cons["finalCluster"]).astype(str)
    score, call = adapter(final)
    # all pool cells, in the input order
    pos = full.obs_names.get_indexer(a.obs_names)
    n = full.n_obs
    s, c, g = np.full(n, score.min() - 1.0), np.zeros(n, bool), np.full(n, DROPPED, object)
    s[pos], c[pos], g[pos] = score, call, final
    lab = full.obs["label"].astype(str).values
    sl, cl = label_arm(g, lab)
    for path, sv, cv in ((sys.argv[2], s, c), (sys.argv[2].replace(".parquet", "_lab.parquet"), sl, cl)):
        pd.DataFrame(
            {"cell_id": full.obs_names.astype(str), "score": sv, "call": cv, "group": g}
        ).to_parquet(path, index=False)
    print("n", n, "kept", kept, "final_clusters", len(set(final)), "rare_called", int(c.sum()), "called_lab", int(cl.sum()))
