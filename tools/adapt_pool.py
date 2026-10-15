"""Tool inputs for one pool.

    python tools/adapt_pool.py POOL.h5ad PILOTS.parquet GENEMAP.csv OUT_DIR
Loads the pool blind (opaque cell ids, obs label only) and writes OUT_DIR/py/<pool>.h5ad (genes with counts in
at least 3 cells) and OUT_DIR/r/<pool>.h5ad (the 2000 highly variable genes). var['feature_name'] holds gene
symbols from GENEMAP.csv (written by data/build_pools.py); the pool files carry Ensembl ids only.
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiddenrare.candidates import hvg_genes
from hiddenrare.io import load_pool

if __name__ == "__main__":
    pool, table, genemap, out = sys.argv[1:5]
    pid = os.path.splitext(os.path.basename(table))[0]
    a = load_pool(pool, table, pid)
    a.obs = a.obs[["label"]].copy()
    a.X = sp.csr_matrix(a.X)
    v = pd.read_csv(genemap, dtype=str).set_index("gene_id")
    assert a.var_names.isin(v.index).all()
    a.var = pd.DataFrame(
        {"feature_name": v.loc[a.var_names, "feature_name"].astype(str).values},
        index=a.var_names,
    )
    for sub, b in (
        ("py", a[:, np.asarray((a.X > 0).sum(0)).ravel() >= 3].copy()),
        ("r", a[:, hvg_genes(a)].copy()),
    ):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
        b.write_h5ad(f"{out}/{sub}/{pid}.h5ad")
    print(pid, a.n_obs)
