"""MARS (Brbic 2020), https://github.com/snap-stanford/mars commit f6f03ba2c1f42221e337e934dad52a1d085b917c.
usage: run_mars.py pool.h5ad out.parquet [--n-extra 3] [--seed 3] [--device cpu]
Published model code unmodified; only import shims for the 2019 API (scanpy.api, sklearn k_means_). Needs TOOLS_SRC (the mars checkout is TOOLS_SRC/mars).
The preprocessing, score and call below are ours; this wrapper gave the paper's row MARS (our reimplementation) on the 10-atlas set.
MARS needs labeled data, so it is label-aware by construction: the pool with obs['label'] is the labeled
experiment and the same cells are the unlabeled experiment (n_clusters = n_labels + n_extra).
score = squared MARS-embedding distance of the cell to its nearest labeled class mean (high = novel).
call  = cell sits in a MARS cluster whose centroid is farther from every labeled class mean than the 95th
percentile of cell-to-own-class-mean distances."""

import os, sys, types, argparse, warnings
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, torch, scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.environ["TOOLS_SRC"], "mars"))
sys.modules["scanpy.api"] = sc  # shim: scanpy.api removed in scanpy>=1.5
import sklearn.cluster as skc  # shim: only used by kmeans++ mode (not default)

sys.modules["sklearn.cluster.k_means_"] = skc
skc.k_means_ = types.SimpleNamespace()
from args_parser import get_parser
from model.mars import MARS
from model.experiment_dataset import ExperimentDataset

p = argparse.ArgumentParser()
p.add_argument("pool")
p.add_argument("out")
p.add_argument("--n-extra", type=int, default=3)
p.add_argument("--device", default="cpu")
p.add_argument("--seed", type=int, default=3)
o = p.parse_args()

a = ad.read_h5ad(o.pool)
a.X = sp.csr_matrix(a.X).astype(np.float32)
# MARS preprocess_data minus the two Smart-seq2 cell filters (they would drop 10x cells we must score):
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
sc.pp.scale(a, max_value=10, zero_center=True)
X = np.nan_to_num(
    np.asarray(a.X.toarray() if sp.issparse(a.X) else a.X, dtype=np.float32)
)
cats = sorted(a.obs["label"].astype(str).unique())
y = (
    a.obs["label"]
    .astype(str)
    .map({c: i for i, c in enumerate(cats)})
    .values.astype(np.int64)
)
ids = np.array(a.obs_names)

params = get_parser().parse_args([])
params.device = o.device
params.manual_seed = o.seed
np.random.seed(o.seed)
torch.manual_seed(o.seed)
lab = ExperimentDataset(X, ids, a.var_names, "pool_labeled", y)
unl = ExperimentDataset(X, ids, a.var_names, "pool_unlabeled", y)
pre = ExperimentDataset(X, ids, a.var_names, "pool_pretrain")
m = MARS(len(cats) + o.n_extra, params, [lab], unl, pre, hid_dim_1=1000, hid_dim_2=100)
adata, lm = m.train(evaluation_mode=False, save_all_embeddings=False)
E = pd.DataFrame(adata.obsm["MARS_embedding"], index=adata.obs_names).loc[ids].values
cl = adata.obs["MARS_labels"].astype(int).loc[ids].values
Lt = np.stack(
    [E[y == k].mean(0) for k in range(len(cats))]
)  # labeled class means in MARS space (landmarks are tau-repelled, so not used for distances)
d2 = lambda A, B: ((A[:, None, :] - B[None]) ** 2).sum(2)
score = d2(E, Lt).min(1)
own = ((E - Lt[y]) ** 2).sum(1)
cen = {
    k: E[cl == k].mean(0, keepdims=True) for k in np.unique(cl)
}  # MARS cluster centroid
clu_nov = {k: float(d2(c, Lt).min()) for k, c in cen.items()}
call = np.array([clu_nov[k] for k in cl]) > np.quantile(own, 0.95)
pd.DataFrame({"cell_id": ids, "score": score, "call": call}).to_parquet(
    o.out, index=False
)
print(
    "cluster novelty",
    {k: round(v, 2) for k, v in clu_nov.items()},
    "thr",
    round(float(np.quantile(own, 0.95)), 2),
    "n_called",
    int(call.sum()),
)
