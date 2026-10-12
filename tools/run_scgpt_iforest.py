"""scGPT, https://github.com/bowang-lab/scGPT commit cebd6fae655b9c585a4807daa3ac31bb764f06b4, whole-human checkpoint; cell embeddings + IsolationForest. usage: run_scgpt_iforest.py pool.h5ad out.parquet [--device cpu] [--ckpt DIR] [--seed 0]
Official scgpt.tasks.embed_data (raw counts in, default max_length 1200, batch 64, 'cls' embedding, L2-normalised) then
sklearn IsolationForest at its defaults (n_estimators=100, contamination='auto'). Unsupervised: scGPT+IF has no label input, so no label-aware option.
score = -IsolationForest.score_samples (high = isolated); call = IsolationForest.predict == -1.
Gene symbols: var['feature_name'] if present (CELLxGENE h5ad), else var_names."""
import os, argparse, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if not hasattr(os, "sched_getaffinity"):        # Linux-only call inside scGPT; empty set -> num_workers=0 (spawn can't pickle its local Dataset on macOS)
    os.sched_getaffinity = lambda pid: set()
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad, torch
from sklearn.ensemble import IsolationForest
from scgpt.tasks import embed_data
p = argparse.ArgumentParser(); p.add_argument("pool"); p.add_argument("out"); p.add_argument("--device", default="cpu")
p.add_argument("--ckpt", default=os.environ.get("SCGPT_CKPT"))
p.add_argument("--seed", type=int, default=0); o = p.parse_args()
np.random.seed(o.seed); torch.manual_seed(o.seed)
a = ad.read_h5ad(o.pool)
if "feature_name" in a.var: a.var_names = a.var["feature_name"].astype(str).values
a = a[:, ~a.var_names.duplicated()].copy()
emb = embed_data(a, o.ckpt, gene_col="index", device=o.device, use_fast_transformer=False, return_new_adata=False)   # return_new_adata=True passes dtype= to AnnData, which anndata >= 0.12 rejects
E = np.asarray(emb.obsm["X_scGPT"]); assert E.shape[0] == a.n_obs and np.isfinite(E).all()
f = IsolationForest(random_state=o.seed).fit(E)
pd.DataFrame({"cell_id": a.obs_names.values, "score": -f.score_samples(E), "call": f.predict(E) == -1}).to_parquet(o.out, index=False)
print("n_called", int((f.predict(E) == -1).sum()))
