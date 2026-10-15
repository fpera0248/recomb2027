"""FiRE, https://github.com/princethewinner/FiRE commit 5e0828375373f7a5af6381a66c214d61da8a8bf6 (python extension built unmodified from python/FiRE).
usage: python run_fire.py pool.h5ad out.parquet; writes out.parquet (as published, pool-wide) and out_lab.parquet (per visible label).
Published pipeline (example/jurkat_simulation.py): utils/preprocess.ranger_preprocess(ngenes_keep=1000), FiRE(L=100, M=50), score,
call = score >= Q3 + 1.5 IQR. Label-aware variant: the same preprocess, FiRE and IQR rule within each label with at least 20 cells; other labels
score 0 and are not called; score = average-tie rank percentile of the raw score within the label.
Needs TOOLS_SRC (the FiRE checkout) and FIRE_BUILD (directory holding the compiled FiRE module).
"""

import os
import sys
import warnings

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "4")
warnings.filterwarnings("ignore")
sys.path[:0] = [
    os.path.dirname(os.path.abspath(__file__)),
    os.environ["FIRE_BUILD"],
    os.path.join(os.environ["TOOLS_SRC"], "FiRE", "utils"),
]

import numpy as np
import scipy.sparse as sp
from scipy import stats

np.int = int
np.float = (
    float  # utils/preprocess.py still uses np.int and np.float, removed in numpy 1.24
)
import FiRE
import preprocess as pp

from common import read_pool, write_out


def fire(X, genes):
    d, _ = pp.ranger_preprocess(X, genes, ngenes_keep=1000, verbose=False)
    m = FiRE.FiRE(L=100, M=50)
    m.fit(d.astype(np.float32))
    return np.array(m.score(d.astype(np.float32)))


def iqr_call(s):
    return s >= np.percentile(s, 75) + 1.5 * stats.iqr(s)


if __name__ == "__main__":
    a = read_pool(sys.argv[1])
    X = (a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float64)
    genes = np.array(a.var_names)
    s = fire(X, genes)
    write_out(a.obs_names, s, iqr_call(s), sys.argv[2])
    lab = a.obs["label"].astype(str).values
    score, call = np.zeros(len(lab)), np.zeros(len(lab), bool)
    for L in np.unique(lab):
        i = np.where(lab == L)[0]
        if len(i) < 20:
            continue
        try:
            r = fire(X[i], genes)
        except Exception as e:
            print("label failed", L, e)
            continue
        if not np.any(r):
            continue
        score[i] = stats.rankdata(r) / len(i)
        call[i] = iqr_call(r)
    write_out(a.obs_names, score, call, sys.argv[2].replace(".parquet", "_lab.parquet"))
    print("n", len(s), "called", int(iqr_call(s).sum()), "called_lab", int(call.sum()))
