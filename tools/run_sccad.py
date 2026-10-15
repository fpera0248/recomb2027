"""scCAD, https://github.com/xuyp-csu/scCAD commit ee59678e29b0f04eba04f033264691b8d0657c6c (scCAD.py unmodified).
usage: python run_sccad.py pool.h5ad out.parquet; writes out.parquet (as published, label-blind) and out_lab.parquet (label-aware)
Settings: scCAD(data=dense float32 raw counts, normalization=True, seed=2023, merge_h=50, overlap_h=0.7, rare_h=0.01, save_full=False),
rows ordered by sha1(pool id NUL barcode). As published: score = overlap of the cell's sub-cluster, call = score >= 0.7.
Label-aware: sccad_label.convert. Needs PYTHONHASHSEED=2023 (scCAD builds its gene list from a set).
"""

import os
import sys
import tempfile
import warnings

for k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(k, "4")
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.environ["TOOLS_SRC"], "scCAD")]

import numpy as np
import scipy.sparse as sp
import scCAD as sccad

from common import opaque_order, read_pool, write_out
from sccad_label import convert

if __name__ == "__main__":
    pool, out = sys.argv[1], sys.argv[2]
    pid = os.path.basename(pool).replace(".h5ad", "")
    a = read_pool(pool)
    o, inv = opaque_order(pid, a.obs_names)
    X = a.X[o]
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    cells = np.array(a.obs_names)[o]
    result, overlap, sub, degs = sccad.scCAD(
        data=X,
        dataName=pid,
        cellNames=cells,
        geneNames=np.array(a.var_names),
        normalization=True,
        seed=2023,
        merge_h=50,
        overlap_h=0.7,
        rare_h=0.01,
        save_full=False,
        save_path=tempfile.gettempdir() + "/",
    )
    overlap = np.asarray(overlap, float)
    sub = np.asarray(sub).astype(int)
    score = overlap[sub]
    write_out(cells[inv], score[inv], (score >= 0.7)[inv], out)
    s2, c2 = convert(score, sub, a.obs["label"].astype(str).values[o])
    write_out(cells[inv], s2[inv], c2[inv], out.replace(".parquet", "_lab.parquet"))
    print(
        "n",
        len(cells),
        "subclusters",
        len(overlap),
        "called",
        int((score >= 0.7).sum()),
        "called_lab",
        int(c2.sum()),
    )
