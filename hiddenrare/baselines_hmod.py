"""The other in-house baselines, same input and output as hiddenrare.baselines: isolation forest on scVI, local outlier factor,
classifier uncertainty and the three self-training loops. The code is hmod/methods.py of the original runs, imported unchanged from
supplement/further/method_at_submission (seeds and parameters are fixed there).

    python -m hiddenrare.baselines_hmod POOL.h5ad PILOTS.parquet OUT_ROOT [lof_pca uncertainty dup_loop nodup_loop classweight_loop iforest_scvi]
writes OUT_ROOT/<method>/<pool>.parquet with cell_id, score, call and group (-1 = no group)
iforest_scvi needs scvi-tools, which neither pinned environment contains.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "supplement",
        "further",
        "method_at_submission",
        "hmod",
    ),
)
import methods as M  # noqa: E402

from . import baselines  # noqa: E402

NAMES = {
    "lof_pca": "m3_lof_pca",
    "uncertainty": "m5_uncertainty",
    "dup_loop": "m6_dup_loop",
    "nodup_loop": "m7_nodup_loop",
    "classweight_loop": "m8_classweight_loop",
    "iforest_scvi": "m4_iforest_scvi",  # last: needs scvi-tools, so the others are written before a missing install stops the run
}
baselines.METHODS.update({k: getattr(M, v) for k, v in NAMES.items()})

if __name__ == "__main__":
    baselines.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or list(NAMES))
