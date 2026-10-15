"""Final score: 0.75 * rank(group scorer) + 0.25 * rank(Leiden subclustering score).

python -m hiddenrare.blend SCORER.parquet LEIDEN_SUB.parquet OUT.parquet
"""

import os
import sys

import pandas as pd

W = 0.75  # weight of the scorer, fixed in the pre-registration


def combo(a, b, w=W):
    ra, rb = a.rank(pct=True), b.reindex(a.index).fillna(b.min()).rank(pct=True)
    return w * ra + (1 - w) * rb


def blend(scorer_file, leiden_file, out):
    a = pd.read_parquet(scorer_file).set_index("cell_id").score
    b = pd.read_parquet(leiden_file).set_index("cell_id").score
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pd.DataFrame({"cell_id": a.index, "score": combo(a, b).values}).to_parquet(
        out, index=False
    )


if __name__ == "__main__":
    s = pd.Series([3.0, 2, 1], index=list("xyz"))
    assert list(combo(s, s[::-1], 0.5).values) == [1, 2 / 3, 1 / 3]
    if len(sys.argv) > 1:
        blend(*sys.argv[1:4])
