"""Label-aware variant of scCAD: a sub-cluster that holds at least 10% of a visible label is not rare for that label."""
import numpy as np
import pandas as pd


def convert(score, sub, label, share=0.10, h=0.7):
    """Cells in a sub-cluster holding >= share of their label get score - 1 and no call; others keep the score and call = score >= h."""
    d = pd.DataFrame({"s": sub, "l": label})
    big = (d.groupby(["s", "l"]).s.transform("size") / d.groupby("l").s.transform("size")).values >= share
    return score - big, (~big) & (score >= h)


if __name__ == "__main__":
    # label A (10 cells) sits entirely in sub-cluster 0, so it is penalised; label B (100) has a 5-cell sub-cluster 1 (5% < 10%), so it is kept
    sc = np.r_[np.full(10, .9), np.full(95, 0.), np.full(5, .8)]
    sb = np.r_[np.zeros(10), np.full(95, 2), np.ones(5)]
    lb = np.r_[["A"] * 10, ["B"] * 100]
    s, c = convert(sc, sb, lb)
    assert np.allclose(s[:10], -.1) and not c[:10].any() and np.allclose(s[-5:], .8) and c[-5:].all() and not c[10:105].any()
