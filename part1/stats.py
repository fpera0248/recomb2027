"""Tests used on the Part I augmentation experiments."""

import itertools

import numpy as np
import pandas as pd

from .signflip import signflip_p


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def max_t(D):
    """One-sided studentized max-T over the columns of D (types x tests), exact over all 2^types sign flips.
    Returns the single-step adjusted p of each column (controls the family-wise error over the columns).
    """
    D = D.dropna(axis=1, how="all").dropna()
    n, X = len(D), D.values
    F = np.array(list(itertools.product((-1, 1), repeat=n)), float)
    with np.errstate(all="ignore"):
        m = (F @ X) / n
    sd = np.sqrt(np.maximum(((X**2).sum(0) - n * m**2) / (n - 1), 1e-18))
    T = m / sd * np.sqrt(n)
    mx, obs = T.max(1), T[-1]
    return pd.Series([float(np.mean(mx >= t - 1e-9)) for t in obs], index=D.columns)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    D = pd.DataFrame(
        {
            "win": 0.02 + 0.005 * rng.standard_normal(8),
            "null": 0.02 * rng.standard_normal(8),
            "loser": -0.3 + 0.05 * rng.standard_normal(8),
        }
    )
    pa = max_t(D)
    assert pa["win"] < 0.05 and pa["loser"] > 0.5
    assert abs(signflip_p(np.ones(3), np.arange(3))[0] - 0.25) < 1e-12
    assert list(bh([0.01, 0.04, 0.03]).round(3)) == [0.03, 0.04, 0.04]
