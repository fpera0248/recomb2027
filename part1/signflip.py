"""Sign-flip test over cell types."""

import itertools
import numpy as np


def signflip_p(d, groups):
    """Exact two-sided sign-flip p over cell types, on per-type mean differences."""
    dt = np.array([d[groups == g].mean() for g in np.unique(groups)])
    obs = abs(dt.mean())
    if (
        len(dt) > 16
    ):  # 2^n sign vectors is too many: Monte-Carlo (200k draws, fixed seed); p resolution 5e-6
        S = np.random.default_rng(0).choice((-1.0, 1.0), size=(200_000, len(dt)))
        with np.errstate(all="ignore"):
            T = np.abs(S @ dt) / len(
                dt
            )  # macOS Accelerate raises spurious matmul flags
        return float((1 + np.sum(T >= obs - 1e-12)) / (1 + len(S))), dt
    null = [abs((dt * s).mean()) for s in itertools.product((-1, 1), repeat=len(dt))]
    return float(np.mean(np.array(null) >= obs - 1e-12)), dt
