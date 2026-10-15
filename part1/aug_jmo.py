"""Just Mix Once (JM1) adapted to single-class rare-cell-type augmentation."""

import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_reference(Z, y, seed, max_iter=2000, C=1.0):
    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed)
    clf.fit(Z, y)
    n_classes = int(np.max(y)) + 1
    P = np.zeros((Z.shape[0], n_classes), dtype=np.float64)
    P[:, clf.classes_] = clf.predict_proba(Z)
    return P

def augment(Z, y, c, k, rng, **params):
    """Generate k JM1-style synthetic rows for class c."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    d = Z.shape[1]

    if k <= 0:
        return (np.empty((0, d)), np.empty((0,), dtype=y.dtype), np.empty((0,)))

    P = params.get("P")
    if P is None:
        seed = params.get("seed", int(rng.integers(0, 2**31 - 1)))
        P = fit_reference(
            Z, y, seed,
            max_iter=params.get("ref_max_iter", 2000),
            C=params.get("ref_C", 1.0),
        )

    beta_a = params.get("beta_a", 2.0)
    beta_b = params.get("beta_b", 5.0)
    coupled = params.get("coupled", True)

    idx_c = np.flatnonzero(y == c)
    n = idx_c.size
    if n == 0:
        raise ValueError(f"class {c} has no training cells to mix")

    pred_c = np.argmax(P[idx_c], axis=1)
    minority = idx_c[pred_c != c]   # g:      misclassified -> "difficult"  (Sec. 2, B)
    majority = idx_c[pred_c == c]   # g-bar:  correctly classified -> "easy"

    cross_group = minority.size > 0 and majority.size > 0
    pool_g = minority if cross_group else idx_c
    pool_h = majority if cross_group else idx_c

    Zadd = np.empty((k, d), dtype=np.float64)
    for t in range(k):
        i = rng.choice(pool_g)
        if cross_group:
            j = rng.choice(pool_h)          # disjoint from pool_g by construction
        elif n >= 2:
            rest = pool_h[pool_h != i]      # degenerate split: avoid a trivial i == j
            j = rng.choice(rest) if rest.size > 0 else i
        else:
            j = i                            # n == 1: only one point exists

        if coupled and rng.random() < 0.5:
            alpha = rng.uniform(0.0, 1.0)    # Beta(1,1) == U(0,1)
        elif coupled:
            alpha = rng.beta(beta_a, beta_b)
        else:
            alpha = rng.uniform(0.0, 1.0)

        Zadd[t] = alpha * Z[i] + (1.0 - alpha) * Z[j]   # Eq. 4

    yadd = np.full(k, c, dtype=y.dtype)   # Eq. 4: y_mix = y (hard, class-conditional)
    wadd = np.ones(k, dtype=np.float64)   # one weight unit per synthetic cell
    return Zadd, yadd, wadd

if __name__ == "__main__":
    d = 50
    n_bg_classes = 6
    n_bg = 60
    c = 0  # probe class under test

    def make_problem(n_probe, seed):
        # deliberately overlapping clusters (small center spread, large noise)
        # so the reference model actually misclassifies SOME but not all of a
        # probe class's own points -- exercising the cross_group=True branch
        # of augment(), not just the degenerate fallback.
        r = np.random.default_rng(seed)
        centers = r.normal(scale=2.0, size=(n_bg_classes + 1, d))
        Zs, ys = [], []
        for cls in range(1, n_bg_classes + 1):
            pts = centers[cls] + r.normal(scale=2.5, size=(n_bg, d))
            Zs.append(pts)
            ys.append(np.full(n_bg, cls))
        probe_pts = centers[0] + r.normal(scale=2.5, size=(n_probe, d))
        Zs.append(probe_pts)
        ys.append(np.zeros(n_probe, dtype=int))
        return np.concatenate(Zs).astype(np.float64), np.concatenate(ys).astype(np.int64)

    def on_some_segment(z, A, B, tol=1e-6):
        """True iff z = alpha*a + (1-alpha)*b for some a in A, b in B, alpha in [0,1]."""
        for a in A:
            ab = B - a
            denom = np.sum(ab * ab, axis=1)
            num = np.sum((z - a) * ab, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(denom > 1e-12, num / denom, 0.0)
            proj = a + t[:, None] * ab
            resid = np.linalg.norm(proj - z, axis=1)
            if np.any((t >= -1e-6) & (t <= 1 + 1e-6) & (resid < tol)):
                return True
        return False

    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        k = 256 - n
        Z, y = make_problem(n, seed=100 + n)

        ref_seed = int(np.random.default_rng(200 + n).integers(0, 2**31 - 1))
        P = fit_reference(Z, y, ref_seed)
        assert P.shape == (Z.shape[0], int(y.max()) + 1)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)

        rng = np.random.default_rng(300 + n)
        Zadd, yadd, wadd = augment(Z, y, c, k, rng, P=P)

        assert Zadd.shape == (k, d)
        assert yadd.shape == (k,)
        assert wadd.shape == (k,)
        assert np.all(yadd == c)
        assert np.isclose(wadd.sum(), k)
        assert np.all(wadd == 1.0)

        idx_c = np.flatnonzero(y == c)
        pts_c = Z[idx_c]
        pred_c = np.argmax(P[idx_c], axis=1)
        minority, majority = pts_c[pred_c != c], pts_c[pred_c == c]
        A, B = (minority, majority) if (minority.shape[0] and majority.shape[0]) else (pts_c, pts_c)

        if n == 1:
            assert np.allclose(Zadd, pts_c[0]), "n=1 must degenerate to exact duplication"
        else:
            for z in Zadd:
                assert on_some_segment(z, A, B), f"n={n}: synthetic point off every real-pair segment"
            assert np.unique(np.round(Zadd, 6), axis=0).shape[0] > 1, f"n={n}: no variety in synthetic points"

        print(f"n={n:4d} k={k:4d} cross_group={bool(minority.shape[0] and majority.shape[0])} OK")

    # Directly exercise the cross_group=True branch (Eq. 4 as literally
    # written, g != g-bar) with a hand-built split
    n_classes_tiny = 3
    Z_tiny = np.array([
        [0.0, 0.0],   # class c=0, correctly-classified half (g-bar)
        [0.2, -0.1],  # class c=0, correctly-classified half (g-bar)
        [5.0, 5.0],   # class c=0, misclassified half (g)
        [5.2, 4.9],   # class c=0, misclassified half (g)
        [1.0, 1.0],   # class 1 filler
        [2.0, 2.0],   # class 2 filler
    ])
    Z_tiny = np.pad(Z_tiny, ((0, 0), (0, d - 2)))
    y_tiny = np.array([0, 0, 0, 0, 1, 2])
    P_tiny = np.zeros((6, n_classes_tiny))
    P_tiny[0] = [0.9, 0.05, 0.05]   # argmax = 0 = c  -> g-bar
    P_tiny[1] = [0.8, 0.1, 0.1]     # argmax = 0 = c  -> g-bar
    P_tiny[2] = [0.1, 0.8, 0.1]     # argmax = 1 != c -> g
    P_tiny[3] = [0.1, 0.1, 0.8]     # argmax = 2 != c -> g
    P_tiny[4] = [0.1, 0.8, 0.1]
    P_tiny[5] = [0.1, 0.1, 0.8]
    rng_tiny = np.random.default_rng(7)
    Zadd_t, yadd_t, wadd_t = augment(Z_tiny, y_tiny, 0, 10, rng_tiny, P=P_tiny)
    assert np.all(yadd_t == 0)
    assert np.isclose(wadd_t.sum(), 10)
    g_bar_pts, g_pts = Z_tiny[:2], Z_tiny[2:4]
    for z in Zadd_t:
        assert on_some_segment(z, g_pts, g_bar_pts), "cross_group branch not on the g/g-bar segment"
    print("cross_group=True branch OK (explicit split)")

    # no-op edge case: k == 0 must return empty arrays of the right shape, no crash
    Z, y = make_problem(4, seed=999)
    Zadd, yadd, wadd = augment(Z, y, c, 0, np.random.default_rng(1))
    assert Zadd.shape == (0, d) and yadd.shape == (0,) and wadd.shape == (0,)

    print("all JM1 self-checks passed")
