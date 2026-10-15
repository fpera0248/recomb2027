"""ISMOTE augmentation"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def _ismote_point(Xq, Xj, alpha, beta):
    diff = Xq - Xj
    Xnew1 = Xq + alpha * diff
    if np.linalg.norm(Xnew1 - Xq) >= 0.5 * np.linalg.norm(diff):
        return Xq + alpha * diff - beta * diff
    return Xq + alpha * diff + beta * diff

def augment(Z, y, c, k, rng, **params):
    """ISMOTE oversampling of class `c`, producing exactly `k` synthetic rows."""
    y = np.asarray(y)
    Xc = np.asarray(Z, dtype=np.float64)[y == c]
    n = Xc.shape[0]
    d = Xc.shape[1]

    if n == 1:
        # degenerate case: X_j can only be X_q -> X_new == X_q
        Zadd = np.repeat(Xc, k, axis=0)
        yadd = np.full(k, c, dtype=y.dtype)
        wadd = np.ones(k, dtype=np.float64)
        return Zadd, yadd, wadd

    n_neighbors = min(params.get('n_neighbors', 5), n - 1)
    idx = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Xc).kneighbors(
        Xc, return_distance=False
    )  # (n, n_neighbors+1), includes the query point itself

    Zadd = np.empty((k, d), dtype=np.float64)
    q_choices = rng.integers(n, size=k)
    for i, q in enumerate(q_choices):
        neighbors = idx[q][idx[q] != q][:n_neighbors]
        j = neighbors[rng.integers(n_neighbors)]
        alpha = rng.uniform(0.0, 1.0)
        beta = rng.uniform(0.0, 1.0)
        Zadd[i] = _ismote_point(Xc[q], Xc[j], alpha, beta)

    yadd = np.full(k, c, dtype=y.dtype)
    wadd = np.ones(k, dtype=np.float64)
    return Zadd, yadd, wadd

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    d = 50

    # --- Eq. (2)/(3) branch geometry, hand-picked alpha/beta (paper's own equations) ---
    Xq = np.zeros(d); Xq[0] = 10.0
    Xj = np.zeros(d)  # X_q - X_j = 10 on coord 0 only; distance(X_q, X_j) = 10

    # "+" branch fires when alpha < 0.5 (dist(Xnew1,Xq)=alpha*10 < 5); t=alpha+beta>0
    # always -> literally always beyond X_q, away from X_j (outside the segment).
    out_plus = _ismote_point(Xq, Xj, alpha=0.3, beta=0.2)
    assert out_plus[0] > Xq[0], "'+' branch must land beyond X_q per printed Eq. (3)"
    assert np.allclose(out_plus[1:], 0.0)

    # "-" branch fires when alpha >= 0.5; t=alpha-beta can be negative -> can land
    # back inside [X_j, X_q] when beta > alpha, as here.
    out_minus = _ismote_point(Xq, Xj, alpha=0.7, beta=0.9)
    assert 0.0 < out_minus[0] < Xq[0], "this alpha/beta should land inside [X_j,X_q]"
    assert np.allclose(out_minus[1:], 0.0)

    # every generated point is X_q plus a multiple of (X_q - X_j): no off-line component
    for a, b in [(0.1, 0.9), (0.9, 0.1), (0.5, 0.5)]:
        p = _ismote_point(Xq, Xj, a, b)
        assert np.allclose(p[1:], 0.0)

    # --- shapes / weights for each probe count, with a large
    # --- "other class" cluster present to confirm it is never touched ---
    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        k = 256 - n
        r = np.random.default_rng(n)
        Zc = r.normal(size=(n, d))
        Zother = r.normal(size=(400, d)) + 50.0  # far away; must have zero influence
        Z = np.vstack([Zc, Zother])
        y = np.array([7] * n + [3] * 400)  # class id 7 is the probe class here
        Zadd, yadd, wadd = augment(Z, y, c=7, k=k, rng=r)

        assert Zadd.shape == (k, d)
        assert yadd.shape == (k,)
        assert wadd.shape == (k,)
        assert np.all(yadd == 7)
        assert np.allclose(wadd, 1.0)
        assert np.isclose(wadd.sum(), k), (n, wadd.sum(), k)
        assert Zadd.max() < 15.0, "must never drift toward the other class's cluster"

        if n == 1:
            assert np.allclose(Zadd, Zc[0]), "n=1 must degenerate to exact duplication"
        else:
            assert np.var(Zadd) > 0.0  # not degenerate once >=1 real neighbour exists

    print("all ISMOTE (Li et al. 2025) self-checks passed")
