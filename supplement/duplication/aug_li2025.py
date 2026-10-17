"""
ISMOTE augmentation: Li, Yang, Song, Duan & Ren, "An improved SMOTE algorithm for
enhanced imbalanced data classification by expanding sample generation space",
Scientific Reports 15:23521 (2025). DOI 10.1038/s41598-025-09506-w.
Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC12222711/

The equations are not in the HTML text (PMC renders them as images); transcriptions are exact.

Equation (2) (base sample, "flowchart, steps and pseudo-code" section):
    X_new1 = X_q + alpha * (X_q - X_j),      alpha ~ Uniform(0, 1)

Equation (3) (final sample, same section):
    if distance(X_new1, X_q) >= 0.5 * distance(X_q, X_j):
        X_new = X_q + alpha*(X_q - X_j) - beta*(X_q - X_j)
    else:
        X_new = X_q + alpha*(X_q - X_j) + beta*(X_q - X_j)
    beta ~ Uniform(0, 1)

Algorithm 1 (pseudocode, same section), steps 1-14:
    1. W+ = minority-class samples only.
    2. For n_samples iterations: pick X_q uniformly at random from W+.
    3. Find the k nearest neighbours of X_q *within W+ only* (k=5 by default;
       "Select the k nearest neighbors of X_q ... k = 5 by default").
    4. Pick X_j uniformly at random from those k neighbours.
    5. Apply Eq. (2) then Eq. (3) to get X_new; add it to the output set.
ISMOTE never looks at the majority class (or, here, any other class) anywhere in
Algorithm 1 -- W+ is filtered out in step 1 and nothing else is touched again. So
augment() below only ever reads Z[y == c]; the other 30 classes present in Z/y are
read (to slice them out) but never used in the arithmetic, matching the paper exactly.

Hyperparameters and paper defaults:
    k (n_neighbors)   default 5   ("Select the k nearest neighbors of X_q ... k=5
                                    by default")
    alpha             ~ U(0,1), redrawn every synthetic point (Eq. 2)
    beta              ~ U(0,1), redrawn every synthetic point (Eq. 3)
    0.5 threshold     fixed constant in Eq. (3), not presented as tunable -> hardcoded

ADAPTATIONS for this setting (paper is binary minority/majority; the paper's own text
says this generalizes: "The ISMOTE algorithm can also be applied to multi-class
classification tasks by adjusting the proportion of synthetic samples" -- so treating
each rare class c as "the minority class" and every other row as irrelevant needs no
change to the method itself):
  1. n_neighbors is capped at n-1 for a probe class with n < 6 real cells (Algorithm 1
     step 3 assumes >=k same-class points exist; you cannot have more neighbours than
     other points in the class). This is a bound, not a method change.
  2. n=1 is outside what Algorithm 1 can execute (step 3, "find the k nearest
     neighbors of X_q", has zero candidates). Literal limit taken: with no other
     same-class point, the only value X_j can take is X_q itself, so X_q - X_j = 0 in
     both Eq. (2) and Eq. (3) regardless of alpha/beta, and every branch collapses to
     X_new = X_q exactly. Implemented as exact duplication of the single real cell for
     all k synthetic units (weight 1.0 each, deterministic, zero-variance).
  3. ISMOTE is a hard-label, single-class method (see above: it never mixes in another
     class's rows), so every synthetic row gets yadd == c and weight 1.0; the soft-
     label-via-several-rows mechanism the runner supports is not needed here.

DISCREPANCY NOTE: the paper's prose (the paragraph right before Algorithm 1, and the
Fig. 2 caption) describes the "+" branch as pulling the new point "around the
neighboring sample X_j" and the "-" branch as holding it "around the seed sample
X_q". Taken literally, Eq. (3)'s own algebra does not do that: writing t such that
X_new = X_q + t*(X_q - X_j), the "+" branch gives t = alpha+beta, which is strictly
positive whenever alpha,beta > 0, so X_new always lands beyond X_q, on the far side
from X_j -- i.e. always outside the [X_q, X_j] segment, moving away from the
neighbour, not toward it. The "-" branch (t = alpha-beta) can land on either side.
This looks like a sign slip in the source paper (mirrors Eq. (1)'s own printed "+"
in "X_new = X_q + alpha*(X_q + X_j)", which does not read as a segment interpolation
either). We implement the equations exactly as printed
(Algorithm 1's pseudocode image, PMC12222711) rather than silently "fixing" the sign
to match the prose gloss; the self-check below checks the literal, printed geometry.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _ismote_point(Xq, Xj, alpha, beta):
    """Eq. (2) then Eq. (3), exactly as printed in Algorithm 1 steps 7-11."""
    diff = Xq - Xj
    Xnew1 = Xq + alpha * diff
    if np.linalg.norm(Xnew1 - Xq) >= 0.5 * np.linalg.norm(diff):
        return Xq + alpha * diff - beta * diff
    return Xq + alpha * diff + beta * diff


def augment(Z, y, c, k, rng, **params):
    """
    ISMOTE oversampling of class `c`, producing exactly `k` synthetic rows.

    Z : (n_cells, 50) float64, all TRAIN cells (PCA-50), every class
    y : (n_cells,) int class labels
    c : int, the class to oversample
    k : int, number of synthetic rows required
    rng : numpy.random.Generator
    params: n_neighbors (paper default 5; capped at n_c-1, see module docstring)

    Returns (Zadd, yadd, wadd): Zadd (k,50), yadd (k,) all == c, wadd (k,) all 1.0,
    summing to exactly k. ISMOTE (Algorithm 1) never reads another class, so no
    soft-labelled rows are produced.
    """
    y = np.asarray(y)
    Xc = np.asarray(Z, dtype=np.float64)[y == c]
    n = Xc.shape[0]
    d = Xc.shape[1]

    if n == 1:
        # degenerate case (see module docstring): X_j can only be X_q -> X_new == X_q
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

    # --- shapes / weights for every probe count used here, with a large
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
