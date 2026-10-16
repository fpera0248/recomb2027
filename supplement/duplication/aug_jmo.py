"""Just Mix Once (JM1) adapted to single-class rare-cell-type augmentation.

PAPER
    Giannone, Havrylov, Massiah, Yilmaz & Jiao, "Just Mix Once: Worst-group
    Generalization by Group Interpolation", arXiv:2210.12195 (extended
    version of the NeurIPS 2021 Workshop on Distribution Shifts paper,
    https://neurips.cc/virtual/2021/35437).
    Fetched 2026-09-24 via curl (public pages, no login):
        https://arxiv.org/abs/2210.12195           (abstract/metadata)
        https://arxiv.org/html/2210.12195v1        (full text, method + appendix)

METHOD, AS THE PAPER STATES IT
    Phase I -- group identification (Sec. 2 "Background" / Sec. 3 "(Optional)
    Groups Identification", reused verbatim from JTT, Liu et al. 2021):
    train an ERM classifier f_hat with early stopping at a small
    "identification epoch" T (chosen on a small group-annotated val set).
    The globally misclassified buffer
        B := {(x_b, y_b) : f_hat_T(x_b) != y_b}                (Sec. 2, def. of B)
    is the "minority/difficult" partition g; its complement D\\B in the same
    class is the "majority/easy" partition g-bar. (When oracle spurious-group
    labels exist, JM1 skips Phase I and uses them directly -- not this
    setting: there is no spurious attribute here, only class rarity, so g /
    g-bar can only come from Phase I.)

    Phase II -- class-conditional interpolation (Sec. 3, Eq. 4): for two
    SAME-LABEL samples h_g^i in g and h_gbar^j in g-bar (y_g^i = y_gbar^j = y):
        h_mix = alpha * h_g^i + (1 - alpha) * h_gbar^j,  y_mix = y     (Eq. 4)
    Note y_mix = y exactly: unlike vanilla MixUp, JM1 is class-conditional,
    so the output label is a HARD label, never a soft alpha-weighted blend
    of two different classes (see Theorem 1 / Eq. 6-7: mixing is meant to
    interpolate the within-class group factor c, not the class itself).

    alpha's distribution (Sec. 4 "Settings" + Table 11, Appendix C): plain
    alpha ~ Beta(1,1) [= U(0,1)] under-performs JTT, so JM1's reported
    setting for the two vision benchmarks (CUB, CelebA -- the closest
    analogue to a generic feature-vector task; MultiNLI used uniform-only
    because its Phase II ran for only ~4 epochs, which has no analogue
    here) is a "coupled" schedule: alpha ~ U(0,1) for half of the draws and
    alpha ~ Beta(2,5) for the other half.

    Table 11 defaults (CUB / CelebA columns): batch size 32/64,
    identification epoch 40/1, epochs (phase I) 200/40, epochs (phase II)
    200/40, alpha U/B (coupled, as above), learning rate 1e-5, loss = CE,
    number of groups 4, mixing layer = "random early layer output" of a
    pretrained ResNet50, weight decay 1.0/0.1. These are SGD-training
    hyperparameters for a neural net trained over many epochs; see
    ADAPTATIONS for what does and doesn't carry over to one shot, sklearn,
    tabular augmentation.

ADAPTATIONS (this setting has no images/text, no SGD loop, no spurious
attribute, and classes as rare as n=1)
  1. No neural net / no epochs -> no "identification epoch T". The task
     spec pins the identification (reference) model to
     sklearn.linear_model.LogisticRegression(max_iter=2000), the same
     family as the downstream classifier. We use that at C=1 (matching the
     downstream classifier's C=1, since JM1's own weight-decay values are
     for a ResNet/BERT encoder and do not transfer). This is a full-ERM fit,
     not an early-stopped one -- but on features this imbalanced (a class
     with 1-128 cells against neighbours with up to 500) an ERM multinomial
     LR already tends to misclassify a probe class's own training points
     purely from class-prior/L2 pressure, which is exactly the functional
     signal JTT/JM1's early stopping is designed to produce (systematic
     failure on the minority pattern). So convergence vs. early stopping is
     a difference in *how* the underfitting on the rare class arises, not
     in *whether* it arises. `ref_max_iter` is still exposed as a knob to
     mimic true early stopping if this proxy ever stops holding on some
     dataset.
  2. No spurious attribute: g and g-bar (Phase I's partitions) are computed
     GLOBALLY on all 31 classes by the reference model as the paper defines
     them, then restricted to probe class c to get the two same-class pools
     the paper's Eq. 4 mixes. If the reference model's argmax on class c's
     own n cells is unanimous (all correct or all wrong -- likely for very
     small n), one of the two pools is empty and Eq. 4 cannot literally
     apply (there is no second, same-class partition to draw from). Most
     literal fallback: mix within all of class c's own cells instead of
     across a nonexistent split. This is not a new method, it is Eq. 4 with
     g = g-bar = {cells of class c} when the identified split degenerates.
  3. n = 1: with only one real cell in the class there is only one distinct
     point available for EITHER side of Eq. 4, cross-group or fallback. The
     method degenerates to h_mix = alpha*h + (1-alpha)*h = h for every
     alpha: JM1 collapses to exact duplication (plain oversampling) of the
     single point, k times. This is a real limit of Eq. 4, not a bug.
  4. Hard labels only, by Eq. 4 (see above) -> every synthetic row gets
     weight 1.0 and m = k rows exactly; no soft-label row-splitting is
     needed for this method.
  5. "Coupled" alpha-sampling was defined per-epoch in the paper (half the
     TRAINING EPOCHS use Beta(1,1), the other half Beta(2,5)). There is no
     epoch axis in a single augmentation call, so we apply the same 50/50
     mixture PER DRAW instead: over many draws this reproduces the same
     marginal distribution over alpha that the paper's per-epoch schedule
     produces.
  6. fit_reference() is exposed so a caller fits the reference LR once per
     seed on the full (Z, y) and passes its probability matrix P in via
     `augment(..., P=P)`, reused across all 8 probe classes -- the group
     split only ever depends on argmax(P), never on k or c beyond indexing.

# ponytail: the reference model is a converged LogisticRegression standing in
# for JM1's early-stopped ERM net (adaptation #1 above); ceiling is that on a
# very well-separated / high-capacity feature space it could start fitting
# even the rarest classes, silently sending every probe class down the
# fallback path. Upgrade: cap `ref_max_iter` low (e.g. 5-10) to force genuine
# underfitting if that is ever observed.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_reference(Z, y, seed, max_iter=2000, C=1.0):
    """JM1 Phase I identification model (Sec. 2 / Sec. 3-I): an ERM
    classifier fit on the full labelled train set. Returns P, an
    (n_cells x n_classes) probability matrix indexed by raw label id
    (0..max(y)), so argmax(P, axis=1) is the model's predicted label and
    disagreement with y is JM1's error/misclassification buffer B.
    """
    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed)
    clf.fit(Z, y)
    n_classes = int(np.max(y)) + 1
    P = np.zeros((Z.shape[0], n_classes), dtype=np.float64)
    P[:, clf.classes_] = clf.predict_proba(Z)
    return P


def augment(Z, y, c, k, rng, **params):
    """Generate k JM1-style synthetic rows for class c.

    params:
      P            : precomputed (n_cells x n_classes) reference-model
                     probabilities from fit_reference(); fit once per seed
                     and reused across probe classes (see module docstring).
                     If omitted, fit_reference is called internally.
      beta_a, beta_b : Beta(a, b) parameters for the "hard" half of the
                     coupled alpha schedule (paper default: 2, 5).
      coupled      : True = paper's reported U(0,1)/Beta(2,5) 50/50 mix
                     (default); False = plain U(0,1), the paper's own
                     ablation baseline (Sec. 4), for comparison.
      ref_max_iter, ref_C, seed : passed to fit_reference() when P is None.
    """
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
            j = i                            # n == 1: only one point exists (adaptation #3)

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
    # written, g != g-bar) with a hand-built split, since realistic data at
    # these class sizes tends to land every probe-class point in one bucket
    # (adaptation #2's fallback branch, already covered above).
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
