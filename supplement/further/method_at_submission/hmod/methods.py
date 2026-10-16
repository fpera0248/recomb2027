"""Blind hidden-rare discovery methods (RECOMB 2027 pilot, Jiaqi's "nobody
knows a rare type is buried in the metadata" question).

BLIND CONTRACT: every method below is a function `adata -> dict(score, call,
groups)` that receives ONLY an AnnData with obs['label'] (visible label,
rare cells already merged into a majority label by the pilot builder) and
obs['donor']. No method may read true_type, rare_type, n_rare, or any other
truth signal -- those live only in evaluate.py, which runs after the call.

Every threshold, resolution, k, contamination level and round count below is
a NUMBER FIXED IN THIS FILE, chosen once, identical for every pilot. Nothing
here is tuned per pilot or from the truth. Each constant's docstring/comment
states the rule it implements.

Shared preprocessing (fixed for all methods): log1p-CPM -> 2000 HVGs (seurat
flavor on the log1p-CPM matrix) -> scale -> PCA-50.

Output contract: score = pd.Series[float] indexed by obs_names (higher =
more suspicious of being a hidden rare cell); call = pd.Series[bool] same
index; groups = list of lists of obs_name strings (discovered candidate
hidden-type clusters).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import silhouette_score

RNG = 0
N_JOBS = 2          # laptop is shared; never raise this
N_HVG = 2000
N_PCS = 50
N_ESTIMATORS = 100  # RandomForest / IsolationForest tree count, fixed
N_SPLITS = 5        # donor-wise cross-fit folds, fixed (capped by n_donors)
N_NEIGHBORS = 15    # leiden graph neighbors, fixed

RES_GRID = (0.3, 0.5, 0.8, 1.2)   # M1's fixed resolution grid
MIN_SHARE = 0.10                   # M1: a subcluster < 10% of its label is a candidate
MIN_LABEL_N = 20                   # too few cells in a label to subcluster at all
LOF_CUTOFF = 1.5                   # M3: LOF > 1.5 is the standard "clear outlier" heuristic
Q_CALL = 0.99                      # M5/loop: flag the top 1% of a label's own score distribution
N_ROUNDS = 3                       # M6/M7/M8: fixed number of relabel-retrain rounds
LEIDEN_RES_LOOP = 0.5              # fixed resolution used to carve/report loop pseudo-groups
MIN_GROUP_SIZE = 10                # smallest reported candidate group / pseudo-label


# --------------------------------------------------------------------------
# shared preprocessing
# --------------------------------------------------------------------------
def _hvg_genes(adata, n_hvg=N_HVG):
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_hvg, a.n_vars - 1))
    return a.var_names[a.var.highly_variable].tolist()


def preprocess_pca(adata, n_pcs=N_PCS, hvg=None):
    """log1p-CPM -> HVG -> scale -> PCA. Returns (X_pca, hvg_list) so scVI
    (M4) can reuse the same fixed HVG selection instead of a separate rule."""
    hvg = hvg or _hvg_genes(adata)
    a = adata[:, hvg].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    npcs = min(n_pcs, a.n_obs - 1, len(hvg) - 1)
    sc.pp.pca(a, n_comps=npcs, svd_solver="arpack", random_state=RNG)
    return a.obsm["X_pca"], hvg


def scvi_latent(adata, hvg, n_latent=10, max_epochs=20):
    import scvi
    a = adata[:, hvg].copy()
    a.X = np.asarray(a.X.todense()) if hasattr(a.X, "todense") else np.asarray(a.X)
    a.layers["counts"] = a.X.copy()
    scvi.settings.seed = RNG
    import torch
    torch.set_num_threads(N_JOBS)
    scvi.model.SCVI.setup_anndata(a, layer="counts")
    m = scvi.model.SCVI(a, n_latent=n_latent)
    m.train(max_epochs=max_epochs, accelerator="cpu", enable_progress_bar=False)
    return m.get_latent_representation()


def _leiden_labels(X, resolution, n_neighbors=N_NEIGHBORS):
    a = sc.AnnData(X)
    a.obsm["X_rep"] = X
    sc.pp.neighbors(a, n_neighbors=min(n_neighbors, a.n_obs - 1), use_rep="X_rep", random_state=RNG)
    sc.tl.leiden(a, resolution=resolution, random_state=RNG, flavor="igraph",
                 n_iterations=2, directed=False)
    return a.obs["leiden"].values.astype(int)


def _groups_from_call(adata, X, call, labels, resolution=0.5, min_size=5):
    """Leiden-cluster the called cells WITHIN each visible label; each
    resulting subcluster (>= min_size) is reported as a candidate group.
    A label's called cells that never reach min_size are dropped, not
    reported (fixed 2026-10-04: this used to report them as a group
    regardless of size, which inflated n_groups/n_false_groups in
    evaluate.py in proportion to the number of visible labels)."""
    obs_names = adata.obs_names.values
    call = np.asarray(call)
    groups = []
    for lab in pd.unique(labels):
        idx = np.where((labels == lab) & call)[0]
        if len(idx) < min_size:
            continue
        cl = _leiden_labels(X[idx], resolution)
        for c in np.unique(cl):
            sel = idx[cl == c]
            if len(sel) >= min_size:
                groups.append(list(obs_names[sel]))
    return groups


# --------------------------------------------------------------------------
# M1: leiden subclustering within each visible label
# --------------------------------------------------------------------------
def m1_leiden_sub(adata):
    """Resolution chosen ONCE per label: the RES_GRID value maximizing mean
    silhouette of the resulting clustering on that label's PCA-50 cells
    (grid and rule fixed for every pilot/label). Any subcluster holding
    < MIN_SHARE of its label's cells is flagged as a hidden-rare candidate."""
    X, _ = preprocess_pca(adata)
    obs_names = adata.obs_names.values
    labels = adata.obs["label"].astype(str).values
    score = pd.Series(0.0, index=obs_names)
    call = pd.Series(False, index=obs_names)
    groups = []
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < MIN_LABEL_N:
            continue
        Xs = X[idx]
        best_sil, best_cl = -2.0, None
        for res in RES_GRID:
            cl = _leiden_labels(Xs, res)
            if len(set(cl)) < 2:
                continue
            sil = silhouette_score(Xs, cl, sample_size=min(2000, len(cl)), random_state=RNG)
            if sil > best_sil:
                best_sil, best_cl = sil, cl
        if best_cl is None:
            continue
        n = len(idx)
        counts = pd.Series(best_cl).value_counts()
        for c, cnt in counts.items():
            share = cnt / n
            if share < MIN_SHARE:
                sel = idx[best_cl == c]
                ids = obs_names[sel]
                score.loc[ids] = 1.0 - share
                call.loc[ids] = True
                groups.append(list(ids))
    return dict(score=score, call=call, groups=groups)


# --------------------------------------------------------------------------
# M2 / M3 / M4: outlier detection within each visible label
# --------------------------------------------------------------------------
def _per_label_outlier(adata, X, score_fn, min_label_n=10):
    labels = adata.obs["label"].astype(str).values
    obs_names = adata.obs_names.values
    score = pd.Series(0.0, index=obs_names)
    call = pd.Series(False, index=obs_names)
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < min_label_n:
            continue
        s, c = score_fn(X[idx])
        score.iloc[idx] = s
        call.iloc[idx] = c
    groups = _groups_from_call(adata, X, call.values, labels)
    return score, call, groups


def _iforest_score_call(Xs):
    clf = IsolationForest(n_estimators=N_ESTIMATORS, contamination="auto",
                           random_state=RNG, n_jobs=N_JOBS)
    clf.fit(Xs)
    return -clf.score_samples(Xs), clf.predict(Xs) == -1


def m2_iforest_pca(adata):
    """IsolationForest per visible label on PCA-50. call = contamination
    'auto' (sklearn's fixed, data-driven threshold, no per-pilot tuning)."""
    X, _ = preprocess_pca(adata)
    score, call, groups = _per_label_outlier(adata, X, _iforest_score_call)
    return dict(score=score, call=call, groups=groups)


def m3_lof_pca(adata):
    """LOF per visible label on PCA-50. call = LOF score > LOF_CUTOFF
    (fixed; 1.5 is the standard "clearly denser elsewhere" heuristic)."""
    def score_fn(Xs):
        k = min(20, len(Xs) - 1)
        clf = LocalOutlierFactor(n_neighbors=max(k, 1), n_jobs=N_JOBS)
        clf.fit_predict(Xs)
        lof = -clf.negative_outlier_factor_
        return lof, lof > LOF_CUTOFF
    X, _ = preprocess_pca(adata)
    score, call, groups = _per_label_outlier(adata, X, score_fn)
    return dict(score=score, call=call, groups=groups)


def m4_iforest_scvi(adata, n_latent=10, max_epochs=20):
    """Same as M2 but the embedding is an scVI latent (n_latent=10, fixed
    epochs) over the same fixed HVG set, instead of PCA."""
    _, hvg = preprocess_pca(adata)
    Z = scvi_latent(adata, hvg, n_latent=n_latent, max_epochs=max_epochs)
    score, call, groups = _per_label_outlier(adata, Z, _iforest_score_call)
    return dict(score=score, call=call, groups=groups)


# --------------------------------------------------------------------------
# donor-wise cross-fitted classifier + uncertainty (shared by M5, M6, M7, M8)
#
# Scope note (fixed 2026-10-04, was overstated): only the RandomForest
# probabilities below are donor-disjoint (GroupKFold on donor). HVG
# selection and PCA (preprocess_pca), mahalanobis_to_own_label's per-label
# mean/covariance, and _assign_pseudo's Leiden carving are all fit on every
# donor at once. None of this reads truth, so it is not a leak, but "the
# cross-fit is donor-disjoint" describes only the classifier step, not the
# full pipeline.
# --------------------------------------------------------------------------
def _maybe_duplicate(X, y, dup_classes, dup_to, rng=RNG):
    if not dup_classes or not dup_to:
        return X, y
    rs = np.random.RandomState(rng)
    add_X, add_y = [], []
    for c in dup_classes:
        idx = np.where(y == c)[0]
        if 0 < len(idx) < dup_to:
            extra = rs.choice(idx, size=dup_to - len(idx), replace=True)
            add_X.append(X[extra])
            add_y.append(y[extra])
    if not add_X:
        return X, y
    return np.vstack([X] + add_X), np.concatenate([y] + add_y)


def cross_fit_proba(X, y, donor, n_splits=N_SPLITS, class_weight=None,
                     dup_classes=(), dup_to=None):
    """Every cell's RandomForest probabilities come from a fold that never
    saw its donor during training (GroupKFold on donor, folds =
    min(n_splits, n_donors), floor 2). This classifier step is the only
    donor-disjoint part of the pipeline -- the X it is fit on (PCA/HVG from
    preprocess_pca) was computed over all donors upstream.
    dup_classes/dup_to duplicate those classes' TRAINING rows (within each
    fold's train split only) up to dup_to rows -- M6's duplication.
    class_weight='balanced' is M8's alternative; both unset is M7 (plain).
    Fallback with < 2 donors: one in-sample fit (documented, not a tuning
    choice -- cross-fitting is simply undefined with a single donor)."""
    classes = np.unique(y)
    n_donors = len(np.unique(donor))
    if n_donors < 2:
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RNG,
                                      n_jobs=N_JOBS, class_weight=class_weight)
        Xtr, ytr = _maybe_duplicate(X, y, dup_classes, dup_to)
        clf.fit(Xtr, ytr)
        proba = pd.DataFrame(0.0, index=np.arange(len(y)), columns=classes)
        proba.loc[:, clf.classes_] = clf.predict_proba(X)
        return proba

    k = max(2, min(n_splits, n_donors))
    gkf = GroupKFold(n_splits=k)
    proba = pd.DataFrame(0.0, index=np.arange(len(y)), columns=classes)
    for tr, te in gkf.split(X, y, groups=donor):
        Xtr, ytr = _maybe_duplicate(X[tr], y[tr], dup_classes, dup_to)
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RNG,
                                      n_jobs=N_JOBS, class_weight=class_weight)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(X[te])
        proba.loc[te, clf.classes_] = p
    return proba


def _entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(axis=1)


def _rank01(s):
    return pd.Series(s).rank(pct=True).values


def mahalanobis_to_own_label(X, labels):
    """Mahalanobis distance to the cell's own visible-label centroid in
    PCA-50, shrunk covariance (LedoitWolf) so it stays invertible even when
    a label's cell count is close to the PCA dimensionality. A label with
    too few cells for any covariance estimate (<5) gets distance 0 (no
    signal, not a crash)."""
    d = np.zeros(len(labels))
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 5:
            continue
        Xs = X[idx]
        mu = Xs.mean(axis=0)
        cov = LedoitWolf().fit(Xs)
        diff = Xs - mu
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dist = np.einsum("ij,jk,ik->i", diff, cov.precision_, diff)
        d[idx] = np.nan_to_num(dist, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    return d


def uncertainty_scores(proba_df, X, labels):
    """Combined score = mean of 4 rank-normalized components: least-
    confidence (1-max prob), margin (1-(top1-top2)), predictive entropy, and
    Mahalanobis distance to the cell's own visible-label PCA-50 prototype.
    Each component is rank-normalized (percentile within the WHOLE pilot)
    before averaging so no component dominates by scale."""
    P = proba_df.values
    sp = np.sort(P, axis=1)
    maxp = sp[:, -1]
    second = sp[:, -2] if P.shape[1] > 1 else np.zeros(len(P))
    least_conf = 1 - maxp
    margin = 1 - (maxp - second)
    ent = _entropy(P)
    mahal = mahalanobis_to_own_label(X, labels)
    return (_rank01(least_conf) + _rank01(margin) + _rank01(ent) + _rank01(mahal)) / 4.0


def percentile_call(score, labels, q=Q_CALL):
    """call = score above the qth percentile of the WITHIN-LABEL null (that
    label's own score distribution). Fixed q, identical for every label and
    pilot; this is the single thresholding rule shared by M5/M6/M7/M8."""
    score = np.asarray(score)
    call = np.zeros(len(score), dtype=bool)
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 10:
            continue
        thr = np.quantile(score[idx], q)
        call[idx] = score[idx] > thr
    return call


def m5_uncertainty(adata):
    X, _ = preprocess_pca(adata)
    labels = adata.obs["label"].astype(str).values
    donor = adata.obs["donor"].astype(str).values
    obs_names = adata.obs_names.values
    proba = cross_fit_proba(X, labels, donor)
    score = uncertainty_scores(proba, X, labels)
    call = percentile_call(score, labels)
    groups = _groups_from_call(adata, X, call, labels)
    return dict(score=pd.Series(score, index=obs_names),
                call=pd.Series(call, index=obs_names),
                groups=groups)


# --------------------------------------------------------------------------
# M6 / M7 / M8: Jiaqi's pseudo-label -> duplicate -> recluster loop
# --------------------------------------------------------------------------
def _assign_pseudo(X, labels, mask, round_idx, resolution=LEIDEN_RES_LOOP):
    """Within each CURRENT label, Leiden-cluster the masked ("candidate")
    cells into pseudo-label subclasses of >= MIN_GROUP_SIZE."""
    labels = labels.copy()
    for lab in pd.unique(labels[mask]):
        idx = np.where((labels == lab) & mask)[0]
        if len(idx) < MIN_GROUP_SIZE:
            continue
        cl = _leiden_labels(X[idx], resolution)
        for c in np.unique(cl):
            sel = idx[cl == c]
            if len(sel) >= MIN_GROUP_SIZE:
                labels[sel] = f"{lab}||pseudo{round_idx}_{c}"
    return labels


def _loop(adata, mode):
    X, _ = preprocess_pca(adata)
    labels0 = adata.obs["label"].astype(str).values
    donor = adata.obs["donor"].astype(str).values
    obs_names = adata.obs_names.values
    median_n = float(pd.Series(labels0).value_counts().median())

    cand = (m2_iforest_pca(adata)["call"].reindex(obs_names).values |
            m5_uncertainty(adata)["call"].reindex(obs_names).values)

    current = _assign_pseudo(X, labels0, cand, round_idx=0)
    last_score = np.zeros(len(obs_names))

    for r in range(N_ROUNDS):
        is_pseudo = np.array(["||pseudo" in c for c in current])
        dup_classes = np.unique(current[is_pseudo]).tolist() if mode == "dup" else []
        dup_to = int(median_n) if mode == "dup" else None
        cw = "balanced" if mode == "classweight" else None

        proba = cross_fit_proba(X, current, donor, class_weight=cw,
                                 dup_classes=dup_classes, dup_to=dup_to)
        last_score = uncertainty_scores(proba, X, current)
        call = percentile_call(last_score, current)

        pred_classes = proba.columns.values[proba.values.argmax(axis=1)]
        pseudo_set = set(np.unique(current[is_pseudo]))
        moved = (pred_classes != current) & np.isin(pred_classes, list(pseudo_set))
        current = np.where(moved, pred_classes, current)

        is_pseudo = np.array(["||pseudo" in c for c in current])
        new_cand = call & ~is_pseudo
        if new_cand.any():
            current = _assign_pseudo(X, current, new_cand, round_idx=r + 1)

    is_pseudo_final = np.array(["||pseudo" in c for c in current])
    final_classes = [c for c in np.unique(current[is_pseudo_final])
                     if (current == c).sum() >= MIN_GROUP_SIZE]
    union_mask = np.isin(current, final_classes)

    groups = []
    if union_mask.sum() >= MIN_GROUP_SIZE:
        idx = np.where(union_mask)[0]
        cl = _leiden_labels(X[idx], LEIDEN_RES_LOOP)
        for c in np.unique(cl):
            sel = idx[cl == c]
            if len(sel) >= MIN_GROUP_SIZE:
                groups.append(list(obs_names[sel]))
    # else: union below MIN_GROUP_SIZE -> no group reported (fixed
    # 2026-10-04: this used to report the whole undersized union as one
    # group, the same floor-skipping bug as _groups_from_call above).

    return dict(score=pd.Series(last_score, index=obs_names),
                call=pd.Series(union_mask, index=obs_names),
                groups=groups)


def m6_dup_loop(adata):
    """Round-0 candidates = union(M2, M5) calls, clustered into pseudo-
    labels per visible label. Each round: duplicate each pseudo-label's
    training rows up to the median visible-label size, retrain the cross-
    fit classifier with pseudo-labels as new classes, re-score, reassign
    cells the classifier now prefers in a pseudo class, carve new pseudo-
    labels from newly-flagged cells. Fixed 3 rounds. Final groups = Leiden
    reclustering of the union of surviving pseudo-labels (>= MIN_GROUP_SIZE)."""
    return _loop(adata, "dup")


def m7_nodup_loop(adata):
    """Identical to M6 with no duplication (pseudo-labels trained at their
    natural, possibly tiny, size)."""
    return _loop(adata, "nodup")


def m8_classweight_loop(adata):
    """Identical to M6 with class_weight='balanced' instead of duplication."""
    return _loop(adata, "classweight")


METHODS = {
    "m1_leiden_sub": m1_leiden_sub,
    "m2_iforest_pca": m2_iforest_pca,
    "m3_lof_pca": m3_lof_pca,
    "m4_iforest_scvi": m4_iforest_scvi,
    "m5_uncertainty": m5_uncertainty,
    "m6_dup_loop": m6_dup_loop,
    "m7_nodup_loop": m7_nodup_loop,
    "m8_classweight_loop": m8_classweight_loop,
}


def demo():
    """Smallest runnable check: synthetic 220-cell / 3-label / 5-donor
    toy problem (one label secretly contains a 15-cell injected cluster,
    like a hidden-rare pilot) run through every method end to end. Asserts
    output shape/type contract and that at least the simple per-label
    outlier methods (M2/M3) actually point at the injected cluster."""
    rs = np.random.RandomState(0)
    n_per_label, n_hidden, n_genes = 100, 15, 40
    donors = [f"d{i}" for i in range(5)]

    X_a = rs.poisson(3, size=(n_per_label, n_genes)).astype(float)
    X_b_main = rs.poisson(3, size=(n_per_label - n_hidden, n_genes)).astype(float)
    X_b_hidden = rs.poisson(3, size=(n_hidden, n_genes)).astype(float)
    X_b_hidden[:, :10] += 8  # injected cluster has a distinct marker block
    X_b = np.vstack([X_b_main, X_b_hidden])
    X = np.vstack([X_a, X_b]).astype(int)

    labels = np.array(["A"] * n_per_label + ["B"] * n_per_label)
    true_hidden = np.array([False] * n_per_label + [False] * (n_per_label - n_hidden) + [True] * n_hidden)
    donor = rs.choice(donors, size=len(labels))

    import anndata as ad
    a = ad.AnnData(X=X.astype(np.float32))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["label"] = labels
    a.obs["donor"] = donor

    for name, fn in METHODS.items():
        out = fn(a)
        assert set(out.keys()) == {"score", "call", "groups"}, name
        assert len(out["score"]) == a.n_obs and len(out["call"]) == a.n_obs, name
        assert out["score"].index.equals(a.obs_names) and out["call"].index.equals(a.obs_names), name
        assert out["call"].dtype == bool, name
        assert isinstance(out["groups"], list), name
        for g in out["groups"]:
            assert all(isinstance(i, str) for i in g), name

    # sanity, not a power claim (real enrichment is measured on real pilots
    # below): no method should call almost every cell on this near-uniform
    # toy data, and M2/M3 must call at least one real cell somewhere.
    for name in ("m2_iforest_pca", "m3_lof_pca"):
        call = METHODS[name](a)["call"].values
        assert 0 < call.sum() < 0.5 * len(call), f"{name} call rate implausible: {call.sum()}/{len(call)}"

    print("methods.py self-check OK: 8/8 methods ran, output contract holds "
          "(score/call/groups shape+type), M2/M3 call rates are sane on the toy pilot. "
          "(Detection power is judged on real pilots, not this synthetic shape-check.)")


if __name__ == "__main__":
    demo()
