"""Foundation-model duplication analysis: does duplication combined with foundation-model embeddings give a more
effective rare-type classifier than PCA-50?

Dataset: AIDA Singapore, full cohort, ~/work/aida_sg/aida_sg.h5ad
(355,046 cells, 216 donors, 32 cell types; raw counts in X). FM embeddings: GF-6L-30M frozen,
~/work/gf6l_emb/emb/emb_%04d.npy, 256-dim, order-matched to h5ad rows
(manifest.json: src_h5ad=aida_sg.h5ad, n_cells=355046, shard=10000; 36 files of 10000/10000/.../5046
sum exactly to 355046, confirming contiguous in-order sharding).

Protocol (gen16-style, read at a matched operating point): 8 probe rare types (the 8
rarest AIDA Singapore types for which an 80/20 donor-disjoint split still leaves >=128 train-donor
cells in every seed: mature B cell, CD16-neg CD56-bright NK, B cell, plasmacytoid DC, plasma cell,
pre-conventional DC, innate lymphoid cell, CD141+ mDC), RUNGS=[1,2,4,8,16,32,64,128] permuted onto
the 8 types per seed (bijection, reshuffled every seed), 3 seeds, donor-disjoint 80/20 train/eval
donor split per seed. Arms: none, dup, classweight (lr/rf only), jitter (s=1.0, pooled abundant sd),
smote (k=5); every augmented arm tops each probe type up to TARGET=128. Classifiers: lr/knn/rf/mlp
(stat16.readout, imported not copied) + a plain nearest-prototype readout (real cells only:
augmentation cannot move a mean-based readout, so it is not re-run per arm). Two representations on the SAME cells: FM (256-dim, frozen) and
PCA-50 of log1p-CPM (TruncatedSVD, fit per seed on that seed's real training pool).

Matched operating point (primary): top-NEVc eval cells by score, NEVc = that type's held-out count.
Default threshold (secondary): argmax recall, ncall, precision, FPR.
"""
import os, sys, json, warnings, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.utils.extmath')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.linear_model._linear_loss')
# Known limit: TruncatedSVD's randomized power-iteration step, and LogisticRegression(class_weight=
# 'balanced') on a near-quasi-separable n=1 class, can overflow float32 in an intermediate lbfgs
# step before regularization + re-normalization pulls the final answer back to something finite
# (verified by hand: 256-dim synthetic 1-vs-100 case above converges, predict_proba all finite,
# max |coef| 0.74). Final output is asserted finite everywhere below, so this is cosmetic -- swap
# TruncatedSVD's algorithm='arpack' or drop class_weight='balanced' if a real dataset ever trips
# the assertion.
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.path.insert(0, HR_WORK + '/cfgen_natrare')
from stat16 import dup, jitter, smote, lognorm, readout  # reused, not copied
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

AIDA_H5 = HR_WORK + '/aida_sg/aida_sg.h5ad'
EMB_DIR = HR_WORK + '/gf6l_emb/emb'
OUT = os.path.dirname(os.path.abspath(__file__))

PROBE = ['mature B cell', 'CD16-negative, CD56-bright natural killer cell, human', 'B cell',
         'plasmacytoid dendritic cell', 'plasma cell', 'pre-conventional dendritic cell',
         'innate lymphoid cell', 'CD141-positive myeloid dendritic cell']
RUNGS = [1, 2, 4, 8, 16, 32, 64, 128]
TARGET = 128
SEEDS = [0, 1, 2]
TRAIN_DONOR_FRAC = 0.8
ABUND_TRAIN_CAP, ABUND_EVAL_CAP = 100, 40
CLFS = ['lr', 'knn', 'rf', 'mlp']


def load_fm_emb():
    files = sorted(f for f in os.listdir(EMB_DIR) if f.startswith('emb_') and f.endswith('.npy'))
    return np.concatenate([np.load(os.path.join(EMB_DIR, f)) for f in files], axis=0)


def load_obs():
    import scanpy as sc
    a = sc.read_h5ad(AIDA_H5, backed='r')
    obs = a.obs[['cell_type', 'donor_id']].copy()
    obs['cell_type'] = np.asarray(obs['cell_type'].astype(str).values, dtype=object)
    obs['donor_id'] = np.asarray(obs['donor_id'].astype(str).values, dtype=object)
    obs.index = np.arange(len(obs))
    return obs, a


def smoke_test(FM, obs):
    """ONE runnable check: FM embeddings must separate known abundant cell types far above chance.
    kNN on a small sample across the 3 most abundant types; chance = 1/3."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    ct = np.asarray(obs.cell_type.values, dtype=object)
    top3 = pd.Series(ct).value_counts().index[:3]
    idx = np.concatenate([np.where(ct == t)[0][:400] for t in top3])
    y = ct[idx]
    Xtr, Xte, ytr, yte = train_test_split(FM[idx], y, test_size=0.3, random_state=0, stratify=y)
    acc = KNeighborsClassifier(15).fit(Xtr, ytr).score(Xte, yte)
    assert acc > 0.6, f'FM embedding order/content sanity check FAILED: 3-class kNN acc={acc:.3f} (chance 0.33)'
    print(f'[smoke] FM embedding sanity: 3-class top-type kNN acc={acc:.3f} (chance 0.33) -- PASS')


def pooled_sd(Z, y, min_n=100):
    ab = [c for c in np.unique(y) if (y == c).sum() >= min_n]
    return np.sqrt(np.mean([Z[y == c].var(0) for c in ab], axis=0))


def embed_arm(name, Z, y, probe_ids, rung_of, seed, target, sd=None):
    """Top each probe class up to `target`, built directly in embedding space (no gene-space
    renorm -- that step is specific to log1p-CPM and does not apply here)."""
    if name in ('none', 'classweight'):
        return Z, y
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    eZ, ey = [], []
    for c in probe_ids:
        k = target - int(rung_of[c])
        if k <= 0:
            continue
        if name == 'dup':
            Xa, ya, _ = dup(Z, y, c, k, rng)
        elif name == 'jitter':
            Xa, ya, _ = jitter(Z, y, c, k, rng, s=1.0, sd=sd)
        elif name == 'smote':
            Xa, ya, _ = smote(Z, y, c, k, rng, kn=5)
        else:
            raise ValueError(name)
        eZ.append(Xa); ey.append(ya)
    if eZ:
        return np.vstack([Z] + eZ), np.concatenate([y] + ey)
    return Z, y


def score_rows(P, pred, yev, probe_ids, nevc, seed, arm, clf, rep, rung_of, n_c):
    rows = []
    for c in probe_ids:
        t = yev == c
        N = int(nevc[c])
        order = np.argsort(-P[:, c])
        matched = float(t[order[:N]].mean()) if N > 0 else np.nan
        argmax_hit = (pred == c)
        rec_argmax = float((pred[t] == c).mean()) if t.sum() else np.nan
        ncall = int(argmax_hit.sum())
        prec_argmax = float(t[argmax_hit].mean()) if ncall else np.nan
        fpr_argmax = float((argmax_hit & ~t).sum() / max((~t).sum(), 1))
        rows.append(dict(seed=seed, rep=rep, arm=arm, clf=clf, type=c, n_train=int(rung_of[c]),
                          n_eval=N, rec200=matched, rec_argmax=rec_argmax, prec_argmax=prec_argmax,
                          ncall=ncall, fpr_argmax=fpr_argmax))
    return rows


def run_seed(seed, obs, FM, LAB2I, ALL_ABUND):
    rng = np.random.default_rng(seed)
    donors = np.array(obs.donor_id.unique().tolist())  # force real ndarray: pandas' ArrowStringArray
    rng.shuffle(donors)                                 # is not safely in-place shufflable (Numpy warns)
    n_train_d = int(len(donors) * TRAIN_DONOR_FRAC)
    train_donors, eval_donors = set(donors[:n_train_d]), set(donors[n_train_d:])
    is_train = obs.donor_id.isin(train_donors).values
    is_eval = ~is_train
    ct = obs.cell_type.values

    perm = rng.permutation(RUNGS)
    rung_of_name = dict(zip(PROBE, perm))
    rung_of = {LAB2I[p]: int(rung_of_name[p]) for p in PROBE}
    probe_ids = [LAB2I[p] for p in PROBE]

    train_idx_parts, y0_parts, nevc = [], [], {}
    eval_idx_parts, yev_parts = [], []
    cap_report = []
    for p in PROBE:
        c = LAB2I[p]
        pool = np.where((ct == p) & is_train)[0]
        r = rung_of[c]
        if r > len(pool):
            cap_report.append((p, r, len(pool)))
            r = max([x for x in RUNGS if x <= len(pool)], default=0)
            rung_of[c] = r
        draw = rng.choice(pool, size=r, replace=False) if r > 0 else np.array([], dtype=int)
        train_idx_parts.append(draw); y0_parts.append(np.full(len(draw), c))
        ev = np.where((ct == p) & is_eval)[0]
        nevc[c] = len(ev)
        eval_idx_parts.append(ev); yev_parts.append(np.full(len(ev), c))
    for a in ALL_ABUND:
        c = LAB2I[a]
        tr_pool = np.where((ct == a) & is_train)[0]
        tr = rng.choice(tr_pool, size=min(ABUND_TRAIN_CAP, len(tr_pool)), replace=False)
        train_idx_parts.append(tr); y0_parts.append(np.full(len(tr), c))
        ev_pool = np.where((ct == a) & is_eval)[0]
        ev = rng.choice(ev_pool, size=min(ABUND_EVAL_CAP, len(ev_pool)), replace=False)
        eval_idx_parts.append(ev); yev_parts.append(np.full(len(ev), c))

    train_idx = np.concatenate(train_idx_parts); y0 = np.concatenate(y0_parts)
    eval_idx = np.concatenate(eval_idx_parts); yev = np.concatenate(yev_parts)
    nevc_arr = np.zeros(len(LAB2I), dtype=int)
    for c, n in nevc.items(): nevc_arr[c] = n

    return dict(seed=seed, train_idx=train_idx, y0=y0, eval_idx=eval_idx, yev=yev,
                rung_of=rung_of, probe_ids=probe_ids, nevc=nevc_arr, cap_report=cap_report)


def pca50_fit_transform(h5, train_idx, eval_idx):
    """log1p-CPM then a per-seed TruncatedSVD(50), fit on the real training pool, projecting both
    pools (same cells as the FM arm, same-processing rule)."""
    all_idx = np.concatenate([train_idx, eval_idx])
    order = np.argsort(all_idx)
    uniq_idx, inv = np.unique(all_idx, return_inverse=True)
    Xraw = h5[uniq_idx].X
    Xraw = np.asarray(Xraw.todense() if hasattr(Xraw, 'todense') else Xraw, dtype=np.float32)
    Xln = lognorm(Xraw)
    pos = {u: i for i, u in enumerate(uniq_idx)}
    tr_pos = np.array([pos[i] for i in train_idx]); ev_pos = np.array([pos[i] for i in eval_idx])
    svd = TruncatedSVD(n_components=50, random_state=0)
    Ztr = svd.fit_transform(Xln[tr_pos])
    Zev = svd.transform(Xln[ev_pos])
    assert np.isfinite(Ztr).all() and np.isfinite(Zev).all(), 'non-finite PCA-50 output'
    return Ztr.astype(np.float32), Zev.astype(np.float32)


def fit_and_score(name, clf, Z0, y0, Zev, yev, probe_ids, rung_of, nevc, seed, rep, n_c):
    sd = pooled_sd(Z0, y0) if name == 'jitter' else None
    Z, y = embed_arm(name, Z0, y0, probe_ids, rung_of, seed, TARGET, sd=sd)
    if name == 'classweight':
        if clf not in ('lr', 'rf'):
            return []
        m = (LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced') if clf == 'lr'
             else RandomForestClassifier(n_estimators=500, random_state=seed, class_weight='balanced', n_jobs=2))
        m.fit(Z, y)
        P = np.full((len(Zev), len(n_c)), -np.inf); P[:, m.classes_] = m.predict_proba(Zev)
        pred = m.classes_[m.predict_proba(Zev).argmax(1)]
    else:
        P, pred = readout(clf, Z, y, np.ones(len(y)), Zev, seed)
        if P.shape[1] < len(n_c):
            Pf = np.full((len(Zev), len(n_c)), -np.inf); Pf[:, np.unique(y)] = P; P = Pf
    finite = np.isfinite(P) | (P == -np.inf)
    assert finite.all(), f'non-finite score from {name}/{clf}/{rep}/seed{seed}'
    return score_rows(P, pred, yev, probe_ids, nevc, seed, name, clf, rep, rung_of, n_c)


def prototype_score(Z0, y0, Zev, yev, probe_ids, rung_of, nevc, seed, rep, n_c):
    classes = np.unique(y0)
    proto = np.stack([Z0[y0 == c].mean(0) for c in classes])
    d2 = ((Zev[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
    P = np.full((len(Zev), len(n_c)), -np.inf); P[:, classes] = -d2
    pred = classes[(-d2).argmax(1)]
    return score_rows(P, pred, yev, probe_ids, nevc, seed, 'none', 'prototype', rep, rung_of, n_c)


def ovr_check(Z0, y0, Zev, yev, probe_ids, rung_of, nevc, seed, rep):
    """One-vs-rest addendum: one-vs-rest LR vs the multiclass LR already scored, for arms none/dup."""
    rows = []
    for arm in ('none', 'dup'):
        Z, y = embed_arm(arm, Z0, y0, probe_ids, rung_of, seed, TARGET)
        for c in probe_ids:
            yb = (y == c).astype(int)
            if yb.sum() < 2 or yb.sum() == len(yb):
                continue
            m = LogisticRegression(max_iter=2000, random_state=seed).fit(Z, yb)
            score = m.predict_proba(Zev)[:, 1]
            t = yev == c; N = int(nevc[c])
            order = np.argsort(-score)
            matched = float(t[order[:N]].mean()) if N > 0 else np.nan
            rows.append(dict(seed=seed, rep=rep, arm=arm, type=c, n_train=int(rung_of[c]),
                              n_eval=N, rec200_ovr=matched))
    return rows


def standardize(Z0, Zev):
    """Per-feature z-score from the real training pool, applied to both pools. Needed because the raw
    FM embedding scale (std ~0.02, max 0.6) is far below sklearn LogisticRegression's default C=1
    regularization scale and overflowed lbfgs's line search (observed: RuntimeWarning overflow in
    sklearn linear_loss matmul on the first run); PCA-50 gets the same treatment for a fair, same-
    processing comparison."""
    mu, sd = Z0.mean(0), Z0.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    return ((Z0 - mu) / sd).astype(np.float32), ((Zev - mu) / sd).astype(np.float32)


def main():
    print('loading obs + FM embeddings...', flush=True)
    obs, h5 = load_obs()
    FM = load_fm_emb()
    assert FM.shape[0] == len(obs), (FM.shape, len(obs))
    smoke_test(FM, obs)

    LAB = sorted(obs.cell_type.unique())
    LAB2I = {l: i for i, l in enumerate(LAB)}
    ALL_ABUND = [l for l in LAB if l not in PROBE]
    assert len(ALL_ABUND) + len(PROBE) == len(LAB) == 32

    all_rows, ovr_rows, caps = [], [], []
    for seed in SEEDS:
        print(f'--- seed {seed} ---', flush=True)
        S = run_seed(seed, obs, FM, LAB2I, ALL_ABUND)
        caps += [(seed, p, r, avail) for p, r, avail in S['cap_report']]
        for rep, (Z0, Zev) in (('fm', standardize(FM[S['train_idx']], FM[S['eval_idx']])),
                                ('pca50', standardize(*pca50_fit_transform(h5, S['train_idx'], S['eval_idx'])))):
            n_c = np.arange(len(LAB))
            for name in ('none', 'dup', 'classweight', 'jitter', 'smote'):
                for clf in CLFS:
                    all_rows += fit_and_score(name, clf, Z0, S['y0'], Zev, S['yev'], S['probe_ids'],
                                               S['rung_of'], S['nevc'], seed, rep, n_c)
            all_rows += prototype_score(Z0, S['y0'], Zev, S['yev'], S['probe_ids'], S['rung_of'],
                                         S['nevc'], seed, rep, n_c)
            if rep == 'fm':
                ovr_rows += ovr_check(Z0, S['y0'], Zev, S['yev'], S['probe_ids'], S['rung_of'],
                                       S['nevc'], seed, rep)
            print(f'  {rep} done, {len(all_rows)} rows so far', flush=True)

    I2LAB = {v: k for k, v in LAB2I.items()}
    df = pd.DataFrame(all_rows); df['type_name'] = df['type'].map(I2LAB)
    df.to_csv(os.path.join(OUT, 'results.csv'), index=False)
    ov = pd.DataFrame(ovr_rows); ov['type_name'] = ov['type'].map(I2LAB)
    ov.to_csv(os.path.join(OUT, 'results_ovr.csv'), index=False)
    with open(os.path.join(OUT, 'rung_caps.json'), 'w') as f:
        json.dump(caps, f, indent=1)
    print(f'wrote {len(df)} rows -> results.csv, {len(ov)} rows -> results_ovr.csv, '
          f'{len(caps)} rung caps -> rung_caps.json')


if __name__ == '__main__':
    main()
