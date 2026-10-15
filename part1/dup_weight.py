"""Duplication versus sample weighting: the same logistic fit with weights in place of copies."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
TARGET, NEV = 256, 200

def one(seed):
    d = np.load(f'split_seed{seed}.npz'); Z, y, Zev, yev, probe, n_c = d['Ztr'], d['ytr'], d['Zev'], d['yev'], d['probe'], d['n_c']
    rng = np.random.default_rng([seed, sum(map(ord, 'upsample'))])          # statistical's upsample stream
    draws = np.concatenate([rng.choice(np.where(y == c)[0], TARGET - int(n_c[c]), replace=True) for c in probe if TARGET > n_c[c]])
    w = np.ones(len(y)); np.add.at(w, draws, 1)
    assert w.sum() == len(y) + len(draws)
    fit = lambda X, t, sw=None: LogisticRegression(max_iter=2000, random_state=seed).fit(X, t, sample_weight=sw)
    a = fit(np.vstack([Z, Z[draws]]), np.concatenate([y, y[draws]])); b = fit(Z, y, w)
    Pa, Pb = a.predict_proba(Zev), b.predict_proba(Zev)
    rows = [dict(seed=seed, c=int(c), n=int(n_c[c]), rec_dup=(yev[np.argsort(-Pa[:, c])[:NEV]] == c).mean(),
                 rec_w=(yev[np.argsort(-Pb[:, c])[:NEV]] == c).mean()) for c in probe]
    for r in rows: r.update(max_dP=float(np.abs(Pa - Pb).max()), max_dcoef=float(np.abs(a.coef_ - b.coef_).max()),
                            coef_scale=float(np.abs(a.coef_).max()), argmax_agree=float((Pa.argmax(1) == Pb.argmax(1)).mean()))
    return rows

if __name__ == '__main__':
    R = pd.DataFrame([r for o in Parallel(n_jobs=int(os.environ.get('NJ', 7)))(delayed(one)(s) for s in range(20)) for r in o])
    R.to_csv('dup_weight.csv', index=False)
    old = pd.read_csv('statistical.csv'); old = old[(old.method == 'upsample') & old.probe][['seed', 'c', 'rec200']]
    x = R.merge(old, on=['seed', 'c'])
    print(f"parity with statistical upsample: max |d rec200| = {(x.rec_dup - x.rec200).abs().max()}")
    print(f"dup vs weight, 20 seeds x 8 types: max |d rec200| = {(R.rec_dup - R.rec_w).abs().max():.4f}, "
          f"mean d = {(R.rec_dup - R.rec_w).mean():+.5f}")
    g = R.groupby('seed').first()
    print(f"max |d prob| over all eval cells = {g.max_dP.max():.2e}; max |d coef| = {g.max_dcoef.max():.2e} "
          f"(coef scale {g.coef_scale.median():.1f}); argmax agreement min = {g.argmax_agree.min():.5f}")
