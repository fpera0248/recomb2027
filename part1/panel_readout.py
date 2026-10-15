"""Four-classifier readout (lr, knn, rf, mlp) of the saved generator and statistical cells of one dataset.
python3 panel_readout.py run | report"""

import sys, os, glob, numpy as np, pandas as pd

HL = os.environ.get('GENE_DIR', 'hlca_genes')
CSV = os.environ.get('READOUT_CSV', 'panel_readout.csv')  # panel run: per-dataset dirs
from statistical import readout, gene_method

METHODS = sorted(
    {os.path.basename(f).split('_', 1)[1][:-4] for f in glob.glob(f'{HL}/s*_*.npz')}
    - {'base', 'sd3'}
    | {'sd3_nosingle'}
)
_fn = lambda s, a: next(
    (
        f
        for f in (f'{HL}/s{s}_{a}.npz',)
        + ((f'{HL}/s{s}_sd3.npz',) if a == 'sd3_nosingle' else ())
        if os.path.exists(f)
    ),
    None,
)


def _base(seed):
    b = np.load(f'{HL}/s{seed}_base.npz')
    comp, pmean = b['comp'], b['pmean']
    Ztr, ytr = b['Ztr'], b['ytr']
    Zev, yev = (b['Xev'] - pmean) @ comp.T, b['yev']
    return b, comp, pmean, Ztr, ytr, Zev, yev


def _rec_tie(p, t, k):
    thr = np.sort(p)[::-1][k - 1]
    hi = p > thr
    tie = p == thr
    return float((t[hi].sum() + t[tie].mean() * (k - hi.sum())) / k)


def _score(Z, y, Zev, yev, cts, n_c, probe, NEVc, seed, method):
    rows = []
    for k in ('lr', 'knn', 'rf', 'mlp'):
        P, pred = readout(k, Z, y, np.ones(len(y)), Zev, seed)
        if P.shape[1] < len(cts):
            Pf = np.full((len(Zev), len(cts)), -np.inf)
            Pf[:, np.unique(y)] = P
            P = Pf
        for c in range(len(cts)):
            t = yev == c
            nevc = int(NEVc[c])
            rows.append(
                dict(
                    seed=seed,
                    method=method,
                    clf=k,
                    c=c,
                    name_=cts[c],
                    n=int(n_c[c]),
                    probe=bool(c in set(probe.tolist())),
                    nev_used=nevc,
                    rec200=float(t[np.argsort(-P[:, c])[:nevc]].mean()),
                    rec200_tie=_rec_tie(P[:, c], t, nevc),
                    rec_argmax=float((pred[t] == c).mean()),
                    ncall=int((pred == c).sum()),
                )
            )
    return rows


def fit(seed, method):
    b, comp, pmean, Ztr, ytr, Zev, yev = _base(seed)
    a = np.load(_fn(seed, method))
    Z = np.vstack([Ztr, (a['X'] - pmean) @ comp.T])
    y = np.concatenate([ytr, a['y']])
    return _score(
        Z, y, Zev, yev, b['cts'], b['n_c'], b['probe'], b['NEVc'], seed, method
    )


def fit_noise(seed, method):
    b, comp, pmean, Ztr, ytr, Zev, yev = _base(seed)
    n_c, probe = b['n_c'], b['probe']
    up = np.load(f'{HL}/s{seed}_upsample.npz')
    T = int(n_c[probe[0]] + (up['y'] == probe[0]).sum())
    A = gene_method(method, b['Xtr'], ytr, probe, n_c, seed, T)
    Z = np.vstack([Ztr] + [(v[0] - pmean) @ comp.T for v in A.values()])
    y = np.concatenate([ytr] + [v[1] for v in A.values()])
    return _score(Z, y, Zev, yev, b['cts'], n_c, probe, b['NEVc'], seed, method)


if __name__ == '__main__' and sys.argv[1] == 'run':
    from joblib import Parallel, delayed

    seeds = range(len(glob.glob(f'{HL}/s*_base.npz')))
    jobs = [
        delayed(fit)(s, a) for s in seeds for a in METHODS if _fn(s, a)
    ]  # sdiff can be budget-dropped for later seeds
    jobs += [
        delayed(fit_noise)(s, a)
        for s in seeds
        for a in ('jitter_s1.0', 'jitternz_s1.0', 'smote_k5')
    ]
    out = Parallel(n_jobs=int(os.environ.get('NJ', 3)), verbose=5)(jobs)
    pd.DataFrame([r for o in out for r in o]).to_csv(CSV, index=False)
    print('wrote panel_readout.csv')

if __name__ == '__main__' and sys.argv[1] == 'report':
    import signflip

    S = pd.read_csv(CSV)
    idx = ['seed', 'name_', 'n']
    for k, Sk in S.groupby('clf', sort=False):
        Pk = Sk[Sk.probe]
        w = Pk.pivot_table(index=idx, columns='method', values='rec200')
        grp = w.index.get_level_values('name_').values
        print(
            f"\n### {k}  ({Sk.seed.nunique()} seeds x {w.index.get_level_values('name_').nunique()} hlca probe types)"
        )
        print(f"{'method':14s}{'rec200':>8s}{'d vs dup':>10s}{'types+':>8s}{'p':>8s}")
        for a in sorted(w.columns, key=lambda a: -w[a].mean()):
            if 'upsample' not in w.columns:
                continue
            m = w[a].notna().values
            d = (w[a] - w['upsample']).values[m]
            p, dt = signflip.signflip_p(d, grp[m])
            print(
                f"{a:14s}{w[a].mean():8.3f}{d.mean():+10.3f}{int((dt > 0).sum()):>5d}/{len(dt)}{p:8.3f}"
                + (f'  ({m.sum()} of {len(m)} units)' if not m.all() else '')
            )
