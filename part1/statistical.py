"""Statistical augmentation conditions (duplication, jitter, SMOTE, published schemes) on the cached AIDA split.

Every condition tops a probe type up to TARGET cells in gene space (log1p-CPM), renormalizes and projects with the cached PCA; readout as in aida.py.
  python3 statistical.py run 0 20     fit every method on seeds 0..19
  python3 statistical.py report       paired analysis against duplication, memorization (DCR) table
"""
import sys, os, time, importlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

TARGET, NEV, KNN = 256, 200, 15
REP = os.environ.get('REP', 'split')   # splittrain = train-only HVGs (HVG_TRAIN=1 split_cache.py)

ANCHORS = None

def dup(Z, y, c, k, rng):
    pos = np.where(y == c)[0]; i = rng.choice(pos, k, replace=True)
    if ANCHORS is not None: ANCHORS.append(i)
    return Z[i], np.full(k, c), np.ones(k)

def jitter(Z, y, c, k, rng, s, sd):
    Zd, yd, wd = dup(Z, y, c, k, rng); return Zd + rng.normal(size=Zd.shape) * s * sd, yd, wd

def jitter_nz(Z, y, c, k, rng, s, sd):
    """jitter on the nonzero entries only (gene space)"""
    Zd, yd, wd = dup(Z, y, c, k, rng); return Zd + (Zd > 0) * rng.normal(size=Zd.shape) * s * sd, yd, wd

def smote(Z, y, c, k, rng, kn):
    X = Z[y == c]; n = len(X)
    if n == 1: return dup(Z, y, c, k, rng)
    kk = n - 1 if kn is None else min(kn, n - 1)
    ind = NearestNeighbors(n_neighbors=kk + 1).fit(X).kneighbors(X)[1]
    i = rng.integers(n, size=k); j = ind[i, rng.integers(1, kk + 1, size=k)]
    if ANCHORS is not None: ANCHORS.append(np.where(y == c)[0][i])
    return X[i] + rng.random((k, 1)) * (X[j] - X[i]), np.full(k, c), np.ones(k)

def lognorm(C):
    C = np.maximum(np.asarray(C, np.float32), 0)
    return np.log1p(C / np.maximum(C.sum(1, keepdims=True), 1) * 1e4).astype(np.float32)

def renorm(X):
    """lognorm(expm1(X)) without overflow (sdiff decoder output reaches 1768 at smoke budgets)"""
    # clip to the valid log1p-CPM ceiling log1p(1e4): sdiff's ReLU decoder reaches ~65, and unclipped
    # expm1 turns such rows into one-gene cells; identity on every real (renormalized) row
    X = np.clip(np.asarray(X, np.float64), 0, np.log1p(1e4)); m = X.max(1, keepdims=True)
    E = np.exp(X - m) - np.exp(-m); S = E.sum(1, keepdims=True)
    return np.log1p(E / np.where(S > 0, S, 1) * 1e4).astype(np.float32)

def gene_method(name, X, y, probe, n_c, seed, T=TARGET, extra=None):
    f = methods(X, y, seed)[name]; rng = np.random.default_rng([seed, sum(map(ord, name))]); out = {}
    for c in probe:
        k = T - int(n_c[c])
        if f is None or k <= 0: continue
        Xa, ya, wa = f(X, y, c, k, rng, **(extra or {}))
        assert abs(wa.sum() - k) < 1e-6 and Xa.shape[1] == X.shape[1], (name, c)
        out[int(c)] = (renorm(Xa), np.asarray(ya, int), np.asarray(wa, float))
    return out

def readout(k, Z, y, w, Zev, seed):
    """Per-class scores (column c = class c) and argmax calls."""
    if k == 'lr':
        m = LogisticRegression(max_iter=2000, random_state=seed).fit(Z, y, sample_weight=None if np.all(w == 1) else w)
    else:
        assert np.all(w == 1), k    # soft weights have no kNN analogue, and sklearn 1.5's MLP takes no sample_weight
        if k == 'knn':
            pred = KNeighborsClassifier(n_neighbors=KNN).fit(Z, y).predict(Zev)
            P = np.stack([-NearestNeighbors(n_neighbors=min(KNN, int((y == c).sum()))).fit(Z[y == c]).kneighbors(Zev)[0].mean(1)
                          for c in np.unique(y)], 1)
            return P, pred
        m = (RandomForestClassifier(n_estimators=500, random_state=seed) if k == 'rf'
             else MLPClassifier(max_iter=500, random_state=seed)).fit(Z, y)                   # sklearn default MLP, (100,) relu adam
    P = m.predict_proba(Zev); return P, m.classes_[P.argmax(1)]

def methods(Z, y, seed):
    ab = [c for c in np.unique(y) if (y == c).sum() >= 100]
    sd = np.sqrt(np.mean([Z[y == c].var(0) for c in ab], 0))        # pooled within-class SD per feature (gene)
    A = {'real': None, 'upsample': dup,
         'jitter_s0.25': lambda *a: jitter(*a, s=0.25, sd=sd), 'jitter_s0.5': lambda *a: jitter(*a, s=0.5, sd=sd),
         'jitter_s1.0': lambda *a: jitter(*a, s=1.0, sd=sd), 'jitter_s2.0': lambda *a: jitter(*a, s=2.0, sd=sd),
         'jitter_s4.0': lambda *a: jitter(*a, s=4.0, sd=sd),
         'jitternz_s0.5': lambda *a: jitter_nz(*a, s=0.5, sd=sd), 'jitternz_s1.0': lambda *a: jitter_nz(*a, s=1.0, sd=sd),
         'jitternz_s2.0': lambda *a: jitter_nz(*a, s=2.0, sd=sd),
         'smote_k5': lambda *a: smote(*a, kn=5), 'smote_kall': lambda *a: smote(*a, kn=None)}
    for mod, grid in (('aug_li2025', 'LI_GRID'), ('aug_jmo', 'JMO_GRID')):
        if not os.path.exists(mod + '.py'): continue
        m = importlib.import_module(mod)
        for tag, params in getattr(m, grid, {'': {}}).items():
            A[mod[4:] + tag] = (lambda m, p: lambda *a, **kw: m.augment(*a, **p, **kw))(m, dict(params))
    return A

def fit_method(seed, name):
    d = np.load(f'{REP}_seed{seed}.npz'); Z, y, Zev, yev = d['Ztr'], d['ytr'], d['Zev'], d['yev']
    X, pm, W = d['Xtr'], d['pmean'], d['comp']
    assert np.allclose((X - pm) @ W.T, Z, atol=1e-3), 'cache Xtr does not project to Ztr'
    probe, cts, n_c = d['probe'], d['cts'], d['n_c']; NC = len(cts)
    Zs, ys, ws, dcr = [Z], [y], [np.ones(len(y))], []
    t = time.time(); extra = {}
    if name.startswith('jmo'):
        import aug_jmo; extra['P'] = aug_jmo.fit_reference(X, y, seed)
    for c, (Xa, ya, wa) in gene_method(name, X, y, probe, n_c, seed, extra=extra).items():
        Za = (Xa - pm) @ W.T
        Zs.append(Za); ys.append(ya); ws.append(wa)
        own = Z[y == c]; dist = NearestNeighbors(n_neighbors=1).fit(own).kneighbors(Za)[0][:, 0]
        dcr.append(dict(c=int(c), dcr_med=float(np.median(dist)), frac_copy=float((dist < 1e-9).mean())))
    gen_sec = time.time() - t
    Zf, yf, wf = np.vstack(Zs), np.concatenate(ys), np.concatenate(ws)
    assert np.array_equal(np.unique(yf), np.arange(NC))
    rows = []
    for k in os.environ.get('CLF', 'lr').split(','):
        P, pred = readout(k, Zf, yf, wf, Zev, seed)
        for c in range(NC):
            true = yev == c
            rows.append(dict(seed=seed, method=name, clf=k, c=c, name_=cts[c], n=int(n_c[c]), probe=bool(c in set(probe.tolist())),
                             rec_argmax=float((pred[true] == c).mean()),
                             rec200=float(true[np.argsort(-P[:, c])[:NEV]].mean()), called=int((pred == c).sum())))
    ref = np.median(np.concatenate([NearestNeighbors(n_neighbors=2).fit(Z[y == c]).kneighbors(Z[y == c])[0][:, 1]
                                    for c in np.unique(y) if (y == c).sum() >= 100]))
    for r in dcr: r.update(seed=seed, method=name, n=int(n_c[r['c']]), dcr_ratio=r['dcr_med'] / ref, gen_sec=gen_sec)
    return rows, dcr

if __name__ == '__main__' and sys.argv[1] == 'run':
    from joblib import Parallel, delayed
    seeds = range(int(sys.argv[2]), int(sys.argv[3]))
    names = list(methods(*[np.load(f'{REP}_seed0.npz')[k] for k in ('Ztr', 'ytr')], 0))
    if os.environ.get('METHODS'): names = [a for a in names if a in os.environ['METHODS'].split(',')]
    print('methods:', names, flush=True); t0 = time.time()
    out = Parallel(n_jobs=int(os.environ.get('NJ', 6)), verbose=5)(delayed(fit_method)(s, a) for s in seeds for a in names)
    tag = os.environ.get('TAG', '')
    pd.DataFrame([r for o in out for r in o[0]]).to_csv(f'statistical{tag}.csv', index=False)
    pd.DataFrame([r for o in out for r in o[1]]).to_csv(f'statistical{tag}_dcr.csv', index=False)
    print(f'done {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__' and sys.argv[1] == 'report':
    import signflip
    tag = os.environ.get('TAG', '')
    S = pd.read_csv(f'statistical{tag}.csv'); Dc = pd.read_csv(f'statistical{tag}_dcr.csv'); G = pd.read_csv('aida.csv')
    par = S[(S.method == 'real') & (S.get('clf', 'lr') == 'lr')].merge(G[G.method == 'real'], on=['seed', 'c'])
    print(f"real-method parity with aida on seeds 0-2: max |d rec200| = {(par.rec200_x - par.rec200_y).abs().max()}")
    def table(w, title, seeds_note):
        grp = w.index.get_level_values('name_').values; sd = w.index.get_level_values('seed').values
        nn = w.index.get_level_values('n').values
        print(f"\n### {title}  ({seeds_note})\n{'method':22s}{'mean':>7s}{'d vs up':>9s}{'n<=8':>8s}{'n>=16':>8s}{'types+':>8s}{'type p':>8s}{'seeds+':>8s}{'seed p':>8s}")
        r = np.random.default_rng(0)
        for a in sorted(w.columns, key=lambda a: -w[a].mean()):
            d = (w[a] - w['upsample']).values; pt, dt = signflip.signflip_p(d, grp)
            ds = np.array([d[sd == s].mean() for s in np.unique(sd)])
            ps = np.mean(np.abs((r.choice([-1, 1], size=(100000, len(ds))) * ds).mean(1)) >= abs(ds.mean()) - 1e-12)
            print(f"{a:22s}{w[a].mean():7.4f}{d.mean():+9.4f}{d[nn <= 8].mean():+8.4f}{d[nn >= 16].mean():+8.4f}"
                  f"{int((dt > 0).sum()):>5d}/8{pt:8.4f}{int((ds > 0).sum()):>5d}/{len(ds)}{ps:8.4f}")
    idx = ['seed', 'name_', 'n']
    if 'clf' in S:   # four-classifier run: same augmented cells (identical DCR shows it), four readouts side by side
        x = Dc.merge(pd.read_csv('statistical_dcr.csv'), on=['seed', 'method', 'c'])
        print(f"augmentation parity with statistical.csv: {len(x)} rows, max |d dcr_med| = {(x.dcr_med_x - x.dcr_med_y).abs().max()}")
        if not (S.clf == 'lr').any(): S = pd.concat([pd.read_csv('statistical.csv').assign(clf='lr'), S])
        for k, Sk in S.groupby('clf', sort=False):
            Pk = Sk[Sk.probe]
            table(Pk.pivot_table(index=idx, columns='method', values='rec200'), f'{k}: rare rec200', f'{Sk.seed.nunique()} seeds')
            table(Pk.pivot_table(index=idx, columns='method', values='rec_argmax'), f'{k}: rare argmax recall', f'{Sk.seed.nunique()} seeds')
            ab = Sk[~Sk.probe].groupby(['seed', 'method']).rec200.mean().unstack()
            print(f'{k}: abundant rec200 mean d vs upsample:', (ab.sub(ab['upsample'], axis=0)).mean().round(4).to_dict())
        sys.exit()
    P = S[S.probe]
    table(P.pivot_table(index=idx, columns='method', values='rec200'), 'rare probe types, rec200 (matched operating point)', f'{S.seed.nunique()} seeds')
    table(P.pivot_table(index=idx, columns='method', values='rec_argmax'), 'rare probe types, argmax recall (threshold moves freely)', f'{S.seed.nunique()} seeds')
    ab = S[~S.probe].groupby(['seed', 'method']).rec200.mean().unstack()
    print('\n### abundant types, rec200, mean d vs upsample (side effect on the other 23 types)')
    print((ab.sub(ab['upsample'], axis=0)).mean().sort_values().round(4).to_string())
    w3 = P[P.seed < 3].pivot_table(index=idx, columns='method', values='rec200')
    g3 = G[G.probe].pivot_table(index=idx, columns='method', values='rec200')
    for a in g3.columns:
        if a not in ('real', 'upsample'): w3[a + '*'] = g3[a]
    table(w3, "rare, seeds 0-2 only, with aida's generators (*) on the identical split", 'bit-exact real-method parity')
    Dc['band'] = np.where(Dc.n <= 8, 'n<=8', 'n>=16')
    print('\n### memorization: added cell to nearest real cell of its class, / median real-real NN distance (0 = exact copy)')
    print(Dc.pivot_table(index='method', columns='band', values='dcr_ratio', aggfunc='median').round(3).join(
        Dc.groupby('method').gen_sec.median().rename('gen_sec').round(2)).to_string())
