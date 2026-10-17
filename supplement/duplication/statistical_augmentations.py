"""Jiaqi's statistical upsampling arms on dup_vs_generators's exact donor-disjoint split (cached by replay_split.py).
Every arm tops each probe type up to TARGET cells, built in the shared gene space (log1p-CPM) and renormalized,
then projected with the cache's PCA (gene_arm; same processing + encoder for every arm); readout is dup_vs_generators's
LogisticRegression(max_iter=2000) and rec200. Soft-labelled rows carry sample_weight (sums to k per type).
  python3 statistical_augmentations.py run 0 20     fit every arm on seeds 0..19
  python3 statistical_augmentations.py report       paired analysis vs upsample, memorization (DCR) table
"""
import sys, os, time, importlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

TARGET, NEV, KNN = 256, 200, 15
REP = os.environ.get('REP', 'rep16')   # rep16t = train-only HVGs (HVG_TRAIN=1 replay_split.py)

ANCHORS = None   # set to [] to record each dup/jitter/smote row's source-cell index (latent_unification anchor-library path)

def dup(Z, y, c, k, rng):
    pos = np.where(y == c)[0]; i = rng.choice(pos, k, replace=True)
    if ANCHORS is not None: ANCHORS.append(i)
    return Z[i], np.full(k, c), np.ones(k)

def jitter(Z, y, c, k, rng, s, sd):
    Zd, yd, wd = dup(Z, y, c, k, rng); return Zd + rng.normal(size=Zd.shape) * s * sd, yd, wd

def jitter_nz(Z, y, c, k, rng, s, sd):
    """jitter on the nonzero entries only (gene space): zeros stay zero, so the added cells keep real sparsity."""
    Zd, yd, wd = dup(Z, y, c, k, rng); return Zd + (Zd > 0) * rng.normal(size=Zd.shape) * s * sd, yd, wd

def smote(Z, y, c, k, rng, kn):
    """Chawla 2002: x + u (x_nn - x), x_nn one of the kn nearest same-class neighbours, u ~ U(0,1).
    n=1 has no neighbour, so it degenerates to duplication. kn=None means every other cell (within-class mixup)."""
    X = Z[y == c]; n = len(X)
    if n == 1: return dup(Z, y, c, k, rng)
    kk = n - 1 if kn is None else min(kn, n - 1)
    ind = NearestNeighbors(n_neighbors=kk + 1).fit(X).kneighbors(X)[1]
    i = rng.integers(n, size=k); j = ind[i, rng.integers(1, kk + 1, size=k)]
    if ANCHORS is not None: ANCHORS.append(np.where(y == c)[0][i])
    return X[i] + rng.random((k, 1)) * (X[j] - X[i]), np.full(k, c), np.ones(k)

def lognorm(C):
    """THE shared normalizer (dup_vs_generators/m27 line 99): counts -> log1p-CPM over the HVGs. It is scale-free, so
    lognorm(expm1(X)) puts any non-negative log-space matrix back on the same scale as the real cells."""
    C = np.maximum(np.asarray(C, np.float32), 0)
    return np.log1p(C / np.maximum(C.sum(1, keepdims=True), 1) * 1e4).astype(np.float32)

def renorm(X):
    """lognorm(expm1(X)) without overflow (sdiff decoder output reaches 1768 at smoke budgets):
    e^x - 1 = e^m (e^(x-m) - e^-m) with m the row max, and the e^m cancels in the row scaling."""
    # clip to the valid log1p-CPM ceiling log1p(1e4): sdiff's ReLU decoder reaches ~65, and unclipped
    # expm1 turns such rows into one-gene cells; identity on every real (renormalized) row
    X = np.clip(np.asarray(X, np.float64), 0, np.log1p(1e4)); m = X.max(1, keepdims=True)
    E = np.exp(X - m) - np.exp(-m); S = E.sum(1, keepdims=True)
    return np.log1p(E / np.where(S > 0, S, 1) * 1e4).astype(np.float32)

def gene_arm(name, X, y, probe, n_c, seed, T=TARGET, extra=None):
    """Same processing for every arm: the statistical arms are built on the real training
    cells in the shared gene space (log1p-CPM, 2000 HVGs), clipped at 0 and renormalized, so they reach
    every encoder exactly like a generator's cells. Same rng seeding and probe order as fit_arm."""
    f = arms(X, y, seed)[name]; rng = np.random.default_rng([seed, sum(map(ord, name))]); out = {}
    for c in probe:
        k = T - int(n_c[c])
        if f is None or k <= 0: continue
        Xa, ya, wa = f(X, y, c, k, rng, **(extra or {}))
        assert abs(wa.sum() - k) < 1e-6 and Xa.shape[1] == X.shape[1], (name, c)
        out[int(c)] = (renorm(Xa), np.asarray(ya, int), np.asarray(wa, float))   # JMO rows can carry another class's soft label
    return out

def readout(k, Z, y, w, Zev, seed):
    """Per-class scores (column c = class c) and argmax calls. CLF env picks readouts: lr (dup_vs_generators's), knn, rf, mlp."""
    if k == 'lr':
        m = LogisticRegression(max_iter=2000, random_state=seed).fit(Z, y, sample_weight=None if np.all(w == 1) else w)
    else:
        assert np.all(w == 1), k    # soft weights have no kNN analogue, and sklearn 1.5's MLP takes no sample_weight
        if k == 'knn':   # cfgen_knn.py protocol: 15-NN vote for argmax; tie-free score = -mean distance to class c's 15 nearest cells
            pred = KNeighborsClassifier(n_neighbors=KNN).fit(Z, y).predict(Zev)
            P = np.stack([-NearestNeighbors(n_neighbors=min(KNN, int((y == c).sum()))).fit(Z[y == c]).kneighbors(Zev)[0].mean(1)
                          for c in np.unique(y)], 1)
            return P, pred
        m = (RandomForestClassifier(n_estimators=500, random_state=seed) if k == 'rf'       # gen10-15's RF_TREES
             else MLPClassifier(max_iter=500, random_state=seed)).fit(Z, y)                   # sklearn default MLP, (100,) relu adam
    P = m.predict_proba(Zev); return P, m.classes_[P.argmax(1)]

def arms(Z, y, seed):
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

def fit_arm(seed, name):
    d = np.load(f'{REP}_seed{seed}.npz'); Z, y, Zev, yev = d['Ztr'], d['ytr'], d['Zev'], d['yev']
    X, pm, W = d['Xtr'], d['pmean'], d['comp']
    assert np.allclose((X - pm) @ W.T, Z, atol=1e-3), 'cache Xtr does not project to Ztr'
    probe, cts, n_c = d['probe'], d['cts'], d['n_c']; NC = len(cts)
    Zs, ys, ws, dcr = [Z], [y], [np.ones(len(y))], []
    t = time.time(); extra = {}
    if name.startswith('jmo'):
        import aug_jmo; extra['P'] = aug_jmo.fit_reference(X, y, seed)
    for c, (Xa, ya, wa) in gene_arm(name, X, y, probe, n_c, seed, extra=extra).items():
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
            rows.append(dict(seed=seed, arm=name, clf=k, c=c, name_=cts[c], n=int(n_c[c]), probe=bool(c in set(probe.tolist())),
                             rec_argmax=float((pred[true] == c).mean()),
                             rec200=float(true[np.argsort(-P[:, c])[:NEV]].mean()), called=int((pred == c).sum())))
    # memorization reference: median nearest-neighbour distance between real cells of the same abundant class
    ref = np.median(np.concatenate([NearestNeighbors(n_neighbors=2).fit(Z[y == c]).kneighbors(Z[y == c])[0][:, 1]
                                    for c in np.unique(y) if (y == c).sum() >= 100]))
    for r in dcr: r.update(seed=seed, arm=name, n=int(n_c[r['c']]), dcr_ratio=r['dcr_med'] / ref, gen_sec=gen_sec)
    return rows, dcr

if __name__ == '__main__' and sys.argv[1] == 'run':
    from joblib import Parallel, delayed
    seeds = range(int(sys.argv[2]), int(sys.argv[3]))
    names = list(arms(*[np.load(f'{REP}_seed0.npz')[k] for k in ('Ztr', 'ytr')], 0))
    if os.environ.get('ARMS'): names = [a for a in names if a in os.environ['ARMS'].split(',')]
    print('arms:', names, flush=True); t0 = time.time()
    out = Parallel(n_jobs=int(os.environ.get('NJ', 6)), verbose=5)(delayed(fit_arm)(s, a) for s in seeds for a in names)
    tag = os.environ.get('TAG', '')
    pd.DataFrame([r for o in out for r in o[0]]).to_csv(f'stat16{tag}.csv', index=False)
    pd.DataFrame([r for o in out for r in o[1]]).to_csv(f'stat16{tag}_dcr.csv', index=False)
    print(f'done {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__' and sys.argv[1] == 'report':
    import polh
    tag = os.environ.get('TAG', '')
    S = pd.read_csv(f'stat16{tag}.csv'); Dc = pd.read_csv(f'stat16{tag}_dcr.csv'); G = pd.read_csv('gen16.csv')
    par = S[(S.arm == 'real') & (S.get('clf', 'lr') == 'lr')].merge(G[G.arm == 'real'], on=['seed', 'c'])
    print(f"real-arm parity with gen16 on seeds 0-2: max |d rec200| = {(par.rec200_x - par.rec200_y).abs().max()}")
    def table(w, title, seeds_note):
        grp = w.index.get_level_values('name_').values; sd = w.index.get_level_values('seed').values
        nn = w.index.get_level_values('n').values
        print(f"\n### {title}  ({seeds_note})\n{'arm':22s}{'mean':>7s}{'d vs up':>9s}{'n<=8':>8s}{'n>=16':>8s}{'types+':>8s}{'type p':>8s}{'seeds+':>8s}{'seed p':>8s}")
        r = np.random.default_rng(0)
        for a in sorted(w.columns, key=lambda a: -w[a].mean()):
            d = (w[a] - w['upsample']).values; pt, dt = polh.signflip_p(d, grp)
            ds = np.array([d[sd == s].mean() for s in np.unique(sd)])
            ps = np.mean(np.abs((r.choice([-1, 1], size=(100000, len(ds))) * ds).mean(1)) >= abs(ds.mean()) - 1e-12)
            print(f"{a:22s}{w[a].mean():7.4f}{d.mean():+9.4f}{d[nn <= 8].mean():+8.4f}{d[nn >= 16].mean():+8.4f}"
                  f"{int((dt > 0).sum()):>5d}/8{pt:8.4f}{int((ds > 0).sum()):>5d}/{len(ds)}{ps:8.4f}")
    idx = ['seed', 'name_', 'n']
    if 'clf' in S:   # classifier-robustness run: same augmented cells (DCR parity proves it), four readouts side by side
        x = Dc.merge(pd.read_csv('stat16_dcr.csv'), on=['seed', 'arm', 'c'])
        print(f"augmentation parity with stat16.csv: {len(x)} rows, max |d dcr_med| = {(x.dcr_med_x - x.dcr_med_y).abs().max()}")
        if not (S.clf == 'lr').any(): S = pd.concat([pd.read_csv('stat16.csv').assign(clf='lr'), S])
        for k, Sk in S.groupby('clf', sort=False):
            Pk = Sk[Sk.probe]
            table(Pk.pivot_table(index=idx, columns='arm', values='rec200'), f'{k}: RARE rec200', f'{Sk.seed.nunique()} seeds')
            table(Pk.pivot_table(index=idx, columns='arm', values='rec_argmax'), f'{k}: RARE argmax recall', f'{Sk.seed.nunique()} seeds')
            ab = Sk[~Sk.probe].groupby(['seed', 'arm']).rec200.mean().unstack()
            print(f'{k}: ABUNDANT rec200 mean d vs upsample:', (ab.sub(ab['upsample'], axis=0)).mean().round(4).to_dict())
        sys.exit()
    P = S[S.probe]
    table(P.pivot_table(index=idx, columns='arm', values='rec200'), 'RARE probe types, rec200 (matched operating point)', f'{S.seed.nunique()} seeds')
    table(P.pivot_table(index=idx, columns='arm', values='rec_argmax'), 'RARE probe types, argmax recall (threshold moves freely)', f'{S.seed.nunique()} seeds')
    ab = S[~S.probe].groupby(['seed', 'arm']).rec200.mean().unstack()
    print('\n### ABUNDANT types, rec200, mean d vs upsample (side effect on the other 23 types)')
    print((ab.sub(ab['upsample'], axis=0)).mean().sort_values().round(4).to_string())
    w3 = P[P.seed < 3].pivot_table(index=idx, columns='arm', values='rec200')
    g3 = G[G.probe].pivot_table(index=idx, columns='arm', values='rec200')
    for a in g3.columns:
        if a not in ('real', 'upsample'): w3[a + '*'] = g3[a]
    table(w3, "RARE, seeds 0-2 only, with gen16's generators (*) on the identical split", 'bit-exact real-arm parity')
    Dc['band'] = np.where(Dc.n <= 8, 'n<=8', 'n>=16')
    print('\n### MEMORIZATION: added cell to nearest real cell of its class, / median real-real NN distance (0 = exact copy)')
    # frac_copy is not reported: kneighbors distances carry ~1e-7 float error, so a 1e-9 cutoff undercounts copies
    print(Dc.pivot_table(index='arm', columns='band', values='dcr_ratio', aggfunc='median').round(3).join(
        Dc.groupby('arm').gen_sec.median().rename('gen_sec').round(2)).to_string())
