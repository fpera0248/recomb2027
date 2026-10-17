"""m27 hlca classifier-robustness readout: m27_gen_hlca's saved arm cells (m27_hl/, gene-space
log1p-CPM, from SAVE_GENE=m27_hl python3 m27_gen_hlca.py) re-read under stat16's four readouts
(lr/knn/rf/mlp), plus jitter_s1.0 and smote_k5 built in gene space and renormalized (stat16.gene_arm), then the same PCA. rec200 uses the per-class NEVc saved by m27_gen_hlca.py (hlca fallback: a
probe type with < 200 held-out cells is scored at its own held-out count, not 200).
  python3 m27_hlclf.py run      -> m27_hlclf.csv
  python3 m27_hlclf.py report
"""
import sys, os, glob, numpy as np, pandas as pd
HL = os.environ.get('M27_HL', 'm27_hl'); CSV = os.environ.get('M27_CSV', 'm27_hlclf.csv')   # panel run: per-dataset dirs
from stat16 import readout, gene_arm
ARMS = sorted({os.path.basename(f).split('_', 1)[1][:-4] for f in glob.glob(f'{HL}/s*_*.npz')} - {'base', 'sd3'} | {'sd3_nosingle'})
# sd3 (all-class fit) succeeds only when the seed has no singleton type, so there it IS the no-singleton fit
_fn = lambda s, a: next((f for f in (f'{HL}/s{s}_{a}.npz',) + ((f'{HL}/s{s}_sd3.npz',) if a == 'sd3_nosingle' else ()) if os.path.exists(f)), None)

def _base(seed):
    b = np.load(f'{HL}/s{seed}_base.npz')
    comp, pmean = b['comp'], b['pmean']
    Ztr, ytr = b['Ztr'], b['ytr']
    Zev, yev = (b['Xev'] - pmean) @ comp.T, b['yev']
    return b, comp, pmean, Ztr, ytr, Zev, yev

def _rec_tie(p, t, k):
    """rec200 with ties at the cutoff split evenly (expected value under random tie-breaking); argsort's
    arbitrary order among P==0 ties can decide rec200 under RF/MLP at n=1."""
    thr = np.sort(p)[::-1][k - 1]; hi = p > thr; tie = p == thr
    return float((t[hi].sum() + t[tie].mean() * (k - hi.sum())) / k)

def _score(Z, y, Zev, yev, cts, n_c, probe, NEVc, seed, arm):
    rows = []
    for k in ('lr', 'knn', 'rf', 'mlp'):
        P, pred = readout(k, Z, y, np.ones(len(y)), Zev, seed)
        if P.shape[1] < len(cts):   # a non-probe type with no training cells (donor split) drops its column: re-index so column c = class c
            Pf = np.full((len(Zev), len(cts)), -np.inf); Pf[:, np.unique(y)] = P; P = Pf
        for c in range(len(cts)):
            t = yev == c; nevc = int(NEVc[c])
            rows.append(dict(seed=seed, arm=arm, clf=k, c=c, name_=cts[c], n=int(n_c[c]),
                             probe=bool(c in set(probe.tolist())), nev_used=nevc,
                             rec200=float(t[np.argsort(-P[:, c])[:nevc]].mean()),
                             rec200_tie=_rec_tie(P[:, c], t, nevc),
                             rec_argmax=float((pred[t] == c).mean()),
                             ncall=int((pred == c).sum())))   # eval cells called c at argmax: tests the threshold reading
    return rows

def fit(seed, arm):
    b, comp, pmean, Ztr, ytr, Zev, yev = _base(seed)
    a = np.load(_fn(seed, arm))
    Z = np.vstack([Ztr, (a['X'] - pmean) @ comp.T]); y = np.concatenate([ytr, a['y']])
    return _score(Z, y, Zev, yev, b['cts'], b['n_c'], b['probe'], b['NEVc'], seed, arm)

def fit_noise(seed, arm):
    """jitter_s1.0 / smote_k5 built in gene space on the real training cells, renormalized, then projected
    with the same PCA as every generator arm (stat16.gene_arm)."""
    b, comp, pmean, Ztr, ytr, Zev, yev = _base(seed)
    n_c, probe = b['n_c'], b['probe']
    up = np.load(f'{HL}/s{seed}_upsample.npz'); T = int(n_c[probe[0]] + (up['y'] == probe[0]).sum())
    A = gene_arm(arm, b['Xtr'], ytr, probe, n_c, seed, T)
    Z = np.vstack([Ztr] + [(v[0] - pmean) @ comp.T for v in A.values()]); y = np.concatenate([ytr] + [v[1] for v in A.values()])
    return _score(Z, y, Zev, yev, b['cts'], n_c, probe, b['NEVc'], seed, arm)

if __name__ == '__main__' and sys.argv[1] == 'run':
    from joblib import Parallel, delayed
    seeds = range(len(glob.glob(f'{HL}/s*_base.npz')))
    jobs = [delayed(fit)(s, a) for s in seeds for a in ARMS if _fn(s, a)]   # sdiff can be budget-dropped for later seeds
    jobs += [delayed(fit_noise)(s, a) for s in seeds for a in ('jitter_s1.0', 'jitternz_s1.0', 'smote_k5')]
    out = Parallel(n_jobs=int(os.environ.get('NJ', 3)), verbose=5)(jobs)
    pd.DataFrame([r for o in out for r in o]).to_csv(CSV, index=False)
    print('wrote m27_hlclf.csv')

if __name__ == '__main__' and sys.argv[1] == 'report':
    import polh
    S = pd.read_csv(CSV)
    idx = ['seed', 'name_', 'n']
    for k, Sk in S.groupby('clf', sort=False):
        Pk = Sk[Sk.probe]
        w = Pk.pivot_table(index=idx, columns='arm', values='rec200')
        grp = w.index.get_level_values('name_').values
        print(f"\n### {k}  ({Sk.seed.nunique()} seeds x {w.index.get_level_values('name_').nunique()} hlca probe types)")
        print(f"{'arm':14s}{'rec200':>8s}{'d vs dup':>10s}{'types+':>8s}{'p':>8s}")
        for a in sorted(w.columns, key=lambda a: -w[a].mean()):
            if 'upsample' not in w.columns: continue
            m = w[a].notna().values; d = (w[a] - w['upsample']).values[m]; p, dt = polh.signflip_p(d, grp[m])
            print(f"{a:14s}{w[a].mean():8.3f}{d.mean():+10.3f}{int((dt > 0).sum()):>5d}/{len(dt)}{p:8.3f}" + (f'  ({m.sum()} of {len(m)} units)' if not m.all() else ''))
