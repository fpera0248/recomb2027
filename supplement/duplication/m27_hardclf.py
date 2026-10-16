"""m27_hard's saved arm cells (m27_hard/, PCA-50 via each seed's own base.npz comp/pmean) re-read
under stat16's four readouts (lr, knn, rf, mlp), PLUS jitter_s1.0 and smote_k5 built in gene space,
renormalized, then the same PCA (stat16.gene_arm). Unlike sd3clf16.py, this does NOT depend on rep16_seed*.npz (that
cache belongs to gen16's OLD 8-probe split); m27_hard/s{seed}_base.npz alone supplies the real
train/eval cells and the PCA basis for this split.
  python3 m27_hardclf.py run      -> m27_hardclf.csv
  python3 m27_hardclf.py report   -> m27_hard_report.txt-style table (also printed)
"""
import sys, os, glob, numpy as np, pandas as pd
from stat16 import readout, gene_arm, NEV, TARGET

D = 'm27_hard'
GEN_ARMS = sorted({os.path.basename(f)[len('s0_'):-4] for f in glob.glob(f'{D}/s0_*.npz')} - {'base'})
STAT_ARMS = ['jitter_s1.0', 'jitternz_s1.0', 'smote_k5']

def fit(seed, arm):
    b = np.load(f'{D}/s{seed}_base.npz')
    Ztr, ytr, cts, n_c, probe = b['Ztr'], b['ytr'], b['cts'], b['n_c'], b['probe']
    Zev = (b['Xev'] - b['pmean']) @ b['comp'].T; yev = b['yev']
    if arm in STAT_ARMS:   # gene space + renormalized, then the same PCA (stat16.gene_arm; Jiaqi 2026-09-28)
        A = gene_arm(arm, b['Xtr'], ytr, probe, n_c, seed)
        Z = np.vstack([Ztr] + [(v[0] - b['pmean']) @ b['comp'].T for v in A.values()]); y = np.concatenate([ytr] + [v[1] for v in A.values()])
    else:
        a = np.load(f'{D}/s{seed}_{arm}.npz')
        Z = np.vstack([Ztr, (a['X'] - b['pmean']) @ b['comp'].T]); y = np.concatenate([ytr, a['y']])
    rows = []
    for k in ('lr', 'knn', 'rf', 'mlp'):
        P, pred = readout(k, Z, y, np.ones(len(y)), Zev, seed)
        for c in range(len(cts)):
            t = yev == c
            rows.append(dict(seed=seed, arm=arm, clf=k, c=c, name_=cts[c], n=int(n_c[c]), probe=c in set(probe.tolist()),
                             rec200=float(t[np.argsort(-P[:, c])[:NEV]].mean()), rec_argmax=float((pred[t] == c).mean())))
    return rows

if __name__ == '__main__' and sys.argv[1] == 'run':
    from joblib import Parallel, delayed
    ARMS = GEN_ARMS + STAT_ARMS
    print('arms:', ARMS, flush=True)
    out = Parallel(n_jobs=int(os.environ.get('NJ', 3)), verbose=5)(delayed(fit)(s, a) for s in range(3) for a in ARMS)
    pd.DataFrame([r for o in out for r in o]).to_csv(f'{D}clf.csv', index=False)

if __name__ == '__main__' and sys.argv[1] == 'report':
    import polh
    S = pd.read_csv(f'{D}clf.csv')
    for k in ('lr', 'knn', 'rf', 'mlp'):
        Sk = S[(S.clf == k) & S.probe]
        w = Sk.pivot_table(index=['seed', 'name_', 'n'], columns='arm', values='rec200')
        if 'upsample' not in w.columns: continue
        g = w.index.get_level_values('name_').values
        print(f"\n### {k}: m27_hard probe types, rec200 (matched operating point)")
        print(f"{'arm':16s}{'mean':>7s}{'d vs up':>9s}{'p':>7s}")
        for a in sorted(w.columns, key=lambda a: -w[a].mean()):
            d = (w[a] - w['upsample']).values; p, dt = polh.signflip_p(d, g)
            print(f"{a:16s}{w[a].mean():7.4f}{d.mean():+9.4f}{p:7.4f}")
