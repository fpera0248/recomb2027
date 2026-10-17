"""m27_repl fix-up: rerun section A's distribution metrics in a form that can judge
duplication. m27_repl.py's gamma grid (0.1..100) is ~0 between distinct cells in unscaled PCA-50, so it
counted exact copies. Here: one RBF kernel at the reference's own median squared distance (median
heuristic, fixed per type so every arm is scored by the same kernel), CFGen's widest kernel gamma=0.01
alone, exact W2, energy. Means are over units EVERY arm has; n_c=1 rows reported separately.
  OMP_NUM_THREADS=2 python3 replicate_metrics_fix.py > m27_repl_fix_report.txt
"""
import numpy as np, pandas as pd, ot, polh
from statistical_augmentations import gene_arm
ARMS = ['real', 'jitter_s1.0', 'upsample', 'sd3_nosingle', 'nb', 'synth_w1', 'synthbal_w1',
        'scvi', 'sdiff', 'lat_boot', 'lat_gauss']
T = 256

def jitter_s1(b, seed):   # gene space + renormalized (statistical_augmentations.gene_arm)
    return {c: v[0] for c, v in gene_arm('jitter_s1.0', b['Xtr'], b['ytr'], b['probe'], b['n_c'], seed, T).items()}

d2 = lambda A, B: ((A[:, None] - B[None]) ** 2).sum(-1)
def mmd(G, R, g):
    return float(np.exp(-g * d2(G, G)).mean() + np.exp(-g * d2(R, R)).mean() - 2 * np.exp(-g * d2(G, R)).mean())
def w2(G, R):
    return float(np.sqrt(max(ot.emd2(np.full(len(G), 1 / len(G)), np.full(len(R), 1 / len(R)), d2(G, R)), 0)))
def energy(G, R):
    return float(2 * np.sqrt(d2(G, R)).mean() - np.sqrt(d2(G, G)).mean() - np.sqrt(d2(R, R)).mean())

rows = []
for seed in range(3):
    b = np.load(f'mk16/s{seed}_base.npz')
    Xtr, ytr, Xev, yev, comp, pmean = b['Xtr'], b['ytr'], b['Xev'], b['yev'], b['comp'], b['pmean']
    proj = lambda X: (X - pmean) @ comp.T
    added = {'real': {}, 'jitter_s1.0': jitter_s1(b, seed)}
    for arm in ARMS:
        if arm not in added:
            a = np.load(f'mk16/s{seed}_{arm}.npz'); added[arm] = {int(c): a['X'][a['y'] == c] for c in np.unique(a['y'])}
    for c in map(int, b['probe']):
        Rz = proj(Xev[yev == c]); dr = d2(Rz, Rz); gmed = 1 / np.median(dr[np.triu_indices(len(Rz), 1)])
        for arm in ARMS:
            add = added[arm].get(c)
            G = proj(np.vstack([Xtr[ytr == c], add]) if add is not None and len(add) else Xtr[ytr == c])
            rows.append(dict(seed=seed, name_=str(b['cts'][c]), n=int(b['n_c'][c]), arm=arm, n_G=len(G),
                             mmd_med=mmd(G, Rz, gmed), mmd_g001=mmd(G, Rz, 0.01), w2=w2(G, Rz), energy=energy(G, Rz)))
    print(f'seed {seed} done', flush=True)
R = pd.DataFrame(rows); R.to_csv('m27_repl_fix.csv', index=False)

def table(df, m, label):
    piv = df.pivot_table(index=['seed', 'name_'], columns='arm', values=m).dropna()   # units every arm has
    print(f'\n### {label} (lower = closer to held-out real cells; {len(piv)} units, '
          f'{piv.index.get_level_values(1).nunique()} types)')
    up = piv['upsample']
    for a in sorted(piv.columns, key=lambda a: piv[a].mean()):
        if a == 'upsample': print(f'  {a:14s}{piv[a].mean():10.4f}        --'); continue
        d = up - piv[a]; p, dt = polh.signflip_p(d.values, d.index.get_level_values('name_').values)
        print(f'  {a:14s}{piv[a].mean():10.4f}  d vs dup {d.mean():+.4f}  {int((dt > 0).sum())}/{len(dt)} types better  p {p:.4f}')

for sub, lab in ((R[R.n >= 2], 'n_c >= 2'), (R[R.n == 1], 'n_c = 1 only')):
    print(f'\n======== {lab} ========')
    for m, l in (('mmd_med', 'MMD^2, median-heuristic kernel'), ('mmd_g001', 'MMD^2, gamma=0.01 only'),
                 ('w2', 'exact 2-Wasserstein'), ('energy', 'energy distance')):
        table(sub, m, l)
