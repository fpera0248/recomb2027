"""Marker / DE-gene recovery for every augmentation arm on dup_vs_generators's donor-disjoint split (2000 HVGs, log1p-CPM).
Truth: the probe type's held-out-donor eval cells vs the real training cells of every other type.
Each arm's class = its n_c real training cells + its added cells; 'rest' is the same real cells for every arm.
more_real = every pre-deletion training cell of the type (what more cells from the same donors would give).
Jitter / SMOTE are built in gene space and renormalized (statistical_augmentations.gene_arm), like every generator arm.
  SAVE_GENE=mk16 python3 dup_vs_generators_savegene.py ; python3 marker_recovery.py [dir]   -> mk16.csv + table
"""
import sys, glob, os, numpy as np, pandas as pd, polh
from statistical_augmentations import gene_arm
from scipy import stats
from sklearn.neighbors import NearestNeighbors
K, DET = 50, 0.25
D = sys.argv[1] if len(sys.argv) > 1 else 'mk16'

def lfc(mc, mr): return np.log2((np.expm1(np.maximum(mc, 0)) + 1e-9) / (np.expm1(np.maximum(mr, 0)) + 1e-9))

def stat_arms(b, seed, T):
    """jitter / smote_k5 built in gene space on the real training cells and renormalized (statistical_augmentations.gene_arm):
    the same normalizer every generator's cells went through."""
    return {name: {c: v[0] for c, v in gene_arm(name, b['Xtr'], b['ytr'], b['probe'], b['n_c'], seed, T).items()}
            for name in ('jitter_s0.5', 'jitter_s1.0', 'jitter_s2.0', 'jitternz_s1.0', 'smote_k5')}

rows = []
for bf in sorted(glob.glob(f'{D}/s*_base.npz')):
    seed = int(os.path.basename(bf)[1:].split('_')[0]); b = np.load(bf)
    if os.path.exists(f'rep16_seed{seed}.npz') and D == 'mk16':
        assert np.allclose(np.load(f'rep16_seed{seed}.npz')['Ztr'], b['Ztr'], atol=1e-4), 'gen16m split/PCA != replay16'
    Xtr, ytr, probe, n_c = b['Xtr'], b['ytr'], b['probe'], b['n_c']
    up = np.load(f'{D}/s{seed}_upsample.npz'); T = int(n_c[probe[0]] + (up['y'] == probe[0]).sum())
    arms = {os.path.basename(f)[len(f's{seed}_'):-4]: np.load(f) for f in glob.glob(f'{D}/s{seed}_*.npz') if not f.endswith('_base.npz')}
    added = {a: {int(c): d['X'][d['y'] == c] for c in np.unique(d['y'])} for a, d in arms.items()}
    added.update(stat_arms(b, seed, T))
    for c in probe:
        c = int(c); real_c = Xtr[ytr == c]; mr = Xtr[ytr != c].mean(0)
        G = b['Xev'][b['yev'] == c]; mg, vg = G.mean(0), G.var(0, ddof=1)
        cand = np.where((G > 0).mean(0) >= DET)[0]; expr = (mg > .01) | (mr > .01)
        lg = lfc(mg, mr); top_g = set(cand[np.argsort(-lg[cand])][:K])
        sets = {'real': real_c, 'more_real': b['Xfull'][b['yfull'] == c]}
        sets.update({a: np.vstack([real_c, v[c]]) for a, v in added.items() if c in v})
        for a, C in sets.items():
            mc = C.mean(0); la = lfc(mc, mr); top = set(cand[np.argsort(-la[cand])][:K])
            vr = np.mean(C.var(0, ddof=1)[expr]) / np.mean(vg[expr]) if len(C) > 1 else np.nan
            rows.append(dict(seed=seed, c=c, type=str(b['cts'][c]), n=int(n_c[c]), arm=a, n_cells=len(C),
                             jaccard50=len(top & top_g) / len(top | top_g), spearman_lfc=stats.spearmanr(la[expr], lg[expr])[0],
                             mean_err=float(np.abs(mc - mg)[expr].mean()), var_ratio=vr, xmax=float(C.max())))
R = pd.DataFrame(rows); R.to_csv(f'{D}.csv', index=False)
print(f"{R.seed.nunique()} seeds x {R.c.nunique()} probe types; truth = held-out-donor eval cells; rest = real train cells of other types")
print(f"{'arm':14s}{'units':>6s}{'jacc50':>8s}{'d vs up':>9s}{'p':>7s}{'rho lfc':>9s}{'d vs up':>9s}{'p':>7s}{'|mean err|':>11s}{'d vs up':>9s}{'p':>7s}{'var ratio':>10s}")
U = R[R.arm == 'upsample'].set_index(['seed', 'c'])
for a, g in sorted(R.groupby('arm'), key=lambda t: -t[1].jaccard50.mean()):
    g = g.set_index(['seed', 'c']); cols = []
    for m in ('jaccard50', 'spearman_lfc', 'mean_err'):
        d = (g[m] - U[m]).dropna(); p = polh.signflip_p(d.values, d.index.get_level_values('c').values)[0] if a != 'upsample' else 1.0
        cols += [g[m].mean(), d.mean(), p]
    flag = f'  (max value {g.xmax.max():.0f}, real cells <= 9)' if g.xmax.max() > 20 else ''
    print(f"{a:14s}{len(g):6d}{cols[0]:8.3f}{cols[1]:+9.3f}{cols[2]:7.3f}{cols[3]:9.3f}{cols[4]:+9.3f}{cols[5]:7.3f}"
          f"{cols[6]:11.4f}{cols[7]:+9.4f}{cols[8]:7.3f}{g.var_ratio.median():10.2f}{flag}")
