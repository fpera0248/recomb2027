"""Paired analysis of run_fm_dup.py's results.csv / results_ovr.csv.
Pairing unit = (seed, type, clf); sign-flip test over the 8 probe types (polh.signflip_p);
BH within each family of comparisons; n_train==1 units reported separately per project rule."""
import os, sys, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.path.insert(0, HR_WORK + '/cfgen_natrare')
import polh

OUT = os.environ.get('OUT', HR_WORK + '/cfgen_natrare/j1004/fm')  # fm3 backbones: OUT=fm3/<dir>
S = pd.read_csv(f'{OUT}/results.csv')
OVR = pd.read_csv(f'{OUT}/results_ovr.csv')


def bh(pvals, names):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p)
    q = np.empty(m); q[order] = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(q, 1)
    return dict(zip(names, out))


def contrast(df, filt_a, filt_b, value, label):
    """filt_a/filt_b: boolean masks selecting the two arms of the contrast on df (same idx otherwise).
    Pairs on (seed, type, clf); prints mean diff, sign-flip p over 8 types, BH-ready p."""
    a = df[filt_a].set_index(['seed', 'type_name', 'clf'])[value].rename('a')
    b = df[filt_b].set_index(['seed', 'type_name', 'clf'])[value].rename('b')
    j = pd.concat([a, b], axis=1, join='inner')
    d = (j['a'] - j['b']).values
    groups = j.index.get_level_values('type_name').values
    if len(d) == 0:
        print(f'{label:55s} NO PAIRED ROWS'); return None
    p, dt = polh.signflip_p(d, groups)
    print(f'{label:55s} n={len(d):4d}  mean d={d.mean():+.4f}  types+={int((dt>0).sum())}/{len(dt)}  p={p:.4f}')
    return p


print('=' * 100)
print('MAIN: matched operating point (rec200), n_train >= 2 only (n=1 units reported separately below)')
print('=' * 100)
M = S[S.n_train >= 2].copy()
pvals, names = [], []
for clf in ['lr', 'knn', 'rf', 'mlp']:
    Mc = M[M.clf == clf]
    print(f'\n--- classifier = {clf} ---')
    p = contrast(Mc, (Mc.rep == 'fm') & (Mc.arm == 'dup'), (Mc.rep == 'pca50') & (Mc.arm == 'dup'),
                 'rec200', '1. FM+dup vs PCA50+dup (does the FM embedding help duplication?)')
    pvals.append(p); names.append(f'{clf}:fmdup_vs_pcadup')
    p = contrast(Mc, (Mc.rep == 'fm') & (Mc.arm == 'dup'), (Mc.rep == 'fm') & (Mc.arm == 'none'),
                 'rec200', '2. FM+dup vs FM+none (does dup help on top of FM?)')
    pvals.append(p); names.append(f'{clf}:fmdup_vs_fmnone')
    p = contrast(Mc, (Mc.rep == 'pca50') & (Mc.arm == 'dup'), (Mc.rep == 'pca50') & (Mc.arm == 'none'),
                 'rec200', '3. PCA50+dup vs PCA50+none (reference: does dup help on PCA, as in the main experiments?)')
    pvals.append(p); names.append(f'{clf}:pcadup_vs_pcanone')
    p = contrast(Mc, (Mc.rep == 'fm') & (Mc.arm == 'none'), (Mc.rep == 'pca50') & (Mc.arm == 'none'),
                 'rec200', '4. FM+none vs PCA50+none (does the FM embedding help with NO augmentation?)')
    pvals.append(p); names.append(f'{clf}:fmnone_vs_pcanone')
    for arm in ['jitter', 'smote']:
        p = contrast(Mc, (Mc.rep == 'fm') & (Mc.arm == arm), (Mc.rep == 'fm') & (Mc.arm == 'dup'),
                     'rec200', f'5. FM+{arm} vs FM+dup')
        pvals.append(p); names.append(f'{clf}:fm{arm}_vs_fmdup')
    if clf in ('lr', 'rf'):
        p = contrast(Mc, (Mc.rep == 'fm') & (Mc.arm == 'classweight'), (Mc.rep == 'fm') & (Mc.arm == 'dup'),
                     'rec200', '6. FM+classweight vs FM+dup')
        pvals.append(p); names.append(f'{clf}:fmcw_vs_fmdup')

print('\n' + '=' * 100)
print('INTERACTION: does the FM embedding change the VALUE of duplication? (delta_dup = dup - none, '
      'compared between rep=fm and rep=pca50; positive = dup buys MORE under FM)')
print('=' * 100)
for clf in ['lr', 'knn', 'rf', 'mlp']:
    Mc = M[M.clf == clf]
    dup_fm = Mc[(Mc.rep == 'fm') & (Mc.arm == 'dup')].set_index(['seed', 'type_name'])['rec200']
    none_fm = Mc[(Mc.rep == 'fm') & (Mc.arm == 'none')].set_index(['seed', 'type_name'])['rec200']
    dup_pca = Mc[(Mc.rep == 'pca50') & (Mc.arm == 'dup')].set_index(['seed', 'type_name'])['rec200']
    none_pca = Mc[(Mc.rep == 'pca50') & (Mc.arm == 'none')].set_index(['seed', 'type_name'])['rec200']
    idx = dup_fm.index.intersection(none_fm.index).intersection(dup_pca.index).intersection(none_pca.index)
    delta_fm = (dup_fm[idx] - none_fm[idx]).values
    delta_pca = (dup_pca[idx] - none_pca[idx]).values
    d = delta_fm - delta_pca
    groups = idx.get_level_values('type_name').values
    p, dt = polh.signflip_p(d, groups)
    print(f'{clf:5s} delta_dup(FM)={delta_fm.mean():+.4f}  delta_dup(PCA50)={delta_pca.mean():+.4f}  '
          f'diff={d.mean():+.4f}  types+={int((dt>0).sum())}/{len(dt)}  p={p:.4f}')

qmap = bh([p for p in pvals if p is not None], [n for n, p in zip(names, pvals) if p is not None])
print('\nBH-adjusted q (main family, 4 classifiers x up to 6 contrasts):')
for n, q in sorted(qmap.items(), key=lambda x: x[1]):
    print(f'  {n:30s} q={q:.4f}')

print('\n' + '=' * 100)
print('n_train == 1 units (reported separately, RULES)')
print('=' * 100)
N1 = S[S.n_train == 1]
print(N1.groupby(['rep', 'arm', 'clf'])['rec200'].agg(['mean', 'count']).to_string())

print('\n' + '=' * 100)
print('PROTOTYPE readout (real cells only -- augmentation cannot move a mean-based readout, '
      'oversampling-is-variance-inflation.md) vs best classifier+arm, matched point, n_train>=2')
print('=' * 100)
proto = M[M.clf == 'prototype'].groupby(['rep'])['rec200'].mean()
best = M[M.clf != 'prototype'].groupby(['rep', 'arm', 'clf'])['rec200'].mean().reset_index()
for rep in ['fm', 'pca50']:
    b = best[best.rep == rep].sort_values('rec200', ascending=False).iloc[0]
    print(f'{rep:6s} prototype={proto[rep]:.4f}   best(arm={b.arm},clf={b.clf})={b.rec200:.4f}')

print('\n' + '=' * 100)
print('Default threshold (secondary): argmax recall, precision, FPR -- C3')
print('=' * 100)
print(M.groupby(['rep', 'arm', 'clf'])[['rec_argmax', 'prec_argmax', 'fpr_argmax', 'ncall']].mean()
      .round(4).to_string())

print('\n' + '=' * 100)
print('C2: one-vs-rest vs multiclass, LR, FM rep (results_ovr.csv vs results.csv)')
print('=' * 100)
mc = S[(S.rep == 'fm') & (S.clf == 'lr') & (S.n_train >= 2)].set_index(['seed', 'type_name', 'arm'])['rec200']
ov = OVR[OVR.n_train >= 2].set_index(['seed', 'type_name', 'arm'])['rec200_ovr']
j = pd.concat([mc, ov], axis=1, join='inner')
d = (j['rec200_ovr'] - j['rec200']).values
groups = j.index.get_level_values('type_name').values
p, dt = polh.signflip_p(d, groups)
print(f'rec200_ovr - rec200_multiclass: mean d={d.mean():+.4f}  types+={int((dt>0).sum())}/{len(dt)}  p={p:.4f}')
print(j.groupby('arm').mean().round(4).to_string())
