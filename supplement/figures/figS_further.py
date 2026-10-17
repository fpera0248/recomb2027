"""Supplement figure: further analyses (supp.tex, section 'Further analyses'). Post hoc, exploratory.
Data: ~/work/jiaqi1011/out/scale_queue/scale.csv (time, memory), out/w2_fm/res_{pca30?,scvi_F,scgpt_F}.csv + paired_tests.csv (embeddings),
out/w6_annot_atac/p1_summary.csv + p1_atlas_union_AP.csv (random label errors).
Run: OMP_NUM_THREADS=4 nice -n 10 python3 figS_further.py"""
import sys, os, itertools, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.path.insert(0, HR_WORK + '/recomb27/v2/figs'); sys.path.insert(0, HR_WORK + '/recomb27/rev1015/figaudit'); sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
from paperstyle import *
apply_style()
J = HR_WORK + '/jiaqi1011/out/'
ok = lambda x, v, d: abs(x - v) < 0.5 * 10 ** -d + 1e-9   # x equals printed v at d decimals
def pflip(x):   # exact two-sided sign-flip over atlas values
    s = [abs(sum(g * v for g, v in zip(c, x))) for c in itertools.product([1, -1], repeat=len(x))]
    return float(np.mean(np.array(s) >= abs(sum(x)) - 1e-12))

# ---- (a) time ----
S = pd.read_csv(J + 'w1_scale/scale.csv'); S = S[(S.status == 'ok') & (S.seed == 0)]
# step ids as stored in scale.csv: candidate_dump = candidate groups, candidate_marker_features = marker features, candidate_halo_features = halo features, m1_leiden_sub = Leiden subclustering,
# learn14_predict = learned scorer, our_method = final blend (our method)
steps = ['sr3', 'sr3m', 'sr3h', 'm1_leiden_sub', 'learn14_predict', 'frozen4']
W = S[S.step.isin(steps)].pivot(index='n', columns='step', values='wall_s'); M = S[S.step.isin(steps)].pivot(index='n', columns='step', values='max_rss_mb')
ns = [5000, 10000, 20000, 50000, 100000, 200000]; assert list(W.index) == ns and not W.isna().any().any()
full = W[steps].sum(axis=1); cand = W['sr3']; leid = W['m1_leiden_sub']; peak = M[steps].max(axis=1) / 1000
assert ok(full[200000], 342, 0) and ok(leid[200000], 56.5, 1) and ok(cand[200000], 216.5, 1) and ok(W['learn14_predict'][200000], 27.7, 1)
for n, v in zip(ns, [6.0, 14.7, 19.6, 35.9, 76.8, 216.5]): assert ok(cand[n], v, 1), n
assert ok(peak[5000], 1.1, 1) and ok(peak[200000], 12.1, 1)

# ---- (b) embeddings ----
def atlas_ap(f, arm, stage):
    d = pd.read_csv(J + 'w2_fm/' + f); d = d[(d.arm == arm) & (d.stage == stage)]; assert len(d) == 29
    return d.groupby('atlas').bl_union_AP.mean()
B = {}
for lab, f, arm in [('scVI', 'res_scvi_F.csv', 'scvi_F'), ('scGPT', 'res_scgpt_F.csv', 'scgpt_F')]:
    B[lab] = {st: atlas_ap(f, arm, st) - atlas_ap(f, 'pca', st) for st in ('retrain', 'frozen')}
    assert len(B[lab]['retrain']) == 10
R = {k: B[k]['retrain'] for k in B}; Fz = {k: B[k]['frozen'] for k in B}
assert ok(R['scVI'].mean(), .020, 3) and ok(R['scGPT'].mean(), -.101, 3) and (R['scVI'] > 0).sum() == 7 and (R['scGPT'] > 0).sum() == 1
assert ok(Fz['scVI'].mean(), .025, 3) and ok(Fz['scGPT'].mean(), -.065, 3) and (Fz['scVI'] > 0).sum() == 7 and (Fz['scGPT'] > 0).sum() == 3
p = {k: pflip(list(R[k])) for k in R}; assert ok(p['scGPT'], .031, 3) and ok(min(2 * p['scGPT'], 1), .0625, 3) and ok(p['scVI'], .268, 3)   # Holm over two: scGPT x2, scVI x1 (larger p, monotone)
assert ok(pflip(list(Fz['scVI'])), .104, 3) and ok(pflip(list(Fz['scGPT'])), .102, 3)
at = list(R['scVI'].index); assert list(R['scGPT'].index) == at

# ---- (c) random label errors ----
P = pd.read_csv(J + 'w6_annot_atac/p1_summary.csv').set_index('group'); E = pd.read_csv(J + 'w6_annot_atac/p1_atlas_union_AP.csv')
rates = [0, 5, 10, 20]; grp = ['R05', 'R10', 'R20']
m = {k: [P['AP_clean_' + k].iloc[0]] + [P.loc[g, 'AP_' + ('ours' if k == 'ours' else 'leiden')] for g in grp] for k in ('ours', 'leiden')}
assert ok(m['ours'][0], .564, 3) and ok(m['leiden'][0], .456, 3)
for k, v in [('ours', [.133, .162, .209]), ('leiden', [.041, .074, .101])]:
    for g, x in zip(grp, v): assert ok(P.loc[g, 'drop_' + k], x, 3), (k, g)
for g, x in zip(grp, [.016, .020, .000]): assert ok(P.loc[g, 'level_gap'], x, 3)
assert ok(P.loc['R20', 'p_D_holm'], .006, 3)
# per-atlas curves (ours, Leiden) from the atlas table, the mean must match the summary
A = {}
for k in ('ours', 'leiden'):
    A[k] = np.array([[E[(E.group == g) & (E.method == k)].set_index('atlas').value.reindex(at if False else sorted(E.atlas.unique())).values] for g in ['ID'] + grp]).reshape(4, 10)
    assert np.allclose(A[k].mean(1), m[k], atol=5e-4), k
    A[k + '_se'] = A[k].std(1, ddof=1) / np.sqrt(10)

# ---- draw ----
W_, H_ = COL2, 2.6
fig = plt.figure(figsize=(W_, H_), layout=None)
ax_ = lambda x0, y0, x1, y1: fig.add_axes([x0 / W_, y0 / H_, (x1 - x0) / W_, (y1 - y0) / H_])
a = ax_(0.55, 0.5, 2.3, 2.2); b = ax_(3.05, 0.5, 4.3, 2.2); c = ax_(5.0, 0.5, 6.4, 2.2)
co, cl = color('ours'), color('leiden')
a.plot(ns, full, '-o', color=co, ms=3, label='Our method, all steps'); a.plot(ns, cand, '--s', color=co, ms=3, mfc='white', label='Our method, candidate groups')
a.plot(ns, leid, '-^', color=cl, ms=3, label='Leiden subclustering')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Cells'); a.set_ylabel('Time (s)')
a.set_xticks([5000, 20000, 200000]); a.set_xticklabels(['5,000', '20,000', '200,000']); a.minorticks_off()
a.legend(loc='upper left', handlelength=1.6)
a.set_ylim(1.5, 2000)
xs = {'scVI': 0, 'scGPT': 1}
for i, t in enumerate(at): b.plot([0, 1], [R['scVI'][t], R['scGPT'][t]], '-', color='#BBBBBB', lw=LW_THIN, zorder=1)
for k, x in xs.items():
    b.plot(np.full(10, x), R[k], 'o', color='#555555', ms=3, mfc='white', mew=.6, zorder=2)
    b.plot(x, R[k].mean(), 'D', color=co, ms=5, mec='black', mew=.5, zorder=3)
b.axhline(0, color='black', ls='--', lw=LW_THIN); b.set_xlim(-.5, 1.5); b.set_xticks([0, 1]); b.set_xticklabels(['scVI', 'scGPT'])
b.set_ylabel('Change in AP from PCA'); b.margins(y=.1)
for k, x in xs.items(): b.text(x, b.get_ylim()[1], f'{R[k].mean():+.3f}'.replace('-', '\u2212'), ha='center', va='top', fontsize=FS_MIN, color=co)
for k, col, mk in [('ours', co, 'o'), ('leiden', cl, '^')]:
    c.errorbar(range(4), m[k], yerr=A[k + '_se'], fmt='-' + mk, color=col, ms=3.5, lw=LW, label='Our method' if k == 'ours' else 'Leiden subclustering', zorder=3)
c.set_xticks(range(4)); c.set_xticklabels([f'{r}%' for r in rates]); c.set_xlim(-.2, 3.2)
c.set_xlabel('Cells with a random label'); c.set_ylabel('AP of hidden subtypes'); c.legend(loc='upper right', handlelength=1.0, handletextpad=0.4, borderaxespad=0.2)
c.set_ylim(.25, .72)
for L, x in zip('abc', (0.0, 2.55, 4.45)): panel_letter(fig, L, x, 2.28)
save(fig, 'figS_further')

# ---- caption ----
open('figS_further_caption.tex', 'w').write(
 "Three of the further analyses (Section~\\ref{supp:further}), all post hoc and exploratory: time against size (a), other embeddings in place of PCA (b) and random errors in the broad labels (c). "
 "\\textbf{(a)}~Wall time against the number of cells for one run per size on a 12-core Mac mini, with both axes logarithmic. "
 "Filled red circles joined by a line: our method, all steps summed (342 s at 200,000 cells). "
 "Open red squares with a dashed line: generating candidate groups alone (216.5 s). "
 "Gray triangles: Leiden subclustering alone (56.5 s). "
 "Building the subsample is not counted. "
 "Each size is a different subsample of one atlas, so the curves show cost, not accuracy. "
 "\\textbf{(b)}~Change in average precision (AP) when scVI or scGPT embeddings replace PCA in our method, with the regressor retrained in each space by leaving out one atlas. "
 "Each open circle is one atlas (n = 10, mean over its subsamples, 29 in total) and a gray line joins the two changes of the same atlas. "
 "Red diamonds are the means over atlases, printed at the top, and the dashed line marks no change. "
 "\\textbf{(c)}~AP of the hidden subtypes against the share of cells given a random other broad label, at 0 (clean), 5, 10 and 20\\%. "
 "Red circles: our method. Gray triangles: Leiden subclustering. Each marker is the mean over 10 atlases and bars show the standard error over atlases. "
 "The bars are wide because atlases differ in level, and the paired losses are in the text.\n")
print('ok')
