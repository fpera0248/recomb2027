"""Marker genes: marker / DE recovery of every augmentation arm, re-derived from the SAVED per-unit outputs.
README: python3 run.py [DIR_WITH_mk16.csv_and_m27_de.csv] [OUT_DIR]   (seconds, 1 core, <200 MB)
Inputs  : p1data/mk16.csv (top-50 by log2FC of arm means, genes detected in >=25% of truth cells; truth = held-out-donor
          eval cells of the type vs real train cells of other types) and p1data/m27_de.csv (Wilcoxon + Welch, BH<0.05 & lfc>0.25,
          truths A=vs real train rest, B=vs held-out rest). 3 seeds x 8 AIDA probe types, donor-disjoint split, 2000 HVGs log1p-CPM.
Same processing (Table S2): every arm is built/renormalized in gene space through statistical_arms.renorm (ceiling log1p(1e4)); the xmax
          assert below only checks that renorm ran, NOT that the clip ran (renorm bounds xmax either way). Clip evidence is circumstantial:
          clip_chain.sh lane A reran marker_recovery/de_marker_recovery after the clip and values moved off the pre-clip numbers. HVGs: 2000 chosen on all
          cells before the donor split, as in Table S2 (no train-only-HVG variant). All arms except more_real are topped to 256 cells;
          more_real = all pre-deletion train cells of the type (median ~600), an UNMATCHED upper reference, not a matched gain.
Metric labels: marker_recovery jaccard50 = top-50 genes by log2FC of arm means (genes detected in >=25% of truth cells). de_marker_recovery jaccard50 = top-50
          up-regulated genes ranked by the Wilcoxon (or Welch) statistic; BH<0.05 & lfc>0.25 feeds only n_sig/precision/fp_rate.
          sd3_nosingle has 21 of 24 units: quote only its PAIRED difference. Table S2 names: nb=scDesign3 without copula,
          sd3_nosingle=unmodified scDesign3, synth_w1=CFGen, synthbal_w1=CFGen class-balanced, jitter=noise on copies,
          jitternz=noise on nonzero genes, real=no augmentation, upsample=duplication.
REPORTING RULE (fixed before re-deriving; NOT a pre-registration, all numbers were already in the reports): unit = probe type (seed-averaged); p = exact two-sided sign-flip over the 8 type means
          (floor 2/256 = 0.0078). Arm A "recovers MORE than" B only if d>0 and p<0.05 in marker_recovery AND in de_marker_recovery Wilcoxon truth A
          AND truth B; "LESS" symmetric; else "no difference shown". Reference = duplication ('upsample') and pre-aug 'real'.
          No arm may be credited with recovering markers beyond 'more_real' (extra real donors' cells). Recall-free metric
          (set overlap), so the rare-class F1 rule does not apply. Never say "no augmentation beats real" without naming the test.
Output  : OUT/e4_markers.csv (one row per arm x metric), printed table.
"""
import sys, os, itertools, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
D = sys.argv[1] if len(sys.argv) > 1 else HR_WORK + '/recomb27/rev1015/p1data'
O = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
S = np.array(list(itertools.product((-1, 1), repeat=8)))

def p_sf(d):                                   # d: Series indexed by type -> exact sign-flip p over type means
    t = d.groupby(level='type').mean().values; assert len(t) == 8
    return float(np.mean(np.abs((S * t).sum(1)) / 8 >= abs(t.mean()) - 1e-12))

def vs(R, m, ref):
    P = R.pivot_table(index=['seed', 'type'], columns='arm', values=m)
    return {a: (P[a].mean(), (P[a] - P[ref]).mean(), p_sf((P[a] - P[ref]).dropna()) if a != ref else 1.0) for a in P}

mk = pd.read_csv(f'{D}/mk16.csv'); de = pd.read_csv(f'{D}/m27_de.csv')
assert mk.xmax.max() < np.log1p(1e4) + .05, 'renorm ceiling violated (does not test the clip)'
assert mk.seed.nunique() == 3 and mk.type.nunique() == 8
rows = []
for ref in ('upsample', 'real'):
    for a, (m, d, p) in vs(mk, 'jaccard50', ref).items():
        rows.append(dict(metric='mk16_top50_logFC_jaccard', ref=ref, arm=a, mean=m, d=d, p=p))
    for (tr, te), g in de.groupby(['truth', 'test']):
        for a, (m, d, p) in vs(g, 'jaccard50', ref).items():
            rows.append(dict(metric=f'm27_de_{te}_truth{tr}_jaccard50', ref=ref, arm=a, mean=m, d=d, p=p))
R = pd.DataFrame(rows)
# BH over non-reference arms within each metric x ref (conservative: one family, not Table S2 method groups)
def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty(len(p)); n = len(p)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)
R['q_bh'] = np.nan
for _, ix in R[R.arm != R.ref].groupby(['metric', 'ref']).groups.items(): R.loc[ix, 'q_bh'] = bh(R.loc[ix, 'p'])
# fp_rate (Wilcoxon, truth A): dose effect vs pseudoreplication
fp = de[(de.truth == 'A') & (de.test == 'wilcoxon')].groupby(['arm', 'type']).fp_rate.mean().groupby('arm').mean().round(3)
print('fp_rate wilcoxon truthA (mean over types):', fp.to_dict())
for a, v in fp.items(): R.loc[len(R)] = dict(metric='m27_de_wilcoxon_truthA_fp_rate', ref='none', arm=a, mean=v, d=np.nan, p=np.nan, q_bh=np.nan)
R.to_csv(f'{O}/e4_markers.csv', index=False)
W = R[(R.ref == 'upsample') & R.metric.isin(['mk16_top50_logFC_jaccard', 'm27_de_wilcoxon_truthA_jaccard50', 'm27_de_wilcoxon_truthB_jaccard50'])]
def verdict(g):
    s = [(x.d > 0 and x.p < .05) - (x.d < 0 and x.p < .05) for x in g.itertuples()]
    if len(s) < 3: return 'not tested (<3 metrics)'
    return 'MORE' if all(v > 0 for v in s) else 'LESS' if all(v < 0 for v in s) else 'none shown'
T = W.pivot(index='arm', columns='metric', values='d').round(3)
T['p_max'] = W.groupby('arm').p.max().round(3); T['verdict_vs_dup'] = W.groupby('arm').apply(verdict)
print(T.sort_values('mk16_top50_logFC_jaccard', ascending=False).to_string())

print('\nWelch rows vs duplication (supplementary, not in the rule):')
print(R[(R.ref == 'upsample') & R.metric.str.contains('welch') & R.arm.isin(['sd3_nosingle', 'real', 'more_real'])][['metric', 'arm', 'd', 'p', 'q_bh']].round(3).to_string())
print('\nscDesign3 vs real:'); print(R[(R.ref == 'real') & (R.arm == 'sd3_nosingle')][['metric', 'd', 'p']].round(3).to_string())
print('\nunits per arm (mk16):', mk.groupby('arm').size().to_dict())
