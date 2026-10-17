"""de_marker_recovery: differential-expression marker recovery for every augmentation arm, on dup_vs_generators's exact
donor-disjoint split and marker_recovery's saved arm cells (2000 HVGs, log1p-CPM). Question: does duplication (or scDesign3, or anything) recover better marker genes for rare types
than the real cells alone, and does any method inflate significance (pseudoreplication)?

Group = the probe type's n_c real training cells + that arm's added cells (identical across arms
for 'rest'). Two truths:
  A: held-out-donor eval cells of the type vs the SAME rest (real training cells of other types)
  B: fully donor-disjoint -- eval cells of the type vs eval cells of all other types
Tests: Mann-Whitney U (one-sided, up-regulated) and Welch t (one-sided), BH-FDR per test.
Metrics per (seed, probe type, arm, truth, test): top-50 Jaccard (ranked by test statistic,
restricted to lfc>0), average precision of the arm's ranking for truth's significant set
(FDR<0.05 & lfc>0.25), n_sig, precision of significant calls vs truth, and fp_rate = fraction of
truth NON-markers the arm calls significant (the pseudoreplication/p-value-inflation readout).

  python3 de_marker_recovery.py run [marker_recovery]     -> m27_de.csv
  python3 de_marker_recovery.py report         -> m27_de_report.txt (+ stdout)
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')
import sys, glob, time, warnings, numpy as np, pandas as pd, polh
from scipy import stats
from statistical_arms import gene_arm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

K, LFC_THR, FDR_THR = 50, 0.25, 0.05
ARMS_NEEDED = ['real', 'upsample', 'jitter_s1.0', 'jitternz_s1.0', 'smote_k5', 'sd3_nosingle', 'nb', 'lat_boot',
               'lat_gauss', 'synthbal_w1', 'synth_w1', 'scvi', 'scvi_cond', 'sdiff', 'more_real']


def lfc(mc, mr):
    return np.log2((np.expm1(np.maximum(mc, 0)) + 1e-9) / (np.expm1(np.maximum(mr, 0)) + 1e-9))


def stat_arms(b, seed, T):
    """jitter_s1.0 / smote_k5 built in gene space and renormalized (statistical_arms.gene_arm)."""
    return {name: {c: v[0] for c, v in gene_arm(name, b['Xtr'], b['ytr'], b['probe'], b['n_c'], seed, T).items()}
            for name in ('jitter_s1.0', 'jitternz_s1.0', 'smote_k5')}


def de_test(G, R):
    """G vs R, log1p-CPM. One-sided (G > R, i.e. up-regulated in G). Returns per-gene arrays."""
    with np.errstate(all='ignore'):
        mw = stats.mannwhitneyu(G, R, axis=0, alternative='greater')
        tw = stats.ttest_ind(G, R, axis=0, equal_var=False, alternative='greater')
    p_w = np.nan_to_num(mw.pvalue, nan=1.0); p_t = np.nan_to_num(tw.pvalue, nan=1.0)
    return dict(stat_w=np.nan_to_num(mw.statistic, nan=0.0), q_w=multipletests(p_w, method='fdr_bh')[1],
                stat_t=np.nan_to_num(tw.statistic, nan=0.0), q_t=multipletests(p_t, method='fdr_bh')[1],
                lfc=lfc(G.mean(0), R.mean(0)))


def sig_set(d, qk):
    return set(np.where((d[qk] < FDR_THR) & (d['lfc'] > LFC_THR))[0].tolist())


def topk(d, sk):
    idx = np.where(d['lfc'] > 0)[0]
    return set(idx[np.argsort(-d[sk][idx])][:K].tolist())


def unit_metrics(d_arm, d_truth, test):
    sk, qk = ('stat_w', 'q_w') if test == 'wilcoxon' else ('stat_t', 'q_t')
    sig_a, sig_t = sig_set(d_arm, qk), sig_set(d_truth, qk)
    top_a, top_t = topk(d_arm, sk), topk(d_truth, sk)
    jacc = len(top_a & top_t) / len(top_a | top_t) if (top_a | top_t) else np.nan
    y_true = np.zeros(len(d_truth['lfc']), bool)
    if sig_t: y_true[list(sig_t)] = True
    ap = average_precision_score(y_true, d_arm[sk]) if y_true.any() else np.nan
    prec = len(sig_a & sig_t) / len(sig_a) if sig_a else np.nan
    non_marker = np.ones(len(d_truth['lfc']), bool)
    if sig_t: non_marker[list(sig_t)] = False
    fp_rate = float((d_arm[qk][non_marker] < FDR_THR).mean()) if non_marker.any() else np.nan
    return dict(jaccard50=jacc, ap=ap, n_sig=len(sig_a), n_sig_truth=len(sig_t), precision_sig=prec, fp_rate=fp_rate)


def do_unit(bf, seed, c):
    b = np.load(bf)
    Xtr, ytr, Xev, yev = b['Xtr'], b['ytr'], b['Xev'], b['yev']
    arms_npz = {os.path.basename(f)[len(f's{seed}_'):-4]: np.load(f)
                for f in glob.glob(f'mk16/s{seed}_*.npz') if not f.endswith('_base.npz')}
    T = int(b['n_c'][b['probe'][0]] + (arms_npz['upsample']['y'] == b['probe'][0]).sum())
    added = {a: {int(cc): d['X'][d['y'] == cc] for cc in np.unique(d['y'])} for a, d in arms_npz.items()}
    added.update(stat_arms(b, seed, T))
    real_c = Xtr[ytr == c]
    rest_A, rest_B = Xtr[ytr != c], Xev[yev != c]
    Gc = Xev[yev == c]
    d_truthA, d_truthB = de_test(Gc, rest_A), de_test(Gc, rest_B)
    sets = {'real': real_c, 'more_real': b['Xfull'][b['yfull'] == c]}
    sets.update({a: np.vstack([real_c, v[c]]) for a, v in added.items() if c in v})
    rows = []
    for a, C in sets.items():
        if a not in ARMS_NEEDED: continue
        d_A, d_B = de_test(C, rest_A), de_test(C, rest_B)
        for truth_name, d_truth, d_arm in (('A', d_truthA, d_A), ('B', d_truthB, d_B)):
            for test in ('wilcoxon', 'welch'):
                m = unit_metrics(d_arm, d_truth, test)
                rows.append(dict(seed=seed, c=c, type=str(b['cts'][c]), n=int(b['n_c'][c]), n_cells=len(C),
                                  arm=a, truth=truth_name, test=test, **m))
    return rows


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'run':
    from joblib import Parallel, delayed
    D = sys.argv[2] if len(sys.argv) > 2 else 'mk16'
    units = []
    for bf in sorted(glob.glob(f'{D}/s*_base.npz')):
        seed = int(os.path.basename(bf)[1:].split('_')[0])
        b = np.load(bf)
        for c in b['probe']: units.append((bf, seed, int(c)))
    print(f'{len(units)} (seed,type) units', flush=True); t0 = time.time()
    out = Parallel(n_jobs=3, verbose=10)(delayed(do_unit)(*u) for u in units)
    R = pd.DataFrame([r for o in out for r in o])
    R.to_csv('m27_de.csv', index=False)
    print(f'done {time.time()-t0:.0f}s, {len(R)} rows -> m27_de.csv', flush=True)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'report':
    import io
    buf = io.StringIO()
    def P(*a): print(*a); print(*a, file=buf)
    R = pd.read_csv('m27_de.csv')
    P(f"m27_de: {R.seed.nunique()} seeds x {R.c.nunique()} probe types x {R.arm.nunique()} arms, "
      f"truth A/B x wilcoxon/welch = {len(R)} rows")
    P("truth A = held-out eval cells of the type vs real training cells of other types (same rest as every arm)")
    P("truth B = held-out eval cells of the type vs held-out eval cells of all other types (fully donor-disjoint)")
    P(f"significant = BH-FDR<{FDR_THR} & log2FC>{LFC_THR}; jaccard50/AP restricted to up-regulated (lfc>0) candidates")

    def table(w, title):
        grp = w.index.get_level_values('type').values; nn = w.index.get_level_values('n').values
        P(f"\n### {title}")
        P(f"{'arm':14s}{'mean':>8s}{'d vs up':>9s}{'p_up':>7s}{'d vs real':>10s}{'p_real':>8s}"
          f"{'n<=8':>8s}{'n>=16':>8s}")
        for a in sorted(w.columns, key=lambda a: -w[a].mean() if w[a].notna().any() else 1e9):
            mu = w[a].notna() & w['upsample'].notna()
            du2 = (w.loc[mu, a] - w.loc[mu, 'upsample']).values; gu = grp[mu.values]
            pu, _ = polh.signflip_p(du2, gu) if mu.any() else (np.nan, None)
            mr = w[a].notna() & w['real'].notna()
            dr2 = (w.loc[mr, a] - w.loc[mr, 'real']).values; gr = grp[mr.values]
            pr, _ = polh.signflip_p(dr2, gr) if mr.any() else (np.nan, None)
            lo = w.loc[nn <= 8, a].mean(); hi = w.loc[nn >= 16, a].mean()
            P(f"{a:14s}{w[a].mean():8.3f}{du2.mean() if len(du2) else np.nan:+9.3f}{pu:7.3f}"
              f"{dr2.mean() if len(dr2) else np.nan:+10.3f}{pr:8.3f}{lo:8.3f}{hi:8.3f}")

    for truth in ('A', 'B'):
        for test in ('wilcoxon', 'welch'):
            sub = R[(R.truth == truth) & (R.test == test)]
            idx = ['seed', 'type', 'n']
            for metric in ('jaccard50', 'ap', 'precision_sig', 'fp_rate', 'n_sig'):
                w = sub.pivot_table(index=idx, columns='arm', values=metric)
                table(w, f"truth {truth}, {test}: {metric}")

    P("\n### pseudoreplication check: fp_rate (fraction of truth's NON-marker genes an arm calls "
      "significant) -- upsample vs real, both truths/tests")
    for truth in ('A', 'B'):
        for test in ('wilcoxon', 'welch'):
            sub = R[(R.truth == truth) & (R.test == test)]
            w = sub.pivot_table(index=['seed', 'type', 'n'], columns='arm', values='fp_rate')
            grp = w.index.get_level_values('type').values
            mu = w['upsample'].notna() & w['real'].notna()
            dd = (w.loc[mu, 'upsample'] - w.loc[mu, 'real']).values; gg = grp[mu.values]
            p, dt = polh.signflip_p(dd, gg) if mu.any() else (np.nan, None)
            P(f"truth {truth} {test:9s} upsample fp_rate={w['upsample'].mean():.4f}  real fp_rate={w['real'].mean():.4f}  "
              f"d={dd.mean():+.4f}  types+={int((dt>0).sum()) if dt is not None else 0}/8  p={p:.4f}")

    with open('m27_de_report.txt', 'w') as f: f.write(buf.getvalue())
