"""CELLxGENE panel analysis. Specified before any panel dataset had finished and before any panel result was read.

Inputs: <PANEL_DIR>/<dataset>/clf.csv (m27_hlclf, every arm x 4 readouts) + real.csv (no augmentation), for
datasets with a 'done' marker. Probe rows only. Unit of analysis = dataset; inside a dataset the unit is the
probe type (8 per dataset), averaged over the 3 seeds (rungs rotate across seeds).

Within a dataset, for every arm a != upsample and readout k: d = rec(a) - rec(upsample) per (seed, type), per-type
means, exact two-sided sign-flip p over the 8 types (polh.signflip_p, floor 0.0078), BH within family (secondary).
MDE per test = 1.92 x SD of the type means (80% power for an 8/8 exact result, no multiplicity).

Families (fixed now):
  PUBLISHED generators: scvi, scvi_cond, synth_w1 (CFGen), synthbal_w1 (CFGen class-balanced), sdiff (scDiffusion), nb,
    sd3_nosingle (scDesign3; absent from the panel, SKIP_SD3=1, so HLCA/AIDA's published family is larger than the panel's)
  OURS (decoder resampling, sampler-vs-decoder decomposition): lat_gauss, lat_boot
  SPREAD (statistical upsampling): jitter_s1.0, jitternz_s1.0, smote_k5
  NOAUG: real
Primary (metric rec200 = recall in the top-NEVc eval cells, the matched operating point):
  P2 headline, per dataset: one-sided studentized max-T sign-flip (Westfall-Young single step, exact 256 flips) over
     every published arm x readout; FWER 0.05 per dataset. Count datasets where any published generator beats duplication.
  P1 per arm x readout across datasets: mean d, 95% bootstrap CI (marginal), one-sided 95% upper bound, median,
     sign-flip p over datasets with BH across rows; non-inferiority of duplication: upper bound of (arm - dup) < DELTA.
  P3 duplication vs no augmentation: upsample - real at rec200 AND rec_argmax, own BH family (4 readouts x 2 metrics).
Sensitivity: rec200_tie (ties at the cutoff split evenly); rung n=1 units dropped; rungs n<=8 only;
  rungs n>=16 only (post hoc, added after the 10-dataset interim look). Secondary: by tissue.

  python3 panel_analysis.py            -> panel_report.txt, panel_tab.csv, panel_across.csv
"""
import os, sys, glob, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import polh
PANEL_DIR = os.environ.get('PANEL_DIR', 'panel_runs'); OUTP = os.environ.get('PANEL_OUT', 'panel')
DELTA = 0.02   # non-inferiority margin in recall, fixed before any panel result
FAM = {'published': ['scvi', 'scvi_cond', 'synth_w1', 'synthbal_w1', 'sdiff', 'nb', 'sd3_nosingle'], 'ours': ['lat_gauss', 'lat_boot'],
       'spread': ['jitter_s1.0', 'jitternz_s1.0', 'smote_k5'], 'noaug': ['real']}
FAMOF = {a: f for f, v in FAM.items() for a in v}
SUB = {'all': lambda X: X, 'no_n1': lambda X: X[X.n > 1], 'n_le8': lambda X: X[X.n <= 8],
       'n_ge16': lambda X: X[X.n >= 16]}   # added post hoc after the 10-dataset interim look (RF dup is pathological at n<=2)

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out

def maxT(D):
    """One-sided studentized max-T over the columns of D (types x tests), exact over all 2^types sign flips.
    Returns the single-step adjusted p of each column (controls FWER over the columns)."""
    D = D.dropna(axis=1, how='all').dropna(); n = len(D); X = D.values
    F = np.array(list(itertools.product((-1, 1), repeat=n)), float)
    with np.errstate(all='ignore'): m = (F @ X) / n                       # sum of squares is flip-invariant
    sd = np.sqrt(np.maximum(((X ** 2).sum(0) - n * m ** 2) / (n - 1), 1e-18))
    T = m / sd * np.sqrt(n); mx = T.max(1); obs = T[-1]                    # last row of product = no flip
    return pd.Series([float(np.mean(mx >= t - 1e-9)) for t in obs], index=D.columns)

def load():
    E = pd.read_csv(f'{HERE}/panel_eligible.csv').set_index('dataset_id')
    S = []
    for d in sorted(glob.glob(f'{PANEL_DIR}/*/done')):
        ds = os.path.basename(os.path.dirname(d)); R = os.path.dirname(d)
        x = pd.concat([pd.read_csv(f'{R}/clf.csv'), pd.read_csv(f'{R}/real.csv')]); x = x[x.probe]
        x['ds'] = ds; x['tissue'] = E.tissue.get(ds, 'n/a'); S.append(x)
    if not S: return None
    S = pd.concat(S)
    assert set(S.arm) <= set(FAMOF) | {'upsample'}, f'unknown arms {set(S.arm) - set(FAMOF)}'
    dup = S.duplicated(['ds', 'seed', 'name_', 'clf', 'arm'])
    assert not dup.any(), f'duplicate rows in {S[dup].ds.unique()}'
    return S

def within(S, metric, sub):
    rows, P2 = [], []
    for ds, X in S.groupby('ds'):
        X = SUB[sub](X)
        w = X.pivot_table(index=['seed', 'name_'], columns=['clf', 'arm'], values=metric)
        R, TM = [], {}
        for (k, a) in w.columns:
            if a == 'upsample' or (k, 'upsample') not in w.columns: continue
            m = w[(k, a)].notna() & w[(k, 'upsample')].notna(); dd = (w[(k, a)] - w[(k, 'upsample')])[m]
            grp = dd.index.get_level_values('name_').values; p, dt = polh.signflip_p(dd.values, grp)
            if FAMOF[a] == 'published': TM[(k, a)] = dd.groupby(level='name_').mean()
            R.append(dict(ds=ds, tissue=X.tissue.iloc[0], metric=metric, sub=sub, clf=k, arm=a, fam=FAMOF[a],
                          d=float(dt.mean()), pos=int((dt > 0).sum()), types=len(dt), p=p, mde=1.92 * dt.std(ddof=1),
                          units=int(m.sum()), seeds=dd.index.get_level_values('seed').nunique()))
        if not R: print(f'  skipped {ds[:8]} for {metric}/{sub}: no paired rows'); continue
        R = pd.DataFrame(R); R['q'] = R.groupby('fam').p.transform(lambda s: bh(s.values)); rows.append(R)
        if TM:
            pa = maxT(pd.DataFrame(TM)); r = pa.idxmin()
            P2.append(dict(ds=ds, tissue=X.tissue.iloc[0], metric=metric, sub=sub, p_maxT=pa.min(), tests=len(pa),
                           clf=r[0], arm=r[1], d=float(TM[r].mean()), pos=int((TM[r] > 0).sum())))
    return pd.concat(rows), pd.DataFrame(P2)

def across(T):
    out = []
    for (mt, sb, k, a), g in T.groupby(['metric', 'sub', 'clf', 'arm']):
        d = g.d.values; b = np.random.default_rng(0).choice(d, (5000, len(d))).mean(1)
        p = polh.signflip_p(d, np.arange(len(d)))[0] if len(d) > 1 else np.nan
        out.append(dict(metric=mt, sub=sb, clf=k, arm=a, fam=FAMOF[a], n_ds=len(d), n_partial=int((g.seeds < 3).sum()),
                        mean_d=d.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), lo95=np.percentile(b, 5),
                        hi95=np.percentile(b, 95), median_d=np.median(d), med_mde=g.mde.median(), p_ds=p,
                        ds_win=int(((g.q < .05) & (g.d > 0)).sum()), ds_loss=int(((g.q < .05) & (g.d < 0)).sum())))
    A = pd.DataFrame(out); A['q_ds'] = A.groupby(['metric', 'sub']).p_ds.transform(lambda s: bh(s.fillna(1).values))
    A['noninf'] = A.hi95 < DELTA; return A

def report(S):
    W = [within(S, m, sb) for m in ('rec200', 'rec200_tie', 'rec_argmax') if m in S for sb in SUB]
    T = pd.concat([w[0] for w in W]); P2 = pd.concat([w[1] for w in W])
    A = across(T); T.to_csv(f'{OUTP}_tab.csv', index=False); A.to_csv(f'{OUTP}_across.csv', index=False)
    P2.to_csv(f'{OUTP}_p2.csv', index=False)
    Q = [l.strip() for l in open(f'{HERE}/panel_queue.txt') if l.strip()]
    st = lambda f: [q for q in Q if os.path.exists(f'{PANEL_DIR}/{q}/{f}')]
    fl = [q for q in st('failed') if q not in st('done')]
    M = S.ds.nunique()
    L = [f'CELLxGENE panel: {M} datasets done, {len(fl)} failed {[q[:8] for q in fl]}, {len(Q) - M - len(fl)} pending, of {len(Q)} queued;'
         f' {S.tissue.nunique()} tissues; arms {sorted(S.arm.unique())}']
    for mt, sb in (('rec200', 'all'), ('rec200_tie', 'all'), ('rec200', 'no_n1'), ('rec200', 'n_le8'), ('rec200', 'n_ge16'), ('rec_argmax', 'all')):
        g = P2[(P2.metric == mt) & (P2['sub'] == sb)]
        if g.empty: continue
        win = g[g.p_maxT < .05]
        L.append(f'\nP2{" HEADLINE" if (mt, sb) == ("rec200", "all") else ""} ({mt}, {sb}): datasets where any published generator beats '
                 f'duplication (max-T over {int(g.tests.median())} arm x readout tests, FWER .05): {len(win)} of {len(g)}'
                 f' (at most {0.05 * len(g):.1f} expected by chance if none ever does; the test is conservative)')
        for r in win.itertuples(): L.append(f'   {r.ds[:8]} {r.tissue:18s} {r.arm:12s} {r.clf:4s} d {r.d:+.3f} {r.pos}/8 p {r.p_maxT:.4f}')
    for mt, sb in (('rec200', 'all'), ('rec200_tie', 'all'), ('rec200', 'no_n1'), ('rec200', 'n_le8'), ('rec200', 'n_ge16'), ('rec_argmax', 'all')):
        a = A[(A.metric == mt) & (A['sub'] == sb)]
        if a.empty: continue
        L.append(f'\n=== P1 {mt} ({sb}): d = arm - duplication over datasets; CI marginal; q_ds = BH over this table;'
                 f' noninf = one-sided 95% upper bound < {DELTA}')
        L.append(f'{"family":10s}{"arm":14s}{"clf":5s}{"n_ds":>5s}{"mean":>8s}{"95% CI":>18s}{"up95":>7s}{"median":>8s}{"medMDE":>7s}'
                 f'{"p_ds":>9s}{"q_ds":>9s}{"BHwin":>6s}{"BHloss":>7s}{"noninf":>7s}{"partial":>8s}')
        for r in a.sort_values(['fam', 'arm', 'clf']).itertuples():
            L.append(f'{r.fam:10s}{r.arm:14s}{r.clf:5s}{r.n_ds:5d}{r.mean_d:+8.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]{r.hi95:+7.3f}{r.median_d:+8.3f}'
                     f'{r.med_mde:7.3f}{r.p_ds:9.2g}{r.q_ds:9.2g}{r.ds_win:6d}{r.ds_loss:7d}{str(r.noninf):>7s}{r.n_partial:8d}')
    p3 = A[(A.arm == 'real') & (A['sub'] == 'all') & A.metric.isin(['rec200', 'rec_argmax'])].copy()
    if not p3.empty:
        p3['q3'] = bh(p3.p_ds.fillna(1).values)
        L.append('\n=== P3 duplication - no augmentation (own BH family); rec200 = matched operating point, rec_argmax = default threshold')
        L.append(f'{"metric":12s}{"clf":5s}{"n_ds":>5s}{"dup-noaug":>10s}{"95% CI":>18s}{"p_ds":>9s}{"q":>9s}{"dup wins":>9s}{"noaug wins":>11s}')
        for r in p3.sort_values(['metric', 'clf']).itertuples():
            L.append(f'{r.metric:12s}{r.clf:5s}{r.n_ds:5d}{-r.mean_d:+10.3f}  [{-r.hi:+.3f}, {-r.lo:+.3f}]{r.p_ds:9.2g}{r.q3:9.2g}{r.ds_loss:9d}{r.ds_win:11d}')
    t = T[(T.metric == 'rec200') & (T['sub'] == 'all') & (T.fam == 'published')].groupby('tissue').apply(
        lambda g: pd.Series(dict(ds=g.ds.nunique(), mean_d=g.d.mean(), bh_win=int(((g.q < .05) & (g.d > 0)).sum()))), include_groups=False)
    L.append('\n=== published generators by tissue (rec200, all readouts pooled)\n' + t.astype({'ds': int, 'bh_win': int}).round(3).to_string())
    open(f'{OUTP}_report.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':   # maxT self-check: a lone 8/8 win beside big losers is detectable
        rng = np.random.default_rng(1); D = pd.DataFrame({'win': 0.02 + 0.005 * rng.standard_normal(8), 'null': 0.02 * rng.standard_normal(8),
                                                          'loser': -0.3 + 0.05 * rng.standard_normal(8)})
        pa = maxT(D); print(pa.round(4).to_dict()); assert pa['win'] < .05 and pa['loser'] > .5; sys.exit(0)
    S = load()
    if S is None: sys.exit('no finished panel datasets yet')
    report(S)
