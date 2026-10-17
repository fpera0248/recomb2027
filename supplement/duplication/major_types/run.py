"""Cost to major types: does chasing rare-type performance sacrifice the non-rare (major) types?
README: reads SAVED per-class tables only (no refit, no ssh). For every Part I condition vs no augmentation ('real'),
per classifier, over seeds: mean recall of non-rare types (probe==False; all have n_train=500) at the default
threshold (rec_argmax) and at the matched point (rec200), and macro-F1 over ALL types at the default threshold.
Sources (all under the same processing: each file's arms came from one pipeline run):
  panel : released part1_results/panel_runs/*/{clf,real}.csv  (88 datasets, 3 seeds, lr/knn/rf/mlp, has ncall+nev_used -> exact F1)
  aida  : rev1015/p1data/m27_base.csv (AIDA 31 types, 3 seeds, lr, nev=200/class, has called -> exact F1, incl. scDesign3)
          rev1015/p1data/sd3clf16.csv (AIDA, 4 clfs, recall only, no F1)
F1 per type = 2TP/(ncall+nev), TP=rec_argmax*nev. Macro over types with nev>0.
PRE-DECLARED RULE (per condition x classifier, delta = condition - real, mean over seeds then datasets; 95% CI by
bootstrap over datasets, AIDA: over seeds): SACRIFICE if CI upper < 0 and mean < -0.01; NO-SACRIFICE (non-inferior)
if CI lower > -0.01; else INCONCLUSIVE. Applied separately to nonrare recall@argmax, nonrare-only F1, nonrare recall@200 (matched), macro-F1.
Usage: python3 run.py [--smoke] ; env PANEL=dir AIDA=dir OUT=dir
"""
import os, sys, glob
import numpy as np, pandas as pd
os.environ.setdefault('OMP_NUM_THREADS', '4')
W = os.environ.get('HR_WORK', os.path.expanduser('~/work'))
PANEL = os.environ.get('PANEL', f'{W}/recomb27/v2/p1/part1_results/panel_runs')
AIDA = os.environ.get('AIDA', f'{W}/recomb27/rev1015/p1data')
OUT = os.environ.get('OUT', f'{W}/jiaqi1016/E2_major_types/out' + ('_smoke' if '--smoke' in sys.argv else ''))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

def per_group(d, keys):
    """d needs rec_argmax, rec200, nev, ncall, probe, n -> one row per keys group.
    major = ~probe & n>=500 (at the train cap); nt = ~probe & n<500 (natural-rare, never augmented); rare = probe."""
    d = d[d.nev > 0].copy()
    if 'ncall' not in d: d['ncall'] = np.nan
    d['f1'] = 2 * d.rec_argmax * d.nev / (d.ncall + d.nev)
    g = d.groupby(keys).f1.mean().rename('macro_f1').to_frame()
    for tag, m in [('', ~d.probe & (d.n >= 500)), ('nt_', ~d.probe & (d.n < 500))]:
        x = d[m]
        g[tag + 'nr_rec'] = x.groupby(keys).rec_argmax.mean()
        g[tag + 'nr_f1'] = x.groupby(keys).f1.mean()
        g[tag + 'nr_rec200'] = x.groupby(keys).rec200.mean()
    g['rare_rec'] = d[d.probe].groupby(keys).rec_argmax.mean()
    return g.reset_index()

def load_panel():
    fs = sorted(glob.glob(f'{PANEL}/*/clf.csv'))
    if '--smoke' in sys.argv: fs = fs[:2]
    out = []
    for f in fs:
        d = pd.concat([pd.read_csv(f), pd.read_csv(f.replace('clf.csv', 'real.csv'))])
        d['ds'] = os.path.basename(os.path.dirname(f)); d = d.rename(columns={'nev_used': 'nev'})
        out.append(per_group(d, ['ds', 'seed', 'clf', 'arm']))
    return pd.concat(out)

def load_aida():
    """Three AIDA-31 sets. m27_base: lr, F1, sdiff DROPPED (pre-clip). anchor: pca latent, 4 clf, recall only, post-clip,
    all arms incl sdiff and scDesign3. stat16: lr+knn/rf/mlp, 20 seeds, F1, noise and SMOTE."""
    d = pd.read_csv(f'{AIDA}/m27_base.csv'); d = d[d.arm != 'sdiff']
    d['clf'] = 'lr'; d['nev'] = 200; d['ncall'] = d.called
    assert (d.groupby(['seed', 'arm']).called.sum() == 6200).all()
    g1 = per_group(d, ['seed', 'clf', 'arm']); g1['ds'] = 'AIDA31_lr'
    a = pd.read_csv(f'{AIDA}/m27_latent_anchor.csv'); a = a[a.latent == 'pca'].copy(); a['nev'] = 1
    g2 = per_group(a, ['seed', 'clf', 'arm']); g2['ds'] = 'AIDA31_anchor'
    s = pd.concat([pd.read_csv(f'{AIDA}/stat16.csv').assign(clf='lr') if 'clf' not in pd.read_csv(f'{AIDA}/stat16.csv', nrows=1) else pd.read_csv(f'{AIDA}/stat16.csv'),
                   pd.read_csv(f'{AIDA}/stat16_clf.csv')]).drop_duplicates(['seed', 'arm', 'clf', 'c'])
    s = s[s.arm.isin(['real', 'upsample', 'smote_k5', 'jitter_s1.0'])].copy(); s['nev'] = 200; s['ncall'] = s.called
    assert (s.groupby(['seed', 'arm', 'clf']).called.sum() == 6200).all()
    g3 = per_group(s, ['seed', 'clf', 'arm']); g3['ds'] = 'AIDA31_stat16'
    return pd.concat([g1, g2, g3])

from scipy import stats
MARGIN = 0.01
PRIMARY = ['nr_rec', 'nr_rec200', 'nr_f1', 'nt_nr_rec', 'nt_nr_rec200', 'nt_nr_f1']

def ci(x, how):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan, np.nan, np.nan, len(x), 0
    if how == 'boot':
        m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        lo, hi = np.percentile(m, 2.5), np.percentile(m, 97.5)
    else:  # t-interval over seeds (df=n-1); AIDA is one dataset so this covers seed noise only
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); lo, hi = x.mean() - h, x.mean() + h
    return x.mean(), lo, hi, len(x), int((x < 0).sum())

def verdict(m, lo, hi):
    if np.isnan(m): return 'NA'
    return 'SACRIFICE' if hi < 0 and m < -MARGIN else 'WITHIN-0.01-MARGIN' if lo > -MARGIN else 'INCONCLUSIVE'

def summarize(G, tag):
    panel = tag.startswith('panel')
    base = G[G.arm == 'real'].set_index(['ds', 'seed', 'clf'])
    rows = []
    for (arm, clf), g in G[G.arm != 'real'].groupby(['arm', 'clf']):
        g = g.set_index(['ds', 'seed', 'clf'])
        for met in PRIMARY + ['macro_f1']:
            if met not in g or g[met].isna().all(): continue
            dlt = (g[met] - base[met].reindex(g.index)).dropna().reset_index()
            if panel:
                unit = dlt.groupby('ds')[met].mean()
                if len(unit) < 10: continue
                m, lo, hi, n, neg = ci(unit, 'boot')
            else:
                unit = dlt.set_index('seed')[met]
                m, lo, hi, n, neg = ci(unit, 'boot' if n_seeds(unit) >= 20 else 't')
            v = 'DESCRIPTIVE' if met == 'macro_f1' else verdict(m, lo, hi)
            rows.append(dict(set=tag, arm=arm, clf=clf, metric=met, delta_vs_real=m, lo=lo, hi=hi, n_units=n, n_neg=neg, sig_loss=bool(hi < 0), verdict=v))
    return pd.DataFrame(rows)

def n_seeds(u): return len(u)

P, A = load_panel(), load_aida()
P.to_csv(f'{OUT}/panel_groups.csv', index=False); A.to_csv(f'{OUT}/aida_groups.csv', index=False)
R = pd.concat([summarize(P, 'panel')] + [summarize(A[A.ds == k], k) for k in A.ds.unique()])
R = R[R.n_units > 0]
R.to_csv(f'{OUT}/E2_summary.csv', index=False)
print(R.groupby(['set', 'metric', 'verdict']).size().to_string())
