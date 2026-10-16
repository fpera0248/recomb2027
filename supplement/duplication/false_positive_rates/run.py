"""E3 (J22a/b): cell-level FPR per Part I condition for rare (probe) types, from saved predictions. No refit.
README: python run.py [--smoke]. Outputs fpr_rows*.csv (probe31 rows), fpr_summary*.csv (probe31), fpr_panel_summary.csv,
fpr_j21_summary.csv in this directory. Reads tgz (probe31_runs.csv, panel_runs/*/{clf,real}.csv) and ref_j21/{aida,hlca,lupus}_full.csv.
Definitions. N_eval = sum over classes of called (probe31) or nev_used (panel; equals sum(ncall) where ncall exists).
 Probe types have exactly 200 eval cells (part1/split_cache.py lines 74-102: takes ep[:NEV], dies if fewer).
 default threshold: FP = called - rec_argmax*nev ; FPR = FP/(N_eval-nev)
 matched cut (nev called for nev true): FP = nev*(1-rec200); FPR = FP/(N_eval-nev); TPR = recall, FP = FN.
 Matched FPR diff = -(recall diff)*nev/(N_eval-nev), so it restates the matched-recall test, not new evidence.
RULE (frozen v2, changed from v1 on checker evidence): method differs from duplication ('upsample') for a classifier only if the
 paper's own test rejects: exact/MC sign-flip over per-type mean paired differences (part1/signflip.py), BH q<0.05 over all
 method x classifier tests of the table (probe31 also over the paper's 20 primary tests). v1 used a percentile bootstrap CI
 with no multiplicity control; checkers showed it disagrees with the paper (mlp/real raw p 0.033, BH q 0.066).
 Bootstrap CIs (over types) are kept as descriptive. Default-threshold FPR is descriptive, always shown with default TPR.
 Metric for rare types is TPR + FPR, never F1."""
import io, sys, tarfile, os, glob, importlib.util
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

SMOKE = '--smoke' in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
TGZ = '/Users/fperaltacastro/work/recomb2027-assets/part1_results.tgz'
J21 = '/Users/fperaltacastro/work/recomb27/monday_q/donorfix/build/ovr_t/ref_j21'
sp = importlib.util.spec_from_file_location('sf', '/Users/fperaltacastro/work/recomb2027-r8i3wt/part1/signflip.py')
sf = importlib.util.module_from_spec(sp); sp.loader.exec_module(sf)
sfx = '_smoke' if SMOKE else ''


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def boot(units, vals, rng, B=2000):
    """units: array of unit ids per row; 95% percentile CI of the mean, resampling units."""
    g = {u: vals[units == u] for u in np.unique(units)}
    ks = list(g)
    bs = [np.mean(np.concatenate([g[ks[i]] for i in rng.integers(0, len(ks), len(ks))])) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def summarize(p, keys, unit, cols, base='upsample', seed=0):
    """p: probe rows with columns keys + [arm, clf, seed] + cols. keys identify a pair (seed, type, dataset...);
    unit = column(s) naming the type unit for the sign-flip. Returns per (clf, arm) means, diffs vs base, p, CIs."""
    rng = np.random.default_rng(seed); out = []
    p = p.copy(); p['_u'] = p[unit].astype(str).agg('|'.join, axis=1) if isinstance(unit, list) else p[unit].astype(str)
    for clf, g in p.groupby('clf'):
        b = g[g.arm == base].set_index(keys)
        for arm, h in g.groupby('arm'):
            h = h.set_index(keys)
            r = dict(clf=clf, arm=arm, n_rows=len(h), n_types=h._u.nunique())
            for m in cols:
                r[m] = h[m].mean()
                if arm == base: continue
                dv = (h[m] - b[m]).dropna().to_frame('d').join(h._u)
                dv = dv.reset_index(drop=True) if False else dv
                r[m + '_diff_dup'] = dv.d.mean()
                r[m + '_diff_lo'], r[m + '_diff_hi'] = boot(dv._u.values, dv.d.values, rng)
                if m == 'fpr_matched':
                    r['p_signflip'], _ = sf.signflip_p(dv.d.values, dv._u.values)
                    r['n_pairs'] = len(dv)
            out.append(r)
    S = pd.DataFrame(out)
    t = S.arm != base
    S.loc[t, 'q_bh'] = bh(S.loc[t, 'p_signflip'].values)
    S['differs_from_dup'] = S.q_bh < 0.05
    return S


# ---- A. probe31 (31 types, 9 conditions) ----
with tarfile.open(TGZ) as t:
    d = pd.read_csv(io.BytesIO(t.extractfile('part1_results/probe31_runs.csv').read()))
    if not SMOKE:
        panel = []
        for m in t.getmembers():
            if m.name.startswith('part1_results/panel_runs/') and m.name.endswith(('/clf.csv', '/real.csv')):
                x = pd.read_csv(io.BytesIO(t.extractfile(m).read())); x['ds'] = m.name.split('/')[2]; panel.append(x)
if SMOKE:
    d = d[(d.seed < 2) & (d.clf == 'lr')]
k = ['seed', 'arm', 'clf']
d['N'] = d.groupby(k).called.transform('sum')
assert d.groupby(['seed', 'clf']).N.nunique().max() == 1, 'N_eval must not depend on arm'
p = d[d.probe].copy(); NEV = 200
tp = p.rec_argmax * NEV
assert np.allclose(tp, tp.round(), atol=1e-6), 'probe eval size != 200'
neg = p.N - NEV
p['fpr_default'] = (p.called - tp) / neg
p['fpr_matched'] = NEV * (1 - p.rec200) / neg
p['tpr_default'], p['tpr_matched'] = p.rec_argmax, p.rec200
p.to_csv(f'{HERE}/fpr_rows{sfx}.csv', index=False)
COLS = ['fpr_default', 'fpr_matched', 'tpr_default', 'tpr_matched']
S = summarize(p, ['seed', 'c'], 'c', COLS)
PRIM = {'jitter_s0.5', 'jitter_s1.0', 'jitternz_s1.0', 'smote_k5', 'real'}
pm = S.arm.isin(PRIM)
S.loc[pm, 'q_bh_paper20'] = bh(S.loc[pm, 'p_signflip'].values)
S.to_csv(f'{HERE}/fpr_summary{sfx}.csv', index=False)
print('PROBE31'); print(S.drop(columns=['n_rows']).round(5).to_string())
if SMOKE: sys.exit()

# ---- B. 88-dataset panel (generators + statistical conditions) ----
q = pd.concat(panel); q['N'] = q.groupby(['ds', 'seed', 'arm', 'clf']).nev_used.transform('sum')
q = q[q.probe].copy(); neg = q.N - q.nev_used
q['fpr_matched'] = q.nev_used * (1 - q.rec200) / neg; q['tpr_matched'] = q.rec200
hasn = q.ncall.notna()
q['fpr_default'] = np.where(hasn, (q.ncall - q.rec_argmax * q.nev_used) / neg, np.nan); q['tpr_default'] = np.where(hasn, q.rec_argmax, np.nan)
q['ct'] = q.ds + ':' + q.c.astype(str)
SP = summarize(q, ['ds', 'seed', 'c'], 'ct', COLS)
SP['n_datasets'] = q.ds.nunique()
SP.to_csv(f'{HERE}/fpr_panel_summary.csv', index=False)
print('PANEL'); print(SP.drop(columns=['n_rows']).round(5).to_string())

# ---- C. AIDA / HLCA / lupus generators (existing J21 refit metrics, multiclass, n>=2) ----
J = []
for ds in ('aida', 'hlca', 'lupus'):
    x = pd.read_csv(f'{J21}/{ds}_full.csv', usecols=['dataset', 'seed', 'arm', 'clf', 'mode', 'c', 'n', 'probe', 'default_fpr', 'matched_fpr', 'default_tpr', 'matched_tpr'])
    x = x[(x['mode'] == 'multiclass') & x.probe & (x.n >= 2)]
    if ds == 'lupus': x = x[x.arm != 'sdiff']
    x = x.rename(columns={'default_fpr': 'fpr_default', 'matched_fpr': 'fpr_matched', 'default_tpr': 'tpr_default', 'matched_tpr': 'tpr_matched'})
    x['ct'] = x.c.astype(str)
    s = summarize(x, ['seed', 'c'], 'ct', COLS); s.insert(0, 'dataset', ds); J.append(s)
SJ = pd.concat(J); SJ.to_csv(f'{HERE}/fpr_j21_summary.csv', index=False)
print('J21'); print(SJ.drop(columns=['n_rows']).round(5).to_string())
