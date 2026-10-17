"""One-versus-rest classifiers: does the published-generator-versus-duplication verdict hold under one-versus-rest?
Reads the saved per-row metrics ({aida,hlca,lupus}_full.csv, written by ../metrics/run_e1.py; not in the repository, set J21_METRICS to their folder).
Same rows and conventions as our earlier one-versus-rest versus multiclass comparison
(not included): probe types only, zero-call precision
scored 0, lupus drops sdiff (invalid cells), n=1 rows kept apart (n2plus is the tested group),
exact sign-flip over rare types, BH within each (dataset, mode) family.
Statistic: generator minus duplication ('upsample'), per (seed, type, n, clf), averaged to one
value per type, then sign-flip over types.
usage: python3 dup_vs_gen_one_vs_rest.py  -> prints tables, writes j21_tests.csv here.
Inputs: aida_full.csv, hlca_full.csv, lupus_full.csv, the per-row metrics that duplication/metrics/one_vs_rest_metrics.py writes
(one run per dataset, each needs that dataset's generated cells, which are not released). Put the three files next to
this script or set J21_METRICS to the folder that holds them. No other input is read."""
import itertools, os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.environ.get('J21_METRICS', os.path.join(HERE, '..', 'metrics'))
_miss = [f for f in ('aida', 'hlca', 'lupus') if not os.path.exists(f'{IN}/{f}_full.csv')]
if _miss:
    raise SystemExit(f'missing {IN}/{{{",".join(_miss)}}}_full.csv: write them with duplication/metrics/run_e1.py or set J21_METRICS')
GENS = {'synth_w1': 'CFGen (shipped)', 'synthbal_w1': 'CFGen (balanced)', 'scvi': 'scVI',
        'scvi_cond': 'scVI (conditional)', 'sdiff': 'scDiffusion', 'nb': 'scDesign3 NB', 'sd3_nosingle': 'scDesign3'}
METRICS = ['matched_recall', 'ap']


def signflip_p(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    obs = abs(v.sum())
    n_ge = sum(abs((np.array(s) * v).sum()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=len(v)))
    return n_ge / 2 ** len(v), len(v)


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    q = p * len(p) / r; qs = np.minimum.accumulate(q[o][::-1])[::-1]; out = np.empty(len(p)); out[o] = np.minimum(qs, 1)
    return out


rows = []
for ds in ['aida', 'hlca', 'lupus']:
    d = pd.read_csv(f'{IN}/{ds}_full.csv')
    d = d[(d.dataset == ds) & (d.probe == True)]
    if ds == 'lupus':
        d = d[d.arm != 'sdiff']
    for pfx in ('default', 'matched'):
        z = (d[f'{pfx}_TP'] + d[f'{pfx}_FP']) == 0
        d.loc[z, f'{pfx}_precision'] = 0.0
    for mode in ['multiclass', 'ovr']:
        m = d[(d['mode'] == mode) & (d.n >= 2)]
        dup = m[m.arm == 'upsample'].set_index(['seed', 'c', 'n', 'clf'])
        fam = []
        for g, gname in GENS.items():
            if g not in set(m.arm):
                continue
            x = m[m.arm == g].set_index(['seed', 'c', 'n', 'clf'])
            j = x.join(dup, rsuffix='_dup', how='inner')
            assert len(j) == len(x) == len(dup), (ds, mode, g, len(j), len(x), len(dup))
            for clf in ['lr', 'knn', 'rf', 'mlp']:
                jc = j.xs(clf, level='clf')
                for met in METRICS:
                    diff = jc[met] - jc[f'{met}_dup']
                    per_type = diff.groupby(level='c').mean()
                    p, k = signflip_p(per_type.values)
                    fam.append(dict(dataset=ds, mode=mode, generator=gname, clf=clf, metric=met, n_rows=len(diff),
                                    n_types=k, mean_diff=diff.mean(), types_behind=int((per_type < 0).sum()), p=p))
        q = bh([r['p'] for r in fam])
        for r, qq in zip(fam, q):
            r['bh_q'] = qq; rows.append(r)

T = pd.DataFrame(rows)
T.to_csv(f'{HERE}/j21_tests.csv', index=False)
T['call'] = np.where(T.bh_q < 0.05, np.where(T.mean_diff < 0, 'behind', 'ahead'), 'no_diff')
print(T.groupby(['dataset', 'mode', 'call']).size().unstack(fill_value=0))
print('\nsign of mean difference (generator minus duplication), all tests:')
print(T.assign(neg=T.mean_diff < 0).groupby(['dataset', 'mode']).neg.agg(['sum', 'count']))
w = T.pivot_table(index=['dataset', 'generator', 'clf', 'metric'], columns='mode', values='mean_diff')
print('\nmode agreement in sign:', int((np.sign(w.multiclass) == np.sign(w.ovr)).sum()), 'of', len(w))
print('tests where multiclass and ovr calls differ:')
c = T.pivot_table(index=['dataset', 'generator', 'clf', 'metric'], columns='mode', values='call', aggfunc='first')
print(c[c.multiclass != c.ovr].to_string())
print('\nmatched_recall mean diff by dataset/mode/generator (over clf):')
print(T[T.metric == 'matched_recall'].pivot_table(index=['dataset', 'generator'], columns='mode', values='mean_diff').round(3).to_string())
print('\nany generator ahead under ovr at q<0.05:', T[(T['mode'] == 'ovr') & (T.call == 'ahead')][['dataset', 'generator', 'clf', 'metric', 'mean_diff', 'bh_q']].to_string())
