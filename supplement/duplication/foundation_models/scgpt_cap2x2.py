"""scGPT cap effect vs seed noise (C18 control): 2x2 of max_length {1200, 6300} x tokenizer seed {0, 1}.
Same cells/splits/analysis; paired on (seed, type, clf, n_train), n_train>=2, sign-flip over 8 types."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/fperaltacastro/work/cfgen_natrare')
from polh import signflip_p
K = ['seed', 'type', 'clf', 'n_train']
R = {k: pd.read_csv(f'{d}/results.csv') for k, d in
     [('cap0', 'emb_scgpt_cls'), ('full0', 'emb_scgptfull_cls'), ('cap1', 'emb_scgptcaps1_cls'), ('full1', 'emb_scgptfulls1_cls')]}
def fm(run, arm):
    r = R[run]; r = r[(r.rep == 'fm') & (r.arm == arm) & (r.n_train >= 2)]
    return r.set_index(K).rec200
def pca(arm):
    r = R['cap0']; r = r[(r.rep == 'pca50') & (r.arm == arm) & (r.n_train >= 2)]
    return r.set_index(K).rec200
def test(a, b):
    d = (a - b).dropna(); out = []
    for c in ['lr', 'knn', 'rf', 'mlp']:
        x = d.xs(c, level='clf'); p, dt = signflip_p(x.values, x.index.get_level_values('type').values)
        out.append(f'{c} {dt.mean():+.4f} ({(dt > 0).sum()}/{len(dt)}, p {p:.3f})')
    return '  '.join(out)
for arm in ['dup', 'none']:
    print(f'== arm {arm}  (mean of per-type diffs, types+, sign-flip p)')
    print('seed noise, cap   (cap1 - cap0):  ', test(fm('cap1', arm), fm('cap0', arm)))
    print('seed noise, full  (full1 - full0):', test(fm('full1', arm), fm('full0', arm)))
    print('cap effect, seed0 (full0 - cap0): ', test(fm('full0', arm), fm('cap0', arm)))
    print('cap effect, seed1 (full1 - cap1): ', test(fm('full1', arm), fm('cap1', arm)))
    print('cap effect, 2-seed mean:          ', test((fm('full0', arm) + fm('full1', arm)) / 2, (fm('cap0', arm) + fm('cap1', arm)) / 2))
    for run in R: print(f'FM vs PCA50, {run:5s}:              ', test(fm(run, arm), pca(arm)))
assert all((R[k][R[k].rep == 'pca50'].rec200.values == R['cap0'][R['cap0'].rep == 'pca50'].rec200.values).all() for k in R), 'PCA rows differ'
print('PCA-50 rows identical across all 4 runs: OK')
