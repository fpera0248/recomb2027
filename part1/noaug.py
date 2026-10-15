# no-augmentation method on HLCA, same _base/_score as panel_readout.py; compared to its upsample rows
import sys

sys.path.insert(0, '.')
import numpy as np, pandas as pd, signflip
from panel_readout import _base, _score, CSV

rows = []
for s in range(3):
    b, comp, pmean, Ztr, ytr, Zev, yev = _base(s)
    rows += _score(
        Ztr, ytr, Zev, yev, b['cts'], b['n_c'], b['probe'], b['NEVc'], s, 'real'
    )
R = pd.DataFrame(rows)
R.to_csv(sys.argv[1], index=False)
S = pd.concat([pd.read_csv(CSV), R])
S = S[S.probe]
for k in ('lr', 'knn', 'rf', 'mlp'):
    w = S[S.clf == k].pivot_table(
        index=['seed', 'c'], columns='method', values='rec200'
    )
    grp = w.index.get_level_values('c').values
    for a in ('real', 'jitter_s1.0', 'lat_boot', 'lat_gauss'):
        d = (w[a] - w['upsample']).values
        p, dt = signflip.signflip_p(d, grp)
        print(
            k,
            a.ljust(12),
            'rec200',
            round(w[a].mean(), 3),
            'up',
            round(w['upsample'].mean(), 3),
            'd',
            f'{d.mean():+.3f}',
            f'{(dt>0).sum()}/{len(dt)}',
            'p',
            round(p, 3),
        )
