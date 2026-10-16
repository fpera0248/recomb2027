"""Part I panel statistics for fig1_partI. Recomputes, from the raw per-dataset CSVs, the per-dataset difference
(method minus duplication) in rare-type recall, at the matched operating point (rec200) and the default threshold (rec_argmax).
Inputs (read only):
  ~/work/recomb27/rev1015/p1data/panel_runs/<dataset>/clf.csv   (88 datasets; every method and classifier except no augmentation)
  ~/work/recomb27/rev1015/p1data/panel_runs/<dataset>/real.csv  (no augmentation)
  ~/work/recomb27/rev1015/p1data/panel_analysis.py, polh.py      (the pre-specified analysis functions, imported unchanged)
  ~/work/recomb27/rev1015/p1data/panel_eligible.csv
Outputs: data/panel_tab.csv (per dataset x method x classifier), data/panel_across.csv, data/panel_p2.csv (max-T per dataset)."""
import os, sys, glob
H = os.path.expanduser('~/work/recomb27/rev1015/p1data'); sys.path.insert(0, H); sys.dont_write_bytecode = True
import pandas as pd, numpy as np
import panel_analysis as P
E = pd.read_csv(f'{H}/panel_eligible.csv').set_index('dataset_id'); S = []
for d in sorted(glob.glob(f'{H}/panel_runs/*/clf.csv')):
    R = os.path.dirname(d); ds = os.path.basename(R)
    x = pd.concat([pd.read_csv(f'{R}/clf.csv'), pd.read_csv(f'{R}/real.csv')]); x = x[x.probe]; x['ds'] = ds; x['tissue'] = E.tissue.get(ds, 'n/a'); S.append(x)
S = pd.concat(S); print('datasets', S.ds.nunique(), 'tissues', S.tissue.nunique())
out = os.path.dirname(os.path.abspath(__file__)) + '/data'
W = [P.within(S, m, 'all') for m in ('rec200', 'rec_argmax')]
T = pd.concat([w[0] for w in W]); P2 = pd.concat([w[1] for w in W]); A = P.across(T)
T.to_csv(f'{out}/panel_tab.csv', index=False); A.to_csv(f'{out}/panel_across.csv', index=False); P2.to_csv(f'{out}/panel_p2.csv', index=False)
g = P2[P2.metric == 'rec200']; print('max-T datasets with a published-generator win (rec200):', int((g.p_maxT < .05).sum()), 'of', len(g), g[g.p_maxT < .05][['ds', 'arm', 'clf']].values.tolist())
