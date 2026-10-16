"""Replace the clip-affected arms' rows in stat16{TAG}.csv / _dcr.csv with the rerun in stat16{TAG}cj.*.
Every other arm is deterministic and untouched by the renorm clip, so its rows are kept as they are."""
import sys, pandas as pd
t = sys.argv[1]
for suf in ('', '_dcr'):
    old, new = pd.read_csv(f'stat16{t}{suf}.csv'), pd.read_csv(f'stat16{t}cj{suf}.csv')
    keep = old[~old.arm.isin(set(new.arm))]
    assert len(keep) + len(new) == len(old), (t, suf, len(old), len(keep), len(new))
    pd.concat([keep, new]).to_csv(f'stat16{t}{suf}.csv', index=False)
    print(t or 'base', suf or 'rec', f'{len(new)} rows replaced ({sorted(new.arm.unique())})')
