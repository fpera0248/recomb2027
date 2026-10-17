"""Step 1 (C18 V2-316M build): reproduce run_fm_dup.py's exact cell selection for seeds 0-2
by IMPORTING the real function (not re-implemented), so the RNG call sequence is guaranteed
identical to the finished 6L run. fm/run_fm_dup.py is read-only -- this only imports it.

run_seed()'s only inputs that affect which cells are picked are `obs` (cell_type, donor_id,
in aida_sg.h5ad row order) and the RNG seeded from `seed`; the `FM` argument is accepted but
never read inside run_seed's body (verified by inspection), so passing None is exact, not an
approximation.

Writes cells_needed.csv: one row per (seed, role, cell) with both the positional obs index
(0..355045, matches run_fm_dup.py's a.obs.index) and the real obs_name (10x barcode, from
aida_sg.h5ad directly) so embed_v2.py can pull raw counts by either key.
"""
import os, sys
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.dont_write_bytecode = True   # fm/ is read-only; never let the import drop a .pyc there
import numpy as np
import pandas as pd

FM_DIR = HR_WORK + '/cfgen_natrare/j1004/fm'
OUT = HR_WORK + '/cfgen_natrare/j1004/fm3'
sys.path.insert(0, FM_DIR)
import run_fm_dup as R  # read-only import; module-level code is just def's, main() is guarded


def main():
    obs, h5 = R.load_obs()                       # positional index 0..N-1, from aida_sg.h5ad backed='r'
    obs_names = np.asarray(h5.obs_names)          # real 10x barcodes, same row order as `obs`
    assert len(obs_names) == len(obs)

    LAB = sorted(obs.cell_type.unique())
    LAB2I = {l: i for i, l in enumerate(LAB)}
    ALL_ABUND = [l for l in LAB if l not in R.PROBE]
    assert len(ALL_ABUND) + len(R.PROBE) == len(LAB) == 32

    rows = []
    per_seed_summary = []
    for seed in R.SEEDS:
        S = R.run_seed(seed, obs, None, LAB2I, ALL_ABUND)
        tr, ev = S['train_idx'], S['eval_idx']
        assert len(np.intersect1d(tr, ev)) == 0, f'seed {seed}: train/eval overlap'
        assert len(tr) == len(np.unique(tr)) and len(ev) == len(np.unique(ev)), \
            f'seed {seed}: duplicate index within a role'
        for idx in tr:
            rows.append((seed, 'train', int(idx), obs_names[idx]))
        for idx in ev:
            rows.append((seed, 'eval', int(idx), obs_names[idx]))
        per_seed_summary.append((seed, len(tr), len(ev), S['cap_report']))
        print(f'seed {seed}: train {len(tr)} eval {len(ev)} cap_report {S["cap_report"]}', flush=True)

    df = pd.DataFrame(rows, columns=['seed', 'role', 'obs_idx', 'obs_name'])
    df.to_csv(os.path.join(OUT, 'cells_needed.csv'), index=False)

    uniq_idx = df['obs_idx'].unique()
    uniq_names = df['obs_name'].unique()
    assert len(uniq_idx) == len(uniq_names)
    print(f'\nwrote {len(df)} rows -> cells_needed.csv')
    print(f'unique cells across all 3 seeds x {{train,eval}}: {len(uniq_idx)}')
    for seed, ntr, nev, cap in per_seed_summary:
        print(f'  seed {seed}: train={ntr} eval={nev} rung_caps={len(cap)}')

    # Cross-check against any saved split artifact in fm/ (none expected: fm/ only kept
    # results.csv / results_ovr.csv / rung_caps.json / run.log, no raw index arrays).
    saved = [f for f in os.listdir(FM_DIR) if 'idx' in f.lower() or 'split' in f.lower()]
    print(f'\nsaved split/index files found in fm/: {saved or "NONE"}')
    if not saved:
        print('No saved split file to diff against. Equivalence to the actual 6L run instead '
              'comes from calling the literal same run_seed() function (imported, not copied) '
              'with the literal same obs and seed -- there is no second code path that could '
              'diverge.')


if __name__ == '__main__':
    main()
