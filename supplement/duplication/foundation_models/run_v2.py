"""Foundation-model rerun: fm/run_fm_dup.py unchanged, backbone swapped to Geneformer V2-316M (frozen).
Only the 12,719 cells run_seed() selects for seeds 0-2 were embedded (embed_v2.py), so FM is a
full-cohort array that is NaN everywhere else; standardize() asserts every train/eval row is finite,
which fails loudly if the split ever drifts from cells_needed.csv.
POOL=cls|mean picks the pooling (mean matches the 6L run). Output: fm3/<POOL>/results*.csv."""
import os, sys
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, HR_WORK + '/cfgen_natrare/j1004/fm')
import run_fm_dup as R

H = os.path.dirname(os.path.abspath(__file__))
POOL = os.environ.get('POOL', 'mean')
EMB = os.environ.get('EMB', 'emb_v2_316m')  # emb_scgpt for the scGPT backbone (VM, same 12,719 cells)
R.OUT = os.path.join(H, POOL if EMB == 'emb_v2_316m' else f'{EMB}_{POOL}'); os.makedirs(R.OUT, exist_ok=True)
N_OBS = 355046


def load_fm_emb():
    import pandas as pd
    idx = pd.read_csv(os.path.join(H, f'{EMB}.csv'))['obs_idx'].values
    E = np.load(os.path.join(H, f'{EMB}_{POOL}.npy'))
    assert len(idx) == len(E) == 12719 and np.all(np.diff(idx) > 0)
    FM = np.full((N_OBS, E.shape[1]), np.nan, dtype=np.float32)
    FM[idx] = E
    return FM


_smoke, _std = R.smoke_test, R.standardize
def smoke_test(FM, obs):
    m = np.isfinite(FM[:, 0])
    _smoke(FM[m], obs[m].reset_index(drop=True))
def standardize(Z0, Zev):
    assert np.isfinite(Z0).all() and np.isfinite(Zev).all(), 'split selected a cell with no embedding'
    return _std(Z0, Zev)

R.load_fm_emb, R.smoke_test, R.standardize = load_fm_emb, smoke_test, standardize

if __name__ == '__main__':
    print(f'EMB={EMB} POOL={POOL} OUT={R.OUT}', flush=True)
    R.main()
