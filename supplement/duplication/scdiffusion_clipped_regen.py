"""Regenerate scDiffusion cells for m27_hard / m27_mk (lupus) with the clipped renorm (statistical_augmentations.renorm).
The saved s*_sdiff.npz there were renormalized WITHOUT the log1p(1e4) ceiling, which turned decoder rows
above it into one-gene cells; that step is not invertible, so we retrain with the gen scripts' settings:
build_sdiff(base Xtr, ytr, NC+1, seed, 40/60/30 epochs, bs 256, vae_steps 5000), TARGET - n_c per probe type.
Usage: python3 scdiffusion_clipped_regen.py m27_hard 0 1 2 ; old files are moved to pcaspace_bak/<D>_sdiff_unclipped/."""
import sys, os, shutil, time, itertools, numpy as np
from sdiff_mod import build_sdiff, sample_sdiff
D, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
bak = f'pcaspace_bak/{D}_sdiff_unclipped'; os.makedirs(bak, exist_ok=True)
for seed in seeds:
    f = f'{D}/s{seed}_sdiff.npz'; b = np.load(f'{D}/s{seed}_base.npz'); t0 = time.time()
    if os.path.exists(f):
        old = np.load(f)['y']
        if not os.path.exists(f'{bak}/s{seed}_sdiff.npz'): shutil.copy2(f, f'{bak}/s{seed}_sdiff.npz')
    else:   # seed never had sdiff (lupus s2, dropped by a wall-clock outlier): probe order, as the gen scripts
        old = np.concatenate([np.full(256 - int(b['n_c'][c]), int(c)) for c in b['probe'] if 256 - int(b['n_c'][c]) > 0])
    SM = build_sdiff(b['Xtr'], b['ytr'], len(b['cts']) + 1, seed, 40, 60, 30, bs=256, vae_steps=5000,
                     vae_ckpt=f'vae_ckpt/{D}_s{seed}_5000.pt')
    X, y = [], []
    for c, k in ((int(c), len(list(g))) for c, g in itertools.groupby(old)):   # same types, order, counts
        assert k == 256 - int(b['n_c'][c]), (c, k)
        X.append(sample_sdiff(SM, c, k)); y.append(np.full(k, c, np.int64))
    X = np.vstack(X); assert X.max() <= np.log1p(1e4) + 1e-3 and np.array_equal(np.concatenate(y), old)
    np.savez_compressed(f, X=X, y=np.concatenate(y))
    P = np.expm1(X.astype(np.float64)); top = (P.max(1) / P.sum(1))
    print(f'{D} s{seed}: {len(X)} cells in {time.time()-t0:.0f}s; top-gene CPM share median {np.median(top):.3f}, >0.5 {np.mean(top>0.5):.3f}', flush=True)
