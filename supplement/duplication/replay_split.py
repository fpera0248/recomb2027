"""Replay dup_vs_generators.py's data prep + donor-disjoint split + PCA-50 and cache each seed to rep16_seed<s>.npz.
Logic copied from an earlier feature-extraction script (not included), verified bit-exact against dup_vs_generators for seeds 0-2.
Seeds 0-2 are checked against gen16.log/gen16.csv; every seed is checked for 0 donor overlap,
exact rungs, and >= NEV eval cells per probe type. Any mismatch stops the run.
Usage: python3 replay_split.py 0 20
"""
import os, sys, time, numpy as np, pandas as pd, anndata as ad, scanpy as sc
SUPP_DATA = os.environ.get("SUPP_DATA", os.path.expanduser("~/data"))
from sklearn.decomposition import PCA

D = SUPP_DATA + '/pilots/aida/'
HVG, NEV, CAP = 2000, 200, 500
RUNGS = np.array([1, 2, 4, 8, 16, 32, 64, 128])
PROBE = ['double negative T regulatory cell', 'CD141-positive myeloid dendritic cell',
         'innate lymphoid cell', 'pre-conventional dendritic cell', 'plasma cell',
         'plasmacytoid dendritic cell', 'mature B cell',
         'CD16-negative, CD56-bright natural killer cell, human']
EXPECT = {0: (283, 11755, 75503), 1: (283, 11755, 72739), 2: (290, 11755, 70807)}
# P31=1: balanced cyclic design over ALL cell types (seeds 0..NT-1): seed s probes types order[(s + 4k) % NT], k=0..7, at
# RUNGS[k], so every type is a probe exactly 8 times, once at each rung (4 is coprime with a prime NT = 31)
P31 = os.environ.get('P31') == '1'

def die(msg): sys.exit(f'REPLAY MISMATCH, stopping: {msg}')

if __name__ == '__main__':
    t0 = time.time()
    ref = pd.read_csv('gen16.csv'); ref = ref[ref.probe].drop_duplicates(['seed', 'name_'])
    REF_N = {(int(s), nm): int(n) for s, nm, n in ref[['seed', 'name_', 'n']].values}
    A = ad.concat([ad.read_h5ad(D + 'AIDA_AbundantPool_120k.h5ad'), ad.read_h5ad(D + 'AIDA_RarePool.h5ad')],
                  join='inner', index_unique='-')
    A.obs['ct'] = A.obs.cell_type.astype(str)
    A = A[:, np.asarray(A.X.sum(0)).ravel() > 0].copy()
    TRH = os.environ.get('HVG_TRAIN') == '1'   # 1 = pick HVGs per seed on train cells only; dup_vs_generators used the pooled cohort
    def lognorm(X):
        X = np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)
        return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4).astype(np.float32)
    if not TRH: sc.pp.highly_variable_genes(A, n_top_genes=HVG, flavor='seurat_v3', subset=True)
    cts = sorted(A.obs.ct.unique()); ci = {c: i for i, c in enumerate(cts)}; NC = len(cts)
    yall = A.obs.ct.map(ci).values.astype(np.int64)
    dall = A.obs.donor_id.astype(str).values; UD = np.unique(dall)
    if not TRH: Xn = lognorm(A.X); del A
    if P31: order = np.random.default_rng(31).permutation(cts); assert NC == 31
    else: probe = np.array(sorted(ci[c] for c in PROBE))
    print(f'loaded {len(yall):,} cells  {time.time()-t0:.0f}s', flush=True)
    for seed in range(int(sys.argv[1]), int(sys.argv[2])):
        rng = np.random.default_rng(seed)
        if P31:
            PROBE = [order[(seed + 4 * k) % NC] for k in range(len(RUNGS))]; probe = np.array(sorted(ci[c] for c in PROBE))
        perm = rng.permutation(UD); cnt = {c: 0 for c in PROBE}; ev_d = []
        for d in perm:
            if all(v >= NEV for v in cnt.values()): break
            ev_d.append(d); m = dall == d
            for c in PROBE: cnt[c] += int((yall[m] == ci[c]).sum())
        if min(cnt.values()) < NEV: die(f'seed {seed}: ran out of donors before every probe type had {NEV} eval cells')
        ev_mask = np.isin(dall, ev_d); tr_mask = ~ev_mask
        rung_of = ({ci[c]: int(r) for c, r in zip(PROBE, RUNGS)} if P31 else dict(zip(probe.tolist(), rng.permutation(RUNGS).tolist())))
        ev_idx, full_tr, keep = [], [], []
        for c in range(NC):
            ep = rng.permutation(np.where((yall == c) & ev_mask)[0]); ev_idx.append(ep[:NEV])
            rest = rng.permutation(np.where((yall == c) & tr_mask)[0]); full_tr.append(rest)
            keep.append(rest[:rung_of[c]] if c in rung_of else rest[:CAP])
        ev_idx = np.concatenate(ev_idx); tr_idx = np.concatenate(keep)
        n_del = len(np.concatenate(full_tr)) - len(tr_idx)
        ytr = yall[tr_idx]; n_c = np.bincount(ytr, minlength=NC)
        if any(n_c[c] != r for c, r in rung_of.items()): die(f'seed {seed}: a rung did not land')
        if set(dall[tr_idx]) & set(dall[ev_idx]): die(f'seed {seed}: donor overlap')
        if not P31 and seed in EXPECT and (len(ev_d), len(tr_idx), n_del) != EXPECT[seed]:
            die(f'seed {seed}: {(len(ev_d), len(tr_idx), n_del)} != gen16.log {EXPECT[seed]}')
        if not P31 and seed in EXPECT and any(n_c[c] != REF_N[(seed, cts[c])] for c in probe): die(f'seed {seed}: n != gen16.csv')
        if TRH:
            g = sc.pp.highly_variable_genes(A[tr_idx].copy(), n_top_genes=HVG, flavor='seurat_v3', inplace=False).highly_variable.values
            Xtr, Xev = lognorm(A.X[tr_idx][:, g]), lognorm(A.X[ev_idx][:, g])
        else: Xtr, Xev = Xn[tr_idx], Xn[ev_idx]
        pca = PCA(50, random_state=seed).fit(Xtr[rng.choice(len(tr_idx), min(30000, len(tr_idx)), replace=False)])
        fn = f'rep{31 if P31 else 16}{"t" if TRH else ""}_seed{seed}.npz'; Ztr = pca.transform(Xtr)
        if os.path.exists(fn): assert np.array_equal(np.load(fn)['Ztr'], Ztr), f'{fn}: rerun does not reproduce the cached split'
        # Xtr/pmean/comp: statistical_augmentations.gene_arm builds jitter/SMOTE in gene space and projects them (same encoder for every arm)
        np.savez(fn, Ztr=Ztr, ytr=ytr, Zev=pca.transform(Xev), yev=yall[ev_idx], n_c=n_c, probe=probe, cts=np.array(cts),
                 n_ev_donors=len(ev_d), Xtr=Xtr, pmean=pca.mean_, comp=pca.components_)
        print(f'  seed {seed}: {len(ev_d)}/{len(UD)} eval donors, {len(tr_idx):,} train, deleted {n_del:,}, '
              f'overlap 0, rungs ok{" , matches gen16" if seed in EXPECT and not P31 else ""}  {time.time()-t0:.0f}s', flush=True)
