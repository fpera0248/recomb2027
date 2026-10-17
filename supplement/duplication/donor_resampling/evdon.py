"""Eval/train donor concentration per probe type x seed, from a fit-free replay of the donor-disjoint splits.
  python3 evdon.py [--ds aida8,aida31,lupus,hlca] [--out DIR]     (--selfcheck: toy test only)
Env: AIDA_DIR (pools, mini only), REPDIR (rep16t/rep31t npz), LUPUS_H5, HLCA_H5, LUPUS_BASE, HLCA_BASE (from_mini dirs).
Outputs (new names): evdon_df.csv (one row per dataset, seed, probe type), evdon_tot_df.csv (eval donors per seed),
evcells_df_<ds>_s<seed>.npz (eval-cell donor ids aligned with the cached yev; used by donor_boot.py).
Every replay is asserted against the cached arrays (yev, n_c, n_ev_donors) and the logged eval-donor totals."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits as S
E = os.path.expanduser
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
SUPP_DATA = os.environ.get("SUPP_DATA", os.path.expanduser("~/data"))
LOGGED = {'aida8': {0: (283, 619), 1: (283, 619), 2: (290, 619)},
          'lupus': {0: (62, 261), 1: (67, 261), 2: (58, 261)}, 'hlca': {0: (24, 106), 1: (22, 106), 2: (19, 106)}}
CFG = {  # ds: (probe, seeds, p31, cache file pattern)
    'aida8': (S.AIDA_PROBE, range(20), False, lambda s: f"{E(os.environ.get('REPDIR', HR_WORK + '/recomb27/monday_q/donorfix/data'))}/rep16t_seed{s}.npz"),
    'aida31': (None, range(31), True, lambda s: f"{E(os.environ.get('REPDIR', HR_WORK + '/recomb27/monday_q/donorfix/data'))}/rep31t_seed{s}.npz"),
    'lupus': (S.LUPUS_PROBE, range(3), False, lambda s: f"{E(os.environ.get('LUPUS_BASE', HR_WORK + '/recomb27/rev1015/hvg_c3/from_mini/m27_mk_t'))}/s{s}_base.npz"),
    'hlca': (S.HLCA_PROBE, range(3), False, lambda s: f"{E(os.environ.get('HLCA_BASE', HR_WORK + '/recomb27/rev1015/hvg_c3/from_mini/m27_hl_t'))}/s{s}_base.npz")}

def load(ds):
    if ds.startswith('aida'): return S.load_aida(E(os.environ.get('AIDA_DIR', SUPP_DATA + '/pilots/aida')))
    if ds == 'lupus': return S.load_lupus(E(os.environ.get('LUPUS_H5', SUPP_DATA + '/cellxgene/lupus_perez.h5ad')))
    return S.load_hlca(E(os.environ.get('HLCA_H5', SUPP_DATA + '/cellxgene/hlca_m27.h5ad')))

def idcheck(ds, seed, s, base, path):
    """Cell-identity check (hlca, lupus): recompute log1p-CPM of the replayed eval cells on the cached genes straight from the h5ad
    and require equality with the cached Xev (max |diff| < 1e-4). yev/n_c asserts alone do not prove which cells were drawn."""
    import anndata as ad
    A = ad.read_h5ad(path, backed='r'); M = A.X
    h = M[0:50]; h = h.data if hasattr(h, 'data') else np.asarray(h)
    if A.raw is not None and not np.allclose(h % 1, 0): M, vn = A.raw.X, list(A.raw.var_names)   # X is normalised: raw counts live in .raw
    else: vn = list(A.var_names)
    gi = np.array([vn.index(g) for g in base['genes']])
    rows = s['ev_idx'] if ds == 'hlca' else S.KEEP_IDX[s['ev_idx']]; o = np.argsort(rows)
    X = M[np.sort(rows)][:, gi]; X = np.asarray(X.todense() if hasattr(X, 'todense') else X, np.float32)
    Xr = np.empty_like(X); Xr[o] = X; Xn = np.log1p(Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e4)
    dmax = float(np.abs(Xn - base['Xev']).max()); assert dmax < 1e-4, f'{ds} seed {seed}: replayed eval cells != cached Xev (max diff {dmax})'
    print(f'  seed {seed}: cell identity OK (max |Xev - replay| = {dmax:.1e}, {len(rows)} cells)', flush=True)

def run(ds, out, nseeds=None, idc=False):
    PROBE, seeds, p31, cache = CFG[ds]; cts, yall, dall = load(ds)
    print(f'{ds}: {len(yall):,} cells, {len(cts)} types, {len(set(dall))} donors', flush=True)
    rows, tot = [], []
    for seed in list(seeds)[:nseeds]:
        s = S.split(cts, yall, dall, PROBE, seed, p31=p31)
        d = np.load(cache(seed), allow_pickle=True)
        assert list(d['cts']) == cts, 'type list differs from cache'
        assert np.array_equal(yall[s['ev_idx']], d['yev']), f'{ds} seed {seed}: replayed eval labels != cached yev'
        assert np.array_equal(s['n_c'], d['n_c']) and np.array_equal(s['probe'], d['probe']), f'{ds} seed {seed}: n_c/probe differ'
        if 'n_ev_donors' in d.files: assert int(d['n_ev_donors']) == len(s['ev_d']), f'{ds} seed {seed}: n_ev_donors'
        if ds in LOGGED and seed in LOGGED[ds]: assert (len(s['ev_d']), len(set(dall))) == LOGGED[ds][seed], (ds, seed, len(s['ev_d']))
        if idc: idcheck(ds, seed, s, d, E(os.environ.get('LUPUS_H5', SUPP_DATA + '/cellxgene/lupus_perez.h5ad')) if ds == 'lupus' else E(os.environ.get('HLCA_H5', SUPP_DATA + '/cellxgene/hlca_m27.h5ad')))
        evd, trd = dall[s['ev_idx']], dall[s['tr_idx']]; yev, ytr = yall[s['ev_idx']], yall[s['tr_idx']]
        assert not os.path.exists(f'{out}/evcells_df_{ds}_s{seed}.npz'), 'exists'
        np.savez_compressed(f'{out}/evcells_df_{ds}_s{seed}.npz', ev_donor=evd, yev=yev, ev_idx=s['ev_idx'])
        tot.append(dict(ds=ds, seed=seed, n_ev_donors=len(s['ev_d']), n_donors_all=len(set(dall)),
                        n_train_donors=len(set(dall[s['tr_idx']])), n_ev_cells=len(yev)))
        for c in s['probe']:
            ne, ee, te = S.conc(evd[yev == c]); nt, et, tt = S.conc(trd[ytr == c])
            rows.append(dict(ds=ds, seed=seed, type=cts[c], c=int(c), rung=int(s['n_c'][c]), n_ev_cells=int((yev == c).sum()),
                             ev_donors=ne, ev_eff_donors=round(ee, 3), ev_top_share=round(te, 4),
                             tr_cells=int((ytr == c).sum()), tr_donors=nt, tr_eff_donors=round(et, 3), tr_top_share=round(tt, 4)))
        print(f'  seed {seed}: {len(s["ev_d"])} eval donors, matches cache', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(tot)

if __name__ == '__main__':
    if '--selfcheck' in sys.argv: S.selfcheck(); sys.exit()
    a = dict(zip([x for x in sys.argv[1:] if x != '--idcheck'][0::2], [x for x in sys.argv[1:] if x != '--idcheck'][1::2]))
    out = a.get('--out', os.path.dirname(os.path.abspath(__file__))); os.makedirs(out, exist_ok=True)
    nse = int(a['--nseeds']) if '--nseeds' in a else None   # smoke only
    R, T = [], []
    for ds in a.get('--ds', 'aida8,aida31,lupus,hlca').split(','):
        r, t = run(ds, out, nse, '--idcheck' in sys.argv); R.append(r); T.append(t)
    sfx = a.get('--tag', '')
    for f in (f'{out}/evdon{sfx}_df.csv', f'{out}/evdon_tot{sfx}_df.csv'): assert not os.path.exists(f), f'{f} exists'
    pd.concat(R).to_csv(f'{out}/evdon{sfx}_df.csv', index=False); pd.concat(T).to_csv(f'{out}/evdon_tot{sfx}_df.csv', index=False)
    print('wrote', out)
