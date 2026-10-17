"""Donor-split replay WITHOUT fitting. Lines copied verbatim from replay16.py (AIDA, incl. P31) and
m27_gen_{hlca,lupus}_t.py (identical split code); the originals run data prep + model fits at import,
so they cannot be imported. Each loader returns (cts, yall, dall); `split` returns the split pieces.
Checked against cached arrays by evdon.py (yev, n_c, n_ev_donors)."""
import numpy as np, pandas as pd

RUNGS = np.array([1, 2, 4, 8, 16, 32, 64, 128]); NEV = 200
AIDA_PROBE = ['double negative T regulatory cell', 'CD141-positive myeloid dendritic cell',
              'innate lymphoid cell', 'pre-conventional dendritic cell', 'plasma cell',
              'plasmacytoid dendritic cell', 'mature B cell',
              'CD16-negative, CD56-bright natural killer cell, human']
LUPUS_PROBE = ['progenitor cell', 'plasmablast', 'plasmacytoid dendritic cell',
               'non-classical monocyte', 'conventional dendritic cell', 'lymphocyte']
HLCA_PROBE = ['acinar cell', 'bronchus fibroblast of lung', 'tracheobronchial smooth muscle cell', 'B cell',
              'endothelial cell of lymphatic vessel', 'pulmonary artery endothelial cell',
              'respiratory tract hillock cell', 'non-classical monocyte']

def _obs(path):
    import anndata as ad
    o = ad.read_h5ad(path, backed='r').obs
    return o.cell_type.astype(str).values, o.donor_id.astype(str).values

def load_aida(d):
    """replay16.py: concat(AbundantPool_120k, RarePool); row order = abundant then rare (obs only, no X)."""
    a = _obs(d + '/AIDA_AbundantPool_120k.h5ad'); b = _obs(d + '/AIDA_RarePool.h5ad')
    return _ids(np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]))

def load_hlca(path):
    """m27_gen_hlca_t.py: its cell filter dropped no cell (logged), so obs order = cell universe."""
    return _ids(*_obs(path))

def load_lupus(path, ctrl_cap=12000):
    """m27_gen_lupus_t.py lines 77-85 (keep mask) verbatim; kept rows stay in file order."""
    ct, don = _obs(path); PROBE = LUPUS_PROBE
    keep = np.isin(ct, PROBE); rng0 = np.random.default_rng(12345)
    for c in sorted(set(ct) - set(PROBE)):
        ix = np.where(ct == c)[0]
        sel = rng0.choice(ix, min(len(ix), ctrl_cap), replace=False)
        keep[sel] = True
    global KEEP_IDX; KEEP_IDX = np.where(keep)[0]   # file row of each kept cell (cell-identity check)
    return _ids(ct[keep], don[keep])

def _ids(ct, don):
    cts = sorted(set(ct)); ci = {c: i for i, c in enumerate(cts)}
    return cts, np.array([ci[c] for c in ct], dtype=np.int64), np.asarray(don).astype(str)

def split(cts, yall, dall, PROBE, seed, p31=False, order_seed31=31, CAP=500, nev=NEV):
    """replay16.py lines 40-65 / m27_gen_*_t.py: seeded donor permutation, eval donors until every probe type
    has >= nev eval cells, rungs per probe, first nev permuted eval cells per type, kept train = rung / CAP."""
    ci = {c: i for i, c in enumerate(cts)}; NC = len(cts); UD = np.unique(dall)
    if p31:
        assert NC == 31; order = np.random.default_rng(order_seed31).permutation(cts)
        PROBE = [order[(seed + 4 * k) % NC] for k in range(len(RUNGS))]
    probe = np.array(sorted(ci[c] for c in PROBE))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(UD); cnt = {c: 0 for c in PROBE}; ev_d = []
    for d in perm:
        if all(v >= nev for v in cnt.values()): break
        ev_d.append(d); m = dall == d
        for c in PROBE: cnt[c] += int((yall[m] == ci[c]).sum())
    assert min(cnt.values()) >= nev, 'ran out of donors'
    ev_mask = np.isin(dall, ev_d); tr_mask = ~ev_mask
    rung_of = ({ci[c]: int(r) for c, r in zip(PROBE, RUNGS)} if p31
               else dict(zip(probe.tolist(), rng.permutation(RUNGS)[:len(probe)].tolist())))
    ev_idx, keep = [], []
    for c in range(NC):
        ep = rng.permutation(np.where((yall == c) & ev_mask)[0]); ev_idx.append(ep[:nev])
        rest = rng.permutation(np.where((yall == c) & tr_mask)[0])
        keep.append(rest[:rung_of[c]] if c in rung_of else rest[:CAP])
    ev_idx = np.concatenate(ev_idx); tr_idx = np.concatenate(keep)
    assert not set(dall[tr_idx]) & set(dall[ev_idx]), 'donor overlap'
    n_c = np.bincount(yall[tr_idx], minlength=NC)
    assert all(n_c[c] == r for c, r in rung_of.items()), 'rung did not land'
    return dict(ev_d=ev_d, ev_idx=ev_idx, tr_idx=tr_idx, probe=probe, rung_of=rung_of, n_c=n_c, ev_mask=ev_mask)

def conc(donors):
    """distinct donors, effective donors 1/sum p^2, top-donor share."""
    _, k = np.unique(donors, return_counts=True); p = k / k.sum()
    return len(k), float(1 / (p ** 2).sum()), float(p.max())

def selfcheck():
    r = np.random.default_rng(0); n = 6000
    ct = r.choice([f't{i}' for i in range(12)], n, p=np.r_[np.full(4, .05), np.full(8, .1)] / 1.0)
    don = r.choice([f'd{i}' for i in range(120)], n)
    cts, y, d = _ids(ct, don)
    probe = cts[:4]
    s = split(cts, y, d, probe, 3, nev=20, CAP=50)
    assert len(s['ev_idx']) == 20 * len(cts) and set(d[s['ev_idx']]) <= set(s['ev_d'])
    s2 = split(cts, y, d, probe, 3, nev=20, CAP=50)
    assert np.array_equal(s['ev_idx'], s2['ev_idx']) and np.array_equal(s['tr_idx'], s2['tr_idx'])
    assert conc(np.array(['a'] * 3 + ['b']))[0] == 2 and abs(conc(np.array(['a', 'b', 'c', 'd']))[1] - 4) < 1e-9
    assert abs(conc(np.array(['a'] * 3 + ['b']))[2] - .75) < 1e-9
    print('splits selfcheck ok')

if __name__ == '__main__': selfcheck()
