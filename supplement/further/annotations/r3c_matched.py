"""Item 12: annotation improvement at a matched number of classes. Per pool, Leiden subclustering of every broad label at the single global resolution, bisected so the total class count equals (within 1) the class count of our method (A1 of p2_refine, at each threshold), scored with the p2_refine metrics.
  python r3c_matched.py targets PID  (clean env, as score)                                   -> out/small/matched/targets_<PID>.json: our class count per threshold
  python r3c_matched.py grid PID     (env R5, SI_H, SLIM_DIR = data/w6/ID, as run_unit.py)  -> grid_<PID>.parquet (one partition per threshold) + gridinfo_<PID>.json; per threshold the global resolution is bisected (log scale) until the total class count is within 1 of ours
  python r3c_matched.py score PID    (clean env, as p2_refine apply)                         -> out/small/matched/score_<PID>.csv
  python r3c_matched.py summary                                                              -> out/small/matched_summary.csv, matched_pool.csv
  python r3c_matched.py demo         (synthetic, <=1,500 cells)"""
import os, sys, json, numpy as np, pandas as pd
sys.dont_write_bytecode = True
HOME = os.path.expanduser('~'); J = HOME + '/work/jiaqi1011'; OUT = J + '/out/small/matched'; W6 = J + '/out/w6_annot_atac'
MIN_LABEL_N = 20; THR = ('tcal', 't0.3', 't0.5', 't0.7')

def bisect(count, n, lo=0.01, hi=10.0, it=18):
    """log-scale bisection of the resolution until |count(r) - n| <= 1 (count(r) = total classes at r, nondecreasing up to Leiden noise); returns (r, count). Keeps the closest seen, fewer classes on ties."""
    best = None
    for _ in range(it):
        r = float(np.sqrt(lo * hi)); c = count(r)
        if best is None or abs(c - n) < abs(best[1] - n) or (abs(c - n) == abs(best[1] - n) and c < best[1]): best = (r, c)
        if abs(c - n) <= 1: break
        lo, hi = (r, hi) if c < n else (lo, r)
    return best

def target_n(pid):
    """class count of our method (A1 of p2_refine) per threshold, plus the scoring frame."""
    sys.path.insert(0, J + '/code/w6'); sys.path.insert(0, J + '/code/w1')
    from w6common import CLEAN, root
    import p2_refine as P, harness as h
    from score_hlca import targets
    G = pd.read_parquet(f'{W6}/groupP/ID__{pid}.parquet'); R = root('ID'); C = pd.read_parquet(f'{R}/cands/{pid}.cells.parquet'); cells = C.cell_id.values
    d = pd.read_parquet(f'{CLEAN}/hmod/pilots/{pid}.parquet', columns=['obs_name', 'label', 'true_type']).astype({'label': str, 'true_type': str}); si, oid = h.hlib._opaque_order(pid, d.obs_name.values)
    df = d.iloc[si].set_index(oid).reindex(cells); tg = targets(df); ts = json.load(open(f'{W6}/p2_t_cal.json'))['t_cal']; out = {}
    for tn, t in (('tcal', ts), ('t0.3', .3), ('t0.5', .5), ('t0.7', .7)):
        a1, k = P.greedy(G, G['P'].values, t, cells); out[tn] = (P.wrap(a1, df), int(P.wrap(a1, df).nunique()))
    return P, df, tg, cells, out

def targets_cmd(pid):
    os.makedirs(OUT, exist_ok=True); json.dump({tn: n for tn, (_, n) in target_n(pid)[4].items()}, open(f'{OUT}/targets_{pid}.json', 'w'))

def grid(pid):
    """per pool and threshold: the global Leiden resolution (bisected) whose total class count matches ours within 1; needs targets PID first."""
    sys.path.insert(0, os.environ['SI_H']); sys.path.insert(0, HOME + '/work/r5/code')
    import loader, methods as M
    a, _ = loader.load_pilot(pid); X, _h = M.preprocess_pca(a); ids = a.obs_names.values; labels = a.obs['label'].astype(str).values; T = json.load(open(f'{OUT}/targets_{pid}.json'))
    idx = {lab: np.where(labels == lab)[0] for lab in pd.unique(labels)}; cache = {}
    def part(r):
        if r not in cache:
            v = np.array([f'{l}|0' for l in labels], dtype=object)
            for lab, ix in idx.items():
                if len(ix) >= MIN_LABEL_N: v[ix] = [f'{lab}|{c}' for c in M._leiden_labels(X[ix], r)]
            cache[r] = v
        return cache[r]
    D = pd.DataFrame({'cell_id': ids, 'label': labels}); info = {}
    for tn, n in T.items():
        r, c = bisect(lambda r: len(set(part(r))), n); D[f'L_{tn}'] = part(r); info[tn] = dict(res=r, classes=c, target=n)
    D.to_parquet(f'{OUT}/grid_{pid}.parquet.tmp', index=False); os.rename(f'{OUT}/grid_{pid}.parquet.tmp', f'{OUT}/grid_{pid}.parquet'); json.dump(info, open(f'{OUT}/gridinfo_{pid}.json', 'w'))

def score(pid):
    P, df, tg, cells, out = target_n(pid); Gr = pd.read_parquet(f'{OUT}/grid_{pid}.parquet').set_index('cell_id').reindex(cells); info = json.load(open(f'{OUT}/gridinfo_{pid}.json')); rows = []
    assert Gr.notna().all().all(), f'{pid}: grid cell ids do not match the pool cells'
    for tn, (a1, n) in out.items():
        for an, cls in (('A1', a1), ('L_matched', Gr[f'L_{tn}']), ('A0', pd.Series(df.label.values, index=cells))):
            rows.append(dict(pid=pid, atlas=pid.split('_')[1], thr=tn, arm=an, target_classes=n, res=info[tn]['res'] if an == 'L_matched' else np.nan, **P.arm_metrics(df, cls, tg)))
    pd.DataFrame(rows).to_csv(f'{OUT}/score_{pid}.csv', index=False)

def signflip(d, B=20000, seed=0):
    """exact two-sided sign-flip p of the mean over units (all 2^n flips when n<=14, else Monte Carlo)."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d); obs = abs(d.mean())
    if n <= 14: S = np.array([[1 if (i >> k) & 1 else -1 for k in range(n)] for i in range(2 ** n)]); m = np.abs(S @ d / n)
    else: m = np.abs(np.random.default_rng(seed).choice([-1, 1], (B, n)) @ d / n)
    return float((m >= obs - 1e-12).mean())

def summary():
    import glob
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/score_*.csv'))]); R.to_csv(J + '/out/small/matched_pool.csv', index=False); rows = []
    nc = R.pivot_table(index=['pid', 'thr'], columns='arm', values='n_classes'); bad = nc[(nc.A1 - nc.L_matched).abs() > 0.05 * nc.A1].index   # per pool and threshold, class count mismatch above 5% of ours
    print(len(bad), 'of', len(nc), 'pool-thresholds excluded for class mismatch > 5%'); R = R[~R.set_index(['pid', 'thr']).index.isin(bad)]
    for tn, g in R.groupby('thr'):
        W = g.pivot_table(index=['pid', 'atlas'], columns='arm', values=['F1_bm', 'F1_oo', 'ARI', 'NMI', 'n_classes'])
        for m in ('F1_bm', 'F1_oo', 'ARI', 'NMI'):
            dp = (W[(m, 'A1')] - W[(m, 'L_matched')]).reset_index(name='d'); da = dp.groupby('atlas').d.mean()
            rows.append(dict(thr=tn, metric=m, n_pools=len(dp), ours=W[(m, 'A1')].mean(), leiden_matched=W[(m, 'L_matched')].mean(), broad=W[(m, 'A0')].mean(), classes_ours=W[('n_classes', 'A1')].mean(), classes_leiden=W[('n_classes', 'L_matched')].mean(),
                             diff_pool_mean=dp.d.mean(), diff_atlas_mean=da.mean(), n_atlases=len(da), p_atlas_signflip=signflip(da.values)))
    S = pd.DataFrame(rows); S.to_csv(J + '/out/small/matched_summary.csv', index=False); print(S.round(3).to_string())

def demo():
    r, c = bisect(lambda r: int(round(40 * r ** 0.7)), 17); assert abs(c - 17) <= 1 and 0.01 < r < 10, (r, c); r, c = bisect(lambda r: 5, 50); assert c == 5
    assert signflip([1, 1, 1, 1]) == 2 / 16 and signflip([0.0, 0.0]) == 1.0
    sys.path.insert(0, os.environ.get('SI_H', J + '/data/w6/ID/hmod')); import methods as M
    rng = np.random.default_rng(0); X = np.vstack([rng.normal(c, 1, (500, 10)) for c in (0, 6, 12)]); n = [len(set(M._leiden_labels(X, r))) for r in (0.05, 5.0)]
    assert n[1] >= n[0] >= 1, n; print('demo ok', n)

if __name__ == '__main__':
    {'targets': lambda: targets_cmd(sys.argv[2]), 'grid': lambda: grid(sys.argv[2]), 'score': lambda: score(sys.argv[2]), 'summary': summary, 'demo': demo}[sys.argv[1]]()
