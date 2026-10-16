"""Donor-block bootstrap of (duplication - no augmentation) on the evaluation side.
  python3 donor_boot.py run  --ds aida8|lupus|hlca [--clf lr[,knn,..]] [--B 1000] [--seeds 0:20] [--evdir D] [--out D]
  python3 donor_boot.py report --ds .. [--clf lr] [--out D]      (aggregates every per-seed file in --out)
  python3 donor_boot.py selfcheck
Per seed: fit 'real' (no augmentation) and 'upsample' (dup to 256 cells, stat16.gene_arm) with stat16.readout, keep per-cell
scores P and calls, resample the seed's EVAL DONORS with replacement (B draws, same draws for both arms and all types), recompute
each probe type's argmax recall and rec200 (k = true eval cells of the type in the resample; cells of a drawn donor count with
multiplicity), take d = upsample - real, average over probe types and seeds per draw. Report percentile 95% interval next to the
type-level exact sign-flip p (polh.signflip_p, as in the paper tables).
Env: REPDIR (rep16t npz), LUPUS_BASE / HLCA_BASE (m27_*_t dirs: s<seed>_base.npz, s<seed>_upsample.npz), STAT16_DIR, M27_SRC
(dir of m27_hlclf.py + polh.py). New outputs only: boot_df_<ds>_<clf>_s<seed>.npz, boot_df_<ds>_<clf>_summary.csv/_pertype.csv."""
import os, sys, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); E = os.path.expanduser
W = E('~/work/recomb27')
DEF = dict(stat16=W + '/rev1015/hvg1015/parent', src=W + '/monday_q/donorfix/src_mini')
FM = W + '/rev1015/hvg_c3/from_mini'
# logged references: list of csv files (rows without a clf column are LR). lupus: upsample from the 4-classifier reader (exact), real arm LR from m27_lupus_t.csv
REF = {'aida8': [W + '/rev1015/p1data/stat16_trhvg.csv'], 'lupus': [FM + '/m27_sd3clf_t.csv', FM + '/m27_lupus_t.csv'],
       'hlca': [FM + '/m27_hlclf_t.csv', FM + '/m27_hlreal_t.csv']}

def paths():
    sys.path[:0] = [os.environ.get('M27_SRC', DEF['src']), os.environ.get('STAT16_DIR', DEF['stat16'])]

# ---------- bootstrap core (pure numpy; selfchecked) ----------
def metrics_boot(*a):
    with np.errstate(all='ignore'):   # macOS Accelerate raises spurious matmul flags (see polh.py)
        return _metrics_boot(*a)

def _metrics_boot(P, pred, yev, c, don_idx, counts):
    """P: (n, NC) scores, pred: (n,), don_idx: (n,) donor index per cell, counts: (B, nd) donor multiplicities.
    Returns (point_default, point_rec200, boot_default (B,), boot_rec200 (B,)); NaN where the resample has no true-c cell."""
    t = (yev == c); s = P[:, c]; o = np.argsort(-s); n = len(t)
    nd = counts.shape[1]
    ntrue_d = np.bincount(don_idx[t], minlength=nd).astype(float)
    hit_d = np.bincount(don_idx[t & (pred == c)], minlength=nd).astype(float)
    k = counts @ ntrue_d; ok = k > 0
    bd = np.where(ok, (counts @ hit_d) / np.where(ok, k, 1), np.nan)
    M = counts[:, don_idx[o]]                       # multiplicity of every cell, in score order
    cm = np.cumsum(M, axis=1); h = np.cumsum(M * t[o][None, :], axis=1)
    j = np.argmax(cm >= k[:, None], axis=1)         # first position where the top-k is filled
    r = np.arange(len(k)); prev_c = np.where(j > 0, cm[r, np.maximum(j - 1, 0)], 0); prev_h = np.where(j > 0, h[r, np.maximum(j - 1, 0)], 0)
    hits = prev_h + (k - prev_c) * t[o][j]          # partial credit for the cell that straddles the cutoff
    br = np.where(ok, hits / np.where(ok, k, 1), np.nan)
    kk = int(t.sum()); pr = float(t[o[:kk]].mean()); pdft = float((pred[t] == c).mean())
    return pdft, pr, bd, br

def draw_counts(nd, B, seed):
    r = np.random.default_rng([seed, 7]); return np.stack([np.bincount(r.integers(0, nd, nd), minlength=nd) for _ in range(B)]).astype(np.float64)

def draw_union(donors_per_seed, B):
    """One donor-block bootstrap over the UNION of eval donors of all seeds: replicate b draws len(union) donors with replacement,
    and a donor that is in the eval set of several seeds carries the SAME weight in each (seeds share eval donors).
    -> dict seed -> counts (B, n_eval_donors_of_seed) aligned with np.unique(that seed's donor ids)."""
    U = np.unique(np.concatenate(list(donors_per_seed.values()))); r = np.random.default_rng([0, 11])
    C = np.stack([np.bincount(r.integers(0, len(U), len(U)), minlength=len(U)) for _ in range(B)]).astype(np.float64)
    return {s: C[:, np.searchsorted(U, np.unique(d))] for s, d in donors_per_seed.items()}

# ---------- per-seed fits ----------
def fit_scores(ds, seed):
    """-> dict arm -> (P, pred) per clf, plus Zev/yev/probe/cts. Same matrices as stat16.fit_arm / m27_hlclf.fit."""
    import stat16
    if ds == 'aida8':
        d = np.load(f"{E(os.environ.get('REPDIR', W + '/monday_q/donorfix/data'))}/rep16t_seed{seed}.npz")
        Z, y, Zev, yev, X, pm, Wc, probe, cts, n_c = (d[k] for k in ('Ztr', 'ytr', 'Zev', 'yev', 'Xtr', 'pmean', 'comp', 'probe', 'cts', 'n_c'))
        assert np.allclose((X - pm) @ Wc.T, Z, atol=1e-3), 'cache Xtr does not project to Ztr'
        A = stat16.gene_arm('upsample', X, y, probe, n_c, seed)
        Zu = np.vstack([Z] + [(v[0] - pm) @ Wc.T for v in A.values()]); yu = np.concatenate([y] + [v[1] for v in A.values()])
        wu = np.concatenate([np.ones(len(y))] + [v[2] for v in A.values()])
    else:
        os.environ['M27_HL'] = E(os.environ.get('LUPUS_BASE' if ds == 'lupus' else 'HLCA_BASE', W + '/rev1015/hvg_c3/from_mini/' + ('m27_mk_t' if ds == 'lupus' else 'm27_hl_t')))
        from m27_hlclf import _base
        b, Wc, pm, Z, y, Zev, yev = _base(seed); probe, cts = b['probe'], b['cts']
        a = np.load(f"{os.environ['M27_HL']}/s{seed}_upsample.npz")
        Zu = np.vstack([Z, (a['X'] - pm) @ Wc.T]); yu = np.concatenate([y, a['y']]); wu = np.ones(len(yu))
    return dict(real=(Z, y, np.ones(len(y))), upsample=(Zu, yu, wu)), Zev, yev, probe, cts

def run_seed(ds, clf, seed, B, evdir):
    paths(); import stat16
    ev = np.load(f'{evdir}/evcells_df_{ds}_s{seed}.npz'); assert 'provenance' not in ev.files, 'synthetic (fake donor) evcells file'
    arms, Zev, yev, probe, cts = fit_scores(ds, seed)
    assert np.array_equal(ev['yev'], yev), f'{ds} seed {seed}: evdon eval cells do not align with the cache'
    _, don_idx = np.unique(ev['ev_donor'], return_inverse=True)
    allf = {int(f.rsplit('_s', 1)[1][:-4]): f for f in glob.glob(f'{evdir}/evcells_df_{ds}_s*.npz') if 'FAKE' not in f}
    counts = draw_union({sd: np.load(f)['ev_donor'] for sd, f in allf.items()}, B)[seed]   # union over every seed's evcells file in evdir
    out = {}
    for arm, (Z, y, w) in arms.items():
        P, pred = stat16.readout(clf, Z, y, w, Zev, seed)
        if P.shape[1] < len(cts): Pf = np.full((len(Zev), len(cts)), -np.inf); Pf[:, np.unique(y)] = P; P = Pf
        out[arm] = [metrics_boot(P, pred, yev, int(c), don_idx, counts) for c in probe]
    return dict(seed=seed, probe=probe, cts=np.array(cts), nd=counts.shape[1],
                pt=np.array([[[o[0], o[1]] for o in out[a]] for a in ('real', 'upsample')]),     # (arm, type, metric[default, rec200])
                bt=np.array([[[o[2], o[3]] for o in out[a]] for a in ('real', 'upsample')]))     # (arm, type, metric, B)

# ---------- aggregation ----------
def report(ds, clf, out):
    import polh
    fs = sorted(glob.glob(f'{out}/boot_df_{ds}_{clf}_s*.npz'), key=lambda f: int(f.rsplit('_s', 1)[1][:-4]))
    R = [np.load(f, allow_pickle=True) for f in fs]; assert R, 'no per-seed files'
    names = ['recall_default', 'rec200']; rows, summ = [], []
    for i, nm in enumerate(names):
        d_pt = np.concatenate([r['pt'][1, :, i] - r['pt'][0, :, i] for r in R])
        grp = np.concatenate([r['cts'][r['probe']] for r in R])
        p, dt = polh.signflip_p(d_pt, grp)
        bd = np.concatenate([r['bt'][1, :, i, :] - r['bt'][0, :, i, :] for r in R], 0)     # (units, B)
        nanfrac = float(np.isnan(bd).mean()); bm = np.nanmean(bd, axis=0)
        lo, hi = np.nanpercentile(bm, [2.5, 97.5])
        # variance split: SE of the mean difference from donor resampling alone vs the naive across-unit SE
        summ.append(dict(ds=ds, clf=clf, metric=nm, n_seeds=len(R), n_units=len(d_pt), mean_diff=d_pt.mean(),
                         boot_lo=lo, boot_hi=hi, boot_se=float(np.std(bm)), unit_se=float(d_pt.std(ddof=1) / np.sqrt(len(d_pt))),
                         signflip_p=p, types_pos=f'{int((dt > 0).sum())}/{len(dt)}', frac_nan_draws=nanfrac,
                         mean_real=np.concatenate([r['pt'][0, :, i] for r in R]).mean(), mean_dup=np.concatenate([r['pt'][1, :, i] for r in R]).mean()))
        for r in R:
            for j, c in enumerate(r['probe']):
                rows.append(dict(ds=ds, clf=clf, metric=nm, seed=int(r['seed']), type=str(r['cts'][c]), real=r['pt'][0, j, i], dup=r['pt'][1, j, i]))
    for f in (f'{out}/boot_df_{ds}_{clf}_pertype.csv', f'{out}/boot_df_{ds}_{clf}_summary.csv'): assert not os.path.exists(f), f'{f} exists'
    pd.DataFrame(rows).to_csv(f'{out}/boot_df_{ds}_{clf}_pertype.csv', index=False)
    S = pd.DataFrame(summ); S.to_csv(f'{out}/boot_df_{ds}_{clf}_summary.csv', index=False); print(S.round(4).to_string(index=False))
    return S

def check_ref(ds, clf, out):
    """point estimates must reproduce the logged readouts (rec200 / argmax recall) of the original runs, tol 1e-9 for every dataset.
    An (arm, clf) pair without a logged row prints NO REFERENCE and is counted; it never passes silently."""
    P = pd.read_csv(f'{out}/boot_df_{ds}_{clf}_pertype.csv'); Rf = []
    for f in REF[ds]:
        x = pd.read_csv(f)
        if 'clf' not in x: x['clf'] = 'lr'
        Rf.append(x[x.arm.isin(['real', 'upsample']) & x.probe])
    Rf = pd.concat(Rf); Rf = Rf[Rf.clf == clf]
    key = {'recall_default': 'rec_argmax', 'rec200': 'rec200'}; worst, noref = 0.0, 0
    for m, col in key.items():
        for arm, a in (('real', 'real'), ('dup', 'upsample')):
            q = P[P.metric == m]; r = Rf[Rf.arm == a].drop_duplicates(['seed', 'name_']).set_index(['seed', 'name_'])[col]
            v = q.set_index(['seed', 'type'])[arm]; j = v.index.intersection(r.index)
            if not len(j): print(f'NO REFERENCE for {ds}/{clf}/{a}/{m}'); noref += 1; continue
            worst = max(worst, float((v[j] - r[j]).abs().max()))
    tol = 1e-9
    print(f'reference check {ds}/{clf}: max |point - logged| = {worst:.2e} (tol {tol}); pairs without reference: {noref}'); assert worst <= tol; return worst

def selfcheck():
    # 1) with all multiplicities 1 the point estimate and the bootstrap statistic equal the textbook rec200; 2) a toy with known answer
    r = np.random.default_rng(0); n, NC, nd = 600, 3, 12
    yev = np.repeat(np.arange(NC), n // NC); don = r.integers(0, nd, n); P = r.random((n, NC)) + (yev[:, None] == np.arange(NC)) * 0.6
    pred = P.argmax(1); ones = np.ones((1, nd))
    pd_, pr, bd, br = metrics_boot(P, pred, yev, 1, don, ones)
    t = yev == 1; o = np.argsort(-P[:, 1]); assert abs(pr - t[o[:t.sum()]].mean()) < 1e-12 and abs(br[0] - pr) < 1e-12
    assert abs(bd[0] - pd_) < 1e-12 and abs(pd_ - (pred[t] == 1).mean()) < 1e-12
    # doubled donors: every cell twice -> same recall; zero-weight donors can empty the type -> NaN
    _, _, bd2, br2 = metrics_boot(P, pred, yev, 1, don, 2 * ones); assert abs(br2[0] - pr) < 1e-12 and abs(bd2[0] - pd_) < 1e-12
    z = np.zeros((1, nd)); z[0, np.unique(don[~t])[:1]] = 1
    only_other = np.setdiff1d(np.unique(don[~t]), np.unique(don[t]))
    if len(only_other): z[:] = 0; z[0, only_other[0]] = 1; assert np.isnan(metrics_boot(P, pred, yev, 1, don, z)[3][0])
    # brute-force: explicit duplication of cells of drawn donors
    # union draw: a donor in two seeds' eval sets gets the same weight in both
    dps = {0: np.array(list('abcdef')), 1: np.array(list('dcxyab'))}; U2 = draw_union(dps, 7)
    assert U2[0].shape == (7, 6)
    for dn in 'abcd': assert (U2[0][:, np.searchsorted(np.unique(dps[0]), dn)] == U2[1][:, np.searchsorted(np.unique(dps[1]), dn)]).all()
    cnt = draw_counts(nd, 5, 3)
    _, _, bd3, br3 = metrics_boot(P, pred, yev, 1, don, cnt)
    for b in range(5):
        idx = np.repeat(np.arange(n), cnt[b][don].astype(int)); tt = t[idx]; k = int(tt.sum())
        if k == 0: assert np.isnan(br3[b]); continue
        # exact top-k with fractional boundary: break score ties never occur here, so plain top-k equals the multiset rule
        oo = np.argsort(-P[idx, 1], kind='stable'); assert abs(tt[oo[:k]].mean() - br3[b]) < 1e-9, (b, tt[oo[:k]].mean(), br3[b])
        assert abs((pred[idx][tt] == 1).mean() - bd3[b]) < 1e-12
    print('donor_boot selfcheck ok')

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'selfcheck': selfcheck(); sys.exit()
    a = dict(zip(sys.argv[2::2], sys.argv[3::2])); ds = a['--ds']; out = a.get('--out', HERE); evdir = a.get('--evdir', HERE)
    os.makedirs(out, exist_ok=True); paths()
    nseed = {'aida8': 20, 'lupus': 3, 'hlca': 3}[ds]; lo, hi = map(int, a.get('--seeds', f'0:{nseed}').split(':'))
    for clf in a.get('--clf', 'lr').split(','):
        if cmd == 'run':
            for s in range(lo, hi):
                import time; t0 = time.time(); r = run_seed(ds, clf, s, int(a.get('--B', 1000)), evdir)
                assert not os.path.exists(f'{out}/boot_df_{ds}_{clf}_s{s}.npz'), 'exists'; np.savez_compressed(f'{out}/boot_df_{ds}_{clf}_s{s}.npz', **r); print(f'{ds} {clf} seed {s} done {time.time()-t0:.0f}s', flush=True)
        else:
            report(ds, clf, out); check_ref(ds, clf, out)
