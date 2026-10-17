import itertools, glob, os, sys
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
D = os.path.dirname(os.path.abspath(__file__)) + '/data/part1_results/'
J21 = HR_WORK + '/recomb27/monday_q/donorfix/build/ovr_t/ref_j21/'
M27 = HR_WORK + '/recomb27/rev1015/p1data/'


def sfp(d, g):
    u = np.unique(g); dt = np.array([d[g == x].mean() for x in u]); obs = abs(dt.mean())
    if len(dt) > 16:
        S = np.random.default_rng(0).choice((-1.0, 1.0), size=(200_000, len(dt)))
        with np.errstate(all='ignore'):
            T = np.abs(S @ dt) / len(dt)
        return float((1 + np.sum(T >= obs - 1e-12)) / (1 + len(S))), dt
    null = np.array([abs((dt * np.array(s)).mean()) for s in itertools.product((-1, 1), repeat=len(dt))])
    return float(np.mean(null >= obs - 1e-12)), dt


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def add_fpr(p, nev, called):
    neg = p.N - nev
    p['fm'] = nev * (1 - p.rec200) / neg
    p['fd'] = (called - p.rec_argmax * nev) / neg
    p['td'], p['tm'] = p.rec_argmax, p.rec200
    return p


def typeunit(p, keys, unit, base='upsample', mets=('fm', 'fd', 'td', 'tm')):
    out = []
    for clf, g in p.groupby('clf'):
        b = g[g.arm == base].set_index(keys)
        for arm, h in g.groupby('arm'):
            h = h.set_index(keys); r = dict(clf=clf, arm=arm, n=len(h))
            for m in mets: r[m] = h[m].mean()
            if arm != base:
                for m in mets:
                    dd = (h[m] - b[m]).dropna(); r['d_' + m] = dd.mean()
                    if m in ('fm', 'tm'):
                        u = dd.index.get_level_values(unit).values if not isinstance(unit, list) else \
                            np.array(['|'.join(map(str, t)) for t in zip(*[dd.index.get_level_values(x) for x in unit])])
                        r['p_' + m] = sfp(dd.values, u)[0]
                r['npair'] = len(dd)
            out.append(r)
    S = pd.DataFrame(out); t = S.arm != base
    for m in ('fm', 'tm'): S.loc[t, 'q_' + m] = bh(S.loc[t, 'p_' + m])
    return S


part = sys.argv[1] if len(sys.argv) > 1 else 'all'
# ---------- A. probe31 ----------
if part in ('all', 'A'):
    d = pd.read_csv(D + 'probe31_runs.csv')
    d['N'] = d.groupby(['seed', 'arm', 'clf']).called.transform('sum')
    print('N const over arm within seed,clf:', d.groupby(['seed', 'clf']).N.nunique().max(), 'N range', d.N.min(), d.N.max())
    print('dup rows', d.duplicated(['seed', 'arm', 'clf', 'name_']).sum())
    p = d[d.probe].copy()
    print('probe rows/arm,clf', p.groupby(['arm', 'clf']).size().unique(), 'types', p.name_.nunique())
    for col in ('rec_argmax', 'rec200'):
        print(col, '*200 integer:', np.allclose(p[col] * 200, (p[col] * 200).round()))
    p = add_fpr(p, 200, p.called)
    print('fd min', p.fd.min())
    S = typeunit(p, ['seed', 'name_'], 'name_')
    PRIM = ['jitter_s0.5', 'jitter_s1.0', 'jitternz_s1.0', 'smote_k5', 'real']
    pm = S.arm.isin(PRIM)
    S.loc[pm, 'q20_fm'] = bh(S.loc[pm, 'p_fm']); S.loc[pm, 'q20_tm'] = bh(S.loc[pm, 'p_tm'])
    print(S.round(5).to_string())
    print('stat max |d_fm| non-real', S[(S.arm != 'upsample') & (S.arm != 'real')].d_fm.abs().max(),
          'incl real', S[S.arm != 'upsample'].d_fm.abs().max())
    S.to_csv('chkA.csv', index=False)

# ---------- B. panel ----------
if part in ('all', 'B'):
    L = []
    for f in sorted(glob.glob(D + 'panel_runs/*/*.csv')):
        x = pd.read_csv(f); x['ds'] = f.split('/')[-2]; x['src'] = f.split('/')[-1]; L.append(x)
    q = pd.concat(L, ignore_index=True)
    print('datasets', q.ds.nunique(), 'arms', sorted(q.arm.unique()), 'clfs', sorted(q.clf.unique()))
    print('arms in real.csv', q[q.src == 'real.csv'].arm.unique(), 'arms in clf.csv', q[q.src == 'clf.csv'].arm.unique())
    print('dup rows', q.duplicated(['ds', 'seed', 'arm', 'clf', 'name_']).sum())
    print('c<->name_ per ds', q.groupby(['ds', 'c']).name_.nunique().max(), q.groupby(['ds', 'name_']).c.nunique().max())
    q['N'] = q.groupby(['ds', 'seed', 'arm', 'clf']).nev_used.transform('sum')
    q['NC'] = q.groupby(['ds', 'seed', 'arm', 'clf']).ncall.transform(lambda s: s.sum(min_count=1))
    hc = q.ncall.notna()
    print('ds with ncall', q[hc].ds.nunique(), 'N==sum(ncall) where ncall:', (q[hc].N == q[hc].NC).all(),
          'groups with partial ncall', q.groupby(['ds', 'seed', 'arm', 'clf']).ncall.apply(lambda s: s.isna().any() and s.notna().any()).sum())
    print('N const over arm within ds,seed,clf:', q.groupby(['ds', 'seed', 'clf']).N.nunique().max())
    print('nclass const over arm:', q.groupby(['ds', 'seed', 'clf', 'arm']).size().groupby(['ds', 'seed', 'clf']).nunique().max())
    p = q[q.probe].copy()
    print('probe nev_used values', p.nev_used.value_counts().to_dict(), 'probe types', p.groupby('ds').name_.nunique().sum())
    p = add_fpr(p, p.nev_used, p.ncall)
    # builder-style pooled type unit
    p['ct'] = p.ds + ':' + p.name_.astype(str)
    S = typeunit(p, ['ds', 'seed', 'name_'], ['ds', 'name_'])
    # paper-style dataset unit for fm and tm
    rows = []
    for (clf, arm), g in p.groupby(['clf', 'arm']):
        if arm == 'upsample': continue
        b = p[(p.clf == clf) & (p.arm == 'upsample')].set_index(['ds', 'seed', 'name_'])
        h = g.set_index(['ds', 'seed', 'name_'])
        r = dict(clf=clf, arm=arm)
        for m in ('fm', 'tm', 'fd', 'td'):
            dd = (h[m] - b[m]).dropna()
            per = dd.groupby(level=['ds', 'name_']).mean().groupby(level='ds').mean()
            r['dsd_' + m] = per.mean(); r['nds_' + m] = len(per)
            if m in ('fm', 'tm'): r['pds_' + m] = sfp(per.values, np.arange(len(per)))[0]
        rows.append(r)
    R = pd.DataFrame(rows)
    for m in ('fm', 'tm'): R['qds_' + m] = bh(R['pds_' + m])
    S = S.merge(R, on=['clf', 'arm'], how='left')
    print(S.round(5).to_string())
    S.to_csv('chkB.csv', index=False)

# ---------- C. J21 + reconciliation with m27 ----------
if part in ('all', 'C'):
    out = []
    for ds in ('aida', 'hlca', 'lupus'):
        x = pd.read_csv(J21 + f'{ds}_full.csv')
        x = x[(x['mode'] == 'multiclass')]
        # recompute fpr from counts
        for k in ('default', 'matched'):
            f = x[f'{k}_FP'] / (x[f'{k}_FP'] + x[f'{k}_TN'])
            print(ds, k, 'fpr col vs counts max abs', np.nanmax(np.abs(f - x[f'{k}_fpr'])))
        x['N'] = x.groupby(['seed', 'arm', 'clf']).nev_used.transform('sum')
        pr = x[x.probe & (x.n >= 2)].copy()
        print(ds, 'matched_TP+FP==nev on probe', ((pr.matched_TP + pr.matched_FP) == pr.nev_used).all(),
              'TP+FN==nev', ((pr.default_TP + pr.default_FN) == pr.nev_used).all(),
              'FP+TN==N-nev', ((pr.default_FP + pr.default_TN) == pr.N - pr.nev_used).all())
        if ds == 'lupus': pr = pr[pr.arm != 'sdiff']
        pr = pr.rename(columns={'matched_fpr': 'fm', 'default_fpr': 'fd', 'matched_tpr': 'tm', 'default_tpr': 'td'})
        S = typeunit(pr, ['seed', 'name_'], 'name_'); S.insert(0, 'ds', ds); out.append(S)
        # reconcile with m27 (lr only, paper rows)
        m = pd.read_csv(M27 + {'aida': 'm27_base.csv', 'hlca': 'm27_hlca.csv', 'lupus': 'm27_lupus.csv'}[ds])
        if 'nev_used' not in m: m['nev_used'] = 200
        m['N'] = m.groupby(['seed', 'arm']).nev_used.transform('sum') if ds != 'aida' else m.groupby(['seed', 'arm']).called.transform('sum')
        if ds != 'aida':
            m['NC'] = m.groupby(['seed', 'arm']).called.transform('sum'); print(ds, 'm27 N==sum(called)', (m.N == m.NC).all())
        mp = m[m.probe & (m.n >= 2)].copy()
        mp['fm27'] = mp.nev_used * (1 - mp.rec200) / (mp.N - mp.nev_used)
        mp['fd27'] = (mp.called - mp.rec_argmax * mp.nev_used) / (mp.N - mp.nev_used)
        jj = pr[pr.clf == 'lr'][['seed', 'arm', 'name_', 'fm', 'fd', 'tm', 'td']]
        mm = mp.merge(jj, on=['seed', 'arm', 'name_'], how='inner')
        print(ds, 'm27 rows', len(mp), 'matched', len(mm), 'arms m27', sorted(mp.arm.unique()))
        print(ds, 'reconcile |fm27-fm| mean %.5f max %.5f identical %.2f ; rec200 vs tm mean %.4f' % (
            (mm.fm27 - mm.fm).abs().mean(), (mm.fm27 - mm.fm).abs().max(), ((mm.fm27 - mm.fm).abs() < 1e-9).mean(), (mm.rec200 - mm.tm).abs().mean()))
        # per-arm mean diff vs upsample, m27 vs J21, lr
        for src, col in (('m27', 'fm27'), ('j21', 'fm')):
            g = mm.pivot_table(index=['seed', 'name_'], columns='arm', values=col)
            dd = {a: (g[a] - g['upsample']).mean() * 100 for a in g.columns if a != 'upsample'}
            print(ds, src, 'lr d_fm pp', {a: round(v, 3) for a, v in dd.items()})
        # tests on m27 rows directly (paper's source), lr
        rows = []
        g = mp.pivot_table(index=['seed', 'name_'], columns='arm', values='fm27')
        for a in g.columns:
            if a == 'upsample': continue
            dd = (g[a] - g['upsample']).dropna()
            pv = sfp(dd.values, dd.index.get_level_values('name_').values)[0]
            rows.append(dict(ds=ds, arm=a, d_pp=dd.mean() * 100, p=pv, nt=dd.index.get_level_values('name_').nunique()))
        T = pd.DataFrame(rows); T['q_within_lr'] = bh(T.p); print(T.round(4).to_string())
        print(ds, 'm27 dup fm mean', mp[mp.arm == 'upsample'].fm27.mean())
    SJ = pd.concat(out); print(SJ.round(5).to_string()); SJ.to_csv('chkC.csv', index=False)
