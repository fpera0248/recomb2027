"""Prior-shift control for duplication (declared in advance). statistical_arms and polh are imported unchanged.
  REP=<path>/rep16t STAT16_DIR=<dir with statistical_arms.py,polh.py> CLF=lr,knn,rf,mlp python3 priorshift.py run S0 S1 [OUT.csv]
  python3 priorshift.py report OUT.csv           (share, bootstrap CI, signflip, verdict per classifier)
  REP=... python3 priorshift.py selfcheck [seed]  (parity with statistical_arms.fit_arm + shift invariants, LR)
Env: POLH_DIR (dir of polh.py, default STAT16_DIR), NJ (joblib workers over seeds, default 1). Output: new csv only, never an existing file."""
import sys, os
sys.path.insert(0, os.environ.get('STAT16_DIR', '.'))
import numpy as np, pandas as pd
import statistical_arms as S
from sklearn.neighbors import KNeighborsClassifier

def shift_probs(P, ratio):
    """Bayes prior correction: multiply class columns by ratio (len = n classes), renormalize rows."""
    Q = P * ratio[None, :]
    return Q / Q.sum(1, keepdims=True)

def fit_both(seed, clfs):
    """Return per-unit rows for real / upsample / shift on the cached split."""
    d = np.load(f'{S.REP}_seed{seed}.npz'); Z, y, Zev, yev = d['Ztr'], d['ytr'], d['Zev'], d['yev']
    X, pm, W = d['Xtr'], d['pmean'], d['comp']
    assert np.allclose((X - pm) @ W.T, Z, atol=1e-3)
    probe, cts, n_c = d['probe'], d['cts'], d['n_c']; NC = len(cts)
    aug = S.gene_arm('upsample', X, y, probe, n_c, seed)           # {c: (Xa, ya, wa)}, only classes with k = 256 - n_c > 0
    dupc = sorted(aug)
    ratio = np.ones(NC)
    for c in dupc:
        k = S.TARGET - int(n_c[c]); assert k > 0 and len(aug[c][1]) == k
        ratio[c] = S.TARGET / int(n_c[c])                          # target count / n_c
    Zs, ys, ws = [Z], [y], [np.ones(len(y))]
    for c in dupc:
        Xa, ya, wa = aug[c]; Zs.append((Xa - pm) @ W.T); ys.append(ya); ws.append(wa)
    Zu, yu, wu = np.vstack(Zs), np.concatenate(ys), np.concatenate(ws)
    w1 = np.ones(len(y)); rows = []
    for k in clfs:
        Pr, pr = S.readout(k, Z, y, w1, Zev, seed)
        Pu, pu = S.readout(k, Zu, yu, wu, Zev, seed)
        if k == 'knn':   # vote fractions behind readout's predict()
            V = KNeighborsClassifier(n_neighbors=S.KNN).fit(Z, y).predict_proba(Zev); assert np.array_equal(np.unique(y)[V.argmax(1)], pr)
        else: V = Pr
        assert V.shape[1] == NC
        psh = np.unique(y)[shift_probs(V, ratio).argmax(1)]
        for c in dupc:
            true = yev == c
            def row(arm, P, pred):
                called = int((pred == c).sum()); tp = int(((pred == c) & true).sum())
                return dict(seed=seed, clf=k, arm=arm, c=c, name_=cts[c], n=int(n_c[c]), ratio=ratio[c], n_true=int(true.sum()),
                            rec_argmax=float((pred[true] == c).mean()), rec200=float(true[np.argsort(-P[:, c])[:S.NEV]].mean()),
                            called=called, tp=tp, precision=tp / called if called else np.nan)
            rr = row('real', Pr, pr); rows += [rr, row('upsample', Pu, pu), row('shift', Pr * ratio[None, :] if k != 'knn' else Pr, psh)]
            assert rows[-1]['rec200'] == rr['rec200'], ('shift moved rec200', seed, k, c)
    return rows

def run(a, b, out):
    assert not os.path.exists(out), f'{out} exists'
    clfs = os.environ.get('CLF', 'lr,knn,rf,mlp').split(',')
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=int(os.environ.get('NJ', 1)))(delayed(fit_both)(s, clfs) for s in range(a, b))
    df = pd.DataFrame([r for x in res for r in x]); df.insert(0, 'cache', os.path.basename(S.REP)); df.to_csv(out, index=False); print('wrote', out, len(df))

def report(path, B=2000):
    sys.path.insert(0, os.environ.get('POLH_DIR', os.environ.get('STAT16_DIR', '.'))); import polh
    df = pd.read_csv(path); rng = np.random.default_rng(0)
    for cache, dc in df.groupby('cache'):
        print(f'\n## {cache}: {dc.seed.nunique()} seeds, {dc.name_.nunique()} types, {len(dc)//3//dc.clf.nunique()} units per classifier')
        verd = {}
        for k, d in dc.groupby('clf', sort=False):
            W = d.pivot_table(index=['seed', 'c', 'name_'], columns='arm', values='rec_argmax'); nm = W.index.get_level_values('name_').values
            R2 = d.pivot_table(index=['seed', 'c', 'name_'], columns='arm', values='rec200')
            gain, num = (W.upsample - W.real).values, (W['shift'] - W.real).values
            share = lambda g, n: n.mean() / g.mean()
            types = np.unique(nm); idx = {t: np.where(nm == t)[0] for t in types}; bs = []
            for _ in range(B):
                ii = np.concatenate([idx[t] for t in rng.choice(types, len(types))]); bs.append(share(gain[ii], num[ii]) if gain[ii].mean() > 0 else np.nan)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            p_us, _ = polh.signflip_p((W.upsample - W['shift']).values, nm); p_sr, _ = polh.signflip_p(num, nm); p_ur, _ = polh.signflip_p(gain, nm)
            g = gain.mean(); sh = share(gain, num) if g >= 0.02 else np.nan
            v = 'undefined(no gain)' if np.isnan(sh) else 'SUPPORTED' if 0.8 <= sh <= 1.2 and p_us >= 0.05 else 'NOT' if sh < 0.5 or sh > 1.5 else 'PARTIAL'
            ov = 'OVERSHOOT (prior shift alone exceeds duplication gain)' if sh > 1.5 else ''
            amend = 'SUPPORTED(CI in [0.8,1.2])' if (not np.isnan(sh) and 0.8 <= lo and hi <= 1.2) else ov or 'not-supported(CI)'
            verd[k] = v
            print(f"     DECL label {v} | amended (DECL_priorshift_amend.md) {amend}")
            C = d.pivot_table(index=['seed', 'c'], columns='arm', values='called'); P = d.pivot_table(index=['seed', 'c'], columns='arm', values='precision')
            print(f"{k:4s} real {W.real.mean():.3f} up {W.upsample.mean():.3f} shift {W['shift'].mean():.3f} | up-real {g:+.3f} shift-real {num.mean():+.3f} "
                  f"SHARE {sh:.2f} [{lo:.2f},{hi:.2f}] | p(up-shift) {p_us:.3f} p(shift-real) {p_sr:.3f} p(up-real) {p_ur:.3f} | {v}\n"
                  f"     rec200 real {R2.real.mean():.3f} up {R2.upsample.mean():.3f} (up-real {(R2.upsample-R2.real).mean():+.3f}; shift==real) | "
                  f"calls/type real {C.real.mean():.1f} up {C.upsample.mean():.1f} shift {C['shift'].mean():.1f} | precision real {P.real.mean():.3f} up {P.upsample.mean():.3f} shift {P['shift'].mean():.3f} "
                  f"(0-call units real/up/shift {int((C.real==0).sum())}/{int((C.upsample==0).sum())}/{int((C['shift']==0).sum())})")
        vs = list(verd.values()); n_s, n_n = vs.count('SUPPORTED'), vs.count('NOT')
        print('OVERALL:', 'prior shift (>=3 SUPPORTED, none NOT)' if n_s >= 3 and n_n == 0 else 'mostly not a prior shift' if n_n >= 3 else 'mixed', verd)

def selfcheck(seed):
    # synthetic: shift_probs keeps simplex, boosts only the ratio'd column
    P = np.array([[.5, .5, 0.], [.1, .1, .8]]); Q = shift_probs(P, np.array([1., 4., 1.]))
    assert np.allclose(Q.sum(1), 1) and Q[0].argmax() == 1 and Q[1].argmax() == 2
    CL = os.environ.get('CLF', 'lr,knn,rf,mlp').split(',')
    os.environ['CLF'] = ','.join(CL)   # statistical_arms.fit_arm reads CLF at call time
    rows = pd.DataFrame(fit_both(seed, CL)); ref, _ = S.fit_arm(seed, 'real'); refu, _ = S.fit_arm(seed, 'upsample')
    for arm, rf in (('real', ref), ('upsample', refu)):
        R = pd.DataFrame(rf)
        for k in CL:   # parity for EVERY classifier in CLF (default all four)
            Rk = R[R.clf == k] if 'clf' in R else R
            m = rows[(rows.arm == arm) & (rows.clf == k)].merge(Rk, on='c', suffixes=('', '_s'))
            assert len(m) and (m.rec_argmax == m.rec_argmax_s).all() and (m.rec200 == m.rec200_s).all() and (m.called == m.called_s).all(), (arm, k)
    for k in CL:
        sh = rows[(rows.arm == 'shift') & (rows.clf == k)]; re = rows[(rows.arm == 'real') & (rows.clf == k)]
        assert (sh.rec200.values == re.rec200.values).all() and (sh.called.values >= 0).all(), k
    print('selfcheck OK: parity with stat16.fit_arm (real/upsample) for', CL, '; shift rec200 == real, knn vote argmax == pred;', len(rows), 'rows')

def donors(src, out):
    """Per-rung donor table from m27_donor.csv (AIDA rep16, 20 seeds x 8 types): median donors, mean top-donor share, mean total-variation
    distance of ancestry shares (copied cells vs eval, copied vs pool, pool vs eval). New csv only."""
    assert not os.path.exists(out), f'{out} exists'
    u = pd.read_csv(src); key = ['seed', 'type', 'rung']
    U = u.drop_duplicates(key)[key + ['n_donors', 'top_donor_share']].set_index(key)
    for nm, a, b in (('tv_src_eval', 'src_share', 'eval_share'), ('tv_src_pool', 'src_share', 'pool_share'), ('tv_pool_eval', 'pool_share', 'eval_share')):
        U[nm] = (u.assign(d=(u[a] - u[b]).abs()).groupby(key).d.sum() / 2)
    T = U.reset_index().groupby('rung').agg(units=('seed', 'size'), donors_median=('n_donors', 'median'), donors_min=('n_donors', 'min'), donors_max=('n_donors', 'max'),
                                          top_donor_share=('top_donor_share', 'mean'), tv_src_eval=('tv_src_eval', 'mean'), tv_src_pool=('tv_src_pool', 'mean'), tv_pool_eval=('tv_pool_eval', 'mean'))
    T.round(3).to_csv(out); print(T.round(3).to_string()); return T

if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'run': run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else 'priorshift_df.csv')
    elif c == 'report': report(sys.argv[2])
    elif c == 'donors': donors(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'donor_table_df.csv')
    elif c == 'selfcheck': selfcheck(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
