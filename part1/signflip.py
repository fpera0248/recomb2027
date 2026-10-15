"""Sign-flip test over cell types."""
import itertools
import numpy as np

def fit(Y, F, fixed_only=False):
    m = len(Y); tot = Y.sum(0)
    best = (tot.max() / m, ('fixed', int(tot.argmax())))
    if fixed_only: return best[1]
    for f in range(F.shape[1]):
        v = np.unique(F[:, f])
        for t in (v[:-1] + v[1:]) / 2:
            lo = F[:, f] <= t
            s_lo, s_hi = Y[lo].sum(0), Y[~lo].sum(0)
            sc = (s_lo.max() + s_hi.max()) / m
            if sc > best[0] + 1e-12:
                best = (sc, ('thr', f, float(t), int(s_lo.argmax()), int(s_hi.argmax())))
    return best[1]

def apply(rule, F):
    if rule[0] == 'fixed': return np.full(len(F), rule[1])
    _, f, t, lo, hi = rule
    return np.where(F[:, f] <= t, lo, hi)

def cv_pick(Y, F, groups, fixed_only=False):
    pick = np.empty(len(Y), int); rules = {}
    for g in np.unique(groups):
        te = groups == g
        rules[g] = fit(Y[~te], F[~te], fixed_only)
        pick[te] = apply(rules[g], F[te])
    return pick, rules

def gain(Y, pick, base=0):
    """Mean over units of Y[chosen] - Y[base]."""
    return float((Y[np.arange(len(Y)), pick] - Y[:, base]).mean())

def signflip_p(d, groups):
    """Exact two-sided sign-flip p over cell types, on per-type mean differences."""
    dt = np.array([d[groups == g].mean() for g in np.unique(groups)])
    obs = abs(dt.mean())
    if len(dt) > 16:   # 2^n sign vectors is too many: Monte-Carlo (200k draws, fixed seed); p resolution 5e-6
        S = np.random.default_rng(0).choice((-1.0, 1.0), size=(200_000, len(dt)))
        with np.errstate(all="ignore"): T = np.abs(S @ dt) / len(dt)   # macOS Accelerate raises spurious matmul flags
        return float((1 + np.sum(T >= obs - 1e-12)) / (1 + len(S))), dt
    null = [abs((dt * s).mean()) for s in itertools.product((-1, 1), repeat=len(dt))]
    return float(np.mean(np.array(null) >= obs - 1e-12)), dt

def perm_test(Y, F, groups, B=2000, strata=None, seed=0):
    """T = CV gain(adaptive) - CV gain(fixed method)."""
    rng = np.random.default_rng(seed)
    fx = gain(Y, cv_pick(Y, F, groups, True)[0])
    def T(Fp): return gain(Y, cv_pick(Y, Fp, groups)[0]) - fx
    t_obs = T(F); null = np.empty(B)
    for b in range(B):
        if strata is None: idx = rng.permutation(len(F))
        else:
            idx = np.arange(len(F))
            for s in np.unique(strata):
                w = np.where(strata == s)[0]; idx[w] = rng.permutation(w)
        null[b] = T(F[idx])
    return t_obs, float((1 + (null >= t_obs - 1e-12).sum()) / (1 + B)), null

if __name__ == '__main__':
    # self-check: method 1 wins by 0.1 iff feature 0 > 0; feature 1 is noise.
    r = np.random.default_rng(1); n, G = 48, 8
    groups = np.repeat(np.arange(G), n // G); F = r.normal(size=(n, 2))
    Y = r.normal(0, 0.01, size=(n, 3)); Y[:, 1] += np.where(F[:, 0] > 0, 0.1, -0.1)
    t, p, _ = perm_test(Y, F, groups, B=200)
    assert t > 0.03 and p < 0.05, (t, p)
    t0, p0, _ = perm_test(Y, F[:, [1]], groups, B=200)
    assert p0 > 0.05, (t0, p0)
    assert fit(Y, F)[:2] == ('thr', 0)
    print(f'ok: signal T={t:.3f} p={p:.3f}; noise T={t0:.3f} p={p0:.3f}')
