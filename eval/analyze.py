"""Pool-level tests and tables from the per-pool metrics of score_atl.

  python -m eval.analyze run --scores S.csv [S.csv ...] --manifest M.csv [M.csv ...] --atlases atlases.csv [--drop leak_drop.txt]
        [--split test] [--flags flags.csv] [--gate gate.csv] [--tiefair tf.csv] [--targets targets_a.csv ...] [--out PREFIX]
        [--ours hiddenrare] [--leiden leiden_sub] [--sccad sccad_lab] [--learned scorer]
  python -m eval.analyze scan PRED_ROOT OUT.csv            # score files with NaN or a single unique value
  python -m eval.analyze tiefair PRED_ROOT OUT.csv PILOT_DIR MANIFEST.csv [...]   # tie-fair union AP per pool
  python -m eval.analyze gateprep PRED_ROOT GROUP_ROOT OUT.csv PILOT_DIR MANIFEST.csv [...] [--twin-map twins.csv]

--exclude ATLAS (default 0bf30b02) adds a tumour stratum without that atlas, as in the pre-registered analysis.

Primary test: union_AP, our method minus a comparator, mean over atlases of the pool means, exact two-sided sign-flip
test over atlases. Outputs PREFIX.json and PREFIX.md.
"""
import sys, os, json, glob, itertools, argparse, numpy as np, pandas as pd

from .score_atl import truth_table

OURS, LEIDEN, SCCAD, LEARNED = "hiddenrare", "leiden_sub", "sccad_lab", "scorer"
POOLM = ["union_AP", "E1_union"]
BASELINES = {"leiden_sub", "iforest_pca"}
NA = float("nan")


def exact(x):
    """p = #{s in {+1,-1}^A : |sum s_a d_a| >= |sum d_a| - 1e-12} / 2^A. Inclusive, two-sided, all patterns."""
    x = np.asarray(x, float)
    if not len(x): return NA
    s = np.array(list(itertools.product([1, -1], repeat=len(x))))
    return float((np.abs((s * x).sum(1)) >= abs(x.sum()) - 1e-12).mean())


def boot(a, B=2000, seed=5):
    """atlas-block bootstrap: resample atlases with replacement, percentile 2.5/97.5 of the mean; RandomState(seed) fresh per interval."""
    a = np.asarray(a, float)
    if not len(a): return [NA, NA]
    m = a[np.random.RandomState(seed).randint(0, len(a), (B, len(a)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def tiefair_ap(s, u):
    """expected average precision when ties are broken uniformly at random (exact: hypergeometric within each tied block)."""
    s, u = np.asarray(s, float), np.asarray(u, bool); K = int(u.sum())
    if not K: return NA
    uniq, inv = np.unique(-s, return_inverse=True); b = np.bincount(inv).astype(float); p = np.bincount(inv, weights=u)
    C = np.cumsum(b) - b; c = np.cumsum(p) - p; one = (b == 1) & (p > 0)
    tot = ((c[one] + 1) / (C[one] + 1)).sum()
    for i in np.nonzero((b > 1) & (p > 0))[0]:
        j = np.arange(1, int(b[i]) + 1); tot += p[i] * np.mean((c[i] + 1 + (j - 1) * (p[i] - 1) / (b[i] - 1)) / (C[i] + j))
    return float(tot / K)


def load(args):
    man = pd.concat([pd.read_csv(f) for f in args.manifest]).drop_duplicates("pilot_id")
    if args.split: man = man[man.split == args.split]
    drop = set(l.strip() for f in (args.drop or []) for l in open(f) if l.strip())
    man = man[~man.pilot_id.isin(drop)].set_index("pilot_id")
    if "n_cells" not in man: man["n_cells"] = 20000
    d = pd.concat([pd.read_csv(f) for f in args.scores]).drop_duplicates(["pilot_id", "method", "metric", "label", "target"])
    return man, drop, d[d.pilot_id.isin(man.index)]


def tumour_flags(atlases, args):
    st = pd.read_csv(args.atlases, dtype=str).set_index("name").stratum
    return {a: st[str(a)].lower().startswith("tum") for a in atlases}


def build(d, man, flags, ours, tgt=None):
    """pool x method tables per metric. Pool value = mean over the pool's rows (target metrics: mean over targets). A missing score for our
    method, and any NaN or constant score file in flags, count as chance (prevalence) on the pool metrics; other missing pools stay missing."""
    P = d.groupby(["metric", "pilot_id", "method"]).rprec.mean().reset_index(); W = {m: g.pivot(index="pilot_id", columns="method", values="rprec") for m, g in P.groupby("metric")}
    K = d[d.metric == "union_AP"].groupby("pilot_id").n_rare.max()
    if tgt is not None:      # pool universe = manifest pools with >= 1 target (targets_<name>.csv), so a pool no method scored stays in T1/T2 at chance for our method
        Kt = tgt.groupby("pilot_id").n.sum()                  # targets are disjoint (L,T) sets, so the sum of n is the union size
        pools = sorted(set(man.index) & set(Kt.index)); K = K.reindex(pools).fillna(Kt.reindex(pools))
    else: pools = sorted(set(d.pilot_id)); K = K.reindex(pools)
    prev = (K / man.n_cells.reindex(pools)).rename("prev")
    for m in POOLM:
        w = W[m].reindex(pools)
        for _, r in flags.iterrows():
            if r.pilot_id in w.index:
                if r.method not in w: w[r.method] = NA
                w.loc[r.pilot_id, r.method] = prev[r.pilot_id]
        if ours not in w: w[ours] = NA
        miss = w.index[w[ours].isna()]; w.loc[miss, ours] = prev[miss]
        W[m] = w
    miss = [p for p in pools if p not in set(d[d.method == ours].pilot_id)]
    return W, prev, miss

def compare(w, a, b, atlas, pools=None, drop_atlas=None):
    s = w[[a, b]].dropna() if a in w and b in w else pd.DataFrame(columns=[a, b])
    if pools is not None: s = s.loc[s.index.intersection(pools)]
    if drop_atlas is not None: s = s[atlas.loc[s.index].values != drop_atlas]
    if not len(s): return dict(a=a, b=b, n_pools=0, n_atlases=0, diff=NA, p=NA, atlas_ahead=0, a_mean=NA, b_mean=NA, pool_weighted_diff=NA, **{"pass": False})
    at = atlas.loc[s.index].values; A = (s[a] - s[b]).groupby(at).mean()
    r = dict(a=a, b=b, n_pools=len(s), n_atlases=len(A), a_mean=float(s[a].groupby(at).mean().mean()), b_mean=float(s[b].groupby(at).mean().mean()),
             diff=float(A.mean()), p=exact(A.values), atlas_ahead=int((A > 0).sum()), pool_weighted_diff=float((s[a] - s[b]).mean()))
    r["pass"] = bool(r["diff"] > 0 and r["p"] < 0.05); r["per_atlas_diff"] = {k: float(v) for k, v in A.items()}
    return r


def stratum(w, a, b, atlas, atl_set):
    r = compare(w, a, b, atlas, pools=[p for p in w.index if atlas[p] in atl_set])
    A = np.array(list(r.get("per_atlas_diff", {}).values())); r["boot95"] = boot(A); r.pop("pass", None); return r


def gate_stats(g, atlas, ours, leiden):
    g = g[g.pilot_id.isin(atlas.index)].copy(); g["atl"] = g.pilot_id.map(atlas); g["fire"] = g.max_small >= 10
    tw, re = g[g.kind == "twin"], g[g.kind == "real"].copy(); re["hit"] = re.n_targets >= 10; re["prec"] = re.n_targets / re.n_called
    aeq = lambda df, col: df.groupby(["method", "atl"])[col].mean().groupby("method").mean()
    FC, SE, PR = (aeq(tw, "fire"), aeq(re, "hit"), aeq(re, "prec")) if len(g) else (pd.Series(dtype=float),) * 3
    res = dict(FC={k: float(v) for k, v in FC.items()}, SE={k: float(v) for k, v in SE.items()}, call_precision={k: float(v) for k, v in PR.items()},
               n_twins={k: int(v) for k, v in tw.groupby("method").size().items()}, n_real={k: int(v) for k, v in re.groupby("method").size().items()})
    fa = tw.groupby(["method", "atl"]).fire.mean().unstack(0); n_at = fa.dropna(subset=[c for c in (ours, leiden) if c in fa]).shape[0] if ours in fa and leiden in fa else 0
    res["n_atlases_with_twins"] = n_at
    if n_at < 6: res["gate"] = "NOT EVALUABLE"; return res
    dd = (fa[ours] - fa[leiden]).dropna(); res["D"] = float(dd.mean()); res["D_boot95"] = boot(dd.values)
    pr = re[re.method.isin([ours, leiden])].pivot_table(index="pilot_id", columns="method", values="hit"); pr["atl"] = pr.index.map(atlas)
    se = pr.dropna().groupby("atl").mean().mean(); res["SE_common_pools"] = {k: float(v) for k, v in se.items() if k != "atl"}
    cond_se = res["SE_common_pools"][ours] >= res["SE_common_pools"][leiden] - 0.05; res["SE_condition"] = bool(cond_se)
    if res["FC"][ours] >= 0.95 or res["FC"][leiden] >= 0.95: res["gate"] = "NOT INFORMATIVE"
    else: res["gate"] = "MET" if (res["D_boot95"][1] < 0.05 and cond_se) else "NOT MET"
    return res


def analyze(args):
    ours, leiden, sccad = args.ours, args.leiden, args.sccad
    man, drop, d = load(args); atlas = man.atlas
    flags = pd.read_csv(args.flags) if args.flags else pd.DataFrame(columns=["pilot_id", "method", "reason"])
    tgt = pd.concat([pd.read_csv(f) for f in args.targets]) if args.targets else None
    W, prev, miss = build(d, man, flags, ours, tgt)
    if args.tiefair:
        tf = pd.read_csv(args.tiefair); w = tf.pivot(index="pilot_id", columns="method", values="tf_ap").reindex(W["union_AP"].index)
        for _, r in flags.iterrows():
            if r.pilot_id in w.index and r.method in w: w.loc[r.pilot_id, r.method] = prev[r.pilot_id]
        if ours not in w: w[ours] = NA
        w.loc[w[ours].isna(), ours] = prev[w.index[w[ours].isna()]]; W["union_AP_tiefair"] = w
    pools = list(W["union_AP"].index); tum = tumour_flags(sorted(atlas[pools].unique()), args)
    res = dict(n_pools=len(pools), n_atlases=int(atlas[pools].nunique()), dropped_pools=sorted(drop), ours_at_chance_pools=miss,
               methods=sorted(set(d.method)), tumour_flags=tum)
    # primary + same tests on E1_union
    for m in POOLM:
        t1, t2 = compare(W[m], ours, leiden, atlas), compare(W[m], ours, sccad, atlas)
        n_sc = int(W[m][sccad].dropna().index.map(atlas).nunique()) if sccad in W[m] else 0
        t2["evaluable"] = n_sc >= 6; t2["atlases_with_scorer"] = n_sc
        if not t2["evaluable"]: t2["pass"] = False
        t1["leiden_dropped_pools"] = int(W[m][leiden].isna().sum()) if leiden in W[m] else len(pools)
        res[m] = dict(T1=t1, T2=t2, both=bool(t1["pass"] and t2["pass"]))
    pr = res["union_AP"]; failed = [k for k in ("T1", "T2") if not pr[k]["pass"]]
    res["verdict"] = "PASS" if not failed else "NOT PASSED"; res["failed_part"] = failed
    if not pr["T2"]["evaluable"]: res["failed_part_note"] = "T2 not evaluable (comparator scored on fewer than 6 atlases)"
    # strata (T1/T2 on both metrics)
    sets = {"tumour": {a for a, t in tum.items() if t}, "non_tumour": {a for a, t in tum.items() if not t}}
    sets["tumour_excl_" + args.exclude] = {a for a in sets["tumour"] if not str(a).startswith(args.exclude)}
    res["strata"] = {s: {m: {t: stratum(W[m], ours, c, atlas, aset) for t, c in (("T1", leiden), ("T2", sccad))} for m in POOLM} for s, aset in sets.items()}
    # every other method vs ours (exploratory)
    others = [m for m in sorted(set(d.method)) if m != ours]
    res["vs_ours"] = {m: {k: compare(W[k], ours, m, atlas) for k in POOLM} for m in others}
    for m in others:
        for k in POOLM: res["vs_ours"][m][k].pop("per_atlas_diff", None)
    # hit rate, masked_AP, pool-weighted means
    dt = d[d.metric == "det50"]; hr = {}
    for m in sorted(set(d.method)):
        g = dt[dt.method == m]; pw = W["det50"][m].dropna() if "det50" in W and m in W["det50"] else pd.Series(dtype=float)
        hr[m] = dict(target_weighted=float(g.rprec.mean()) if len(g) else NA, atlas_equal=float(pw.groupby(atlas.loc[pw.index].values).mean().mean()) if len(pw) else NA, n_targets=len(g), n_pools=len(pw))
    res["hit_rate_det50"] = hr
    res["masked_AP"] = {m: dict(atlas_equal=float(W["masked_AP"][m].dropna().groupby(atlas.loc[W["masked_AP"][m].dropna().index].values).mean().mean()), n_pools=int(W["masked_AP"][m].notna().sum()))
                        for m in W["masked_AP"].columns}
    res["masked_AP_vs"] = {t: {k: v for k, v in compare(W["masked_AP"], ours, c, atlas).items() if k != "per_atlas_diff"} for t, c in (("T1", leiden), ("T2", sccad))}
    res["pool_weighted_means"] = {k: {m: float(W[k][m].mean()) for m in W[k].columns} for k in POOLM}
    res["atlas_equal_means"] = {k: {m: float(W[k][m].dropna().groupby(atlas.loc[W[k][m].dropna().index].values).mean().mean()) for m in W[k].columns} for k in POOLM}
    res["per_atlas"] = {m: {t: compare(W[m], ours, c, atlas).get("per_atlas_diff", {}) for t, c in (("T1", leiden), ("T2", sccad))} for m in POOLM}
    res["per_atlas_means"] = {k: {m: {a: float(v) for a, v in W[k][m].dropna().groupby(atlas.loc[W[k][m].dropna().index].values).mean().items()} for m in (ours, leiden, sccad) if m in W[k]} for k in POOLM}
    res["loao"] = {m: {t: {a: {kk: v for kk, v in compare(W[m], ours, c, atlas, drop_atlas=a).items() if kk in ("diff", "p", "n_atlases")} for a in sorted(atlas[pools].unique())}
                       for t, c in (("T1", leiden), ("T2", sccad))} for m in POOLM}
    res["learned_alone"] = {m: {k: v for k, v in compare(W[m], ours, args.learned, atlas).items() if k != "per_atlas_diff"} for m in POOLM}
    tools = [m for m in W["union_AP"].columns if m not in BASELINES | {ours, args.learned} and W["union_AP"][m].notna().any()]
    ok = [p for p in pools if all(pd.notna(W["union_AP"].loc[p, t]) for t in tools)]
    res["vs_leiden_tools_complete_pools"] = dict(tools=tools, n_pools=len(ok), **{m: {k: v for k, v in compare(W[m], ours, leiden, atlas, pools=ok).items() if k != "per_atlas_diff"} for m in POOLM})
    if "union_AP_tiefair" in W:
        res["tiefair"] = dict(T1=compare(W["union_AP_tiefair"], ours, leiden, atlas), T2=compare(W["union_AP_tiefair"], ours, sccad, atlas))
    if args.gate: res["gate"] = gate_stats(pd.read_csv(args.gate), atlas, ours, leiden)
    return res


def f(x, n=3): return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def row(r): return f"| {r['a']} vs {r['b']} | {r['n_pools']} | {r['n_atlases']} | {f(r['a_mean'])} | {f(r['b_mean'])} | {f(r['diff'], 4)} | {r['atlas_ahead']}/{r['n_atlases']} | {f(r['p'], 5)} | {r['pass'] if 'pass' in r else ''} |"


HDR = "| comparison | pools | atlases | a mean | b mean | diff | atlases ahead | exact p | pass |\n|---|---|---|---|---|---|---|---|---|"


def render(res, args):
    L = ["# Analysis", "", f"VERDICT (primary, union_AP): **{res['verdict']}**" + (f" (failed: {', '.join(res['failed_part'])})" if res["failed_part"] else ""),
         f"{res['n_pools']} pools, {res['n_atlases']} atlases, {len(res['dropped_pools'])} pools dropped by leak check, {len(res['ours_at_chance_pools'])} pools where our method is scored at chance.", ""]
    L += ["## Primary and E1_union\n", HDR]
    for m in POOLM:
        for t in ("T1", "T2"): L.append(row(res[m][t]) + f" <- {m} {t}")
    L += ["", f"T2 evaluable: {res['union_AP']['T2']['evaluable']} (comparator scored on {res['union_AP']['T2']['atlases_with_scorer']} atlases). Pools dropped from T1 for Leiden did-not-fit: {res['union_AP']['T1']['leiden_dropped_pools']}.", ""]
    L += ["## Strata (descriptive; bootstrap 95% of atlas mean diff)\n", "| stratum | metric | test | atlases | diff | ahead | boot95 | exact p |\n|---|---|---|---|---|---|---|---|"]
    for s, v in res["strata"].items():
        for m, tt in v.items():
            for t, r in tt.items(): L.append(f"| {s} | {m} | {t} | {r['n_atlases']} | {f(r['diff'], 4)} | {r['atlas_ahead']}/{r['n_atlases']} | [{f(r['boot95'][0])}, {f(r['boot95'][1])}] | {f(r['p'], 4)} |")
    L += ["", "## Every other method vs our method (exploratory, unadjusted)\n", HDR]
    for m, v in res["vs_ours"].items():
        for k in POOLM: L.append(row(v[k]) + f" <- {k}")
    L += ["", "## Hit rate (det50)\n", "| method | target-weighted | atlas-equal | targets | pools |\n|---|---|---|---|---|"]
    L += [f"| {m} | {f(v['target_weighted'])} | {f(v['atlas_equal'])} | {v['n_targets']} | {v['n_pools']} |" for m, v in res["hit_rate_det50"].items()]
    L += ["", "## masked_AP (atlas-equal)\n", "| method | value | pools |\n|---|---|---|"] + [f"| {m} | {f(v['atlas_equal'])} | {v['n_pools']} |" for m, v in res["masked_AP"].items()]
    L += ["", HDR] + [row(res["masked_AP_vs"][t]) + f" <- masked_AP {t}" for t in ("T1", "T2")]
    L += ["", "## Ours vs learned scorer alone, and vs Leiden on pools where every tool completed\n", HDR]
    L += [row(res["learned_alone"][m]) + f" <- {m}" for m in POOLM]
    L += [row(res["vs_leiden_tools_complete_pools"][m]) + f" <- {m} (tools: {', '.join(res['vs_leiden_tools_complete_pools']['tools'])}; {res['vs_leiden_tools_complete_pools']['n_pools']} pools)" for m in POOLM]
    if "tiefair" in res: L += ["", "## Tie-fair union AP\n", HDR] + [row(res["tiefair"][t]) for t in ("T1", "T2")]
    L += ["", "## Pool-weighted and atlas-equal means\n", "| metric | method | pool-weighted | atlas-equal |\n|---|---|---|---|"]
    L += [f"| {k} | {m} | {f(res['pool_weighted_means'][k][m])} | {f(res['atlas_equal_means'][k][m])} |" for k in POOLM for m in res["pool_weighted_means"][k]]
    L += ["", "## Per atlas (our method minus comparator, union_AP / E1_union)\n", "| atlas | T1 union_AP | T2 union_AP | T1 E1_union | T2 E1_union |\n|---|---|---|---|---|"]
    for a in sorted(res["per_atlas"]["union_AP"]["T1"]):
        g = lambda m, t: f(res["per_atlas"][m][t].get(a, NA), 4)
        L.append(f"| {a} | {g('union_AP', 'T1')} | {g('union_AP', 'T2')} | {g('E1_union', 'T1')} | {g('E1_union', 'T2')} |")
    L += ["", "## Leave-one-atlas-out (union_AP diff / p)\n", "| left out | T1 | T2 |\n|---|---|---|"]
    for a in res["loao"]["union_AP"]["T1"]:
        x, y = res["loao"]["union_AP"]["T1"][a], res["loao"]["union_AP"]["T2"][a]
        L.append(f"| {a} | {f(x['diff'], 4)} (p {f(x['p'], 4)}) | {f(y['diff'], 4)} (p {f(y['p'], 4)}) |")
    if "gate" in res:
        g = res["gate"]; L += ["", f"## Null gate: **{g['gate']}**", "", f"atlases with twins: {g['n_atlases_with_twins']}"]
        for k in ("D", "D_boot95", "SE_condition", "SE_common_pools"):
            if k in g: L.append(f"- {k}: {g[k]}")
        L += ["", "| method | FC (twins) | SE (real) | call precision | twins | real |\n|---|---|---|---|---|---|"]
        for m in sorted(set(g["FC"]) | set(g["SE"])): L.append(f"| {m} | {f(g['FC'].get(m, NA))} | {f(g['SE'].get(m, NA))} | {f(g['call_precision'].get(m, NA))} | {g['n_twins'].get(m, 0)} | {g['n_real'].get(m, 0)} |")
    return "\n".join(L) + "\n"


def scan(root, out):
    rows = []
    for fn in sorted(glob.glob(f"{root}/*/*.parquet")):
        s = pd.read_parquet(fn, columns=["score"]).score.values; meth, pid = fn.split(os.sep)[-2], os.path.basename(fn)[:-8]
        if np.isnan(s).any(): rows.append((pid, meth, "nan"))
        elif len(np.unique(s)) == 1: rows.append((pid, meth, "constant"))
    pd.DataFrame(rows, columns=["pilot_id", "method", "reason"]).to_csv(out, index=False); print(len(rows), "flagged")


def union_mask(df, tg):
    u = np.zeros(len(df), bool)
    for L, T, _ in tg: u |= ((df.label == L) & (df.true_type == T)).values
    return u


def tiefair_table(root, out, pilot_dir, manifests):
    man = set(pd.concat([pd.read_csv(m) for m in manifests]).pilot_id); rows, cache = [], {}
    for fn in sorted(glob.glob(f"{root}/*/*.parquet")):
        meth, pid = fn.split(os.sep)[-2], os.path.basename(fn)[:-8]
        if pid not in man: continue
        if pid not in cache: cache[pid] = truth_table(pilot_dir, pid)
        df, tg = cache[pid]; u = union_mask(df, tg)
        if not u.any(): continue
        s = pd.read_parquet(fn).set_index("cell_id").reindex(df.index).score.fillna(0).values; rows.append((pid, meth, tiefair_ap(s, u)))
    pd.DataFrame(rows, columns=["pilot_id", "method", "tf_ap"]).to_csv(out, index=False); print(len(rows), "rows")


def gateprep(a):
    """Per (kind, pool, method): n_called (200), max_small = largest number of called cells in one group whose share of its label is < 0.10,
    n_targets = called cells that are targets (real pools). Top 200 by score, ties at the boundary by ascending cell_id.
    Group files: GROUP_ROOT/<method>/<pool>.parquet with cell_id, group, group_share. Methods without a group file get max_small = NaN."""
    twin = pd.read_csv(a.twin_map).set_index("twin_id").pilot_id.to_dict() if a.twin_map else {}
    man = set(pd.concat([pd.read_csv(m) for m in a.manifest]).pilot_id) | set(twin); rows = []
    for fn in sorted(glob.glob(f"{a.pred_root}/*/*.parquet")):
        meth, pid = fn.split(os.sep)[-2], os.path.basename(fn)[:-8]
        if pid not in man: continue
        p = pd.read_parquet(fn); top = p.iloc[np.lexsort((p.cell_id.values.astype(str), -p.score.values))[:200]]
        gf = f"{a.group_root}/{meth}/{pid}.parquet"; ms = NA
        if os.path.exists(gf):
            g = pd.read_parquet(gf).set_index("cell_id").reindex(top.cell_id); sm = g[g.group_share < 0.10]
            ms = float(sm.group.value_counts().max()) if len(sm.dropna(subset=["group"])) else 0.0
        kind, nt = ("twin", NA) if pid in twin else ("real", NA)
        if kind == "real":
            df, tg = truth_table(a.pilot_dir, pid)
            nt = float(union_mask(df, tg)[df.index.get_indexer(top.cell_id)].sum())
        rows.append((kind, twin.get(pid, pid), pid, meth, len(top), ms, nt))
    pd.DataFrame(rows, columns=["kind", "pilot_id", "source_id", "method", "n_called", "max_small", "n_targets"]).to_csv(a.out, index=False); print(len(rows), "rows")


def demo():
    assert exact([1, 1, 1, 1, 1]) == 2 / 32 and exact([0.0, 0.0]) == 1.0 and exact([]) != exact([])
    s = np.array([3, 2, 2, 2, 1.0]); u = np.array([1, 0, 1, 0, 1], bool); rng = np.random.RandomState(0); mc = []
    from sklearn.metrics import average_precision_score
    for _ in range(4000): mc.append(average_precision_score(u, s + rng.rand(5) * 1e-3 * (s == 2)))
    assert abs(tiefair_ap(s, u) - np.mean(mc)) < 0.01, (tiefair_ap(s, u), np.mean(mc))
    assert abs(tiefair_ap(np.arange(10.), np.arange(10) >= 7) - 1.0) < 1e-12
    # end to end on synthetic scores: 8 atlases x 2 pools, our method missing on one pool
    rows = []; man = []; rng = np.random.RandomState(1)
    for a in "ABCDEFGH":
        for p in range(2):
            pid = f"atl_{a}_p{p}"; man.append((pid, a, "t", 20000))
            for m, mu in ((OURS, .4), (LEIDEN, .2), (SCCAD, .3)):
                for met in POOLM + ["masked_AP", "det50"]:
                    rows.append((pid, m, met, "*" if met in POOLM else "L", "*" if met in POOLM else "T", 500, mu + .05 * rng.rand()))
    S = pd.DataFrame(rows, columns=["pilot_id", "method", "metric", "label", "target", "n_rare", "rprec"]); M = pd.DataFrame(man, columns=["pilot_id", "atlas", "split", "n_cells"])
    S = S[~((S.pilot_id == "atl_A_p0") & (S.method == OURS))]
    import tempfile; td = tempfile.mkdtemp(); S.to_csv(f"{td}/s.csv", index=False); M.to_csv(f"{td}/m.csv", index=False)
    pd.DataFrame({"name": list("ABCDEFGH"), "stratum": ["tumour"] * 4 + ["non-tumour"] * 4}).to_csv(f"{td}/atlases.csv", index=False)
    args = argparse.Namespace(manifest=[f"{td}/m.csv"], scores=[f"{td}/s.csv"], atlases=f"{td}/atlases.csv", split=None, drop=None, flags=None, tiefair=None, gate=None,
                              targets=None, exclude="zzz", ours=OURS, leiden=LEIDEN, sccad=SCCAD, learned=LEARNED)
    r = analyze(args); assert r["failed_part"] == ["T2"] and r["ours_at_chance_pools"] == ["atl_A_p0"] and abs(r["union_AP"]["T1"]["p"] - 2 / 256) < 1e-9  # chance fill sinks atlas A in T2 only
    render(r, args)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if c == "demo": demo(); print("demo ok")
    elif c == "scan": scan(sys.argv[2], sys.argv[3])
    elif c == "tiefair": tiefair_table(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif c == "gateprep":
        ap = argparse.ArgumentParser(); ap.add_argument("pred_root"); ap.add_argument("group_root"); ap.add_argument("out"); ap.add_argument("pilot_dir"); ap.add_argument("manifest", nargs="+")
        ap.add_argument("--twin-map"); gateprep(ap.parse_args(sys.argv[2:]))
    elif c == "run":
        ap = argparse.ArgumentParser(); ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--manifest", nargs="+", required=True); ap.add_argument("--atlases", required=True)
        for k in ("split", "drop", "flags", "targets", "gate", "tiefair"): ap.add_argument("--" + k, nargs="+" if k in ("drop", "targets") else None)
        ap.add_argument("--exclude", default="0bf30b02"); ap.add_argument("--out", default="analysis")
        ap.add_argument("--ours", default=OURS); ap.add_argument("--leiden", default=LEIDEN); ap.add_argument("--sccad", default=SCCAD); ap.add_argument("--learned", default=LEARNED)
        a = ap.parse_args(sys.argv[2:]); r = analyze(a); json.dump(r, open(a.out + ".json", "w"), indent=1, default=str); open(a.out + ".md", "w").write(render(r, a))
        print("verdict", r["verdict"], r["failed_part"], f"({r['n_pools']} pools, {r['n_atlases']} atlases)")
