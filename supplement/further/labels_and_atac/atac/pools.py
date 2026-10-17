"""python pools.py prep|roots --dataset ID --h5ad FILE --sel SELECTION.json --data DIR --out DIR [--k K --coarse C --fine F]  [--tag T --roots NAME] (--coarse/--fine/--k only for the kidney regression; tag goes into the pid)
prep : donor-disjoint pools (greedy by donor size to K = min(3, floor(cells / 8000)) pools, each subsampled with seed 0 to <= 20,000 cells), writes DATA/whitelist.txt (obs names of all pool cells),
       OUT/pools.json (donors, cell counts, target pairs on the pre-GA cells). Metadata only: no X is read.
roots: after build_ga.py (DATA/ga_counts.h5ad): pipeline roots DATA/roots/{clean,rna,ga,gak} (rna only for multiome with integer counts; gak = coarse labels re-derived by cross-fit kNN in GA PCA-50), same
       recipe as the kidney pilot (code/w6/p3_pools.py); rewrites OUT/pools.json with final pairs on cells that have fragments. pid = atc_<dataset8>_p<k>."""
import os, sys, json, shutil, argparse, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
sys.dont_write_bytecode = True; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from select_ds import targets, DROP
CODE_R5 = HR_WORK + "/r5/code"; CAP = 20000; K_NN = 15

def split_donors(don, k):
    """greedy donor-disjoint split balanced by cells (kidney-pilot rule): donors by size, each to the currently smallest pool."""
    sizes = pd.Series(don).value_counts(); pools = [[] for _ in range(k)]; tot = [0] * k
    for d, n in sizes.items(): i = int(np.argmin(tot)); pools[i].append(d); tot[i] += n
    return pools, tot

def pool_cells(obs, coarse, fine, k=None, cap=CAP):
    """obs with donor_id; returns (kept obs, list of (donors, cell positions into kept)). Unknown labels dropped first; subsample seed 0."""
    ok = ~(obs[coarse].astype(str).str.contains(DROP) | obs[fine].astype(str).str.contains(DROP)).values; o = obs[ok]
    k = k or min(3, len(o) // 8000); k = max(1, min(k, o["donor_id"].nunique())); don = o["donor_id"].astype(str).values
    pools, tot = split_donors(don, k); res = []
    for dn in pools:
        idx = np.where(np.isin(don, dn))[0]
        if len(idx) > cap: idx = np.sort(np.random.default_rng(0).choice(idx, cap, replace=False))
        res.append((dn, idx))
    return o, res

def frame(o, idx, coarse, fine, pid):
    return pd.DataFrame(dict(obs_name=o.index.values[idx], label=o[coarse].astype(str).values[idx], true_type=o[fine].astype(str).values[idx], is_rare=0, donor=o["donor_id"].astype(str).values[idx], source_path=f"{pid}.h5ad"))

def knn_votes(X, y, nlab, k=K_NN):
    """leave-one-out cosine kNN vote matrix (same as make_variants.knn_votes)."""
    from sklearn.neighbors import NearestNeighbors
    _, ix = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X).kneighbors(X); ix = ix[:, 1:]; V = np.zeros((len(ix), nlab)); np.add.at(V, (np.repeat(np.arange(len(ix)), k), y[ix].ravel()), 1); return V

def prep(A, sel):
    import anndata as ad
    obs = ad.read_h5ad(A.h5ad, backed="r").obs.copy(); c, f = A.coarse or sel["chosen"]["coarse"], A.fine or sel["chosen"]["fine"]
    o, pl = pool_cells(obs, c, f, A.k); d8 = A.dataset[:8] + A.tag; info = dict(dataset=A.dataset, coarse=c, fine=f, K=len(pl), cap=CAP, pools={}); names = []
    for k, (dn, idx) in enumerate(pl):
        pid = f"atc_{d8}_p{k}"; df = frame(o, idx, c, f, pid); names += list(df.obs_name)
        info["pools"][pid] = dict(donors=dn, n_cells=len(df), pairs_pre_ga=[list(t) for t in targets(df)])
    os.makedirs(A.data, exist_ok=True); os.makedirs(A.out, exist_ok=True); wf = f"{A.data}/whitelist.txt"
    if os.path.exists(wf): assert set(names) <= set(open(wf).read().split("\n")), "pool cells not covered by the existing whitelist (fragments already built for other cells)"
    else: open(wf, "w").write("\n".join(names) + "\n")
    info["tag"] = A.tag; json.dump(info, open(f"{A.out}/pools.json", "w"), indent=1, default=str)
    print("PREP_DONE", {p: (v["n_cells"], len(v["pairs_pre_ga"])) for p, v in info["pools"].items()}, flush=True)

def roots(A, sel):
    import anndata as ad, scipy.sparse as sp
    sys.path.insert(0, f"{CODE_R5}/hmod"); import methods as M
    info = json.load(open(f"{A.out}/pools.json")); A.tag = info["tag"]; c, f = info["coarse"], info["fine"]; a = ad.read_h5ad(A.h5ad, backed="r"); obs = a.obs
    ga = ad.read_h5ad(f"{A.data}/ga_counts.h5ad"); multi = "multiome" in str(sel["row"]["assay"]).lower(); RT = f"{A.data}/{A.roots}"
    raw = a.raw if a.raw is not None else a; rna = False; gnames = set(ga.var_names)
    if multi:
        x = raw.X[:200]; x = x.toarray() if sp.issparse(x) else np.asarray(x); rna = bool(np.allclose(x, np.round(x)) and x.max() > 1)   # counts, not normalised
        common = [g for g in raw.var_names if g in gnames]
    else:    # scATAC-only: same gene universe as the multiome arms (the h5ad genes), not all ~60k Ensembl genes; fall back to all if the h5ad has no gene ids
        hv = set(raw.var_names); common = [g for g in ga.var_names if g in hv]
        if len(common) < 5000: print("WARN few h5ad genes in GA; using all GA genes", len(common), flush=True); common = list(ga.var_names)
    info["rna_arm"] = rna; info["n_common_genes"] = len(common); arms = (["rna"] if rna else []) + ["ga", "gak"]; print("arms", arms, "genes", len(common), flush=True)
    for r in ["clean"] + arms:
        os.makedirs(f"{RT}/{r}/hmod/pilots", exist_ok=True); os.makedirs(f"{RT}/{r}/slim_atlas", exist_ok=True)
        for fn in ("hlib.py", "methods.py", "rawutil.py", "rprec.py"): shutil.copy(f"{CODE_R5}/hmod/{fn}", f"{RT}/{r}/hmod/{fn}")
    gidx = [ga.var_names.get_loc(g) for g in common]; ridx = np.array([raw.var_names.get_loc(g) for g in common]) if rna else None
    for pid, pv in list(info["pools"].items()):
        # same cells as prep (identical rule), then keep those with fragments
        o, pl = pool_cells(obs, c, f, info["K"]); k = int(pid.rsplit("_p", 1)[1]); idx = pl[k][1]; keep = np.isin(o.index.values[idx], ga.obs_names.values); idx = idx[keep]
        df = frame(o, idx, c, f, pid); nm = df.obs_name.values; assert len(df) > 0, pid
        Xg = ga[nm][:, common].X.tocsr().astype(np.float32)
        if rna:
            order = obs.index.get_indexer(nm); so = np.argsort(order); Xr = sp.csr_matrix(raw.X[order[so]][:, ridx])[np.argsort(so)].astype(np.float32)
            ad.AnnData(X=Xr, obs=pd.DataFrame(index=nm), var=pd.DataFrame(index=common)).write_h5ad(f"{RT}/rna/slim_atlas/{pid}.h5ad")
        s = ad.AnnData(X=Xg, obs=pd.DataFrame(index=nm), var=pd.DataFrame(index=common)); s.write_h5ad(f"{RT}/ga/slim_atlas/{pid}.h5ad"); shutil.copy(f"{RT}/ga/slim_atlas/{pid}.h5ad", f"{RT}/gak/slim_atlas/{pid}.h5ad")
        for r in ["clean"] + [x for x in arms if x != "gak"]: df.to_parquet(f"{RT}/{r}/hmod/pilots/{pid}.parquet", index=False)
        X, _ = M.preprocess_pca(s); labs, inv = np.unique(df.label.values, return_inverse=True); new = labs[knn_votes(X, inv, len(labs)).argmax(1)]
        d2 = df.copy(); d2["label"] = new; d2.to_parquet(f"{RT}/gak/hmod/pilots/{pid}.parquet", index=False)
        pv.update(n_cells_final=len(df), pairs=[list(t) for t in targets(df)], knn_acc=float((new == df.label.values).mean()), n_targets_gak=len(targets(d2)))
        print(pid, len(df), "targets", len(pv["pairs"]), "knn acc", pv["knn_acc"], flush=True)
    info["arms"] = arms; json.dump(info, open(f"{A.out}/pools.json", "w"), indent=1, default=str); print("POOLS_DONE")

def demo():
    don = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 30 + ["d"] * 10); p, t = split_donors(don, 2); assert p == [["a", "d"], ["b", "c"]] and t == [60, 60]
    rng = np.random.default_rng(1); n = 50000; obs = pd.DataFrame(dict(donor_id=rng.choice(list("abcdefgh"), n), co=rng.choice(["X", "Y", "unknown"], n), fi=rng.choice(["x1", "x2", "y1"], n)), index=[f"c{i}" for i in range(n)])
    o, pl = pool_cells(obs, "co", "fi"); assert len(o) < n and len(pl) == 3 and all(len(i) <= CAP for _, i in pl) and len({d for dn, _ in pl for d in dn}) == 8   # donor-disjoint, capped, unknown dropped
    assert np.array_equal(pl[0][1], pool_cells(obs, "co", "fi")[1][0][1])                                            # seed 0: deterministic
    o2, pl2 = pool_cells(obs.iloc[:9000], "co", "fi"); assert len(pl2) == 1                                          # floor(cells / 8000) = 1 pool
    y = np.repeat([0, 1], 50); X = np.vstack([rng.normal(0, 1, (50, 5)) + 5, rng.normal(0, 1, (50, 5)) - 5]); assert (knn_votes(X, y, 2).argmax(1) == y).all()

if __name__ == "__main__":
    if len(sys.argv) == 1: demo(); print("demo ok"); sys.exit(0)
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["prep", "roots"]); p.add_argument("--dataset", required=True); p.add_argument("--h5ad", required=True); p.add_argument("--sel", required=True)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--k", type=int); p.add_argument("--coarse"); p.add_argument("--fine"); p.add_argument("--tag", default=""); p.add_argument("--roots", default="roots"); A = p.parse_args()
    sel = json.load(open(A.sel))
    if A.stage == "prep" and str(sel.get("row", {}).get("eligible", True)) != "True": sys.exit(f"ineligible under the amended selection rule: {sel.get('row', {}).get('reason')}")
    {"prep": prep, "roots": roots}[A.stage](A, sel)
