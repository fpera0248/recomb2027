"""Natural-hierarchy pools from full CELLxGENE atlases Same pilot format as build_hlca.py; one tier per atlas:
label = the atlas's coarser author/CL annotation (what a user would have), true_type = its finer annotation. Raw counts from raw/X.
Pools of 20,000 cells (seed 0). mode "block": one pool per value of the block column (min 2,000 cells); mode "pack": donors shuffled and packed
until >= 20,000 cells, each pack one pool (donor-disjoint). At most MAXP pools per atlas, in shuffled order (the cap is logged).
TEST atlases are md5-parity-1 datasets never used to train the learned scorer; DEV atlases are parity 0.
  python build_atlas.py NAME [NAME ...] -> slim_atlas/<name>_p<k>.h5ad, hmod/pilots/atl_<name>_p<k>.parquet, atlas_manifest_<name>.csv"""
import os, sys, numpy as np, pandas as pd, anndata as ad, h5py, scipy.sparse as sp
from anndata.io import sparse_dataset
R5 = os.environ.get("R5") or os.path.expanduser("~/work/r5/data"); SI = R5; AT = f"{R5}/atlas"; OUT = f"{SI}/slim_atlas"; PIL = f"{SI}/hmod/pilots"
N, MINP, MAXP = 20000, 2000, 24   # up to 24 pools per atlas (power)
CFG = {  # name: (label col, finer col, mode, block col, split). Parity-1 atlases accepted after scouting.
    "pbmc": ("cell_type", "annotation_detailed", "pack", "donor_id", "test3"),
    "covid5": ("Cell.group", "Cell.class", "pack", "donor_id", "test3"),
    "combat": ("cell_type_original", "major_subset", "pack", "donor_id", "test3"),
    "mi": ("cell_type", "final_cluster", "pack", "donor_id", "test3"),
}
DROP = {}
CELLDROP = {"pbmc": lambda o: (o.cell_type == "natural killer cell") & (o.annotation_detailed == "B n-sw mem")}   # curation error: B cells labelled NK (checker)


def pools(o, mode, col, rng):
    out = []
    if mode == "block":
        for b in rng.permutation(sorted(o[col].unique())):
            ix = np.where(o[col].values == b)[0]
            if len(ix) >= MINP: out.append((str(b), np.sort(rng.choice(ix, min(N, len(ix)), replace=False))))
    else:
        cur, names = [], []
        for d in rng.permutation(sorted(o[col].unique())):
            cur.append(np.where(o[col].values == d)[0]); names.append(str(d))
            if sum(map(len, cur)) >= N:
                ix = np.concatenate(cur); out.append(("+".join(names)[:200], np.sort(rng.choice(ix, N, replace=False)))); cur, names = [], []
    return out


def build(name):
    lab, fine, mode, col, split = CFG[name]; os.makedirs(OUT, exist_ok=True); f = f"{AT}/{name}.h5ad"
    a = ad.read_h5ad(f, backed="r"); o = a.obs[list(dict.fromkeys([lab, fine, col, "donor_id"]))].astype(str).copy()
    var = (a.raw.var_names if a.raw is not None else a.var_names).copy(); xk = "raw/X" if a.raw is not None else "X"; on = a.obs_names.copy(); a.file.close()
    o = o[~o.donor_id.isin(DROP.get(name, []))]
    if name in CELLDROP: o = o[~CELLDROP[name](o)]
    ix0 = np.where(on.isin(o.index))[0] if len(o) < len(on) else None
    P = pools(o, mode, col, np.random.RandomState(0)); P = [(b, ix0[r] if ix0 is not None else r) for b, r in P]; print(name, len(P), "pools possible; keeping", min(len(P), MAXP), flush=True); man = []
    for k, (b, rows) in enumerate(P[:MAXP]):
        with h5py.File(f, "r") as h: X = sparse_dataset(h[xk])[rows]
        X = sp.csr_matrix(X).astype(np.float32); assert X.shape[0] == len(rows) and np.allclose(X.data[:100000], np.round(X.data[:100000]))
        pp = o.loc[on[rows]]; pn = f"{name}_p{k}"; pid = f"atl_{pn}"
        ad.AnnData(X=X, obs=pd.DataFrame(index=pp.index), var=pd.DataFrame(index=var)).write_h5ad(f"{OUT}/{pn}.h5ad")
        pd.DataFrame({"obs_name": pp.index.values, "label": pp[lab].values, "true_type": pp[fine].values, "is_rare": 0,
                      "donor": pp.donor_id.values, "source_path": f"{pn}.h5ad"}).to_parquet(f"{PIL}/{pid}.parquet", index=False)
        man.append((pid, name, split, b, k, len(pp), pp[lab].nunique())); print(pid, len(pp), flush=True)
    pd.DataFrame(man, columns=["pilot_id", "atlas", "split", "block", "pool", "n_cells", "n_labels"]).to_csv(f"{SI}/atlas_manifest_{name}.csv", index=False)


def demo():
    o = pd.DataFrame({"d": ["a"] * 15000 + ["b"] * 15000 + ["c"] * 1000})
    P = pools(o, "pack", "d", np.random.RandomState(0)); assert len(P) == 1 and len(P[0][1]) == N and len(set(P[0][1])) == N
    P = pools(o, "block", "d", np.random.RandomState(0)); assert sorted(len(p[1]) for p in P) == [15000, 15000]


if __name__ == "__main__":
    demo(); [build(n) for n in sys.argv[1:]]
