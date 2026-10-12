"""Exact cell fingerprints of every test pool against every previously used dataset file, run before any method touches a pool.

    python -m data.leak_check cells NAME [NAME ...] [--panel DIR] [--out DIR] [--thresh 0.01] [--minexpr 50]
pools = $HR_DATA/pools/<NAME>_p*.h5ad; panel = DIR/*.h5ad (default $HR_DATA/panel: the datasets of used_datasets.csv,
downloaded by dataset id). Twin pools are not checked, they are subsets of the same donors. Only the sparsity pattern of X
and the gene ids are read, never a label or a score.

The fingerprint of a cell is the 64-bit sum of fixed per-gene random keys (blake2b of the version-stripped Ensembl id) over
the genes expressed in the cell, restricted to the intersection U of the gene ids of all pools and panel files. The sum
over a set is an order-independent set hash. Cells with fewer than --minexpr expressed genes in U are not fingerprinted.
A pool is dropped when at least --thresh of its cells match a panel cell.
Writes leak_cells.csv (pool, ref, n_match, n_cells, n_fp), leak_drop.txt (pool ids, one per line) and leak_cells_meta.json to --out.
"""
import os, glob, json, hashlib, argparse, warnings, numpy as np, pandas as pd, h5py
warnings.filterwarnings("ignore")
ROOT = os.environ.get("HR_DATA", "work")
_K = {}


def key(g):
    k = _K.get(g)
    if k is None: k = _K[g] = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
    return k


def gene_ids(h):
    from anndata.io import read_elem
    v = read_elem(h["var"]); x = v["feature_id"].astype(str).values if "feature_id" in v else v.index.astype(str).values
    return np.array([s.split(".")[0] for s in x])


def csr_parts(h):
    g = h["X"]; assert isinstance(g, h5py.Group) and "indptr" in g, "X is not sparse"
    enc = g.attrs.get("encoding-type", "csr_matrix"); assert enc == "csr_matrix", f"X encoding {enc} (csc not supported)"
    return g["data"], g["indices"], g["indptr"][...]


def fingerprints(path, U, minexpr, chunk=50000):
    """-> (hash uint64 per cell, fingerprinted mask) for the file's cells over universe U (array of gene ids)."""
    with h5py.File(path, "r") as h:
        ids = gene_ids(h); data, ind, ptr = csr_parts(h); n = len(ptr) - 1
        kv = np.zeros(len(ids), np.uint64); inU = np.isin(ids, U)
        kv[inU] = [key(g) for g in ids[inU]]
        out = np.zeros(n, np.uint64); ne = np.zeros(n, np.int64)
        for s in range(0, n, chunk):
            e = min(n, s + chunk); lo, hi = ptr[s], ptr[e]; d = data[lo:hi]; ix = ind[lo:hi]
            ok = (d != 0) & inU[ix]; kk = np.where(ok, kv[ix], np.uint64(0)); rp = ptr[s:e + 1] - lo
            cs = np.concatenate([[0], np.cumsum(ok)]); ne[s:e] = cs[rp[1:]] - cs[rp[:-1]]
            nz = rp[1:] > rp[:-1]
            if nz.any(): out[s:e][nz] = np.add.reduceat(kk, rp[:-1][nz])
    return out, ne >= minexpr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("names", nargs="*")
    ap.add_argument("--panel", default=f"{ROOT}/panel"); ap.add_argument("--files"); ap.add_argument("--pools")
    ap.add_argument("--thresh", type=float, default=0.01); ap.add_argument("--minexpr", type=int, default=50); ap.add_argument("--out", default=ROOT)
    a = ap.parse_args(); assert a.cmd == "cells"
    pools = a.pools.split(",") if a.pools else sorted(f for n in a.names for f in glob.glob(f"{ROOT}/pools/{n}_p*.h5ad"))
    panel = [os.path.join(a.panel, f) for f in a.files.split(",")] if a.files else sorted(glob.glob(f"{a.panel}/*.h5ad"))
    assert pools, "no pools"; U = None
    for f in pools + panel:
        with h5py.File(f, "r") as h: g = set(gene_ids(h))
        U = g if U is None else U & g
    U = np.array(sorted(U)); print("pools", len(pools), "panel files", len(panel), "|U|", len(U), flush=True)
    assert len(U) >= 5000, f"universe {len(U)} genes < 5000: cannot fingerprint (report, do not guess)"
    P = {}
    for f in pools:
        h_, ok = fingerprints(f, U, a.minexpr); P[os.path.basename(f)[:-5]] = (h_, ok)
    rows, matched = [], {k: np.zeros(len(v[0]), bool) for k, v in P.items()}
    for f in panel:
        ph, pok = fingerprints(f, U, a.minexpr); ph = np.unique(ph[pok]); ref = os.path.basename(f)[:-5]
        for k, (h_, ok) in P.items():
            m = ok & np.isin(h_, ph)
            if m.any(): rows.append((k, ref, int(m.sum()), len(h_), int(ok.sum()))); matched[k] |= m
        print("panel", ref[:8], len(ph), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows, columns=["pool", "ref", "n_match", "n_cells", "n_fp"]).to_csv(f"{a.out}/leak_cells.csv", index=False)
    drop = sorted(k for k, m in matched.items() if m.sum() >= a.thresh * len(m))
    open(f"{a.out}/leak_drop.txt", "w").write("".join(f"atl_{k}\n" for k in drop))
    json.dump(dict(U=len(U), pools=len(pools), panel_files=len(panel), thresh=a.thresh, minexpr=a.minexpr, matched={k: int(m.sum()) for k, m in matched.items()}, dropped=drop),
              open(f"{a.out}/leak_cells_meta.json", "w"), indent=1)
    print("matched cells per pool:", {k: int(m.sum()) for k, m in matched.items()}, "\ndropped:", drop)


if __name__ == "__main__":
    main()
