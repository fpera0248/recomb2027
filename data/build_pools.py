"""Build the test pools of a CELLxGENE atlas: 20,000-cell, donor-disjoint pools with raw counts.

    python -m data.build_pools NAME [NAME ...] [--force]
NAME is the first 8 characters of a dataset id in atlases.csv; the atlas is read from
$HR_DATA/atlas/<dataset_id>.h5ad (from fetch_data.sh full "$HR_DATA", which checks each file's size
and, when ATLAS_MANIFEST.csv has one, its md5; or from download_atlases.sh, which checks neither).
Only obs, raw counts and the two annotation columns are used.

Visible label = the atlas's coarse annotation, truth = its finer annotation. Donors are shuffled
(seed 0) and packed until a pack holds 20,000 cells; 20,000 cells are sampled from each pack.
At most 8 pools per atlas. Before pooling we drop non-10x cells, the donors listed in DROP_DONORS
(donors that also appear in the datasets used to train the scorer), placeholder labels and, for
NON_PRIMARY, cells flagged is_primary_data = False.
Writes under $HR_DATA: pools/<name>_p<k>.h5ad, pilots/atl_<name>_p<k>.parquet,
manifests/atlas_manifest_<name>.csv, targets/targets_<name>.csv, genemap/<name>.csv.
"""

import json
import os
import re
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import sparse_dataset

from eval.targets import targets

ROOT = os.environ.get("HR_DATA", "work")
HERE = os.path.dirname(os.path.abspath(__file__))
N, MAX_POOLS, MIN_DONORS, MIN_CELLS = 20000, 8, 8, 60000
ATLASES = pd.read_csv(f"{HERE}/atlases.csv", dtype=str).set_index("name")

# donors that also appear in a previously used dataset (data/used_datasets.csv)
DROP_DONORS = {
    "fe067858": "CV0902 CV0904 CV0911 CV0915 CV0917 CV0926 CV0929 CV0934 CV0939 CV0940 CV0944 "
    "MH8919176 MH8919177 MH8919178 MH8919179 MH8919226 MH8919227 MH8919282 MH8919283 MH8919332 MH8919333 newcastle65 newcastle74 A35 A36".split()
}
# meta-atlas whose generic donor ids cannot be traced: keep primary-data cells only
NON_PRIMARY = {"fe067858"}
PLACEHOLDER = re.compile(
    r"^(doublets?|unassigned|unclassified|unknown|undefined|low[ _]?quality|removed|nan|none|na|-)$"
)


def pools(o, col, rng):
    """[(pack name, sorted row indices)]: donors shuffled and packed to >= N cells, N cells sampled from each pack."""
    out, cur, names = [], [], []
    for d in rng.permutation(sorted(o[col].unique())):
        cur.append(np.where(o[col].values == d)[0])
        names.append(str(d))
        if sum(map(len, cur)) >= N:
            ix = np.concatenate(cur)
            out.append(
                ("+".join(names)[:200], np.sort(rng.choice(ix, N, replace=False)))
            )
            cur, names = [], []
    return out


def packs(P):
    """Donor packs in the same order as pools(): [(donor names, local row indices)] and the donor order."""
    o = P["o"]
    order = np.random.RandomState(0).permutation(sorted(o.donor_id.unique()))
    by = o.groupby("donor_id").indices
    out, cur, names = [], [], []
    for d in order:
        cur.append(by[d])
        names.append(str(d))
        if sum(map(len, cur)) >= N:
            out.append((names, np.concatenate(cur)))
            cur, names = [], []
    return out, order


def raw_ok(f, xk):
    with h5py.File(f, "r") as h:
        g = h[xk]
        v = np.asarray(
            g["data"][:200000] if "data" in g else g[:200].ravel()[:200000],
            dtype=np.float64,
        )
    return bool(len(v) and np.allclose(v, np.round(v)) and v.min() >= 0)


def prep(name):
    """Obs of the atlas after the drops above: dict(o, pos, f, xk, var, lab, fine, ...)."""
    c = ATLASES.loc[name]
    lab, fine = c.label_col, c.fine_col
    f = f"{ROOT}/atlas/{c.dataset_id}.h5ad"
    a = ad.read_h5ad(f, backed="r")
    cols = list(dict.fromkeys([lab, fine, "donor_id"]))
    miss = [x for x in cols if x not in a.obs.columns]
    if miss:
        raise KeyError(f"missing obs columns {miss}")
    o = a.obs[cols].astype(str).copy()
    assert a.obs_names.is_unique, "obs_names not unique"
    keep = np.ones(len(o), bool)
    if "assay" in a.obs.columns:
        keep &= (
            pd.Series(a.obs["assay"].astype(str).values).str.startswith("10x").values
        )
    keep &= ~o.donor_id.isin(DROP_DONORS.get(name, [])).values
    if name in NON_PRIMARY and "is_primary_data" in a.obs.columns:
        keep &= (a.obs["is_primary_data"].astype(str) == "True").values
    for x in (lab, fine):
        keep &= (
            ~o[x]
            .astype(object)
            .fillna("nan")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda s: bool(PLACEHOLDER.match(s)))
            .values
        )
    vv = a.raw.var if a.raw is not None else a.var
    xk = "raw/X" if a.raw is not None else "X"
    names = (
        vv["feature_name"].astype(str).values
        if "feature_name" in vv
        else (
            a.var["feature_name"].reindex(vv.index).astype(str).values
            if "feature_name" in a.var
            else vv.index.astype(str)
        )
    )
    genemap = pd.DataFrame({"gene_id": vv.index.astype(str), "feature_name": names})
    var = vv.index.copy()
    a.file.close()
    return dict(
        name=name,
        f=f,
        xk=xk,
        var=var,
        genemap=genemap,
        lab=lab,
        fine=fine,
        o=o[keep],
        pos=np.where(keep)[0],
    )


def write_pool(P, pn, pid, r):
    """Write pool pn from local rows r: raw counts h5ad plus the pilot table (obs_name, label, true_type, donor)."""
    rows = P["pos"][r]
    with h5py.File(P["f"], "r") as h:
        X = sparse_dataset(h[P["xk"]])[rows]
    X = sp.csr_matrix(X).astype(np.float32)
    assert X.shape[0] == len(rows) and np.allclose(
        X.data, np.round(X.data)
    ), "non-integer counts in pool"
    pp = P["o"].iloc[r]
    os.makedirs(f"{ROOT}/pools", exist_ok=True)
    os.makedirs(f"{ROOT}/pilots", exist_ok=True)
    ad.AnnData(
        X=X, obs=pd.DataFrame(index=pp.index), var=pd.DataFrame(index=P["var"])
    ).write_h5ad(f"{ROOT}/pools/{pn}.h5ad")
    pd.DataFrame(
        {
            "obs_name": pp.index.values,
            "label": pp[P["lab"]].values,
            "true_type": pp[P["fine"]].values,
            "is_rare": 0,
            "donor": pp.donor_id.values,
            "source_path": f"{pn}.h5ad",
        }
    ).to_parquet(f"{ROOT}/pilots/{pid}.parquet", index=False)
    return pp


def build(name, force=False):
    c = ATLASES.loc[name]
    used = pd.read_csv(f"{HERE}/used_datasets.csv")
    problems = []
    if c.dataset_id in set(used.dataset_id):
        problems.append("dataset id is in used_datasets.csv")
    if c.collection_id in set(used.collection_id):
        problems.append("collection id is in used_datasets.csv")
    P = prep(name)
    o, lab, fine = P["o"], P["lab"], P["fine"]
    if not raw_ok(P["f"], P["xk"]):
        problems.append("non-integer or negative values in the stored counts")
    if o.donor_id.nunique() < MIN_DONORS:
        problems.append(f"{o.donor_id.nunique()} donors after drops (< {MIN_DONORS})")
    sel = pools(o, "donor_id", np.random.RandomState(0))
    assert [x[0] for x in sel] == [
        ("+".join(n))[:200] for n, _ in packs(P)[0]
    ], "pack bookkeeping differs from pools()"
    if len(sel) < 2 or len(o) < MIN_CELLS:
        problems.append(
            f"{len(sel)} packs, {len(o)} cells (need >= 2 pools and >= {MIN_CELLS} cells)"
        )
    sel = sel[:MAX_POOLS]
    man, rows = [], []
    for k, (blk, r) in enumerate(sel):
        pp = o.iloc[r]
        t = targets(
            pd.DataFrame({"label": pp[lab].values, "true_type": pp[fine].values})
        )
        man.append(
            (f"atl_{name}_p{k}", name, "test", blk, k, len(r), pp[lab].nunique())
        )
        rows += [(f"atl_{name}_p{k}", L, T, n) for L, T, n in t]
    if sum(1 for m in man if any(x[0] == m[0] for x in rows)) < 2:
        problems.append("fewer than 2 pools with a target")
    if problems and not force:
        print(name, "not built:", problems)
        return False
    for d in ("manifests", "targets", "genemap"):
        os.makedirs(f"{ROOT}/{d}", exist_ok=True)
    P["genemap"].to_csv(f"{ROOT}/genemap/{name}.csv", index=False)
    pd.DataFrame(rows, columns=["pilot_id", "label", "target", "n"]).to_csv(
        f"{ROOT}/targets/targets_{name}.csv", index=False
    )
    for k, (blk, r) in enumerate(sel):
        write_pool(P, f"{name}_p{k}", f"atl_{name}_p{k}", r)
        print(f"atl_{name}_p{k}", len(r), flush=True)
    pd.DataFrame(
        man,
        columns=["pilot_id", "atlas", "split", "block", "pool", "n_cells", "n_labels"],
    ).to_csv(f"{ROOT}/manifests/atlas_manifest_{name}.csv", index=False)
    print(
        json.dumps(
            dict(
                atlas=name,
                cells=len(o),
                donors=int(o.donor_id.nunique()),
                pools=len(sel),
                problems=problems,
            )
        )
    )
    return True


def _check():
    o = pd.DataFrame({"d": ["a"] * 15000 + ["b"] * 15000 + ["c"] * 1000})
    P = pools(o, "d", np.random.RandomState(0))
    assert len(P) == 1 and len(P[0][1]) == N and len(set(P[0][1])) == N
    assert PLACEHOLDER.match("doublet") and not PLACEHOLDER.match("B cell")


if __name__ == "__main__":
    if not sys.argv[1:] or {"-h", "--help"} & set(sys.argv):
        print(__doc__)
        sys.exit()
    _check()
    ok = [
        build(n, "--force" in sys.argv) for n in sys.argv[1:] if not n.startswith("--")
    ]
    sys.exit(0 if all(ok) else 1)
