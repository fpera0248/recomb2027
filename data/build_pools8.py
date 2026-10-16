"""Build the test pools of the 8-atlas test set (set test8 of ATLAS_MANIFEST.csv): 70 pools, 69 of them with a target.

    python -m data.build_pools8 NAME [NAME ...]
NAME is the name column of atlases8.csv. The atlas is read from $HR_DATA/atlas/<dataset_id>.h5ad
(from fetch_data.sh full8 "$HR_DATA"). Only obs, raw counts and the two annotation columns are used.

This is the engine of the original run (supplement/further/method_at_submission/build_atlas5.py with the round-4 settings of build_atlas6.py, which is not in the repository: the columns in atlases8.csv and DROP/CELLDROP below), not
the one of data/build_pools.py: no assay, placeholder-label or minimum-donor filter. Only the donors in DROP
(reused from other studies or from datasets used for training) and the cells in CELLDROP are removed.
Donors are shuffled (RandomState(0)) and packed until a pack holds 20,000 cells (data.build_pools.pools, the
same code); 20,000 cells are sampled from each pack; at most 24 packs per atlas, in shuffled order.
Writes the layout of data/build_pools.py under $HR_DATA: pools/<name>_p<k>.h5ad, pilots/atl_<name>_p<k>.parquet,
manifests/atlas_manifest_<name>.csv, targets/targets_<name>.csv, genemap/<name>.csv.
Check the result with python -m eval.check_pools8 $HR_DATA.
"""

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

from data import build_pools as bp

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_POOLS = 24
bp.ROOT = os.environ.get("HR_DATA", "work8")  # own default, so the two test sets never share a directory
ATLASES8 = pd.read_csv(f"{HERE}/atlases8.csv", dtype=str).set_index("name")

# donors reused from other studies or shared with training or used atlases
DROP = {
    "ocular": "BCM_22_0200 BCM_22_0389 BCM_22_0434 BCM_22_0458 BCM_22_0500 BCM_22_0769 BCM_22_0784 BCM_22_0890 BCM_23_0131 BCM_23_0358".split(),
    "psccolon": "411c 414c 417c 432c 440c 411C 414C 417C 432C 440C HC-2".split(),
    "hypomap": ["siletti_H18.30.002", "siletti_H19.30.001", "siletti_H19.30.002"],
}
CELLDROP = {  # doublets, unassigned and unknown cells (same columns as label_col and fine_col of atlases8.csv)
    "hspc": lambda o: o["cell state"].isin(["Doublet", "unassigned"])
    | (o.cell_type == "unknown")
}


def prep(name):
    """Like data.build_pools.prep() but with the drops of the original run: dict(o, pos, f, xk, var, genemap, lab, fine, ...)."""
    c = ATLASES8.loc[name]
    lab, fine = c.label_col, c.fine_col
    f = f"{bp.ROOT}/atlas/{c.dataset_id}.h5ad"
    a = ad.read_h5ad(f, backed="r")
    cols = list(dict.fromkeys([lab, fine, "donor_id"]))
    miss = [x for x in cols if x not in a.obs.columns]
    if miss:
        raise KeyError(f"missing obs columns {miss}")
    assert a.obs_names.is_unique, "obs_names not unique"
    o = a.obs[cols].astype(str).copy()
    keep = ~o.donor_id.isin(DROP.get(name, [])).values
    if name in CELLDROP:
        keep &= ~CELLDROP[name](o).values
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


def build(name):
    P = prep(name)
    o, lab, fine = P["o"], P["lab"], P["fine"]
    assert bp.raw_ok(P["f"], P["xk"]), "non-integer or negative values in the stored counts"
    sel = bp.pools(o, "donor_id", np.random.RandomState(0))
    print(name, len(sel), "pools possible; keeping", min(len(sel), MAX_POOLS), flush=True)
    sel = sel[:MAX_POOLS]
    man, rows = [], []
    for k, (blk, r) in enumerate(sel):
        pp = o.iloc[r]
        t = bp.targets(
            pd.DataFrame({"label": pp[lab].values, "true_type": pp[fine].values})
        )
        man.append((f"atl_{name}_p{k}", name, "test", blk, k, len(r), pp[lab].nunique()))
        rows += [(f"atl_{name}_p{k}", L, T, n) for L, T, n in t]
    for d in ("manifests", "targets", "genemap"):
        os.makedirs(f"{bp.ROOT}/{d}", exist_ok=True)
    P["genemap"].to_csv(f"{bp.ROOT}/genemap/{name}.csv", index=False)
    pd.DataFrame(rows, columns=["pilot_id", "label", "target", "n"]).to_csv(
        f"{bp.ROOT}/targets/targets_{name}.csv", index=False
    )
    for k, (blk, r) in enumerate(sel):
        bp.write_pool(P, f"{name}_p{k}", f"atl_{name}_p{k}", r)
        print(f"atl_{name}_p{k}", len(r), flush=True)
    pd.DataFrame(
        man,
        columns=["pilot_id", "atlas", "split", "block", "pool", "n_cells", "n_labels"],
    ).to_csv(f"{bp.ROOT}/manifests/atlas_manifest_{name}.csv", index=False)


if __name__ == "__main__":
    if not sys.argv[1:] or {"-h", "--help"} & set(sys.argv):
        print(__doc__)
        sys.exit()
    bp._check()
    for n in sys.argv[1:]:
        build(n)
