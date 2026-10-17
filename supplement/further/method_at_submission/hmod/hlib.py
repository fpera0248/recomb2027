"""load_pilot(pilot_id) -> (AnnData, true_type) for a hidden-rare pilot built
by stage3_build.py / manifest.csv. The AnnData carries only what a blind
detection method is allowed to see (raw counts, visible label, donor); the
true cell type comes back as a separate Series so nothing passes it by
accident. See h/manifest.csv for pilot_id -> (design, dataset, rare_type, ...).

Leaks fixed here:
 - raw_adata() returns the source slice UNCHANGED
   (including its obsm/uns/varm/layers/var columns) whenever the source h5ad
   has no .raw. Fixed: load_pilot always strips those slots itself.
 - stage3_build.py writes every pilot parquet
   sorted by TRUE type (D1: [m1 block][m2 block][rare block], always in that
   order; D2: one contiguous block per true type), AND obs_names are the
   source h5ad's own row index, so a source file sorted by cell type would
   let obs_name itself encode truth. A FIXED-SEED shuffle (this file's
   previous fix) does not close this: RandomState(seed).permutation(n)
   depends only on n and the seed, so it is the SAME permutation for every
   pilot of a given size and is trivially invertible by anyone who knows
   hlib's seed -- it still lets a method's own code exploit row position.
   Fixed properly: rows are reordered by sort(sha1(pilot_id + source
   obs_name)), a permutation that depends on data a method never sees (the
   source obs_name) combined with the pilot_id, so it cannot be recomputed
   or inverted from anything a method has access to. obs_names are replaced
   with the first 16 hex chars of that same hash (opaque ids) -- the
   source obs_name is never returned. The (opaque id -> source obs_name,
   truth) mapping lives only in this function's locals and in the
   true_type Series this function hands back for evaluation; it is never
   attached to the AnnData a method receives.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import scanpy as sc
from rawutil import raw_adata

R5 = os.environ.get("R5") or os.path.expanduser("~/work/r5/data")
H = os.path.join(R5, "hmod")
PILOTS_DIR = os.path.join(H, "pilots")


def _opaque_order(pilot_id, source_obs_names):
    """sort(sha1(pilot_id + '\\x00' + source_obs_name))[:16] per row.
    Returns (sort_idx, opaque_id_in_sorted_order). Content-derived, so it is
    a different permutation per pilot and cannot be inverted without the
    source obs_name a method never sees."""
    digests = [hashlib.sha1(f"{pilot_id}\x00{on}".encode()).hexdigest() for on in source_obs_names]
    sort_idx = np.argsort(digests)
    opaque_id = np.array([digests[i][:16] for i in sort_idx])
    assert len(set(opaque_id)) == len(opaque_id), f"{pilot_id}: opaque id collision"
    return sort_idx, opaque_id


def load_pilot(pilot_id):
    df = pd.read_parquet(os.path.join(PILOTS_DIR, f"{pilot_id}.parquet"))
    source_path = df["source_path"].iloc[0]

    a = sc.read_h5ad(source_path, backed="r")
    pos = a.obs_names.get_indexer(df["obs_name"].values)
    if (pos < 0).any():
        a.file.close()
        raise KeyError(f"{pilot_id}: {int((pos < 0).sum())} obs_names missing from {source_path}")
    order = np.argsort(pos)  # sorted positions for fast backed h5py access
    sub = raw_adata(a[pos[order]].to_memory())
    a.file.close()
    sub = sub[np.argsort(order)].copy()  # back to df's row order (still truth-sorted here)

    # strip every slot except X/var_names: whatever the source h5ad carried
    # in obsm/varm/obsp/uns/layers/extra var columns must never reach a
    # method, whether or not raw_adata() already dropped it.
    sub.obsm.clear()
    sub.varm.clear()
    sub.obsp.clear()
    sub.uns.clear()
    for k in list(sub.layers.keys()):
        if k is not None:  # this anndata version aliases X itself as layers[None]
            del sub.layers[k]
    sub.var = pd.DataFrame(index=sub.var_names)

    # reorder to the opaque, content-derived permutation and replace
    # obs_names with opaque ids -- both the row order AND the index are now
    # independent of the parquet's truth-sorted order and of the source
    # obs_name. This must happen before obs/true_type are built so every
    # downstream array lines up with the same (sort_idx, opaque_id) pair.
    sort_idx, opaque_id = _opaque_order(pilot_id, df["obs_name"].values)
    sub = sub[sort_idx].copy()
    sub.obs_names = opaque_id

    sub.obs = pd.DataFrame({"label": df["label"].values[sort_idx],
                             "donor": df["donor"].values[sort_idx]}, index=sub.obs_names)
    true_type = pd.Series(df["true_type"].values[sort_idx], index=sub.obs_names, name="true_type")

    return sub, true_type


def list_pilots():
    return pd.read_csv(os.path.join(H, "manifest.csv"))["pilot_id"].tolist()


def demo():
    """Smallest runnable check: load one real pilot, verify the blind
    contract (only label/donor in obs, raw integer counts, rare count
    matches the manifest, no leftover obsm/uns/layers) AND the opaque-order
    fix: obs_names are exactly sort(sha1(pilot_id+source_obs_name))[:16],
    never a source obs_name, with no row/label misalignment introduced by
    the resort."""
    man = pd.read_csv(os.path.join(H, "manifest.csv"))
    pid = man["pilot_id"].iloc[len(man) // 2]
    a, true_type = load_pilot(pid)
    assert a.n_obs == len(true_type)
    assert set(a.obs.columns) == {"label", "donor"}
    assert list(a.obsm.keys()) == [] and list(a.uns.keys()) == [], "no obsm/uns may survive load_pilot"
    assert [k for k in a.layers.keys() if k is not None] == [], \
        "no extra layers (beyond X itself) may survive load_pilot"
    dense = a.X[:20].toarray() if hasattr(a.X, "toarray") else np.asarray(a.X[:20])
    assert np.allclose(dense, np.round(dense)) and (dense >= 0).all(), "raw counts must be non-negative integers"
    row = man[man.pilot_id == pid].iloc[0]
    assert int((true_type == row["rare_type"]).sum()) == row["n_rare"]
    assert row["rare_type"] not in set(a.obs["label"].unique()), \
        "visible label must never equal the true rare type"

    # opaque-order check: recompute the sha1 order independently from the
    # parquet and confirm load_pilot returned exactly that id set/order,
    # that no id equals a source obs_name, and that truth survived the
    # resort aligned to the right row (not shifted/shuffled wrong).
    df = pd.read_parquet(os.path.join(PILOTS_DIR, f"{pid}.parquet"))
    expect_sort_idx, expect_id = _opaque_order(pid, df["obs_name"].values)
    assert np.array_equal(a.obs_names.values, expect_id), \
        "row order/ids must be exactly sort(sha1(pilot_id+obs_name))[:16] -- not the parquet's truth order"
    assert not (set(a.obs_names) & set(df["obs_name"].astype(str))), \
        "opaque ids must never equal a source obs_name"
    expect_true = df["true_type"].values[expect_sort_idx]
    assert np.array_equal(true_type.values, expect_true), \
        "truth misaligned after opaque resort"

    print(f"hlib self-check OK on {pid}: {a.n_obs} cells, {row['n_rare']} rare, design={row['design']}, "
          f"rows/ids = sort(sha1(pilot_id+obs_name))[:16], no stray obsm/uns/layers")


if __name__ == "__main__":
    demo()
