"""Twin pools for the null test: pools drawn from the same donors as a real pool, with the
finer subtypes that could act as targets removed, so a call on a twin is a false call.

    python -m data.build_twins NAME [NAME ...]
The real pools of NAME must exist under $HR_DATA/pilots. Only labels are used.

Per real pool (donor pack):
  1. drop every cell of a (label L, finer type T) pair with T != L, T not a placeholder, unless it is
     large (at least MIN_EXP cells scaled to 20,000) and at least 10% of its label;
  2. add donors from the shuffled donor order until 20,000 cells are kept;
  3. sample 20,000 cells; if the sample still has a target-sized pair (10 to 199 cells, or 10 or more
     and under 10% of the label), remove that pair and redraw, at most 3 times.
A twin is dropped if more than 25% of its pack was removed or it has fewer than 5 labels.
Writes pools/twin_<name>_p<k>.h5ad, pilots/atl_twin_<name>_p<k>.parquet, manifests/atlas_manifest_twin_<name>.csv
and twin_log_<name>.csv.
"""
import sys

import numpy as np
import pandas as pd

from eval.targets import BAD, targets

from .build_pools import MAX_POOLS, N, ROOT, packs, pools, prep, write_pool

MIN_EXP, SHARE, LIM, MINLAB = 230, 0.10, 0.25, 5


def type_table(o, ix, lab, fine):
    df = pd.DataFrame({"l": o[lab].values[ix], "t": o[fine].values[ix]})
    return df, df.groupby(["l", "t"]).size(), df.groupby("l").size()


def keep_mask(o, ix, lab, fine):
    """Step 1 on the pack rows ix."""
    df, nlt, nl = type_table(o, ix, lab, fine)
    n = len(ix)
    c = nlt.reindex(pd.MultiIndex.from_arrays([df.l, df.t])).values
    s = c / nl.reindex(df.l).values
    return (df.t.values == df.l.values) | df.t.isin(BAD).values | ((c * N / n >= MIN_EXP) & (s >= SHARE))


def offenders(o, ix, lab, fine):
    """Step 3 on a sample: target-sized (L, T) pairs."""
    _, nlt, nl = type_table(o, ix, lab, fine)
    return [(l, t) for (l, t), n in nlt.items() if t != l and t not in BAD and n >= 10 and (n <= 199 or n / nl[l] < SHARE)]


def twin(P, order, pk, k):
    """One twin for pool k. Returns dict(status, reason, ..., rows=sample row indices)."""
    o, lab, fine = P["o"], P["lab"], P["fine"]
    by = o.groupby("donor_id").indices
    names = list(pk[k][0])
    pos = list(order).index(names[-1]) + 1
    ix = np.concatenate([by[d] for d in names])
    off = set()
    log = dict(pool=k, n_pack0=len(ix), added=0, passes=0, status="", reason="")
    km = keep_mask(o, ix, lab, fine)
    log["removed_step1"] = int((~km).sum())

    def refresh():
        """Add donors until at least N cells are kept; pairs already removed stay removed."""
        nonlocal ix, km, pos
        while True:
            km = keep_mask(o, ix, lab, fine)
            if off:
                km &= ~np.isin(o[lab].values[ix] + "\x00" + o[fine].values[ix], [l + "\x00" + t for l, t in off])
            if km.sum() >= N:
                return True
            if pos >= len(order):
                return False
            ix = np.concatenate([ix, by[order[pos]]])
            pos += 1
            log["added"] += 1

    if not refresh():
        return dict(log, status="DROP", reason=f"fewer than {N} kept cells after adding all donors")
    for ps in range(4):
        smp = np.sort(np.random.RandomState(1000 + k + 100 * ps).choice(ix[km], N, replace=False))
        bad = offenders(o, smp, lab, fine)
        log["passes"] = ps
        if not bad:
            break
        if ps == 3:
            return dict(log, status="DROP", reason=f"still {len(bad)} target-sized pairs after the last redraw", n_pack=len(ix), n_kept=int(km.sum()))
        off |= set(bad)
        if not refresh():
            return dict(log, status="DROP", reason=f"fewer than {N} kept cells after removing pairs and adding all donors", n_pack=len(ix))
    removed = len(ix) - int(km.sum())
    nlab = o[lab].iloc[smp].nunique()
    log.update(n_pack=len(ix), n_kept=int(km.sum()), removed_total=removed, removed_frac=round(removed / len(ix), 4), offenders=len(off), n_labels=nlab)
    if removed > LIM * len(ix):
        return dict(log, status="DROP", reason=f"removed {removed} > 25% of pack {len(ix)}")
    if nlab < MINLAB:
        return dict(log, status="DROP", reason=f"{nlab} labels (< {MINLAB})")
    return dict(log, status="OK", reason="", rows=smp)


def build_twins(name):
    P = prep(name)
    o, lab = P["o"], P["lab"]
    pk, order = packs(P)
    allp = pools(o, "donor_id", np.random.RandomState(0))
    L, man = [], []
    for k in range(min(len(pk), MAX_POOLS)):
        pid = f"atl_{name}_p{k}"
        real = pd.read_parquet(f"{ROOT}/pilots/{pid}.parquet", columns=["obs_name", "label", "true_type"]).astype(str)
        assert list(o.index[allp[k][1]]) == list(real.obs_name), f"{pid}: rebuilt real pool differs from the one on disk"
        if not targets(real):
            L.append(dict(pool=k, status="SKIP", reason="real pool has no target"))
            continue
        r = twin(P, order, pk, k)
        rows = r.pop("rows", None)
        L.append(r)
        if rows is not None:
            tp = f"twin_{name}_p{k}"  # the pool id seeds the cell ids (hiddenrare.io.opaque_order): another prefix gives other ids
            write_pool(P, tp, f"atl_{tp}", rows)
            man.append((f"atl_{tp}", name, "twin", pid, k, len(rows), o[lab].iloc[rows].nunique()))
            print(f"atl_{tp}", len(rows), flush=True)
    pd.DataFrame(man, columns=["pilot_id", "atlas", "split", "block", "pool", "n_cells", "n_labels"]).to_csv(f"{ROOT}/manifests/atlas_manifest_twin_{name}.csv", index=False)
    d = pd.DataFrame(L)
    d.insert(0, "atlas", name)
    d.to_csv(f"{ROOT}/twin_log_{name}.csv", index=False)
    print(d.to_string())


def _check():
    n = 20000
    o = pd.DataFrame({"l": ["a"] * n, "t": ["a"] * 15000 + ["x"] * 3000 + ["y"] * 150 + ["z"] * 5 + ["Unknown"] * 1845, "donor_id": "d"})
    km = keep_mask(o, np.arange(n), "l", "t")       # x: 3000 cells, 15% of the label, kept; y (150) and z (5) removed; Unknown and T == L kept
    assert km.sum() == 15000 + 3000 + 1845 and not km[18000:18155].any()
    assert offenders(o, np.arange(n), "l", "t") == [("a", "y")]    # y is target-sized; z has fewer than 10 cells; x is large


if __name__ == "__main__":
    _check()
    [build_twins(n) for n in sys.argv[1:]]
