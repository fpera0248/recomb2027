"""Group scorer: gradient boosting on the 41 group features, trained offline on pools with known truth.

The target of a training group is its purity: the share of its members that belong to a hidden
type. A cell's score is the maximum over its groups of P^4 / size, where P is the predicted purity.

    python -m hiddenrare.scorer predict MODEL.pkl CAND_DIR POOL_ID OUT.parquet
    python -m hiddenrare.scorer fit MODEL.pkl CAND_DIR PILOT_DIR POOL_IDS.txt
fit needs the candidate tables of each training pool (candidates, markers and halo stages) and
its pilot table with the truth (is_rare for benchmark pools, label and true_type for natural pools).
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .candidates import K
from .io import opaque_order

POWER, SIZE_EXP = 4, 1.0  # cell score = P**POWER / size**SIZE_EXP
GEOMETRY = [
    "lsize",
    "lshare",
    "dprime",
    "cohesion",
    "coh_corr",
    "stab",
    "lres",
    "space_l",
    "ldonor",
    "donor_excess",
    "dp_pct",
    "coh_pct",
    "dp_z",
    "coh_z",
]
MARKER = ["mk_top1", "mk_top5", "mk_top20", "mk_n40", "mk_n25", "mk_lr5", "mk_neg5"]
HALO = ["hl_top5", "hl_top20", "hl_n25", "hl_mk5", "hl_neg5", "hl_ratio"]


def stability(G):
    """Number of other resolutions (same space and label) with a group of Jaccard >= 0.5 to this one."""
    st = np.zeros(len(G), int)
    for _, g in G.groupby(["space", "label"]):
        sets = [set(m) for m in g.members]
        res, ix = g.res.values, g.index.values
        for a in range(len(ix)):
            hit = {
                res[b]
                for b in range(len(ix))
                if res[b] != res[a]
                and len(sets[a] & sets[b]) >= 0.5 * len(sets[a] | sets[b])
            }
            st[ix[a]] = len(hit)
    return st


def featurize(G):
    X = pd.DataFrame(index=G.index)
    n_l = G["size"] / G.share
    X["lsize"] = np.log(G["size"])
    X["lshare"] = np.log(G.share)
    X["dprime"] = G.dprime
    X["cohesion"] = G.cohesion
    base = (G["size"] - 1) / (n_l - 1)
    cmax = np.minimum(K, G["size"] - 1) / K
    X["coh_corr"] = (G.cohesion - base) / np.maximum(cmax - base, 1e-9)
    X["stab"] = G.stab
    X["lres"] = np.log(G.res)
    X["space_l"] = (G.space == "l").astype(int)
    X["ldonor"] = np.log(G.n_donor)
    X["donor_excess"] = G.top_donor - G.parent_top_donor
    for c, s in (("dprime", "dp"), ("cohesion", "coh")):
        X[f"{s}_pct"] = G[c].rank(pct=True)
        X[f"{s}_z"] = (G[c] - G[c].median()) / (
            G[c].quantile(0.75) - G[c].quantile(0.25) + 1e-9
        )
    for c in MARKER + HALO:
        X[c] = G[c]
        X[c + "_pct"] = G[c].rank(pct=True)
    X["hl_keep"] = G.hl_mk5 / np.maximum(G.mk_top5, 1e-3)
    cols = (
        GEOMETRY
        + MARKER
        + [c + "_pct" for c in MARKER]
        + HALO
        + [c + "_pct" for c in HALO]
        + ["hl_keep"]
    )
    return X[cols].values


def load_groups(cand_dir, pid):
    """Groups with their marker and halo features, or None if a stage is missing."""
    parts = ["markers", "halo"]
    if not all(os.path.exists(f"{cand_dir}/{pid}.{p}.parquet") for p in parts):
        return None
    G = pd.read_parquet(f"{cand_dir}/{pid}.groups.parquet")
    G["stab"] = stability(G)
    return pd.concat(
        [G] + [pd.read_parquet(f"{cand_dir}/{pid}.{p}.parquet") for p in parts], axis=1
    )


def cell_score(G, group_scores, C):
    """Max over a cell's groups of the group's rank-scaled score; cells in no group get -1. The isolation rank only breaks ties."""
    gs = np.asarray(group_scores, float)
    keep = np.isfinite(gs)
    gs = pd.Series(gs[keep]).rank(method="dense").values / max(keep.sum(), 1)
    s = np.full(len(C), -np.inf)
    for m, v in zip(G.members.values[keep], gs):
        np.maximum.at(s, m, v)
    return np.where(np.isfinite(s), s, -1.0) + 1e-6 * C["tb_g"].values


def predict(model_path, cand_dir, pid, out):
    m = pickle.load(open(model_path, "rb"))
    G = load_groups(cand_dir, pid)
    C = pd.read_parquet(f"{cand_dir}/{pid}.cells.parquet")
    P = np.clip(m.predict(featurize(G)), 0, 1)
    s = cell_score(G, P**POWER / G["size"].values ** SIZE_EXP, C)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pd.DataFrame({"cell_id": C.cell_id.values, "score": s}).to_parquet(out, index=False)


def purity_targets(pid, cell_ids, pilot_dir):
    """Per-cell target index (-1 none) and target sizes. Benchmark pools: the hidden type. Natural pools: every target of eval.targets."""
    from eval.targets import targets

    df = pd.read_parquet(f"{pilot_dir}/{pid}.parquet")
    order, ids = opaque_order(pid, df.obs_name.values)
    df = df.iloc[order].set_index(ids).reindex(cell_ids)
    if df.is_rare.astype(bool).any():
        t = np.where(df.is_rare.astype(bool), 0, -1)
        return t, np.array([(t == 0).sum()])
    df = df.astype({"label": str, "true_type": str})
    t = np.full(len(df), -1)
    T = targets(df)
    for i, (lab, typ, _) in enumerate(T):
        t[((df.label == lab) & (df.true_type == typ)).values] = i
    return t, np.array([n for _, _, n in T], int)


def fit(model_path, cand_dir, pilot_dir, pool_ids):
    X, y = [], []
    for pid in pool_ids:
        if not os.path.exists(f"{cand_dir}/{pid}.groups.parquet"):
            continue
        G = load_groups(cand_dir, pid)
        if G is None:
            continue
        C = pd.read_parquet(f"{cand_dir}/{pid}.cells.parquet")
        t, nt = purity_targets(pid, C.cell_id.values, pilot_dir)
        if not len(nt):
            continue
        pur = np.zeros(len(G))
        for j, m in enumerate(G.members.values):
            pur[j] = np.bincount(t[m][t[m] >= 0], minlength=len(nt)).sum() / len(m)
        X.append(featurize(G))
        y.append(pur)
    m = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=0,
    )
    m.fit(np.vstack(X), np.concatenate(y))
    pickle.dump(m, open(model_path, "wb"))
    print("fit on", len(X), "pools,", sum(map(len, y)), "groups")


if __name__ == "__main__":
    ids = lambda f: [l.split()[0] for l in open(f) if l.strip()]
    if sys.argv[1] == "predict":
        predict(*sys.argv[2:6])
    elif sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3], sys.argv[4], ids(sys.argv[5]))
