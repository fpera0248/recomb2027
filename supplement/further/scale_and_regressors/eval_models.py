"""Evaluate every fitted model in out/w1/models on set B (57 pools) and set A (69 pools): sc (scorer alone) and bl (blend with Leiden subclustering), union AP and E1_union per pool.
Resumable: out/w1/eval/<tag>.csv per model."""
import os, sys, pickle, glob, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h, pandas as pd
O = f"{h.H}/jiaqi1011/out/w1"; os.makedirs(f"{O}/eval", exist_ok=True)
for jf in sorted(glob.glob(f"{O}/models/*.json")):
    tag = os.path.basename(jf)[:-5]; out = f"{O}/eval/{tag}.csv"
    if os.path.exists(out): continue
    M = pickle.load(open(f"{O}/models/{tag}.pkl", "rb")); rows = []
    for S in ("B", "A"):
        for p in h.pools(S): r = h.evaluate(M.predict, p, S)[0]; r["set"] = S; rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False); print(tag, flush=True)
