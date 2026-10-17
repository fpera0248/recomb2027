"""Writes out/w1/storage_pools.csv: per set-B test pool, bytes of raw slim h5ad and derived candidate/prediction files (read-only on ~/work/r5/data)."""
import os, glob, sys, pandas as pd; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import harness as h
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
D = HR_WORK + "/r5/data"; sz = lambda f: os.path.getsize(f) if os.path.exists(f) else 0
rows = [dict(pid=p, raw_h5ad=sz(f"{D}/slim_atlas/{p[4:]}.h5ad"), **{k: sz(f"{D}/cands/{p}.{k}.parquet") for k in ("markers", "halo", "groups", "cells")}, preds=sum(sz(f) for f in glob.glob(f"{D}/preds*/*/{p}.parquet"))) for p in h.pools("B")]
d = pd.DataFrame(rows); d.to_csv(HR_WORK + "/jiaqi1011/out/w1/storage_pools.csv" if "--write" in sys.argv else "/dev/stdout", index=False)
