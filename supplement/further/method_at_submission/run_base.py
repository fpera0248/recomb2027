"""python run_base.py PILOT_ID -> si/preds_base/<method>/<pid>.parquet for m1-m8 (unchanged methods.py, CPU only).."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SI); R5 = os.environ.get("R5") or os.path.expanduser("~/work/r5/data")
import numpy as np, pandas as pd
import loader, methods as M
al = lambda v: v.reindex(a.obs_names).values if isinstance(v, pd.Series) else np.asarray(v)   # align by cell id
pid = sys.argv[1]; P = f"{R5}/preds_base"
todo = [m for m in M.METHODS if not os.path.exists(f"{P}/{m}/{pid}.parquet")]
if not todo: sys.exit(0)
a, _true = loader.load_pilot(pid)          # truth never passed to a method
for m in todo:
    p = M.METHODS[m](a.copy()); gid = pd.Series(-1, index=a.obs_names)
    for i, g in enumerate(p["groups"]): gid.loc[g] = i
    d = f"{P}/{m}"; os.makedirs(d, exist_ok=True)
    pd.DataFrame({"cell_id": a.obs_names.values, "score": al(p["score"]).astype(float), "call": al(p["call"]).astype(bool), "group": gid.values}).to_parquet(f"{d}/{pid}.parquet.tmp", index=False)
    os.rename(f"{d}/{pid}.parquet.tmp", f"{d}/{pid}.parquet")
print(pid, a.n_obs, len(todo), flush=True)
