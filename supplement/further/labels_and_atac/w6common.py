"""W6 shared paths and pool list. Set B = manifest split test5, pool index < 3 (29 pools)."""
import os, glob, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
HOME = os.path.expanduser("~"); CLEAN = f"{HR_WORK}/r5/data"; W = os.environ.get("W6_DATA", f"{HR_WORK}/jiaqi1011/data/w6")
OUT = f"{HR_WORK}/jiaqi1011/out/w6_annot_atac"; CODE_R5 = f"{HR_WORK}/r5/code"
RATES = {"05": .05, "10": .10, "20": .20}
VARIANTS = ["ID"] + [f"R{r}s{s}" for r in RATES for s in (0, 1)] + [f"B{r}" for r in RATES] + ["MERGE", "KNNX", "KNNXu", "KNNT10", "KNNT2"]
def pools():
    M = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{CLEAN}/atlas_manifest_*.csv"))])
    M = M[(M.split == "test5") & (M.pool < 3)].sort_values(["pool", "atlas"]); return M[["pilot_id", "atlas", "pool"]]
def root(v): return f"{W}/{v}"
