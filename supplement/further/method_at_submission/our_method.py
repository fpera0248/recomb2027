"""python frozen4.py IDS.txt : our final method, fixed before its test atlases were chosen. Untrained rank combination of two frozen methods:
score = 0.75 * rank_pct(m14p_frozen = the learned scorer) + 0.25 * rank_pct(m1_leiden_sub) -> preds_frozen4/frozen4/<pid>.parquet. Chosen on the 146 earlier
test pools (highest atlas-equal E1_union among w in {0.25, 0.5, 0.75} x {m1, m4}; also highest union_AP and masked_AP among the m1 combos)."""
import sys, os, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
from combo_r2 import combo

if __name__ == "__main__":
    R5 = os.environ.get("R5") or HR_WORK + "/r5/data"; os.makedirs(f"{R5}/preds_frozen4/frozen4", exist_ok=True)
    for pid in [l.split()[0] for l in open(sys.argv[1]) if l.strip()]:
        a = pd.read_parquet(f"{R5}/preds_m14p/m14p_frozen/{pid}.parquet").set_index("cell_id").score
        b = pd.read_parquet(f"{R5}/preds_base/m1_leiden_sub/{pid}.parquet").set_index("cell_id").score
        pd.DataFrame({"cell_id": a.index, "score": combo(a, b, 0.75).values}).to_parquet(f"{R5}/preds_frozen4/frozen4/{pid}.parquet", index=False)
