"""python rank_combination_sweep.py IDS.txt : rank combinations w*rank_pct(m15x_rule) + (1-w)*rank_pct(m11_seprank; the seprank baseline), w in {0.25, 0.5, 0.75},
for every preds_r2/m15* method -> preds_r2c/<m15x_rule>_w<w>/<pid>.parquet. Pools without an m11 prediction are logged and skipped."""
import sys, os, glob, pandas as pd


def combo(a, b, w):
    ra, rb = a.rank(pct=True), b.reindex(a.index).fillna(b.min()).rank(pct=True); return w * ra + (1 - w) * rb


if __name__ == "__main__":
    s = pd.Series([3., 2, 1], index=list("xyz")); assert list(combo(s, s[::-1], 0.5).values) == [1, 2 / 3, 1 / 3]
    ids = [l.split()[0] for l in open(sys.argv[1]) if l.strip()]; miss = set()
    for d in sorted(glob.glob("preds_r2/m15*")):
        for pid in ids:
            f, g = f"{d}/{pid}.parquet", f"preds/m11_seprank/{pid}.parquet"
            if not os.path.exists(g): miss.add(pid); continue
            if not os.path.exists(f): continue
            a = pd.read_parquet(f).set_index("cell_id").score; b = pd.read_parquet(g).set_index("cell_id").score
            for w in (0.25, 0.5, 0.75):
                o = f"preds_r2c/{os.path.basename(d)}_w{w}"; os.makedirs(o, exist_ok=True)
                pd.DataFrame({"cell_id": a.index, "score": combo(a, b, w).values}).to_parquet(f"{o}/{pid}.parquet", index=False)
    print("pools without m11:", len(miss), sorted(miss)[:5])
