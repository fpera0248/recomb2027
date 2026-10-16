"""python qhelp.py CMD ... : small CSV lookups for mini_queue/atac_all.sh. order OUT (eligible dataset ids, kidney regression first, then fragment size ascending) | urls CAND ID (h5ad url, then fragment urls) | humans CAND (id and h5ad url of human datasets) | flags OUT ID (pair override flags)."""
import sys, os, pandas as pd
KID = "09b518f9-da64-44cc-aec8-70a89d55611f"
def main(a):
    c = a[0]
    if c == "order":
        s = pd.read_csv(f"{a[1]}/selection.csv"); s = s[s.eligible.astype(str) == "True"]; f = pd.read_csv(a[2]).set_index("dataset_id").frag_gb
        for d in sorted(s.dataset_id, key=lambda d: (d != KID, f[d])): print(d)
    elif c == "urls":
        r = pd.read_csv(a[1]).set_index("dataset_id").loc[a[2]]; print(r.h5ad_url)
        f = os.path.join(os.environ.get("OUT", ""), "selection.csv")   # no fragment download for a dataset made ineligible after the run order was fixed
        if os.environ.get("OUT") and os.path.exists(f) and str(pd.read_csv(f).set_index("dataset_id").eligible.get(a[2], True)) != "True": return
        [print(u) for u in r.frag_urls.split()]
    elif c == "humans": [print(r.dataset_id, r.h5ad_url) for r in pd.read_csv(a[1]).itertuples() if r.organism == "Homo sapiens"]
    elif c == "flags":
        f = f"{a[1]}/pair_override.csv"
        if os.path.exists(f):
            o = pd.read_csv(f); o = o[o.dataset_id == a[2]]
            if len(o): print(f"--coarse {o.coarse.iloc[0]} --fine {o.fine.iloc[0]}")
def demo():
    import tempfile; d = tempfile.mkdtemp(); pd.DataFrame(dict(dataset_id=["x", KID, "y"], eligible=[True, True, False])).to_csv(f"{d}/selection.csv", index=False)
    pd.DataFrame(dict(dataset_id=["x", KID, "y"], frag_gb=[1, 9, 0.5])).to_csv(f"{d}/c.csv", index=False)
    import io, contextlib; b = io.StringIO()
    with contextlib.redirect_stdout(b): main(["order", d, f"{d}/c.csv"])
    assert b.getvalue().split() == [KID, "x"]
if __name__ == "__main__":
    if len(sys.argv) == 1: demo(); print("demo ok")
    else: main(sys.argv[1:])
