"""Marker-specificity features for every sr3 candidate group (dev). Real cell types carry genes detected in most of their
cells and few cells of the rest of their label; over-split fragments of one population do not. Blind: uses counts + visible labels only.
Detection = count > 0, on genes detected in >= 1% of the pilot's cells. delta_g = pct_in(g) - pct_out(g), out = rest of the same label.
  python sr3m.py PILOT_ID -> $SI/cands/<pid>.markers.parquet (row-aligned with <pid>.groups.parquet)"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): os.environ.setdefault(v, "2")
SI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SI); R5 = os.environ.get("R5") or os.path.expanduser("~/work/r5/data")
import numpy as np, pandas as pd, scipy.sparse as sp
import loader


def markers(a, G):
    X = sp.csr_matrix(a.X); X.data = (X.data > 0).astype(np.float32); X.eliminate_zeros()
    keep = np.asarray(X.mean(0)).ravel() >= 0.01; X = X[:, keep].tocsc().tocsr()
    labels = a.obs["label"].astype(str).values; n = a.n_obs
    lab_ix = {l: np.where(labels == l)[0] for l in pd.unique(labels)}
    lab_sum = {l: np.asarray(X[ix].sum(0)).ravel() for l, ix in lab_ix.items()}
    rows, cols = np.concatenate(G.members.values), np.repeat(np.arange(len(G)), G["size"].values)
    S = (sp.csr_matrix((np.ones(len(rows), np.float32), (cols, rows)), shape=(len(G), n)) @ X).toarray()   # groups x genes detected counts
    out = []
    for j, (l, m) in enumerate(zip(G.label.values, G["size"].values)):
        nl = len(lab_ix[l]); pin = S[j] / m; pout = (lab_sum[l] - S[j]) / max(nl - m, 1); d = np.sort(pin - pout)[::-1]
        lr = np.sort(np.log((pin + 0.01) / (pout + 0.01)))[::-1]
        out.append(dict(mk_top1=d[0], mk_top5=d[:5].mean(), mk_top20=d[:20].mean(), mk_n40=int((d >= 0.4).sum()), mk_n25=int((d >= 0.25).sum()),
                        mk_lr5=lr[:5].mean(), mk_neg5=-np.sort(pin - pout)[:5].mean()))
    return pd.DataFrame(out)


def run(pid):
    d = os.environ.get("SI_CANDS", f"{R5}/cands"); out = f"{d}/{pid}.markers.parquet"
    if os.path.exists(out): return
    t0 = time.time(); a, _true = loader.load_pilot(pid)          # truth is never passed to the method
    G = pd.read_parquet(f"{d}/{pid}.groups.parquet", columns=["label", "size", "members"])
    M = markers(a, G); M.to_parquet(out + ".tmp", index=False); os.rename(out + ".tmp", out)
    print(pid, len(G), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
