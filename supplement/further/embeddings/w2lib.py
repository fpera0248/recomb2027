"""W2 shared helpers: data root, representation patch, embedding IO. Locked code in ~/work/r5/code is imported read-only, never edited.
The swap is ONE monkeypatch of methods.preprocess_pca (sr3 g and l spaces, sr3h halo centroid and m1 all call it); no locked file is changed."""
import os, sys, time, csv, shutil, numpy as np, pandas as pd
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
HOME = os.path.expanduser("~")
R5D = os.environ.get("R5SRC", f"{HR_WORK}/r5/data")                  # locked set B data (read only)
CODE = os.environ.get("R5CODE", f"{HR_WORK}/r5/code")                 # locked code (read only)
ROOT = os.environ.get("W2_ROOT", f"{HR_WORK}/jiaqi1011/data/w2_fm")   # NEW data root: hmod copy + symlinks to inputs
OUT = os.environ.get("W2_OUT", f"{HR_WORK}/jiaqi1011/out/w2_fm")
POOLS = os.environ.get("W2_POOLS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pools_phase1.txt")


def ensure_root():
    """hmod/ = copy of the locked hlib.py etc. (identical to code/hmod), pilots + slim_atlas symlinked from the locked data (read only use)."""
    h = f"{ROOT}/hmod"; os.makedirs(h, exist_ok=True)
    import filecmp                                                        # locked runs imported hmod from R5D/hmod (env.sh SI_H): copy from there, and stop if code/hmod differs
    for f in os.listdir(f"{R5D}/hmod"):
        if f.endswith(".py"):
            if os.path.exists(f"{CODE}/hmod/{f}"): assert filecmp.cmp(f"{CODE}/hmod/{f}", f"{R5D}/hmod/{f}", shallow=False), f"{f}: code/hmod differs from data/hmod"
            if not os.path.exists(f"{h}/{f}"): shutil.copy(f"{R5D}/hmod/{f}", f"{h}/{f}")
    for src, dst in ((f"{R5D}/hmod/pilots", f"{h}/pilots"), (f"{R5D}/slim_atlas", f"{ROOT}/slim_atlas")):
        if not os.path.lexists(dst): os.symlink(src, dst)
    return ROOT


def env():
    """call BEFORE importing loader / sr3 / methods."""
    ensure_root(); os.environ.update(R5=ROOT, SI_H=f"{ROOT}/hmod", SLIM_DIR=f"{ROOT}/slim_atlas", PYTHONDONTWRITEBYTECODE="1")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): os.environ.setdefault(v, "3")
    sys.dont_write_bytecode = True
    if CODE not in sys.path: sys.path.insert(0, CODE)


def pools():
    return [l.split()[0] for l in open(POOLS) if l.strip()]


def arm_dir(arm): return f"{OUT}/arms/{arm}"
def emb_path(space, pid): return f"{OUT}/emb/{space}/{pid}.npy"


def log_time(stage, space, pid, wall, **kw):
    """timing.csv: stage,space,pid,wall_s,device,extra (append; one row per finished unit)."""
    os.makedirs(OUT, exist_ok=True); f = f"{OUT}/timing.csv"; new = not os.path.exists(f)
    with open(f, "a") as fh:
        w = csv.writer(fh)
        if new: w.writerow(["stage", "space", "pid", "wall_s", "host", "extra"])
        w.writerow([stage, space, pid, round(wall, 1), os.uname().nodename, ";".join(f"{k}={v}" for k, v in kw.items())])


def save_emb(space, pid, E, ids):
    p = emb_path(space, pid); os.makedirs(os.path.dirname(p), exist_ok=True)
    assert E.shape[0] == len(ids) and np.isfinite(E).all()
    q = p[:-4] + ".ids.txt"; open(q + ".tmp", "w").write("\n".join(map(str, ids))); os.replace(q + ".tmp", q)     # ids first: the .npy is the commit marker
    np.save(p + ".tmp.npy", E.astype(np.float32)); os.replace(p + ".tmp.npy", p)


def load_emb(space, pid):
    p = emb_path(space, pid); E = np.load(p); ids = open(p[:-4] + ".ids.txt").read().split("\n"); assert len(ids) == E.shape[0]; return E, ids


# ---------------------------------------------------------------- representation patch
def reduce_emb(M, Z, n=50):
    """within-label (or global) reduction of an embedding that mirrors methods.preprocess_pca: z-score each dimension (clip 10), centred PCA,
    min(n, n_obs-1, d-1) components, arpack, seed M.RNG, UNWHITENED scores (so Leiden sees the same variance profile as for the PCA arm)."""
    import scanpy as sc, anndata as ad
    a = ad.AnnData(np.asarray(Z, np.float32)); sc.pp.scale(a, max_value=10)
    sc.pp.pca(a, n_comps=min(n, a.n_obs - 1, a.n_vars - 1), svd_solver="arpack", random_state=M.RNG); return a.obsm["X_pca"]


class Patch:
    """variant F: g = E (raw d), l = reduce_emb(E restricted to the label). H: g = E, l = original HVG PCA-50 of the label.
    P (dimension-matched control): g = reduce_emb(E) global PCA-50, l as F. pca30: original pipeline with 30 components in g and l."""
    def __init__(s, M, space, variant, E=None, ids=None):
        s.M, s.space, s.variant, s.orig = M, space, variant, M.preprocess_pca
        s.E, s.ids = E, (None if ids is None else pd.Index(ids)); s.n_glob = s.n_lab = s.n_lab_orig = 0; s.dim_g = None
        if variant == "P": s.Eg = reduce_emb(M, E, 50)
    def __call__(s, adata, n_pcs=None, hvg=None):
        if s.space == "pca30": s.dim_g = 30; return s.orig(adata, n_pcs=30, hvg=hvg)
        pos = s.ids.get_indexer(adata.obs_names); assert (pos >= 0).all(), "cell ids not found in the stored embedding (row misalignment)"
        if adata.n_obs == len(s.ids) and np.array_equal(pos, np.arange(len(pos))):          # global call (no visible label equals the whole pool)
            s.n_glob += 1; Z = s.Eg if s.variant == "P" else s.E; s.dim_g = Z.shape[1]; return Z, None
        if s.variant == "H": s.n_lab_orig += 1; return s.orig(adata, n_pcs=n_pcs or 50, hvg=hvg)
        s.n_lab += 1; return reduce_emb(s.M, s.E[pos], 50), None
    def check(s, d_expected):
        assert s.M.preprocess_pca is s, "patch not installed"
        if s.space == "pca30": return
        assert s.n_glob >= 3, f"global call hit {s.n_glob} times (expect sr3, sr3h, m1)"
        assert s.dim_g == d_expected, (s.dim_g, d_expected)
        assert (s.n_lab + s.n_lab_orig) >= 1, "no per-label call"


def install(space, variant, pid=None):
    """import locked modules (env() first), patch methods.preprocess_pca, return (modules dict, Patch|None, d_expected)."""
    import loader, sr2, sr3, sr3m, sr3h
    M = sys.modules["methods"]; assert sr2.M is M and M.__file__.startswith(ROOT), (sr2.M, M.__file__)
    if space == "pca": return dict(loader=loader, sr3=sr3, sr3m=sr3m, sr3h=sr3h, M=M), None, None
    if space == "pca30": p = Patch(M, "pca30", variant); M.preprocess_pca = p; return dict(loader=loader, sr3=sr3, sr3m=sr3m, sr3h=sr3h, M=M), p, 30
    E, ids = load_emb(space, pid); p = Patch(M, space, variant, E, ids); M.preprocess_pca = p
    return dict(loader=loader, sr3=sr3, sr3m=sr3m, sr3h=sr3h, M=M), p, (50 if variant == "P" else E.shape[1])
