"""python embed.py SPACE [pid ...]   SPACE: scvi | scgpt | gf | noise.  Blind (no truth). Writes $W2_OUT/emb/<space>/<pid>.npy (+ .ids.txt) and timing.csv. Resumable.
scvi  : methods.scvi_latent copy (locked one cannot early-stop): same 2000 HVG (methods._hvg_genes), n_latent 30, NO batch key, seed 0, max_epochs 400 with
        early_stopping=True (official, validation ELBO), wall cap 15 min via Lightning max_time (cap hit => stops at that epoch, recorded), 3 threads. Pipeline venv.
scgpt : official scgpt.tasks.embed_data, whole-human ckpt, the SAME 2000 HVG as the PCA arm (toolin/r pool = a[:, _hvg_genes]), symbols from var.feature_name,
        max_length=2001 so no gene sampling, batch 64, seeds 0. cls embedding (embed_data L2-normalises). Tools venv (no loader needed). env SCGPT_CKPT, --device.
gf    : Geneformer V1-10M (gf-6L-30M-i2048), the foundation-model duplication pipeline of ~/work/gf6l_emb/src (real tokenizer, gc30M dictionaries, mean over non-padding tokens of hidden_states[5]).
        Laptop python3 (transformers); model_version V1 = gc30M dictionaries with vocab asserted == 25426.
noise : planted random embedding (64-d N(0,1)) for the leakage / plumbing gate."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import embedding_lib as W
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("space"); ap.add_argument("pids", nargs="*"); ap.add_argument("--device", default=os.environ.get("W2_DEVICE", "cpu"))
ap.add_argument("--toolin"); ap.add_argument("--subsample", type=int, default=0, help="plumbing test only: first N cells (output goes to space+_sub)")
ap.add_argument("--ckpt", default=os.environ.get("SCGPT_CKPT", f"{W.HR_WORK}/r5/tools/scgpt_ckpt")); o = ap.parse_args()
pids = o.pids or W.pools()


def load_cells(pid):
    W.env(); import loader; a, _t = loader.load_pilot(pid); del _t; return a


def embed_scvi(pid):
    a = load_cells(pid); import methods as M, scvi, torch
    hvg = M._hvg_genes(a); b = a[:, hvg].copy(); b.X = np.asarray(b.X.todense()) if hasattr(b.X, "todense") else np.asarray(b.X); b.layers["counts"] = b.X.copy()
    if o.subsample: b = b[:o.subsample].copy()
    scvi.settings.seed = M.RNG; torch.set_num_threads(3); scvi.model.SCVI.setup_anndata(b, layer="counts"); m = scvi.model.SCVI(b, n_latent=30)
    t0 = time.time(); ME = int(os.environ.get("SCVI_MAX_EPOCHS", 400)); MT = int(os.environ.get("SCVI_MAX_MIN", 60))    # fixed epoch budget (set from the full-pool plumbing run); wall cap is only a safety net
    m.train(max_epochs=ME, early_stopping=True, accelerator="cpu", enable_progress_bar=False, max_time=f"00:{MT // 60:02d}:{MT % 60:02d}:00"); w = time.time() - t0
    h = m.history["elbo_train"].iloc[:, 0]; ne = len(h)
    return m.get_latent_representation(), list(b.obs_names), dict(epochs=ne, capped=int(w > (MT - 0.5) * 60), epoch_budget=ME, early_stopped=int(ne < ME), elbo_train=round(float(h.iloc[-1]), 2), scvi=scvi.__version__)


def embed_scgpt(pid):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if not hasattr(os, "sched_getaffinity"): os.sched_getaffinity = lambda p: set()
    import warnings; warnings.filterwarnings("ignore"); import anndata as ad, torch
    from scgpt.tasks import embed_data
    np.random.seed(0); torch.manual_seed(0)
    c = [p for p in ((o.toolin,) if o.toolin else (f"{W.ROOT}/toolin/r", f"{W.R5D}/toolin/r")) if os.path.exists(f"{p}/{pid}.h5ad")]
    assert c, f"toolin pool missing: {W.R5D}/toolin/r/{pid}.h5ad"; d = c[0]
    a = ad.read_h5ad(f"{d}/{pid}.h5ad"); assert a.n_vars <= 2100   # a few pools carry 2002 genes (HVG ties)
    if o.subsample: a = a[:o.subsample].copy()
    ids = list(a.obs_names); a.var_names = a.var["feature_name"].astype(str).values; a = a[:, ~a.var_names.duplicated()].copy()
    e = embed_data(a, o.ckpt, gene_col="index", max_length=max(2001, a.n_vars + 1), batch_size=64, device=o.device, use_fast_transformer=False, return_new_adata=False)
    return np.asarray(e.obsm["X_scGPT"]), ids, dict(n_sym=a.n_vars, device=o.device, torch=torch.__version__)


def embed_gf(pid):
    import shutil, tempfile, torch, anndata as ad, scipy.sparse as sp
    GFS = f"{W.HR_WORK}/gf6l_emb/src"; sys.path.insert(0, GFS); import gfcfg as C
    import gfload, types, importlib.util
    def make_tokenizer():
        """gfload.make_tokenizer, but the REAL tokenizer.py is exec'd with ONE in-memory change (pandas>=3 removed positional fallback in Series[int array]):
        adata.var["ensembl_id_collapsed"][coding_miRNA_loc]  ->  .iloc[coding_miRNA_loc]   (2 sites). Nothing else differs; the file on disk is untouched."""
        fake = types.ModuleType("geneformer"); fake.__path__ = [str(C.PKG)]; fake.GENE_MEDIAN_FILE, fake.TOKEN_DICTIONARY_FILE, fake.ENSEMBL_MAPPING_FILE = C.GENE_MEDIAN_FILE, C.TOKEN_DICTIONARY_FILE, C.ENSEMBL_MAPPING_FILE
        sys.modules["geneformer"] = fake; spec = importlib.util.spec_from_file_location("geneformer.tokenizer", C.PKG / "tokenizer.py"); m = importlib.util.module_from_spec(spec); sys.modules["geneformer.tokenizer"] = m
        src = open(C.PKG / "tokenizer.py").read(); assert src.count('adata.var["ensembl_id_collapsed"][coding_miRNA_loc]') == 2
        exec(compile(src.replace('adata.var["ensembl_id_collapsed"][coding_miRNA_loc]', 'adata.var["ensembl_id_collapsed"].iloc[coding_miRNA_loc]'), str(C.PKG / "tokenizer.py"), "exec"), m.__dict__)
        tk = m.TranscriptomeTokenizer(custom_attr_name_dict={"orig_row_idx": "orig_row_idx"}, nproc=1, chunk_size=C.TOK_CHUNK, model_input_size=C.MODEL_INPUT_SIZE, special_token=C.SPECIAL_TOKEN,
                                      collapse_gene_ids=True, gene_median_file=C.GENE_MEDIAN_FILE, token_dictionary_file=C.TOKEN_DICTIONARY_FILE, gene_mapping_file=C.ENSEMBL_MAPPING_FILE)
        assert len(tk.gene_token_dict) == C.VOCAB and "<cls>" not in tk.gene_token_dict and tk.model_input_size == 2048 and tk.special_token is False   # V1 / gc30M guard
        return tk
    from datasets import load_from_disk
    from transformers import BertForMaskedLM
    a = load_cells(pid)
    if o.subsample: a = a[:o.subsample].copy()
    X = sp.csr_matrix(a.X); b = ad.AnnData(X=X); b.var_names = a.var_names
    assert str(b.var_names[0]).startswith("ENSG"), "slim pools must carry Ensembl ids"
    b.var["ensembl_id"] = b.var_names.values; b.obs["n_counts"] = np.asarray(X.sum(1), np.float64).ravel(); b.obs["orig_row_idx"] = np.arange(b.n_obs)
    tk = make_tokenizer(); tmp = tempfile.mkdtemp(prefix="gf_", dir=os.environ.get("TMPDIR", "/tmp")); sd = f"{tmp}/in"; os.makedirs(sd); b.write_h5ad(f"{sd}/s.h5ad")
    tk.tokenize_data(sd, tmp, "tok", file_format="h5ad"); ds = load_from_disk(f"{tmp}/tok.dataset"); assert ds["orig_row_idx"] == list(range(b.n_obs)), "tokenizer dropped or reordered cells"
    ids_l = [np.asarray(v, np.int64) for v in ds["input_ids"]]; lens = np.array([len(x) for x in ids_l]); shutil.rmtree(tmp, ignore_errors=True)
    model = BertForMaskedLM.from_pretrained(C.CKPT, output_hidden_states=True).eval(); assert model.config.vocab_size == 25426 and model.config.hidden_size == 256
    dev = o.device; torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4))); model = model.to(dev); out = np.zeros((len(ids_l), 256), np.float32)
    order = np.argsort(-lens)
    for i in range(0, len(order), C.BATCH):
        sel = order[i:i + C.BATCH]; L = int(lens[sel].max()); x = torch.zeros((len(sel), L), dtype=torch.long); am = torch.zeros_like(x)
        for j, k in enumerate(sel): x[j, :lens[k]] = torch.from_numpy(ids_l[k]); am[j, :lens[k]] = 1
        with torch.no_grad(): h = model(input_ids=x.to(dev), attention_mask=am.to(dev)).hidden_states[C.LAYER_TO_QUANT]
        w = am.to(dev).unsqueeze(-1).to(h.dtype); out[sel] = ((h * w).sum(1) / w.sum(1)).float().cpu().numpy()
    return out, list(a.obs_names), dict(device=dev, frac_trunc2048=round(float((lens >= 2048).mean()), 3), med_len=int(np.median(lens)))


def embed_noise(pid, d=64):
    a = load_cells(pid); return np.random.RandomState(7).standard_normal((a.n_obs, d)).astype(np.float32), list(a.obs_names), {}


FN = dict(scvi=embed_scvi, scgpt=embed_scgpt, gf=embed_gf, noise=embed_noise, noise512=lambda pid: embed_noise(pid, 512))
for pid in pids:
    sp_ = o.space + ("_sub" if o.subsample else "")
    if os.path.exists(W.emb_path(sp_, pid)): print("skip", sp_, pid); continue
    t0 = time.time(); E, ids, extra = FN[o.space](pid); w = time.time() - t0
    assert np.isfinite(E).all() and E.shape[0] == len(ids); W.save_emb(sp_, pid, E, ids); W.log_time("embed", sp_, pid, w, d=E.shape[1], **extra)
    print(sp_, pid, E.shape, round(w, 1), extra, flush=True)
