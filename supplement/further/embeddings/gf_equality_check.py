"""Equality check: our Geneformer embeddings vs the official geneformer EmbExtractor.
usage: gf_equality_check.py --out DIR [--pool PID] [--subsample N] [--device cpu|mps] [--stored DIR]
Pool default = first (alphabetical) of the 10 stored pools; all hold 20000 cells, so none is smaller.
Pipeline = official TranscriptomeTokenizer (same args as w2_embed.py) -> official EmbExtractor(emb_mode cell, emb_layer -1, mean_pool, model_type Pretrained)
on gf-6L-30M-i2048 -> rows re-ordered by orig_row_idx -> compared with out/w2_fm_lap/emb/gf/<pid>.npy.
Deviations from a stock run, both in memory only, no file in the Geneformer checkout is edited:
 (1) tokenizer.py: pandas>=3 removed positional Series[int array]; the 2 sites get .iloc (same patch as our run).
 (2) emb_extractor/perturber_utils hardcode device="cuda" in 3 calls (torch.tensor, Tensor.to); redirected to --device (laptop has no CUDA).
 The package __init__ is replaced by a stub (gc30M dictionaries) because the real one imports bitsandbytes (absent) and points at gc95M files.
EmbExtractor expresses everything our run did (layer -1 = hidden_states[5], mean over non-pad tokens, no CLS/EOS). Our run used batch 16; EmbExtractor forward_batch_size=16 here.
Refuses to overwrite an existing RESULT.txt / npy in --out."""
import os, sys, types, importlib.util, hashlib, argparse, subprocess, tempfile, shutil
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
import numpy as np
HOME = os.path.expanduser("~"); HERE = os.path.dirname(os.path.abspath(__file__))
GF = os.environ.get("GF_ROOT", f"{HOME}/geneformer/Geneformer"); PKG = f"{GF}/geneformer"; D30 = f"{PKG}/gene_dictionaries_30m"; CKPT = f"{GF}/gf-6L-30M-i2048"
MED, TOK, MAP = (f"{D30}/{n}_gc30M.pkl" for n in ("gene_median_dictionary", "token_dictionary", "ensembl_mapping_dict"))
STORED = f"{HR_WORK}/jiaqi1011/out/w2_fm_lap/emb/gf"; BATCH = 16; K = 15

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--pool"); ap.add_argument("--subsample", type=int, default=0)
ap.add_argument("--device", default="cpu"); ap.add_argument("--stored", default=STORED); o = ap.parse_args()
pool = o.pool or sorted(f[:-4] for f in os.listdir(o.stored) if f.endswith(".npy"))[0]
os.makedirs(o.out, exist_ok=True); res_p = f"{o.out}/RESULT.txt"
if os.path.exists(res_p) or os.path.exists(f"{o.out}/official_emb.npy"): sys.exit(f"refusing to overwrite results in {o.out}")
sys.path.insert(0, HERE); import w2lib as W
import torch, pandas as pd, anndata as ad, scipy.sparse as sp
dev = o.device; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))

# ---- official modules via a stub package (see header)
stub = types.ModuleType("geneformer"); stub.__path__ = [PKG]; stub.GENE_MEDIAN_FILE, stub.TOKEN_DICTIONARY_FILE, stub.ENSEMBL_MAPPING_FILE = MED, TOK, MAP; sys.modules["geneformer"] = stub
spec = importlib.util.spec_from_file_location("geneformer.tokenizer", f"{PKG}/tokenizer.py"); tkm = importlib.util.module_from_spec(spec); sys.modules["geneformer.tokenizer"] = tkm
src = open(f"{PKG}/tokenizer.py").read(); PAT = 'adata.var["ensembl_id_collapsed"][coding_miRNA_loc]'; assert src.count(PAT) == 2
exec(compile(src.replace(PAT, PAT.replace("][", "].iloc[", 1)), f"{PKG}/tokenizer.py", "exec"), tkm.__dict__)
tk = tkm.TranscriptomeTokenizer(custom_attr_name_dict={"orig_row_idx": "orig_row_idx"}, nproc=1, chunk_size=512, model_input_size=2048, special_token=False,
                                collapse_gene_ids=True, gene_median_file=MED, token_dictionary_file=TOK, gene_mapping_file=MAP)
assert len(tk.gene_token_dict) == 25426 and "<cls>" not in tk.gene_token_dict
_t, _to = torch.tensor, torch.Tensor.to
torch.tensor = lambda *a, **k: _t(*a, **{**k, "device": dev if k.get("device") == "cuda" else k.get("device")}) if "device" in k else _t(*a, **k)
torch.Tensor.to = lambda self, *a, **k: _to(self, *[dev if x == "cuda" else x for x in a], **k)
from geneformer import perturber_utils as pu, emb_extractor as ee
_lm = pu.load_model; pu.load_model = lambda *a, **k: _lm(*a, **k).to(dev)

# ---- cells (same loader, same ensembl-id AnnData construction as w2_embed.py embed_gf)
W.env(); import loader; a, _ = loader.load_pilot(pool); del _
stored = np.load(f"{o.stored}/{pool}.npy"); ids = open(f"{o.stored}/{pool}.ids.txt").read().split("\n")
ids = [i for i in ids if i != ""] if len(ids) != stored.shape[0] else ids
assert list(a.obs_names[:len(ids)]) == ids[:a.n_obs] and a.n_obs == stored.shape[0], "pool cells differ from stored ids"
n = o.subsample or a.n_obs; a = a[:n].copy(); stored = stored[:n]
X = sp.csr_matrix(a.X); b = ad.AnnData(X=X); b.var_names = a.var_names; assert str(b.var_names[0]).startswith("ENSG")
b.var["ensembl_id"] = b.var_names.values; b.obs["n_counts"] = np.asarray(X.sum(1), np.float64).ravel(); b.obs["orig_row_idx"] = np.arange(n)
tmp = tempfile.mkdtemp(prefix="gfeq_", dir=os.environ.get("TMPDIR", "/tmp")); os.makedirs(f"{tmp}/in"); b.write_h5ad(f"{tmp}/in/s.h5ad")
tk.tokenize_data(f"{tmp}/in", tmp, "tok", file_format="h5ad")
ex = ee.EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode="cell", cell_emb_style="mean_pool", max_ncells=None, emb_layer=-1, emb_label=["orig_row_idx"],
                     forward_batch_size=BATCH, nproc=1, token_dictionary_file=TOK)
df = ex.extract_embs(CKPT, f"{tmp}/tok.dataset", tmp, "emb"); shutil.rmtree(tmp, ignore_errors=True)
df = df.sort_values("orig_row_idx"); assert df["orig_row_idx"].tolist() == list(range(n)), "tokenizer/extractor dropped cells"
E = df.drop(columns="orig_row_idx").to_numpy(np.float32); assert E.shape == stored.shape, (E.shape, stored.shape)
np.save(f"{o.out}/official_emb.npy", E)

# ---- compare
d = np.abs(E - stored); cos = (E * stored).sum(1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(stored, axis=1))
from sklearn.neighbors import NearestNeighbors
def knn(Z): return NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]   # drop self
A, B = knn(E), knn(stored); share = np.mean([len(set(x) & set(y)) / K for x, y in zip(A, B)])
mx = float(d.max()); verdict = "PASS" if mx < 1e-4 else f"FAIL (max abs diff {mx:.3g}, {mx / 1e-4:.1f}x the 1e-4 bar)"
head = subprocess.run(["git", "-C", GF, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
txt = f"""{verdict}
pool {pool}  cells {n}{' (SMOKE subsample)' if o.subsample else ''}  device {dev}  k {K}  batch {BATCH}
max_abs_diff {mx:.6g}  mean_abs_diff {float(d.mean()):.6g}  p99_abs_diff {float(np.quantile(d, .99)):.6g}  cells_with_max_diff_gt_1e-4 {int((d.max(1) > 1e-4).sum())}
mean_cosine {float(cos.mean()):.8f}  min_cosine {float(cos.min()):.8f}
knn15_shared_neighbour_fraction {share:.6f}
geneformer repo {GF} commit {head}; sha256[:16] tokenizer.py {sha(PKG + '/tokenizer.py')} emb_extractor.py {sha(PKG + '/emb_extractor.py')} perturber_utils.py {sha(PKG + '/perturber_utils.py')}
checkpoint {CKPT} model.safetensors {sha(CKPT + '/model.safetensors')}
settings: TranscriptomeTokenizer(model_input_size 2048, special_token False, collapse_gene_ids True, gc30M dicts, chunk 512); EmbExtractor(emb_mode cell, mean_pool, emb_layer -1, Pretrained)
dirty {subprocess.run(['git','-C',GF,'status','--porcelain','--untracked-files=no'],capture_output=True,text=True).stdout.strip() or 'clean'}; torch {torch.__version__} transformers {__import__('transformers').__version__} datasets {__import__('datasets').__version__} pandas {pd.__version__}
deviations (in memory): tokenizer .iloc pandas3 patch (2 sites); cuda->{dev} redirect; stub package __init__ (gc30M paths, no bitsandbytes import)
"""
open(res_p, "x").write(txt); print(txt)
