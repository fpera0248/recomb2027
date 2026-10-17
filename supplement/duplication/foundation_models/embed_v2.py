"""Step 2 (V2-316M build): Geneformer V2-316M frozen cell embeddings for exactly the
12,719 unique cells cells_needed.csv selects (run cells_needed.py first).

Reuses the proven gf6l_emb/src pipeline pattern (01_shard_tokenize.py + 02_embed.py: the REAL,
unmodified geneformer/tokenizer.py loaded without triggering geneformer/__init__.py's eager
ray/optuna/tdigest/seaborn imports) reconfigured for the V2/"95M-series" convention instead of
30M: special_token=True (<cls>/<eos>), model_input_size=4096, the gc95M-vocab dictionaries
(gene_median_dictionary_gc95M.pkl / token_dictionary_gc95M.pkl / ensembl_mapping_dict_gc95M.pkl,
vocab 20275, <pad>=0 <mask>=1 <cls>=2 <eos>=3 -- all confirmed by reading the pickles directly).
There is no gc104M dictionary; these gc95M dicts are the ones tokenizer.py (line 37) pairs
with special_token=True + model_input_size=4096 + vocab 20275, i.e. the same convention family.

aida_sg.h5ad already has raw counts directly in .X and Ensembl ids directly as var_names
(confirmed: 01_shard_tokenize.py writes var['ensembl_id'] = var_names verbatim and asserts
var dim 35477; build.log shows it was built by summing from a 1,265,624-cell source, almost
certainly AIDA_phase1_v2.h5ad). So rawutil.py's .raw-vs-.X branch is moot here -- aida_sg.h5ad
*is* the already-extracted raw-count, Ensembl-indexed derivative. Going back to
AIDA_phase1_v2.h5ad and rawutil.py would re-derive something we already have.

EMBEDDING CONVENTION -- read geneformer/emb_extractor.py before assuming "mean over gene
tokens": EmbExtractor's own class default is `emb_mode="cls"` (line ~414), and for a cls-
bearing model `emb_mode="cell"` explicitly EXCLUDES the cls token and mean-pools the rest --
mean-pooling is the non-default alternative, not what the repo defaults to. `emb_layer=-1` is
also the default, giving layer_to_quant = quant_layers(model) + (-1) = N_LAYERS - 1 (2nd-to-
last layer, matching the 2nd-to-last-layer convention already used for 6L in gf6l_emb/src).
This also matches this project's OWN prior V2-316M artifacts (fm2/run_fm_dup_gf316m.py's
docstring: "1152-dim GF-316M CLS, frozen"). The repo default is therefore CLS at the 2nd-to-
last layer, NOT a mean over gene tokens -- BUT the finished 6L run
(gf6l_emb/src/02_embed.py) mean-pooled in fp32, so a CLS-only V2 output would confound "model
size" with "pooling choice" in any foundation-model comparison. Fixed at zero extra cost: every forward pass
below now saves BOTH poolings (CLS and mean, excluding <cls>/<eos>) from the same hidden_states,
so the comparison can isolate the pooling axis instead of silently mixing it with the model-size
axis. Precision note: bf16 is the MPS-required compute dtype (see load_model()), but hidden
states are upcast to fp32 BEFORE the mean-pool reduction -- summing up to ~4096 bf16 positions
loses real precision a post-hoc cast cannot recover.

CHECKPOINT: the V2-316M weights (18L/1152-hidden/18-head/vocab-20275/max-pos-4096) are not in the
Geneformer checkout. CKPT defaults to the upstream HF folder name ("Geneformer-V2-316M",
ctheodoris/Geneformer) under GF_ROOT and is overridable with GF_V2_CKPT; this script never fetches
the weights itself. main() hard-asserts on CKPT before loading anything and exits with an
actionable message instead of silently running the wrong checkpoint.
"""
import os, sys, json, time, shutil
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
try:
    os.nice(10)   # self-renice as a safety net even if not launched under `nice -n 10`
except Exception:
    pass

SRC_H5AD   = HR_WORK + '/aida_sg/aida_sg.h5ad'                       # READ-ONLY
GF_ROOT    = os.environ.get('GF_ROOT', os.path.expanduser('~/geneformer/Geneformer'))                          # READ-ONLY
PKG        = os.path.join(GF_ROOT, 'geneformer')
GENE_MEDIAN_FILE      = os.path.join(PKG, 'gene_median_dictionary_gc95M.pkl')
TOKEN_DICTIONARY_FILE = os.path.join(PKG, 'token_dictionary_gc95M.pkl')
ENSEMBL_MAPPING_FILE  = os.path.join(PKG, 'ensembl_mapping_dict_gc95M.pkl')
# Upstream HF folder name (ctheodoris/Geneformer).
# Override with GF_V2_CKPT once the weights are placed somewhere -- this script never downloads
# them itself.
CKPT = os.environ.get('GF_V2_CKPT', os.path.join(GF_ROOT, 'Geneformer-V2-316M'))

OUT            = os.path.dirname(os.path.abspath(__file__))
CELLS_CSV      = os.path.join(OUT, 'cells_needed.csv')
SHARD_DIR      = os.path.join(OUT, 'shard_v2')        # temp h5ad, deleted after tokenizing
TOK_DIR        = os.path.join(OUT, 'tok_v2')          # kept: lets you re-embed without re-tokenizing
PART_DIR       = os.path.join(OUT, 'emb_v2_parts')    # per-chunk checkpoints
FINAL_NPY_CLS  = os.path.join(OUT, 'emb_v2_316m_cls.npy')    # repo-default pooling
FINAL_NPY_MEAN = os.path.join(OUT, 'emb_v2_316m_mean.npy')   # 6L-run-matched pooling
FINAL_CSV      = os.path.join(OUT, 'emb_v2_316m.csv')
MANIFEST       = os.path.join(OUT, 'manifest_v2.json')

MODEL_INPUT_SIZE = 4096
SPECIAL_TOKEN    = True
VOCAB            = 20275
HIDDEN           = 1152
N_LAYERS         = 18
N_HEADS          = 18
CLS_ID           = 2              # tokdict['<cls>'], asserted in make_tokenizer()
EOS_ID           = 3              # tokdict['<eos>'], asserted in make_tokenizer()
EMB_LAYER        = -1             # repo default
LAYER_TO_QUANT   = N_LAYERS + EMB_LAYER   # == 17 (2nd-to-last of 18)
CHUNK_CELLS      = int(os.environ.get('V2_CHUNK', 2000))   # checkpoint granularity
BATCH            = int(os.environ.get('V2_BATCH', 8))      # UNVERIFIED guess -- no weights to measure against
DEVICE           = os.environ.get('GF_DEVICE', 'mps')

t0 = time.time()
def log(m): print(f'[{time.time()-t0:8.1f}s] {m}', flush=True)


def real_tokenizer_module():
    """Load the REAL geneformer/tokenizer.py without geneformer/__init__.py's eager imports.
    Same trick as gf6l_emb/src/gfload.py, pointed at the gc95M (V2) dictionaries instead of gc30M."""
    import importlib.util, types
    fake = types.ModuleType('geneformer')
    fake.__path__ = [PKG]
    fake.GENE_MEDIAN_FILE = GENE_MEDIAN_FILE
    fake.TOKEN_DICTIONARY_FILE = TOKEN_DICTIONARY_FILE
    fake.ENSEMBL_MAPPING_FILE = ENSEMBL_MAPPING_FILE
    sys.modules['geneformer'] = fake
    spec = importlib.util.spec_from_file_location('geneformer.tokenizer', os.path.join(PKG, 'tokenizer.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['geneformer.tokenizer'] = m
    spec.loader.exec_module(m)
    return m


def make_tokenizer():
    m = real_tokenizer_module()
    tk = m.TranscriptomeTokenizer(
        custom_attr_name_dict={'orig_row_idx': 'orig_row_idx'},
        nproc=1, chunk_size=512,
        model_input_size=MODEL_INPUT_SIZE, special_token=SPECIAL_TOKEN,
        collapse_gene_ids=True,
        gene_median_file=GENE_MEDIAN_FILE, token_dictionary_file=TOKEN_DICTIONARY_FILE,
        gene_mapping_file=ENSEMBL_MAPPING_FILE,
    )
    assert len(tk.gene_token_dict) == VOCAB, f'token dict is not the V2/95M-series dict: {len(tk.gene_token_dict)}'
    assert tk.gene_token_dict['<cls>'] == 2 and tk.gene_token_dict['<eos>'] == 3
    assert tk.gene_token_dict['<pad>'] == 0 and tk.gene_token_dict['<mask>'] == 1
    assert tk.model_input_size == MODEL_INPUT_SIZE and tk.special_token is True
    return tk


def load_cells_needed():
    df = pd.read_csv(CELLS_CSV)
    uniq = df[['obs_idx', 'obs_name']].drop_duplicates('obs_idx').sort_values('obs_idx').reset_index(drop=True)
    return uniq   # canonical row order for everything downstream: obs_idx ascending


def build_and_tokenize(uniq):
    dpath = TOK_DIR + '.dataset'
    if os.path.exists(dpath):
        from datasets import load_from_disk
        ds = load_from_disk(dpath)
        if ds['orig_row_idx'] == uniq['obs_idx'].tolist():
            log(f'tokenized dataset already on disk, {len(ds)} cells, order OK -- skip tokenizing')
            return ds
        sys.exit(f'ABORT: {dpath} exists but orig_row_idx does not match cells_needed.csv -- '
                 'delete tok_v2.dataset or fix the mismatch, never silently re-tokenize over it')

    import h5py, scipy.sparse as sp, anndata as ad
    f = h5py.File(SRC_H5AD, 'r')
    nvars = int(f['X'].attrs['shape'][1])
    assert nvars == 35477
    indptr_all = f['X']['indptr'][:]
    Xd, Xi = f['X']['data'], f['X']['indices']
    var_index = np.array([x.decode() for x in f['var']['_index'][:]])

    idx = uniq['obs_idx'].values
    rows = []
    for i in idx:
        a, z = int(indptr_all[i]), int(indptr_all[i + 1])
        rows.append((Xd[a:z], Xi[a:z], z - a))
    data_cat = np.concatenate([r[0] for r in rows])
    ind_cat = np.concatenate([r[1] for r in rows])
    indptr = np.concatenate([[0], np.cumsum([r[2] for r in rows])])
    M = sp.csr_matrix((data_cat, ind_cat, indptr), shape=(len(idx), nvars))

    adata = ad.AnnData(X=M)
    adata.var_names = var_index
    adata.var['ensembl_id'] = var_index                                       # tokenizer.py asserts this
    adata.obs['n_counts'] = np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
    adata.obs['orig_row_idx'] = idx.astype(np.int64)                          # the join key
    assert 'filter_pass' not in adata.obs.columns

    if os.path.exists(SHARD_DIR):
        shutil.rmtree(SHARD_DIR)
    os.makedirs(SHARD_DIR)
    adata.write_h5ad(os.path.join(SHARD_DIR, 'shard.h5ad'))
    assert len(os.listdir(SHARD_DIR)) == 1

    tk = make_tokenizer()
    log(f'tokenizer OK: {len(tk.gene_token_dict)} tokens, special_token={tk.special_token}, '
        f'input_size {tk.model_input_size}')
    tk.tokenize_data(SHARD_DIR, OUT, 'tok_v2', file_format='h5ad')
    shutil.rmtree(SHARD_DIR, ignore_errors=True)

    from datasets import load_from_disk
    ds = load_from_disk(dpath)
    assert len(ds) == len(idx), f'{len(ds)} rows, expected {len(idx)}'
    assert ds['orig_row_idx'] == idx.tolist(), 'row ids not 1:1 with cells_needed.csv order'
    L = np.array(ds['length'])
    assert L.min() >= 1 and L.max() <= MODEL_INPUT_SIZE
    log(f'tokenized {len(ds)} cells | len med {np.median(L):.0f} max {L.max()} | '
        f'frac at cap {np.mean(L == MODEL_INPUT_SIZE):.3f}')
    return ds


def check_checkpoint():
    cfg_path = os.path.join(CKPT, 'config.json')
    if not os.path.isdir(CKPT) or not os.path.exists(cfg_path):
        sys.exit(
            f'ABORT: no V2-316M checkpoint at {CKPT}. Confirmed by inspection: this machine\'s '
            f'{GF_ROOT} has gf-6L-30M-i2048, gf-12L-30M-i2048, gf-12L-95M-i4096 (also copied to '
            f'the repo root -- its config.json/model.safetensors are byte-identical, md5 '
            f'a0570023168309de19bb166b33bab134, to gf-12L-95M-i4096/model.safetensors), '
            f'gf-12L-95M-i4096_CLcancer and gf-20L-95M-i4096 (20L, hidden 896) -- none is '
            f'18 layers / hidden 1152 / 18 heads. The weights are public on HF '
            f'(ctheodoris/Geneformer, folder "Geneformer-V2-316M", no login needed) but this '
            f'script does not fetch them. Place them at {CKPT} or point the GF_V2_CKPT env var '
            f'at wherever they land; nothing below this line has been run for real.'
        )
    cfg = json.load(open(cfg_path))
    for k, want in [('hidden_size', HIDDEN), ('num_hidden_layers', N_LAYERS),
                     ('num_attention_heads', N_HEADS), ('vocab_size', VOCAB),
                     ('max_position_embeddings', MODEL_INPUT_SIZE), ('pad_token_id', 0)]:
        assert cfg.get(k) == want, f'{CKPT}/config.json[{k}]={cfg.get(k)!r}, expected {want!r}'
    log(f'checkpoint config OK: {CKPT}')


def load_model():
    import torch
    from transformers import BertForMaskedLM
    check_checkpoint()
    model = BertForMaskedLM.from_pretrained(CKPT, output_hidden_states=True).eval()
    assert model.config.vocab_size == VOCAB and model.config.hidden_size == HIDDEN
    assert model.config.num_hidden_layers == N_LAYERS and model.config.pad_token_id == 0
    ln = [int(n.split('layer.')[1].split('.')[0]) for n, _ in model.named_parameters() if 'layer' in n]
    assert max(ln) + 1 + EMB_LAYER == LAYER_TO_QUANT
    # torch.autocast('mps') fails on torch 2.4.1 (measured) -- cast manually instead of autocast.
    model = model.to(torch.bfloat16)
    dev = DEVICE if (DEVICE != 'mps' or torch.backends.mps.is_available()) else 'cpu'
    model = model.to(dev)
    torch.set_num_threads(2)
    return model, dev


def embed_chunk(model, dev, idlists):
    """idlists: list of 1-D int arrays (already includes <cls>/<eos> from the tokenizer).
    Returns (cls_out, mean_out), each (n, HIDDEN) float32, same order, from ONE forward pass:
      cls_out:  hidden_states[LAYER_TO_QUANT][:, 0, :]        (repo default, EmbExtractor emb_mode='cls')
      mean_out: mean over non-pad, non-cls/eos positions, same layer (EmbExtractor emb_mode='cell')
    Both are saved so a foundation-model comparison can isolate the pooling axis (see module docstring)."""
    import torch
    L = max(len(a) for a in idlists)
    x = torch.zeros((len(idlists), L), dtype=torch.long)
    am = torch.zeros((len(idlists), L), dtype=torch.long)
    for j, a in enumerate(idlists):
        n = len(a)
        assert int(a.max()) < VOCAB, f'token id {int(a.max())} >= vocab {VOCAB} -- MPS will NOT catch this'
        if SPECIAL_TOKEN:
            assert int(a[0]) == CLS_ID and int(a[-1]) == EOS_ID, \
                f'row does not start/end with <cls>/<eos> ({a[0]}, {a[-1]}) -- tokenizer special_token mismatch'
        x[j, :n] = torch.from_numpy(np.asarray(a, dtype=np.int64)); am[j, :n] = 1
    with torch.no_grad():
        h = model(input_ids=x.to(dev), attention_mask=am.to(dev)).hidden_states[LAYER_TO_QUANT]
    cls_out = h[:, 0, :].float().cpu().numpy()

    hf = h.float()                                    # cast BEFORE the reduction: summing up to
    w = am.to(dev).unsqueeze(-1).float().clone()       # ~4096 bf16 positions loses precision a
    w[:, 0, :] = 0                                     # drop <cls>                 post-hoc cast can't recover
    lens = torch.tensor([len(a) for a in idlists])
    for j, n in enumerate(lens):
        w[j, int(n) - 1, :] = 0                        # drop <eos> (last real position)
    mean_out = ((hf * w).sum(1) / w.sum(1)).cpu().numpy()
    return cls_out, mean_out


def main():
    uniq = load_cells_needed()
    log(f'{len(uniq)} unique cells to embed (from cells_needed.csv)')
    ds = build_and_tokenize(uniq)
    model, dev = load_model()   # exits via check_checkpoint() if weights absent

    ids_all = [np.asarray(v, dtype=np.int64) for v in ds['input_ids']]
    lens = np.array([len(a) for a in ids_all])
    order = np.argsort(-lens)   # long first: catches OOM immediately, ~3x faster
    os.makedirs(PART_DIR, exist_ok=True)

    n_parts = (len(order) + CHUNK_CELLS - 1) // CHUNK_CELLS
    for pi in range(n_parts):
        p_cls  = os.path.join(PART_DIR, f'part_{pi:03d}_cls.npy')
        p_mean = os.path.join(PART_DIR, f'part_{pi:03d}_mean.npy')
        pp     = os.path.join(PART_DIR, f'part_{pi:03d}_pos.npy')
        if os.path.exists(p_cls) and os.path.exists(p_mean) and os.path.exists(pp):
            log(f'part {pi:03d} already embedded -- skip'); continue
        sel = order[pi * CHUNK_CELLS:(pi + 1) * CHUNK_CELLS]
        out_cls  = np.full((len(sel), HIDDEN), np.nan, dtype=np.float32)
        out_mean = np.full((len(sel), HIDDEN), np.nan, dtype=np.float32)
        for b in range(0, len(sel), BATCH):
            bi = sel[b:b + BATCH]
            out_cls[b:b + BATCH], out_mean[b:b + BATCH] = embed_chunk(model, dev, [ids_all[k] for k in bi])
            if (b // BATCH) % 100 == 0: log(f'part {pi:03d}: {b}/{len(sel)} cells, max len {lens[bi].max()}')
        assert np.isfinite(out_cls).all() and np.isfinite(out_mean).all(), f'part {pi:03d}: non-finite embedding'
        np.save(pp, sel); np.save(p_cls, out_cls); np.save(p_mean, out_mean)
        log(f'part {pi:03d}/{n_parts - 1} ({len(sel)} cells) embedded on {dev}')

    emb_cls  = np.full((len(order), HIDDEN), np.nan, dtype=np.float32)
    emb_mean = np.full((len(order), HIDDEN), np.nan, dtype=np.float32)
    for pi in range(n_parts):
        sel = np.load(os.path.join(PART_DIR, f'part_{pi:03d}_pos.npy'))
        emb_cls[sel]  = np.load(os.path.join(PART_DIR, f'part_{pi:03d}_cls.npy'))
        emb_mean[sel] = np.load(os.path.join(PART_DIR, f'part_{pi:03d}_mean.npy'))
    assert np.isfinite(emb_cls).all() and np.isfinite(emb_mean).all(), 'consolidation left unfilled rows'

    np.save(FINAL_NPY_CLS, emb_cls)
    np.save(FINAL_NPY_MEAN, emb_mean)
    man = uniq.copy(); man['length'] = lens
    man.to_csv(FINAL_CSV, index=False)
    json.dump({'layer_to_quant': LAYER_TO_QUANT, 'ckpt': CKPT, 'n_cells': len(order), 'hidden': HIDDEN,
               'cls_npy': FINAL_NPY_CLS, 'mean_npy': FINAL_NPY_MEAN,
               'note': "both CLS (repo default) and mean-pooled (excl. cls/eos) saved from the "
                       "same forward pass -- pick whichever matches the axis you're isolating"},
              open(MANIFEST, 'w'), indent=1)
    log(f'wrote {FINAL_NPY_CLS} + {FINAL_NPY_MEAN} {emb_cls.shape} + {FINAL_CSV} '
        f'(row order = obs_idx ascending, matches cells_needed.csv)')


if __name__ == '__main__':
    main()
