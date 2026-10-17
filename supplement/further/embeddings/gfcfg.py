"""Single source of truth for every knob. Imported by all stages.
Any change here changes CFG_HASH, which every stage asserts against the manifest,
so a half-finished run can never be resumed under different settings."""
import hashlib, json, os
HR_WORK = os.environ.get("HR_WORK", os.path.expanduser("~/work"))
from pathlib import Path

SRC_H5AD   = HR_WORK + "/aida_sg/aida_sg.h5ad"          # READ-ONLY
CELLS_CSV  = os.environ.get("GF_CELLS_CSV")   # CSV of the cells to embed (aida_sg_cells.csv); set GF_CELLS_CSV before running a stage that reads it
GF_ROOT    = Path(os.environ.get("GF_ROOT", os.path.expanduser("~/geneformer/Geneformer")))        # READ-ONLY
PKG        = GF_ROOT / "geneformer"
D30        = PKG / "gene_dictionaries_30m"
GENE_MEDIAN_FILE      = D30 / "gene_median_dictionary_gc30M.pkl"
TOKEN_DICTIONARY_FILE = D30 / "token_dictionary_gc30M.pkl"
ENSEMBL_MAPPING_FILE  = D30 / "ensembl_mapping_dict_gc30M.pkl"
CKPT       = str(GF_ROOT / "gf-6L-30M-i2048")

N_CELLS          = int(os.environ.get("GF_NCELLS", 355046))   # smoke-test override only
MODEL_INPUT_SIZE = 2048      # 30M series
SPECIAL_TOKEN    = False     # 30M series has no <cls>/<eos>
VOCAB            = 25426
HIDDEN           = 256
N_LAYERS         = 6
EMB_LAYER        = -1                       # EmbExtractor default
LAYER_TO_QUANT   = N_LAYERS + EMB_LAYER     # == 5  (hidden_states[5], 2nd-to-last)
SHARD            = int(os.environ.get("GF_SHARD", 10000))                         # cells per shard -> 36 shards
TOK_CHUNK        = 512                      # tokenizer.py inner chunk (dense (chunk,19512) float64 = 80 MB)
BATCH            = 16                       # measured optimum on BOTH cpu and mps
THREADS          = 8                        # measured: 5 threads 0.096, 8 threads 0.069, 10 threads 0.070 s/cell
DEVICE           = os.environ.get("GF_DEVICE", "cpu")   # "mps" (~4.1 h) or "cpu" (~6.7 h)

OUT   = Path(os.environ.get("GF_OUT", HR_WORK + "/gf6l_emb"))
SHARD_DIR = OUT / "shards"      # temp h5ad shards, deleted after tokenizing
TOK_DIR   = OUT / "tok"         # tokenized .dataset per shard (kept: lets you re-embed w/o re-tokenizing)
EMB_DIR   = OUT / "emb"         # emb_%04d.npy + idx_%04d.npy + len_%04d.npy per shard
FINAL     = OUT / "gf6L_emb.npz"
MANIFEST  = OUT / "manifest.json"

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]

def cfg_dict():
    return {
        "ckpt": CKPT, "ckpt_sha_safetensors": _sha(CKPT + "/model.safetensors"),
        "token_dict_sha": _sha(TOKEN_DICTIONARY_FILE),
        "gene_median_sha": _sha(GENE_MEDIAN_FILE),
        "ensembl_map_sha": _sha(ENSEMBL_MAPPING_FILE),
        "model_input_size": MODEL_INPUT_SIZE, "special_token": SPECIAL_TOKEN,
        "layer_to_quant": LAYER_TO_QUANT, "shard": SHARD, "n_cells": N_CELLS,
        "pool": "mean_over_attention_mask", "src_h5ad": SRC_H5AD,
    }

def cfg_hash():
    return hashlib.sha256(json.dumps(cfg_dict(), sort_keys=True).encode()).hexdigest()[:16]
