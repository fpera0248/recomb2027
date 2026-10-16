"""Load the REAL, unmodified geneformer/tokenizer.py without triggering
geneformer/__init__.py (which eagerly imports ray/optuna/tdigest/seaborn/... AND
hardwires the WRONG 95M dictionaries as defaults)."""
import importlib.util, sys, types
import gfcfg as C

def real_tokenizer_module():
    fake = types.ModuleType("geneformer")
    fake.__path__ = [str(C.PKG)]
    fake.GENE_MEDIAN_FILE      = C.GENE_MEDIAN_FILE
    fake.TOKEN_DICTIONARY_FILE = C.TOKEN_DICTIONARY_FILE
    fake.ENSEMBL_MAPPING_FILE  = C.ENSEMBL_MAPPING_FILE
    sys.modules["geneformer"] = fake
    spec = importlib.util.spec_from_file_location("geneformer.tokenizer", C.PKG / "tokenizer.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["geneformer.tokenizer"] = m
    spec.loader.exec_module(m)          # executes the ACTUAL file on disk
    return m

def make_tokenizer():
    import pickle
    m = real_tokenizer_module()
    tk = m.TranscriptomeTokenizer(
        custom_attr_name_dict={"orig_row_idx": "orig_row_idx"},
        nproc=1, chunk_size=C.TOK_CHUNK,
        model_input_size=C.MODEL_INPUT_SIZE,
        special_token=C.SPECIAL_TOKEN,
        collapse_gene_ids=True,
        gene_median_file=C.GENE_MEDIAN_FILE,
        token_dictionary_file=C.TOKEN_DICTIONARY_FILE,
        gene_mapping_file=C.ENSEMBL_MAPPING_FILE,
    )
    # GATE 1 -- wrong-dictionary guard. The 95M dict has 20275 entries and <cls>/<eos>;
    # the 30M model's embedding table is 25426 wide so 95M ids would NOT crash, they
    # would silently name the wrong gene for ~99% of tokens.
    assert len(tk.gene_token_dict) == C.VOCAB, f"token dict is not gc30M: {len(tk.gene_token_dict)}"
    assert "<cls>" not in tk.gene_token_dict and "<eos>" not in tk.gene_token_dict
    assert tk.gene_token_dict["<pad>"] == 0 and tk.gene_token_dict["<mask>"] == 1
    assert tk.model_input_size == 2048 and tk.special_token is False
    return tk
