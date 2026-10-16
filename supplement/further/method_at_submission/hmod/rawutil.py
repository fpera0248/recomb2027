"""Shared one-liner: some CELLxGENE h5ads store processed data in .X and the
real raw counts in .raw (AIDA_phase1_v2, lupus_perez); others (hlca_m27, the
93 panel/*.h5ad pulls, the AIDA pilot pools) have raw counts directly in .X.
Always go through this so every script in h/ agrees on which is which."""
import anndata as ad
import pandas as pd


def raw_adata(sub):
    """Given an in-memory AnnData slice, return a new AnnData with raw counts
    in .X (same obs), preferring .raw when present."""
    if sub.raw is not None:
        return ad.AnnData(X=sub.raw.X.copy(), obs=sub.obs, var=pd.DataFrame(index=sub.raw.var_names))
    return sub
