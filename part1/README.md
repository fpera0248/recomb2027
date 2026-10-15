## Part I code

Run the commands from the repository root.

`part1/` holds the code behind the duplication-versus-generators study of known rare types. `part1/run.sh` lists the commands: `aida.py` (all generators and duplication on AIDA), `split_cache.py`, `statistical.py` and `dup_weight.py` (statistical augmentation conditions and sample weighting on the same split), `hlca.py` and `lupus.py` (the same protocol on two more datasets), and `panel_run.sh`, `panel_pull.py`, `panel_readout.py`, `noaug.py` and `panel_analysis.py` (the 88-dataset CELLxGENE panel). `scdesign3.R` is the scDesign3 driver and `diffusion.py` wraps scDiffusion. These runs need the source h5ad files, CFGen and scDiffusion checkouts, `part1/mem_patch.R` for scDesign3 (`MEM_PATCH` points at it; `SD3_MEM=1`, which `scdesign3.R` sets, turns it on), and days of CPU compute; the `paper` tier only reads their result tables through `part1/panel.py` and `part1/probe31.py`.
