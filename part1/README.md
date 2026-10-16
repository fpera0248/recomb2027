## Part I code

Run the commands in `part1/run.sh` from the `part1/` directory.

`part1/` holds the code behind the duplication-versus-generators study of known rare types. `part1/run.sh` lists the commands:

- `aida.py`: all generators and duplication on AIDA.
- `split_cache.py`, `statistical.py` and `dup_weight.py`: statistical augmentation conditions and sample weighting on the same split.
- `hlca.py` and `lupus.py`: the same protocol on two more datasets.
- `panel_run.sh` (which calls `panel_pull.py`, `panel_readout.py` and `noaug.py`) and `panel_analysis.py`: the 88-dataset CELLxGENE panel.

`scdesign3.R` is the scDesign3 driver and `diffusion.py` wraps scDiffusion. These runs need the source h5ad files, CFGen and scDiffusion checkouts, and `part1/mem_patch.R` for scDesign3. `MEM_PATCH` points at it; `SD3_MEM=1`, which `scdesign3.R` sets, turns it on. They also need days of CPU compute.

The `paper` tier only reads their result tables through `part1/panel.py` and `part1/probe31.py`. Notebook 04 reads the 31-type results as `probe31_runs.csv` in `part1_results.tgz`. `probe31_runs.csv` has the columns of the `statistical_p31.csv` that the 31-type `statistical.py run 0 31` step writes, except that `method` is named `arm` (`probe31.table` accepts both). It holds nine of the methods in `statistical_p31.csv`.

`panel_run.sh` sets `SKIP_SD3=1`, so `run.sh` does not run unmodified scDesign3 on the panel. The paper's unmodified scDesign3 results (65 datasets, and the comparison on 8 of them) come from separate runs whose commands and outputs are not in this repository or in `part1_results.tgz`. `hlca.py` has an `ONLY_SD3=1` mode that fits and saves only the scDesign3 methods.

`panel_queue.txt` lists 93 datasets. Five of them (19053a82, 56a4bc14, 7b20c613, 9968be68 and f7c1c579) fail the preflight check of `hlca.py`. `PREFLIGHT=1` checks every seed's donor split and probe sizes and trains nothing. `panel_run.sh` marks them `failed` and logs them in `panel_runs/progress.log`. `run.sh` continues past them; the panel is the other 88.

`panel_queue_shuffled.txt` is the order `panel_run.sh` follows when called without a dataset id; `run.sh` does not use it.
