# Blind recovery of hidden rare subtypes

Code for the RECOMB 2027 paper. Given a pool of cells with a coarse visible label (say "T cell"), the method ranks cells by how likely they belong to a finer subtype that is not labelled. It never sees the finer annotation.

## Method

1. `hiddenrare/candidates.py`: Leiden clustering of each label at 8 resolutions in two spaces (pool PCA-50, and per-label HVG + PCA-50). Clusters with at least 10 cells and under 10% of the label are candidate groups.
2. `hiddenrare/markers.py`, `halo.py`: marker and neighbourhood ("halo") statistics per group. Together with geometry features this gives 41 features per group.
3. `hiddenrare/scorer.py`: a gradient-boosted regressor (`models/refit_dev_nat.pkl`) predicts how pure each group is. A cell's score is the best group it belongs to, plus a small isolation tie-break.
4. `hiddenrare/baselines.py`, `blend.py`: the final score is 0.75 * rank(scorer) + 0.25 * rank(Leiden subclustering score).

Feature-table column names (`mk_*`, `hl_*`) are kept from the training tables because `hiddenrare/scorer.py` reads them by name in the order the model was fit with.

## Install

Python 3.12. `pip install -r requirements.txt`. scikit-learn must be 1.5.2 (the model pickle was written with it). The 10-atlas results were computed with these versions on macOS arm64. Leiden partitions depend on the numerical stack, so on another machine per-cell scores differ (Spearman about 0.8 to 0.9 on one test pool) and pool union AP moves by about 0.01. Rerunning reproduces the pipeline and the evaluation, not the exact per-cell scores of the paper.

## One pool

```
scripts/run_pool.sh POOL.h5ad PILOT.parquet OUT_DIR
```
`POOL.h5ad` holds raw counts. `PILOT.parquet` has columns `obs_name`, `label` and `donor`; the pool id is the parquet file name. Cell ids are opaque hashes of the barcode. Output: `OUT_DIR/preds/hiddenrare/<pool>.parquet` (`cell_id`, `score`), plus the Leiden and isolation-forest baselines and the intermediate tables in `OUT_DIR/cands`.

## Rebuilding the 10-atlas test and the evaluation

```
export HR_DATA=work
data/download_atlases.sh                      # public CELLxGENE files, large
python -m data.build_pools $(cut -d, -f1 data/atlases.csv | tail -n +2)
python -m data.build_twins <names>            # null-gate twin pools
python -m data.leak_check cells <names> --panel work/panel    # panel: each dataset of data/used_datasets.csv downloaded from CELLxGENE into work/panel (not scripted)
```
Then run `scripts/run_pool.sh` on every pool, the published tools with `scripts/run_tool.sh` (inputs from `tools/adapt_pool.py`), and `eval/run_eval.sh $HR_DATA RUN_DIR OUT_DIR` to score and analyse. `eval/signflip.py` recomputes the exact sign-flip p-values independently. `data/manifests` and `data/targets` are the pool lists and target tables of the original run; they are what `build_pools.py` writes.

## Published tools

`tools/sources.txt` lists the official repositories and commits; `tools/fetch_sources.sh` clones them into `$TOOLS_SRC`. The wrappers in `tools/` only adapt input and output. R tools need Seurat 4.4.0 / SeuratObject 4.1.4.

Python tools run in their own environment: `pip install -r tools/requirements.txt; pip install --no-deps $TOOLS_SRC/GiniClust3 $TOOLS_SRC/scGPT`; build FiRE with `python setup.py build_ext --inplace` in `FIRE_BUILD` (Boost headers needed); set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary. `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2. The reported scGPT run used `DEVICE=mps`.

## Datasets used before the test

Datasets used before the test are listed in `data/used_datasets.csv`; the Human Breast Cell Atlas (collection 48259aa8) was also used and is not in that list.

## Not included

CELLxGENE data (downloaded by the builders), the training pools, and the tool checkouts. `hiddenrare/scorer.py fit` is the training script; it needs the training pools. Run it from the repository root (it imports eval/targets.py). Also not included: the runtime and memory table of the paper (per-job `/usr/bin/time -l` logs on the run machine).
