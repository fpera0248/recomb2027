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

## Reproducing the numbers

```
./reproduce.sh paper     # download the release assets, recompute the Part II numbers of the 8-atlas test set, assert the match
./reproduce.sh rerun     # rescore the 69 held-out pools with this repository's scorer
./reproduce.sh full      # rebuild the 10-atlas test pools from CELLxGENE and rerun everything
```
`fetch_data.sh` downloads the assets of a release (`TAG`, default v0.1.0) and checks them against `ASSETS.md5`; for `full` it downloads the source atlases listed in `ATLAS_MANIFEST.csv` and checks their sizes (and md5 where the manifest has one). `python data/build_manifest.py --check` re-verifies the URLs; `--refresh` rewrites the manifest from the CELLxGENE curation API (it then points to the latest dataset versions).

While the repository is private, the plain release URLs return 404. `fetch_data.sh` then falls back to `gh release download` (install the GitHub CLI and log in with read access), or set `ASSET_SRC=DIR` to take the assets from a local directory. Set `TAG` for another release.

| Tier | Input | Exactness | Time |
|---|---|---|---|
| paper | saved predictions of every method on the 69 pools of the 8-atlas test set, and the Part I result tables | exact: values, tests, null-test rates and Part I figures are asserted; the bootstrap interval is checked to 0.005 | 2 minutes, 1.2 GB |
| rerun | frozen candidate tables and Leiden scores | not exact: mean Spearman of per-cell scores 0.88 against the saved ones, atlas-equal union AP 0.3195 against 0.3213; asserted within 0.01 | 2 minutes |
| full | CELLxGENE source files | approximate: atlas-equal means and differences within 0.03 of the paper, same sign, atlas-level exact p below 0.05 (`eval/full_check.py`) | about 30 GB of downloads, hours to days |

The `paper` tier executes `notebooks/01_heldout_test.ipynb` (our method against Leiden subclustering and the in-house baselines), `02_published_detectors.ipynb` (published detectors, MARS and scCAD style metrics), `03_null_test.ipynb` (twin pools with the hidden cells removed) and `04_part1_summary.ipynb` (the 88-dataset panel, the 31-type run, duplication as sample weighting). The notebooks only call `eval/paper.py`, `eval/nulltest.py` and `part1/`. Not recomputed by `paper`: the 10-atlas results (the 10-atlas test, strata, separability tiers, label ablation, donor ablation, CHOIR and the tools that ran only there), scGAD on 32 pools, the random-ranking line of Fig. 1d, and the Part I runs outside notebook 04 (generator training, the full data panel, the AIDA, HLCA and lupus runs); notebook 04 lists what it covers. The rerun tier rescores frozen candidate tables with this repository's scorer; it does not rebuild embeddings or Leiden partitions. Container images, an image build workflow and per-tool environment files are not provided yet. The `paper` and `rerun` tiers use only the 69 pools of the 8-atlas test set; the 10-atlas predictions are produced by `full`.

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

## Part I code

`part1/` holds the code behind the duplication-versus-generators study of known rare types. `part1/run.sh` lists the commands: `aida.py` (all generators and duplication on AIDA), `split_cache.py`, `statistical.py` and `dup_weight.py` (statistical augmentation conditions and sample weighting on the same split), `hlca.py` and `lupus.py` (the same protocol on two more datasets), and `panel_run.sh`, `panel_pull.py`, `panel_readout.py`, `noaug.py` and `panel_analysis.py` (the 88-dataset CELLxGENE panel). `scdesign3.R` is the scDesign3 driver and `diffusion.py` wraps scDiffusion. These runs need the source h5ad files, CFGen and scDiffusion checkouts, mem_patch.R for scDesign3, and days of CPU compute; the `paper` tier only reads their result tables through `part1/panel.py` and `part1/probe31.py`.

## Not included

CELLxGENE data (downloaded by the builders), the training pools, and the tool checkouts. `hiddenrare/scorer.py fit` is the training script; it needs the training pools. Run it from the repository root (it imports eval/targets.py). Also not included: the runtime and memory table of the paper (per-job `/usr/bin/time -l` logs on the run machine).
