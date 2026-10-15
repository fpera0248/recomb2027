# Blind recovery of hidden rare subtypes

Code for the RECOMB 2027 paper. Given a pool of cells with a coarse visible label (say "T cell"), the method ranks cells by how likely they belong to a finer subtype that is not labelled. It never sees the finer annotation.

This repository holds our method (`hiddenrare/`), the evaluation (`eval/`), the builders of the test pools (`data/`), the wrappers for the published detectors we compare against (`tools/`, `scripts/`), the study of known rare types (`part1/`), the hidden-type calls plus duplication experiment (`combination/`) and the notebooks that recompute the numbers of the paper (`notebooks/`).

## Method

1. `hiddenrare/candidates.py`: Leiden clustering of each label at 8 resolutions in two spaces (pool PCA-50, and per-label HVG + PCA-50). Clusters with at least 10 cells and under 10% of the label are candidate groups.
2. `hiddenrare/markers.py`, `halo.py`: marker and neighbourhood ("halo") statistics per group. Together with geometry features this gives 41 features per group.
3. `hiddenrare/scorer.py`: a gradient-boosted regressor (`models/refit_dev_nat.pkl`) predicts how pure each group is. A cell's score is the best group it belongs to, plus a small isolation tie-break.
4. `hiddenrare/baselines.py`, `blend.py`: the final score is 0.75 * rank(scorer) + 0.25 * rank(Leiden subclustering score).

Feature-table column names (`mk_*`, `hl_*`) are kept from the training tables because `hiddenrare/scorer.py` reads them by name in the order the model was fit with.

## Install

Pick one.

**Container (recommended).** Two prebuilt images, built by `.github/workflows/build-images.yml` and pushed to GHCR on every release:

| Image | Contents |
|---|---|
| `ghcr.io/fpera0248/recomb2027-hiddenrare` | our method, the evaluation, the notebooks (Python 3.12) |
| `ghcr.io/fpera0248/recomb2027-tools` | the published detectors: Python tools, CellSIUS, GapClust, SCISSORS (R 4.6.1, Seurat 4.4.0) |

```
docker pull ghcr.io/fpera0248/recomb2027-hiddenrare:latest
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:latest \
    env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
Apptainer and details: `CONTAINER.md`. While the repository is private the images are private too; log in to `ghcr.io` with a token that can read packages.

**Python environment.** Python 3.12, `pip install -r requirements.txt` or `conda env create -f environment_hiddenrare.yml`. The versions are those of the interpreter the 10-atlas results were computed with (macOS arm64); `piplist_hiddenrare.txt` is its full `pip list`. scikit-learn must be 1.5.2 (the model pickle was written with it). Leiden partitions depend on the numerical stack, so on another machine per-cell scores differ (Spearman about 0.8 to 0.9 on one test pool) and pool union AP moves by about 0.01. Rerunning reproduces the pipeline and the evaluation, not the exact per-cell scores of the paper. The published tools have their own environment (`environment_tools.yml`, `Rpackages_tools.csv`, `piplist_tools.txt`); see `ENVIRONMENTS.md`.

## Reproduce the numbers

```
./reproduce.sh paper     # download the release assets, recompute the Part II numbers of the 8-atlas test set, assert the match
./reproduce.sh rerun     # rescore the 69 held-out pools with this repository's scorer
./reproduce.sh full      # rebuild the 10-atlas test pools from CELLxGENE and rerun everything
```

| Tier | Input | Exactness | Time and disk |
|---|---|---|---|
| paper | saved predictions of every method on the 69 pools of the 8-atlas test set, and the Part I result tables | exact: values, tests, null-test rates and Part I figures are asserted; the bootstrap interval is checked to 0.005 | about 2 minutes, 1.2 GB of downloads (measured on the run machine) |
| rerun | frozen candidate tables and Leiden scores | not exact: mean Spearman of per-cell scores 0.88 against the saved ones, atlas-equal union AP 0.3195 against 0.3213; asserted within 0.01 | about 2 minutes, same assets as paper (measured) |
| full | CELLxGENE source files | approximate: atlas-equal means and differences within 0.03 of the paper, same sign, atlas-level exact p below 0.05 (`eval/full_check.py`) | about 30 GB of downloads and 20 GB of pool files, hours to days on 12 cores (estimated from the run machine; not re-timed) |

At the `paper` tier the results match the paper exactly: every table value, test and null-test rate is recomputed from the saved predictions and asserted, so a mismatch fails the run. The `rerun` and `full` tiers rerun code and can differ from the paper as stated in the table. Time estimates for the containers have not been measured, and the images have not been built from this checkout (no Docker on the machine that prepared them); the first CI build is the first test of the Dockerfiles.

`fetch_data.sh` downloads the assets of a release (`TAG`, default v0.1.0) and checks them against `ASSETS.md5`; for `full` it downloads the source atlases listed in `ATLAS_MANIFEST.csv` and checks their sizes (and md5 where the manifest has one). `python data/build_manifest.py --check` re-verifies the URLs; `--refresh` rewrites the manifest from the CELLxGENE curation API (it then points to the latest dataset versions).

While the repository is private, the plain release URLs return 404. `fetch_data.sh` then falls back to `gh release download` (install the GitHub CLI and log in with read access), or set `ASSET_SRC=DIR` to take the assets from a local directory. Set `TAG` for another release.

## Atlases

The source atlases are public CELLxGENE files. `ATLAS_MANIFEST.csv` lists every one with its collection, its `dataset_version_id` download URL, size, md5 where known and the set it belongs to (`name` is the short id used by the builders). `data/atlases.csv` is the list of the 10-atlas test set, `data/used_datasets.csv` the datasets used before the test. Download them with `data/download_atlases.sh` (or `reproduce.sh full`).

## Notebooks

`reproduce.sh paper` executes `notebooks/01_heldout_test.ipynb` (our method against Leiden subclustering and the in-house baselines), `02_published_detectors.ipynb` (published detectors, MARS and scCAD style metrics), `03_null_test.ipynb` (twin pools with the hidden cells removed) and `04_part1_summary.ipynb` (the 88-dataset panel, the 31-type run, duplication as sample weighting). The notebooks only call `eval/paper.py`, `eval/nulltest.py` and `part1/`. Not recomputed by `paper`: the 10-atlas results (the 10-atlas test, strata, separability tiers, label ablation, donor ablation, CHOIR and the tools that ran only there), scGAD on 32 pools, the random-ranking line of Fig. 1d, and the Part I runs outside notebook 04 (generator training, the full data panel, the AIDA, HLCA and lupus runs); notebook 04 lists what it covers. The rerun tier rescores frozen candidate tables with this repository's scorer; it does not rebuild embeddings or Leiden partitions. The `paper` and `rerun` tiers use only the 69 pools of the 8-atlas test set; the 10-atlas predictions are produced by `full`.

## Rebuild the test sets

See "Rebuilding the 10-atlas test and the evaluation" below: `data/build_pools.py` writes the pools and `data/build_twins.py` the null-gate twin pools from the atlases of the manifest; `data/manifests` and `data/targets` are the pool lists and target tables of the original run.

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

## Hidden-type calls plus duplication

`combination/` tests whether calling hidden types and duplicating the called cells helps a classifier find the hidden cells in new donors: our hidden-type calls plus duplication, against Leiden subclustering calls plus duplication, with matched call counts and with true labels as the upper bound. Each pool is split by donor, both methods scored the whole pool with the test donors unlabelled; only training-donor calls become labels, and the classifier never sees test-donor labels. The held-out donors are scored. `combination/DESIGN.md` is the design, fixed before any full-run result.

```
export HR_DATA=assets HR_POOLS=/path/to/pool_h5ads    # HR_DATA: paper-tier assets (pilots/, manifests/, pools.csv, predictions/)
export OUT=work/combination_rows                      # keep outputs out of the source tree
python -m combination.run_one atl_luad_p0 0 --out $OUT/one.csv                 # one pool, one seed: one row per setting
combination/run_all.sh lr                                    # all eligible pools and seeds, then the analysis; also rf and mlp
```
`HR_POOLS` holds `<source_path>` h5ad files (raw counts, all genes) for the 69 pools of the 8-atlas test set, for example `luad_p0.h5ad`. `data/build_pools.py` does not write them (it builds the 10-atlas pools) and the paper-tier assets do not contain them, so the experiment cannot yet be run from the repository and release assets alone. Results will be added here when the full run is finished.

## Published tools

`tools/sources.txt` lists the official repositories and commits; `tools/fetch_sources.sh` clones them into `$TOOLS_SRC`. The wrappers in `tools/` only adapt input and output. R tools need Seurat 4.4.0 / SeuratObject 4.1.4. The `recomb2027-tools` image (`docker/Dockerfile.tools`) installs all of this; `Rpackages_tools.csv` and `piplist_tools.txt` are the exact package sets of the environment the tools ran in.

Python tools run in their own environment: `pip install -r tools/requirements.txt; pip install --no-deps $TOOLS_SRC/GiniClust3 $TOOLS_SRC/scGPT`; build FiRE with `python setup.py build_ext --inplace` in `FIRE_BUILD` (Boost headers needed); set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary. `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2. The reported scGPT run used `DEVICE=mps`.

## Datasets used before the test

Datasets used before the test are listed in `data/used_datasets.csv`; the Human Breast Cell Atlas (collection 48259aa8) was also used and is not in that list.

## Part I code

`part1/` holds the code behind the duplication-versus-generators study of known rare types. `part1/run.sh` lists the commands: `aida.py` (all generators and duplication on AIDA), `split_cache.py`, `statistical.py` and `dup_weight.py` (statistical augmentation conditions and sample weighting on the same split), `hlca.py` and `lupus.py` (the same protocol on two more datasets), and `panel_run.sh`, `panel_pull.py`, `panel_readout.py`, `noaug.py` and `panel_analysis.py` (the 88-dataset CELLxGENE panel). `scdesign3.R` is the scDesign3 driver and `diffusion.py` wraps scDiffusion. These runs need the source h5ad files, CFGen and scDiffusion checkouts, mem_patch.R for scDesign3, and days of CPU compute; the `paper` tier only reads their result tables through `part1/panel.py` and `part1/probe31.py`.

## Not included

CELLxGENE data (downloaded by the builders), the training pools, and the tool checkouts. `hiddenrare/scorer.py fit` is the training script; it needs the training pools. Run it from the repository root (it imports eval/targets.py). Also not included: the runtime and memory table of the paper (per-job `/usr/bin/time -l` logs on the run machine).
