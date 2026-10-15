# Finding rare cell subtypes hidden inside broad labels

Code for the RECOMB 2027 paper. Given a pool of cells with a coarse visible label (say "T cell"), the method ranks cells by how likely they belong to a finer subtype that is not labelled. It never sees the finer annotation.

This repository holds our method (`hiddenrare/`), the evaluation (`eval/`), the builders of the test pools (`data/`), the wrappers for the published detectors we compare against (`tools/`, `scripts/`), the study of known rare types (`part1/`), the hidden-type calls plus duplication experiment (`combination/`) and the notebooks that recompute the numbers of the paper (`notebooks/`).

## Method

1. `hiddenrare/candidates.py`: Leiden clustering of each label at 8 resolutions in two spaces (pool PCA-50, and per-label HVG + PCA-50). Clusters with at least 10 cells and under 10% of the label are candidate groups.
2. `hiddenrare/markers.py`, `halo.py`: marker and neighbourhood ("halo") statistics per group. Together with geometry features this gives 41 features per group.
3. `hiddenrare/scorer.py`: a gradient-boosted regressor (`models/refit_dev_nat.pkl`) predicts how pure each group is. A cell's score is the best group it belongs to, plus a small isolation tie-break.
4. `hiddenrare/baselines.py`, `blend.py`: the final score is 0.75 * rank(scorer) + 0.25 * rank(Leiden subclustering score).

The saved scores of the 8-atlas test set come from an earlier fit of the same recipe that was not kept; `models/refit_dev_nat.pkl` is a refit. The `paper` tier is therefore exact (it reads the saved scores) and the `rerun` tier is a consistency check (`eval/rerun.py`).

Feature-table column names (`mk_*`, `hl_*`) are kept from the training tables because `hiddenrare/scorer.py` reads them by name in the order the model was fit with.

## Install

Pick one.

**Python environment.** Python 3.12, `pip install -r requirements.txt` or `conda env create -f environment_hiddenrare.yml`. The versions are those of the interpreter the 10-atlas results were computed with (macOS arm64); `piplist_hiddenrare.txt` is the dependency closure of `requirements.txt` in that interpreter. scikit-learn must be 1.5.2 (the model pickle was written with it). Leiden partitions depend on the numerical stack, so on another machine per-cell scores differ. Rerunning reproduces the pipeline and the evaluation, not the exact per-cell scores of the paper. The published tools have their own environment (`environment_tools.yml`, `Rpackages_tools.csv`, `piplist_tools.txt`); see `ENVIRONMENTS.md`.

**Container (untested).** Two images, defined in `docker/` and built by `.github/workflows/build-images.yml`, which pushes them to GHCR on every release. They have not been built or run yet. They are amd64 only (emulated on Apple silicon, for example `docker run --platform linux/amd64`):

| Image | Contents |
|---|---|
| `ghcr.io/fpera0248/recomb2027-hiddenrare` | our method, the evaluation, the notebooks (Python 3.12) |
| `ghcr.io/fpera0248/recomb2027-tools` | the published detectors: Python tools, CellSIUS, GapClust, SCISSORS (R 4.6.1, Seurat 4.4.0) |

```
docker pull ghcr.io/fpera0248/recomb2027-hiddenrare:latest
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:latest \
    env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
Apptainer and details: `CONTAINER.md`.

## Reproduce the numbers

```
./reproduce.sh paper     # download the release assets, recompute the Part II numbers of the 8-atlas test set plus the Part I panel and extra checks, assert the match
./reproduce.sh rerun     # rescore the 69 held-out pools with this repository's scorer
./reproduce.sh full      # rebuild the 10-atlas test pools from CELLxGENE and rerun our method, Leiden subclustering and (TOOLS=1) the published tools on them (untested)
```

| Tier | Input | Exactness | Time and disk |
|---|---|---|---|
| paper | saved predictions of every method run on the 69 pools of the 8-atlas test set, and the Part I result tables | exact: values, tests, null-test rates and the Part I numbers of notebook 04 are asserted; the bootstrap interval is checked to 0.005 | about 4 minutes plus the download, 1.2 GB downloaded and about 2.5 GB on disk (measured on a laptop, 4 threads) |
| rerun | frozen candidate tables and Leiden scores | not exact (the saved scores come from an earlier fit that was not kept): mean Spearman of per-cell scores 0.878 against the saved ones, atlas-equal union AP 0.3195 against 0.3213, single pools can move by up to 0.2; asserted within 0.01 and Spearman above 0.8 (`eval/rerun.py`) | about 2 minutes, about 1.05 GB downloaded (pools, predictions, rerun inputs) (measured) |
| full | CELLxGENE source files | approximate: atlas-equal means and differences within 0.03 of the paper, same sign, atlas-level exact p below 0.05 (`eval/full_check.py`) | 29.3 GB of downloads (sum of the manifest sizes); time and disk for the pools and the run are not measured; untested end to end |

`full` does not rebuild the twin pools, run the leak check or the Part I panel, and does not produce the strata, separability tiers, ablations, CHOIR or figures. `TOOLS=1` also runs the published tools: they need their own Python environment (scikit-learn 1.9.1, which conflicts with `requirements.txt`; point `TOOLS_PYTHON` at it, see `ENVIRONMENTS.md`) and `SCGPT_CKPT`, a tool with an output is skipped and failures go to `$OUT/tool_failures.txt`. Without `TOOLS=1` only the Leiden comparison is checked and T2 (against scCAD) is not evaluated. `eval/full_check.py` asserts 57 pools in 10 atlases.

At the `paper` tier every value the notebooks print is recomputed from the saved predictions and asserted (see Paper items for what is partial), so a mismatch fails the run; the null-test AUC recomputes to 0.61 against the 0.60 printed. The `rerun` and `full` tiers rerun code and can differ from the paper as stated in the table. Time estimates for the containers have not been measured, and the images have not been built from this checkout (no Docker on the machine that prepared them); the first run of `build-images.yml` (on a release or manual dispatch) is the first test of the Dockerfiles.

`fetch_data.sh` downloads the assets of a release (`TAG`, default v0.1.0) and checks them against `ASSETS.md5`; for `full` it downloads the source atlases listed in `ATLAS_MANIFEST.csv` and checks their sizes (and md5 where the manifest has one). `python data/build_manifest.py --check` re-verifies the URLs; `--refresh` rewrites the manifest from the CELLxGENE curation API (it then points to the latest dataset versions).

A missing release gives 404 (check `TAG`, default v0.1.0). `fetch_data.sh` then falls back to `gh release download` (install the GitHub CLI and log in). Set `ASSET_SRC=DIR` to take the assets from a local directory, for offline use.

## Atlases

The source atlases are public CELLxGENE files. `ATLAS_MANIFEST.csv` lists every one with its collection, its `dataset_version_id` download URL, size, md5 where known and the set it belongs to (`key` for the 10-atlas set, `name` for the 8-atlas set, is the short id used by the builders; rows without a URL say why in `notes`). `data/atlases.csv` is the list of the 10-atlas test set, `data/used_datasets.csv` the datasets used before the test. Download them with `data/download_atlases.sh` (or `reproduce.sh full`).

## Notebooks

`reproduce.sh paper` executes `notebooks/01_heldout_test.ipynb` (our method against Leiden subclustering and the in-house baselines), `02_published_detectors.ipynb` (published detectors, metrics used in the MARS and scCAD papers), `03_null_test.ipynb` (twin pools with the hidden cells removed: 101 real and twin pairs from 12 atlases, then the 69 pairs of the 8-atlas test set) and `04_part1_summary.ipynb` (the 88-dataset panel, an extra 31-type check, duplication as sample weighting). The notebooks assert numbers and draw no figures. They only call `eval/paper.py`, `eval/nulltest.py` and `part1/`. Not recomputed by `paper`: the 10-atlas results (the 10-atlas test, strata, separability tiers, label ablation, donor ablation, CHOIR and the tools that ran only there), scGAD on 32 pools, the random-ranking line of Fig. 1d, and the Part I runs outside notebook 04 (generator training, the full data panel, the AIDA, HLCA and lupus runs); notebook 04 lists what it covers. The rerun tier rescores frozen candidate tables with this repository's scorer; it does not rebuild embeddings or Leiden partitions. The `paper` and `rerun` tiers use the 69 pools of the 8-atlas test set (notebook 03 also reads the twins of 12 atlases). The 8-atlas test set (69 pools) and the 10-atlas test set (57 pools) are different sets; the 10-atlas predictions are produced by `full` (untested).

## Paper items

`reproduce.sh paper` recomputes numbers; no tier draws a figure. "Reproduced" means the notebook recomputes and asserts the values behind the item, "partial" that only some of them.

| Paper item | Notebook section | Tier | Status |
|---|---|---|---|
| Fig. 1a, augmentation minus duplication over 88 datasets | 04 section 1 | paper | partial: win counts and deep-generator summary; the per-test counts are not recomputed |
| Fig. 1b, default threshold against matched point | 04 section 1 | paper | partial: duplication against no augmentation only |
| Fig. 1c, duplication equals sample weighting | 04 section 2 | paper | reproduced |
| Fig. 1d, union AP of our method, Leiden subclustering, scCAD and the loops | 01 sections 2, 4, 5; 02 section 2 | paper | partial: union AP values; the random-ranking line is not recomputed |
| Fig. 2a, atlas-equal means of the methods | 01 section 5; 02 sections 1 and 2 | paper | partial: every method except classifier uncertainty |
| Fig. 2b, per-atlas differences | 01 sections 2 and 3; 02 section 3 | paper | partial: per-atlas differences against Leiden subclustering are asserted; against scCAD with labels only the mean, atlas count and p |
| Fig. 3, separability tiers and UMAPs | none | none | not in repo (10-atlas results) |
| Fig. 4, hidden-type calls plus duplication | none | none | not in repo (`combination/` code only, results pending) |
| Supp. Fig. S1, example pool | none | none | not in repo |
| Supp. Fig. S2, null twins | 03 (101 pairs, opening cells) and 03 section 2 (69 pairs) | paper | reproduced: rates; the AUC recomputes to 0.61 against the 0.60 printed |
| Supp., published detectors against our method | 02 sections 2 and 3 | paper | reproduced |
| Supp., metrics for the called cells | 02 sections 3 and 4 | paper | reproduced |
| 31-type AIDA run: the values asserted here are not printed in the paper; the paper's 31-type statement in the matched-operating-point paragraph is not recomputed | 04 section 3 | paper | recomputed (other readout) |
| Supp., per-subtype results by tier | none | none | not in repo |
| Supp., datasets | none | none | partial: `ATLAS_MANIFEST.csv`, `data/used_datasets.csv` hold the sources, not the table |
| Supp., duplication study conditions, methods compared, methods missing, comparator checks | none | none | not in repo |
| Label and donor ablations, strata, CHOIR | none | none | not in repo |

## One pool

```
scripts/run_pool.sh POOL.h5ad PILOT.parquet OUT_DIR
```
`POOL.h5ad` holds raw counts. `PILOT.parquet` has columns `obs_name`, `label` and `donor`; the pool id is the parquet file name. Cell ids are opaque hashes of the barcode. Output: `OUT_DIR/preds/hiddenrare/<pool>.parquet` (`cell_id`, `score`), plus the Leiden and isolation-forest baselines and the intermediate tables in `OUT_DIR/cands`.

## Rebuild the test sets

`data/build_pools.py` writes the pools and `data/build_twins.py` the null-gate twin pools of the 10-atlas test set (`data/atlases.csv`) from the atlases of the manifest. `data/manifests` and `data/targets` are reference copies of the pool lists and target tables of the original run: diff them against what `build_pools.py` writes to `$HR_DATA`. The 101 twin pools of the paper (12 atlases) ship as predictions in `predictions.tgz` (`ids/twin_manifest.csv`) and no tier rebuilds them; the twin tier is not part of `full`, and twins rebuilt by `build_twins.py` are not claimed to be identical to the shipped ones.

```
export HR_DATA=work
data/download_atlases.sh                      # public CELLxGENE files, large
python -m data.build_pools $(cut -d, -f1 data/atlases.csv | tail -n +2)
python -m data.build_twins <names>            # null-gate twin pools
python -m data.leak_check cells <names> --panel work/panel    # panel: each dataset of data/used_datasets.csv downloaded from CELLxGENE into work/panel (not scripted)
```
Then run `scripts/run_pool.sh` on every pool, the published tools with `scripts/run_tool.sh` (inputs from `tools/adapt_pool.py`), and `eval/run_eval.sh $HR_DATA RUN_DIR OUT_DIR` to score and analyse. `eval/signflip.py` recomputes the exact sign-flip p-values independently.

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

Python tools run in their own environment (set `TOOLS_PYTHON` for `scripts/run_tool.sh` and `reproduce.sh full`): `pip install -r tools/requirements.txt; pip install --no-deps $TOOLS_SRC/GiniClust3 $TOOLS_SRC/scGPT`; build FiRE by copying `python/FiRE` of its checkout and `tools/fire_setup.py` (as `setup.py`) into `FIRE_BUILD` and running `python setup.py build_ext --inplace` there (Boost headers needed); set `SCGPT_CKPT` to the scGPT whole-human checkpoint dir and `MCL` to the mcl binary. `$TOOLS_SRC/scCAD/scCAD.py` must have md5 a0865c37513aaee5eae13551e48da6f2. The reported scGPT run used `DEVICE=mps`.

## Datasets used before the test

Datasets used before the test are listed in `data/used_datasets.csv`; the Human Breast Cell Atlas (collection 48259aa8) was also used and is not in that list.

## Part I code

`part1/` holds the code behind the duplication-versus-generators study of known rare types. `part1/run.sh` lists the commands: `aida.py` (all generators and duplication on AIDA), `split_cache.py`, `statistical.py` and `dup_weight.py` (statistical augmentation conditions and sample weighting on the same split), `hlca.py` and `lupus.py` (the same protocol on two more datasets), and `panel_run.sh`, `panel_pull.py`, `panel_readout.py`, `noaug.py` and `panel_analysis.py` (the 88-dataset CELLxGENE panel). `scdesign3.R` is the scDesign3 driver and `diffusion.py` wraps scDiffusion. These runs need the source h5ad files, CFGen and scDiffusion checkouts, mem_patch.R for scDesign3, and days of CPU compute; the `paper` tier only reads their result tables through `part1/panel.py` and `part1/probe31.py`.

## Not included and known gaps

- CELLxGENE data (downloaded by the builders), the training pools and the tool checkouts. `hiddenrare/scorer.py fit` is the training script; it needs the training pools. Run it from the repository root (it imports eval/targets.py).
- The 10-atlas predictions: the 10-atlas results (strata, separability tiers, ablations, CHOIR) cannot be rerun from the release.
- Figure scripts; the notebooks assert numbers and draw nothing.
- The 69 pool h5ads of the 8-atlas test set, needed by `combination/` and by any rescoring of those pools from counts.
- The inputs of the Part I runs (source h5ad files, generator checkouts) and the training of the generators.
- Reproductions of the published tools; only their wrappers and pinned environments are included.
- The `full` tier and the container images are untested.
- The runtime and memory table of the paper (per-job `/usr/bin/time -l` logs on the run machine).
