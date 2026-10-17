# Reproduction details

## Atlases

The source atlases are public CELLxGENE files. `ATLAS_MANIFEST.csv` lists every one with its collection where recorded, its `dataset_version_id` download URL, size, md5 where known and the set it belongs to (`key` for the 10-atlas set, `name` for the 8-atlas set, is the short id used by the builders). Two rows have no URL (`hbca`, which is not in the paper, and `hlca`), and both have a note in `notes`. The URLs of five `panel` rows (2adb1f8a, 43aa19d2, 456e8b9b, b2dda353, f7c1c579) are the version the CELLxGENE API listed on 2026-10-12, and the version the run used is not recorded. Download them by `dataset_id` from the page in `collection_url`, which serves the current version.

`data/atlases.csv` lists the 10-atlas test set and `data/used_datasets.csv` lists the datasets used before the test.

Download the 10-atlas test set with `./fetch_data.sh full DIR` (`reproduce.sh full` calls it, and `data/download_atlases.sh` checks neither size nor md5). `full` downloads the 10 rows of set `test10` (29.3 GB) and checks each against its size, and against its md5 for the 3 rows that have one.

The 8-atlas set (`test8`, 69 pools) is the one the `paper` and `rerun` levels check. The 10-atlas set (`test10`, 57 pools) is the one `full` and `data/build_pools.py` build. The two share no dataset or collection.

## Notebooks

`reproduce.sh paper` executes four notebooks:

1. `notebooks/01_heldout_test.ipynb`: our method against Leiden subclustering and the in-house baselines.
2. `02_published_detectors.ipynb`: published detectors, with the metrics used in the MARS and scCAD papers.
3. `03_null_test.ipynb`: twin pools with the hidden cells removed (101 real and twin pairs from 12 atlases, then the 69 pairs of the 8-atlas test set).
4. `04_part1_summary.ipynb`: the 88-dataset panel, an extra 31-type check and duplication as sample weighting.

The notebooks only call `eval/paper.py`, `eval/nulltest.py` and `part1/`. They assert numbers and draw no figures.

The `paper` level does not recompute:

- the 10-atlas results (the 10-atlas test, strata, separability levels, label ablation, CHOIR and the tools that ran only there).
- the donor ablation, which recomputes features on the 69 pools of the 8-atlas test set.
- scGAD on 32 pools.
- the random-ranking line of Figs. 1d and 2a.
- the rebuilt-subsample block of Supp. Table S6 and the GiniClust3 and FiRE bars of Fig. 2a (notebook 02 scores the earlier GiniClust3 and FiRE runs on the original subsamples).
- classifier uncertainty in Figs. 1d and 2a.
- the Part I runs outside notebook 04 (generator training, the full data panel, the AIDA, HLCA and lupus runs). Notebook 04 lists what it covers.

The `rerun` level rescores frozen candidate tables with this repository's scorer. It does not rebuild embeddings or Leiden partitions. Both levels use the 69 pools of the 8-atlas test set (notebook 03 also reads the twins of 12 atlases).

## One pool

```
scripts/run_pool.sh POOL.h5ad PILOTS.parquet OUT_DIR
```
`POOL.h5ad` holds raw counts. `PILOTS.parquet` has columns `obs_name`, `label` and `donor`. The pool id is the parquet file name. Cell ids are opaque hashes of the barcode. Output: `OUT_DIR/preds/hiddenrare/<pool>.parquet` (`cell_id`, `score`), plus the Leiden and isolation-forest baselines and the intermediate tables in `OUT_DIR/cands`.

## Datasets used before the test

`data/used_datasets.csv` lists the datasets used before the test. A Human Breast Cell Atlas file from collection 48259aa8 was also used during development (row `hbca` of `ATLAS_MANIFEST.csv`). Its dataset id was not recorded, so the cell-level leak check may not cover it. The collection itself is in the list (dataset 55003f67), so the collection check of `data/build_pools.py` excludes every dataset of that collection.

## Quick check details

```
pip install -r requirements.txt    # Python 3.12
./reproduce.sh paper
```
- Download: 1.2 GB of release assets, about 2.5 GB on disk.
- Time: about 4 minutes plus the download (measured on a laptop, 4 threads).
- Success: the run prints `paper: all notebooks ran and every assertion held`.
- Checks: the `paper` level recomputes numbers from the saved predictions and result tables and asserts them (the bootstrap interval to 0.005). It does not refit the model. [PAPER_ITEMS.md](PAPER_ITEMS.md) lists what it covers.
- `./reproduce.sh rerun` applies a refit scorer as a consistency check, so its numbers differ a little (atlas-equal union AP 0.3195 against 0.3213).
- Container: the `docker run` line under Install details. The image is `v0.1.2`, and the `paper` and `rerun` levels pass in it.
- Reproduction problems: https://github.com/fpera0248/recomb2027/issues

## Install details

You can run the code in a Python environment or in a container.

**Python environment.** Run `pip install -r requirements.txt` under Python 3.12, or `conda env create -f environment_hiddenrare.yml` (scikit-learn 1.5.2 is required). The pins are the package versions of the Python 3.12 environment (macOS arm64) that computed the 10-atlas results. `ENVIRONMENTS.md` has the platform and score-reproducibility notes and the separate environment of the published tools.

**Container.** Two images are defined in `docker/` and published on GHCR with the release tag `v0.1.2`. They pull without a login. Use `v0.1.2`. The `v0.1.1` images have not been checked, and `latest` and `main` move with later builds. The `recomb2027-hiddenrare` image passes the `paper` and `rerun` levels on a GitHub Actions runner ([container check](https://github.com/fpera0248/recomb2027/actions/runs/38940072528)), as the `v0.1.0` image did. The `recomb2027-tools` image has not been run end to end. Both are amd64 only (emulated on Apple silicon, for example `docker run --platform linux/amd64`):

| Image | Contents |
|---|---|
| `ghcr.io/fpera0248/recomb2027-hiddenrare` | our method, the evaluation, the notebooks (Python 3.12) |
| `ghcr.io/fpera0248/recomb2027-tools` | the published detectors: Python tools, CellSIUS, GapClust, SCISSORS (R 4.6.1, Seurat 4.4.0) |

```
docker pull ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2 \
    env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
For Apptainer and more details, see `CONTAINER.md`.

## Reproduce the numbers

```
./reproduce.sh paper     # download the release assets, recompute the Part II numbers of the 8-atlas test set plus the Part I panel and extra checks, assert the match
./reproduce.sh rerun     # rescore the 69 held-out pools with this repository's scorer
./reproduce.sh full      # rebuild the 10-atlas test pools from CELLxGENE and rerun our method, Leiden subclustering and (TOOLS=1) the published tools on them (untested)
```

| Level | Input | Exactness | Time and disk |
|---|---|---|---|
| paper | saved predictions of every method run on the 69 pools of the 8-atlas test set, and the Part I result tables | exact: values, tests, null-test rates and the Part I numbers of notebook 04 are asserted, and the bootstrap interval is checked to 0.005 | about 4 minutes plus the download, 1.2 GB downloaded and about 2.5 GB on disk (measured on a laptop, 4 threads) |
| rerun | frozen candidate tables and Leiden scores | not exact (refit scorer, see Method): mean Spearman of per-cell scores 0.878 against the saved ones, atlas-equal union AP 0.3195 against 0.3213, single pools can move by up to 0.2. The check asserts atlas-equal union AP within 0.01 and mean Spearman above 0.8 (`eval/rerun.py`) | about 2 to 3 minutes (2 min 10 s on a GitHub Actions runner after the paper level had downloaded the pools and predictions, see `CONTAINER.md`) and about 1.05 GB downloaded for the pools, predictions and rerun inputs (measured) |
| full | CELLxGENE source files | approximate: atlas-equal means and differences within 0.03 of the paper, same sign, atlas-level exact p below 0.05 (`eval/full_check.py`). The tolerance is a chosen one, since the change of Leiden partitions between machines is not measured | 29.3 GB of downloads (sum of the manifest sizes). Time and disk for the pools and the run are not measured. The `full` level is untested end to end |

`full` does not rebuild the twin pools, run the leak check or run the Part I panel. It also skips the strata, separability levels, ablations, CHOIR and figures.

`TOOLS=1` also runs the published tools. They need their own Python environment and `SCGPT_CKPT`. That environment has scikit-learn 1.9.1, which conflicts with `requirements.txt`. Point `TOOLS_PYTHON` at that environment (see `ENVIRONMENTS.md`). `full` skips a tool with an existing output. Failures go to `$OUT/tool_failures.txt`. Without `TOOLS=1` only the Leiden comparison is checked and T2 (against scCAD) is not evaluated. `$OUT/eval/result.md` then still reads NOT PASSED with T2 among the failed parts, because `eval/analyze.py` counts a T2 it cannot evaluate as failed. The level passes or fails on `eval/full_check.py`, which then checks T1 (against Leiden subclustering) only. `eval/full_check.py` asserts 57 pools in 10 atlases.

At the `paper` level a mismatch with any asserted value fails the run (see [PAPER_ITEMS.md](PAPER_ITEMS.md) for what is partial). The `rerun` and `full` levels rerun code, so their results can differ from the paper (see table).

## Releases

| Release | What it holds |
|---|---|
| v0.1.2 | The code for the RECOMB 2027 submission and the container images to use. This is the release to cite. Its `recomb2027-hiddenrare` image passes the `paper` and `rerun` levels, but the `recomb2027-tools` image has not been run end to end. It has no data assets. |
| v0.1.0 | The data assets (the five tarballs listed in `ASSETS.md5`). `fetch_data.sh` downloads from this release by default. Its `recomb2027-hiddenrare` image also passes the `paper` and `rerun` levels. |
| v0.1.1 | Superseded by v0.1.2. It has no data assets. Its container images (tag `v0.1.1`) were built, but no container check has run on them. |

The v0.1.2 code differs from v0.1.0 only in documentation, comments and usage text, the image tag in `Apptainer.def`, the manual container-check workflow, the seeds of `part1/run.sh` (no level runs it), printed text in `part1/panel_readout.py` and notebooks 03 and 04, and an empty-panel check in `data/leak_check.py`. The method code in `hiddenrare/` differs only in one comment line, so the `v0.1.0` and `v0.1.2` images run the same method.

CI: the `ci` workflow runs the self-tests on every push and pull request, and has passed on `main` ([runs](https://github.com/fpera0248/recomb2027/actions/workflows/ci.yml)). The container check is a separate manual workflow. See `CONTAINER.md`.

`fetch_data.sh` downloads the assets of a release (`TAG`, default v0.1.0) and checks them against `ASSETS.md5`. For `full` it also downloads and checks the source atlases (see [Atlases](#atlases)). `data/README.md` explains how to check or refresh `ATLAS_MANIFEST.csv`.

A missing release or asset returns a 404 error (check `TAG`: the assets are on release v0.1.0, while release v0.1.2, the version in `CITATION.cff`, has none). `fetch_data.sh` then falls back to `gh release download`, which needs the GitHub CLI and a login. Offline, set `ASSET_SRC=DIR` to use a local directory of assets.

## Notebook overview

`reproduce.sh paper` executes the four notebooks in `notebooks/`: 01 (held-out test), 02 (published detectors), 03 (null test) and 04 (Part I summary). They assert numbers and draw no figures. [Notebooks](#notebooks) describes each notebook and lists what the `paper` and `rerun` levels do not recompute. This page also covers the source atlases, `ATLAS_MANIFEST.csv`, the two test sets, running one pool and the datasets used before the test.

## Method

1. `hiddenrare/candidates.py`: Leiden clustering of each label at 8 resolutions in two spaces (pool PCA-50, and per-label HVG + PCA-50). Clusters with at least 10 cells and under 10% of the label are candidate groups.
2. `hiddenrare/markers.py`, `halo.py`: marker and neighbourhood ("halo") statistics per group. With geometry features, these make 41 features per group.
3. `hiddenrare/scorer.py`: a gradient-boosted regressor (`models/refit_dev_nat.pkl`) predicts the fraction of each group's cells that belong to hidden subtypes. A cell's score is the best group it belongs to, plus a small isolation tie-break.
4. `hiddenrare/baselines.py`, `blend.py`: the final score is 0.75 * rank(scorer) + 0.25 * rank(Leiden subclustering score).

`models/refit_dev_nat.pkl` is a refit of the same recipe, but we did not keep the fit behind the saved scores of the 8-atlas test set. The `paper` level reads the saved scores and is exact. The `rerun` level applies the refit as a consistency check (`eval/rerun.py`).

We kept feature-table column names (`mk_*`, `hl_*`) from the training tables because `hiddenrare/scorer.py` reads them by name in the order used to fit the model.

## Layout

- `hiddenrare/`: our method (candidate groups, marker and halo features, scorer, blend).
- `models/`: the refit scorer `refit_dev_nat.pkl`.
- `eval/`: the evaluation and the checks of the `paper`, `rerun` and `full` levels.
- `notebooks/`: the four notebooks the `paper` level runs.
- `data/`: builders of the test and twin pools, reference pool lists and targets ([data/README.md](data/README.md) rebuilds the test sets).
- `scripts/`: `run_pool.sh` (our method on one pool) and `run_tool.sh` (one published tool).
- `tools/`: wrappers of the published detectors, their pinned repositories and commits (`sources.txt`) and `fetch_sources.sh`, which clones them ([tools/README.md](tools/README.md)).
- `part1/`: the study of known rare types, duplication against generators ([part1/README.md](part1/README.md)).
- `combination/`: the hidden-type calls plus duplication experiment ([combination/README.md](combination/README.md)).
- `supplement/`: copies of the scripts behind the supplement analyses that no level runs, with only comments, docstrings, path defaults and one escaped string literal edited ([supplement/README.md](supplement/README.md)).
- `docker/`: the two image definitions (see `CONTAINER.md`).
- `.github/workflows/`: CI (self-tests on every push, no level run), the image builds and the manual container check (`paper` and `rerun` levels in the published image).

## Not included and known gaps

- CELLxGENE data (downloaded by the builders), the training pools and the tool checkouts. `hiddenrare/scorer.py fit` is the training script and needs the training pools. Run it from the repository root (it imports eval/targets.py).
- The 10-atlas predictions: the 10-atlas results (strata, separability levels, label ablation, CHOIR) cannot be rerun from the release.
- The rebuilt 8-atlas subsamples and their predictions (our method, Leiden subclustering, GiniClust3 and FiRE): the second block of Supp. Table S6 and the GiniClust3 and FiRE bars of Fig. 2a cannot be rerun from the release.
- Figure scripts, except `supplement/figures/figS_further.py` (Supp. Fig. S5), which no level runs. No level runs the statistics script `supplement/figures/part1_panel_stats.py` either.
- Most inputs of the supplement analyses listed in [PAPER_ITEMS.md](PAPER_ITEMS.md): the AIDA, HLCA and lupus files, the generated cells and the 10-atlas subsamples. Of these inputs, only `part1_results.tgz` and the trained regressor (`models/refit_dev_nat.pkl`) are released. Their scripts are in [`supplement/`](supplement/README.md), but no level runs them.
- The 69 pool h5ads of the 8-atlas test set, needed by `combination/`, Fig. 3b and c, Supp. Figs. S1 and S4, Supp. Tables S8 and S9, the per-subtype results, the donor ablation and any rescoring of those pools from counts. No builder here writes them (`data/build_pools.py` builds the 10-atlas pools), so there is no route to them yet.
- The inputs of the Part I runs and the training of the generators: the AIDA, HLCA and lupus h5ad files that `part1/` reads (`ATLAS_MANIFEST.csv` gives the AIDA download URL, but the HLCA file was pulled from Census 2025-11-08 and has no `dataset_version_id` recorded, see its notes) and the CFGen and scDiffusion checkouts (no commit recorded).
- Reproductions of the published tools: only their wrappers and pinned environments are included.
- RaceID and CHOIR, which ran from the authors' released code on the 10-atlas pools only: no wrapper, commit in tools/sources.txt or image entry is included for them.
- The code of our AnnoGCD and scGAD wrappers. They used the authors' model code (AnnoGCD commit 77b356a unmodified, scGAD commit 3328eda with two edits for a newer scanpy) and re-coded the other steps of the authors' scripts, including each tool's call. The wrappers were on a VM that is no longer available, so only their written declaration survives and this repository cannot rerun either tool. Notebook 02 rescores the saved AnnoGCD predictions (0.066 on 69 subsamples). The scGAD predictions are not in the release files, so this repository cannot rescore the 0.051 on 32 subsamples. Neither tool ran on the 10-atlas set, because both need a GPU and none was available for that round. The release also leaves out the runner of the GiniClust3 (swapped filters) rows.
- The code of the local outlier factor, isolation forest on scVI, classifier uncertainty and self-training loop baselines. `hiddenrare/baselines.py` holds only Leiden subclustering and the isolation forest on PCA.
- The `full` level is untested, and we have not run the `recomb2027-tools` image end to end.
- The 101 twin pools of the null test (12 atlases): they ship only as predictions in `predictions.tgz`, and no level rebuilds them.

## License details

The code in this repository is under the MIT license (`LICENSE`). Third-party code keeps its own license:

- The published tools, cloned by `tools/fetch_sources.sh` at the commits in `tools/sources.txt`: scCAD and FiRE (GPL-3), GapClust (GPL-3, per its DESCRIPTION file), rflann (GPL), GiniClust3, SCISSORS, mars and scGPT (MIT), CellSIUS (Apache-2.0).
- The Python and R packages at the versions in `piplist_hiddenrare.txt`, `piplist_tools.txt` and `Rpackages_tools.csv`, among them leidenalg (GPL-3.0-or-later) and igraph (GPL-2.0-or-later).

The container images contain these programs under their own licenses. The `recomb2027-tools` image keeps the full source of each published tool in `/opt/tools_src`. The source of every Python and R package is available from PyPI, CRAN and Bioconductor at the version installed in the image, and the source of the operating-system packages from the Debian (`recomb2027-hiddenrare`, based on `python:3.12-slim-bookworm`) and Ubuntu (`recomb2027-tools`, based on `rocker/r-ver:4.6.1`) archives. To get the complete corresponding source of any GPL-licensed component of the images, open an issue at https://github.com/fpera0248/recomb2027/issues. The authors will provide the source for three years after each image release.

The release assets are derived from the public CELLxGENE datasets listed in `data/used_datasets.csv` and the CELLxGENE rows of `ATLAS_MANIFEST.csv`, which CELLxGENE distributes under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The assets are under the same license. Please cite the original datasets.
