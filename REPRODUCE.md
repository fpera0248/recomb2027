# Reproduction details

## Tier limits and release assets

The code release is v0.1.1. The release assets and the checked container images stay on v0.1.0 (v0.1.1 has no assets), and the v0.1.0 image holds the v0.1.0 code, which differs from v0.1.1 only in documentation, the image tag in `Apptainer.def`, the manual container-check workflow, the seeds of `part1/run.sh` (no tier runs it) and a print format in notebook 03.

`fetch_data.sh` downloads the assets of a release (`TAG`, default v0.1.0) and checks them against `ASSETS.md5`. For `full` it also downloads and checks the source atlases (see Atlases). `data/README.md` explains how to check or refresh `ATLAS_MANIFEST.csv`.

A missing release or asset gives 404 (check `TAG`; the assets are on release v0.1.0, and release v0.1.1, the version in `CITATION.cff`, has none). `fetch_data.sh` then falls back to `gh release download` (install the GitHub CLI and log in). Set `ASSET_SRC=DIR` to take the assets from a local directory, for offline use.

## Atlases

The source atlases are public CELLxGENE files. `ATLAS_MANIFEST.csv` lists every one with its collection where recorded, its `dataset_version_id` download URL, size, md5 where known and the set it belongs to (`key` for the 10-atlas set, `name` for the 8-atlas set, is the short id used by the builders). Ten rows have no URL, and 5 of them have a note in `notes`. The other 5 are `panel` rows with no recorded version; download them by `dataset_id` from the page in `collection_url`, which serves the current version.

`data/atlases.csv` lists the 10-atlas test set and `data/used_datasets.csv` lists the datasets used before the test.

Download the 10-atlas test set with `./fetch_data.sh full DIR` (`reproduce.sh full` calls it; `data/download_atlases.sh` checks neither size nor md5). `full` downloads the 10 rows of set `test10` (29.3 GB) and checks each against its size, and against its md5 for the 3 rows that have one.

The 8-atlas set (`test8`, 69 pools) is the one the `paper` and `rerun` tiers check. The 10-atlas set (`test10`, 57 pools) is the one `full` and `data/build_pools.py` build. The two share no dataset or collection.

## Notebooks

`reproduce.sh paper` executes four notebooks:

1. `notebooks/01_heldout_test.ipynb`: our method against Leiden subclustering and the in-house baselines.
2. `02_published_detectors.ipynb`: published detectors, with the metrics used in the MARS and scCAD papers.
3. `03_null_test.ipynb`: twin pools with the hidden cells removed (101 real and twin pairs from 12 atlases, then the 69 pairs of the 8-atlas test set).
4. `04_part1_summary.ipynb`: the 88-dataset panel, an extra 31-type check and duplication as sample weighting.

The notebooks only call `eval/paper.py`, `eval/nulltest.py` and `part1/`; they assert numbers and draw no figures.

The `paper` tier does not recompute:

- the 10-atlas results (the 10-atlas test, strata, separability tiers, label ablation, donor ablation, CHOIR and the tools that ran only there);
- scGAD on 32 pools;
- the random-ranking line of Fig. 1d;
- the Part I runs outside notebook 04 (generator training, the full data panel, the AIDA, HLCA and lupus runs). Notebook 04 lists what it covers.

The `rerun` tier rescores frozen candidate tables with this repository's scorer; it does not rebuild embeddings or Leiden partitions. Both tiers use the 69 pools of the 8-atlas test set (notebook 03 also reads the twins of 12 atlases).

## One pool

```
scripts/run_pool.sh POOL.h5ad PILOTS.parquet OUT_DIR
```
`POOL.h5ad` holds raw counts. `PILOTS.parquet` has columns `obs_name`, `label` and `donor`; the pool id is the parquet file name. Cell ids are opaque hashes of the barcode. Output: `OUT_DIR/preds/hiddenrare/<pool>.parquet` (`cell_id`, `score`), plus the Leiden and isolation-forest baselines and the intermediate tables in `OUT_DIR/cands`.

## Datasets used before the test

Datasets used before the test are listed in `data/used_datasets.csv`. A Human Breast Cell Atlas file from collection 48259aa8 was also used during development (row `hbca` of `ATLAS_MANIFEST.csv`). Its dataset id was not recorded, so the cell-level leak check may not cover it. The collection itself is in the list (dataset 55003f67), so the collection check of `data/build_pools.py` excludes every dataset of that collection.
