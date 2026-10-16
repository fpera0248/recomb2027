# Reproduction details

## Atlases

The source atlases are public CELLxGENE files. `ATLAS_MANIFEST.csv` lists every one with its collection where recorded, its `dataset_version_id` download URL, size, md5 where known and the set it belongs to (`key` for the 10-atlas set, `name` for the 8-atlas set, is the short id used by the builders). Ten rows have no URL, and 5 of them have a note in `notes`. The other 5 are `panel` rows with no recorded version; download them by `dataset_id` from the page in `collection_url`, which serves the current version.

`data/atlases.csv` lists the 10-atlas test set and `data/used_datasets.csv` lists the datasets used before the test.

Download the 10-atlas test set with `./fetch_data.sh full DIR` (`reproduce.sh full` calls it; `data/download_atlases.sh` checks neither size nor md5). `full` downloads the 10 rows of set `test10` (29.3 GB) and checks each against its size, and against its md5 for the 3 rows that have one.

The 8-atlas set (`test8`, 69 pools) is the one the `paper` and `rerun` tiers check. The 10-atlas set (`test10`, 57 pools) is the one `full` and `data/build_pools.py` build. The two share no dataset or collection.

## One pool

```
scripts/run_pool.sh POOL.h5ad PILOTS.parquet OUT_DIR
```
`POOL.h5ad` holds raw counts. `PILOTS.parquet` has columns `obs_name`, `label` and `donor`; the pool id is the parquet file name. Cell ids are opaque hashes of the barcode. Output: `OUT_DIR/preds/hiddenrare/<pool>.parquet` (`cell_id`, `score`), plus the Leiden and isolation-forest baselines and the intermediate tables in `OUT_DIR/cands`.

## Datasets used before the test

Datasets used before the test are listed in `data/used_datasets.csv`. A Human Breast Cell Atlas file from collection 48259aa8 was also used during development (row `hbca` of `ATLAS_MANIFEST.csv`). Its dataset id was not recorded, so the cell-level leak check may not cover it. The collection itself is in the list (dataset 55003f67), so the collection check of `data/build_pools.py` excludes every dataset of that collection.
