## Rebuild the test sets

Run the commands from the repository root.

`data/build_pools.py` writes the pools and `data/build_twins.py` the null-test twin pools of the 10-atlas test set. Both read the atlas files of `data/atlases.csv` in `$HR_DATA/atlas`. These come from `./fetch_data.sh full "$HR_DATA"` (below) or from `data/download_atlases.sh`, which does no size or md5 check.

`data/manifests` and `data/targets` are reference copies of the pool lists and target tables of the original run: diff them against what `build_pools.py` writes to `$HR_DATA`.

The 101 twin pools of the paper (12 atlases) ship only as predictions in `predictions.tgz` (`ids/twin_manifest.csv`). No tier rebuilds them, `full` included, and twins rebuilt by `build_twins.py` may differ from the shipped ones.

```
export HR_DATA=work
./fetch_data.sh full "$HR_DATA"               # public CELLxGENE files (29.3 GB), checked against ATLAS_MANIFEST.csv
NAMES=$(cut -d, -f1 data/atlases.csv | tail -n +2)   # the 10 atlases
python -m data.build_pools $NAMES
python -m data.build_twins $NAMES            # null-test twin pools
python -m data.leak_check cells $NAMES --panel work/panel    # panel: see the note below
```

The panel download is not scripted. Download each dataset of `data/used_datasets.csv` from CELLxGENE into `work/panel`, using `download_url` in `ATLAS_MANIFEST.csv`. For the 5 panel rows without one, use the current file on the collection page in `collection_url`.

Then run `scripts/run_pool.sh` on every pool, the published tools with `scripts/run_tool.sh` (inputs from `tools/adapt_pool.py`), and `eval/run_eval.sh $HR_DATA RUN_DIR OUT_DIR` to score and analyse. `eval/signflip.py` recomputes the exact sign-flip p-values independently.

## Check or refresh the manifest

`python data/build_manifest.py --check` re-verifies the URLs in `ATLAS_MANIFEST.csv`. `python data/build_manifest.py --refresh` rewrites the manifest from the CELLxGENE curation API; it then points to the latest dataset versions.
