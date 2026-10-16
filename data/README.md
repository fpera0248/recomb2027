## Rebuild the test sets

Run the commands from the repository root.

`data/build_pools.py` writes the pools and `data/build_twins.py` the null-test twin pools of the 10-atlas test set from the atlas files listed in `data/atlases.csv` (downloaded by `data/download_atlases.sh`; sizes in `ATLAS_MANIFEST.csv`). `data/manifests` and `data/targets` are reference copies of the pool lists and target tables of the original run: diff them against what `build_pools.py` writes to `$HR_DATA`. The 101 twin pools of the paper (12 atlases) ship only as predictions in `predictions.tgz` (`ids/twin_manifest.csv`). No tier rebuilds them, `full` included, and twins rebuilt by `build_twins.py` may differ from the shipped ones.

```
export HR_DATA=work
./fetch_data.sh full "$HR_DATA"               # public CELLxGENE files (29.3 GB), checked against ATLAS_MANIFEST.csv
NAMES=$(cut -d, -f1 data/atlases.csv | tail -n +2)   # the 10 atlases
python -m data.build_pools $NAMES
python -m data.build_twins $NAMES            # null-test twin pools
python -m data.leak_check cells $NAMES --panel work/panel    # panel: each dataset of data/used_datasets.csv downloaded from CELLxGENE into work/panel (not scripted)
```
Then run `scripts/run_pool.sh` on every pool, the published tools with `scripts/run_tool.sh` (inputs from `tools/adapt_pool.py`), and `eval/run_eval.sh $HR_DATA RUN_DIR OUT_DIR` to score and analyse. `eval/signflip.py` recomputes the exact sign-flip p-values independently.
