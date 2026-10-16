## Hidden-type calls plus duplication

Run the commands from the repository root.

`combination/` tests whether calling hidden types and duplicating the called cells helps a classifier find the hidden cells in new donors. It compares:

- our hidden-type calls plus duplication, with the call count matched to Leiden's;
- Leiden subclustering calls plus duplication;
- true labels, as the upper bound.

Each pool is split by donor. Both methods scored the whole pool with test donors unlabelled. Only calls on training donors become labels, so the classifier never sees test-donor labels; it is scored on the held-out donors. `combination/DESIGN.md` is the design, fixed before any full-run result.

```
export HR_DATA=assets HR_POOLS=/path/to/pool_h5ads    # HR_DATA: paper-tier assets (pilots/, manifests/, pools.csv, predictions/)
export OUT=work/combination_rows                      # keep outputs out of the source tree
python -m combination.run_one atl_luad_p0 0 --out $OUT/one.csv                 # one pool, one seed: one row per setting (without --out it writes combination/rows_<pool>_<seed>_<clf>.csv)
combination/run_all.sh lr                                    # all eligible pools and seeds, then the analysis; also rf and mlp
```
`HR_POOLS` holds `<source_path>` h5ad files (raw counts, all genes) for the 69 pools of the 8-atlas test set, for example `luad_p0.h5ad`.

Limits: `data/build_pools.py` builds the 10-atlas pools, not these, and the paper-tier assets omit them, so the experiment cannot yet run from the repository and release assets alone. The finished run (58 pools from 8 atlases) is reported in Supp. Fig. S3 of the paper; its per-pool result rows are not in the repository.
