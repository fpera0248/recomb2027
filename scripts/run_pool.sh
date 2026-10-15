#!/bin/bash
# One pool end to end: candidate groups and features, Leiden subclustering, group scorer, blend.
# usage: scripts/run_pool.sh POOL.h5ad PILOTS.parquet OUT_DIR
# OUT_DIR/cands holds the feature tables; OUT_DIR/preds/{hiddenrare,scorer,leiden_sub,iforest_pca} the cell scores.
set -euo pipefail
pool=$1
table=$2
out=$3
root=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-python3}
model=${MODEL:-$root/models/refit_dev_nat.pkl}
pid=$(basename "$table" .parquet)
export PYTHONPATH=$root PYTHONWARNINGS=ignore
mkdir -p "$out/cands" "$out/preds"
# thread counts of the original run: 2 for the three feature stages, 3 for the rest
for stage in candidates markers halo; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 $py -m hiddenrare.$stage "$pool" "$table" "$out/cands"
done
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMBA_NUM_THREADS=3
$py -m hiddenrare.baselines "$pool" "$table" "$out/preds"
$py -m hiddenrare.scorer predict "$model" "$out/cands" "$pid" "$out/preds/scorer/$pid.parquet"
$py -m hiddenrare.blend "$out/preds/scorer/$pid.parquet" "$out/preds/leiden_sub/$pid.parquet" "$out/preds/hiddenrare/$pid.parquet"
