#!/bin/bash
# All pools and seeds, one classifier (lr is the primary readout; rf and mlp are secondary), then the analysis. Finished pool-seeds are skipped.
# usage: HR_DATA=assets HR_POOLS=/path/to/pool_h5ads combination/run_all.sh [lr|rf|mlp]     rows and logs go to $OUT (default combination/rows); the 69 pool h5ads are not in the release (combination/README.md)
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/.."
CLF=${1:-lr}
OUT=${OUT:-combination/rows}
mkdir -p "$OUT"
PY=${PYTHON:-python3}
ENVV="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1"
for s in 0 1 2; do
  while read -r p <&3; do
    [ -f "$OUT/${p}_${s}_${CLF}.csv" ] || nice -n 10 env $ENVV $PY -m combination.run_one "$p" $s --clf $CLF --out "$OUT/${p}_${s}_${CLF}.csv" < /dev/null >> "$OUT/full_${CLF}.log" 2>&1
  done 3< <(cut -d, -f1 "$here/plan.csv" | tail -n +2 | uniq)
done
nice -n 10 env $ENVV $PY -m combination.analyze "$OUT/*_${CLF}.csv" "$OUT/analysis_${CLF}.csv" "$here/plan.csv" $CLF
