#!/bin/bash
# W6 part 1: label-corruption reruns on the laptop (single machine so ID/variant pairs share one numerical environment).
# usage: w6_p1.sh [POOL_INDEX_MAX=2] [WORKERS=2] ; units ordered pool-index major so partial results cover all atlases early. Resumable.
# Timing: ~1.3-1.5 h for the first pass (150 units), ~4.5 h of lock time for all 435 units + make_variants.
set -eu
H=${HR_WORK:-$HOME/work}/jiaqi1011; C=$H/code/w6; OUT=$H/out/w6_annot_atac; LOG=$H/logs/w6_p1.log; MAXP=${1:-2}; NW=${2:-2}; G=$H/laptop_queue/memgate.sh
mkdir -p $OUT $H/logs
trap 'touch $OUT/P1_FAILED' ERR
while ! mkdir $H/.laptop_lock 2>/dev/null; do sleep 60; done
trap 'rmdir $H/.laptop_lock 2>/dev/null || true' EXIT
unlink $OUT/P1_DONE 2>/dev/null || true; unlink $OUT/P1_FAILED 2>/dev/null || true   # only after the lock, so a stale pass-1 flag cannot mask pass 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONWARNINGS=ignore
free_gb=$(df -g $H | awk 'NR==2{print $4}'); [ "$free_gb" -ge 30 ] || { echo "disk gate: $free_gb GB free" >> $LOG; touch $OUT/P1_FAILED; exit 1; }
python3 -W ignore -c "import sys; sys.path.insert(0,'$C'); import make_variants as m; m.setup_roots()"   # once, before the parallel workers
python3 -W ignore $C/list_units.py $MAXP pools > $OUT/units.txt
xargs -P $NW -n 1 $G python3 -W ignore $C/make_variants.py < $OUT/units.txt >> $LOG 2>&1
python3 -W ignore $C/list_units.py $MAXP units > $OUT/units_vp.txt
xargs -P $NW -L 1 $G python3 -W ignore $C/run_unit.py < $OUT/units_vp.txt >> $LOG 2>&1
echo "units done: $(ls $OUT/units | wc -l)" >> $LOG
touch $OUT/P1_DONE
