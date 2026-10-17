#!/bin/bash
# R3c: annotation improvement at a matched number of classes, 29 set-B pools (pool index < 3). Laptop, takes the laptop lock, 2 workers, memgate per job. Resumable.
# Estimate (not measured): ~~18 bisection steps x all labels per pool (partitions cached, shared across the four thresholds), ~5-15 min per pool -> about 2-5 h with 2 workers. Scoring afterwards is seconds per pool.
set -eu
H=${HR_WORK:-$HOME/work}/jiaqi1011; C=$H/code/small; O=$H/out/small/matched; ID=$H/data/w6/ID; G=$H/laptop_queue/memgate.sh; LOG=$H/logs/r3c_matched.log
mkdir -p $O $H/logs; trap 'touch $O/FAILED' ERR; rm -f $O/FAILED $O/DONE
while ! mkdir $H/.laptop_lock 2>/dev/null; do sleep 60; done
trap 'rmdir $H/.laptop_lock 2>/dev/null || true' EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PIDS=$(ls $H/out/w6_annot_atac/groupP | sed -n 's/^ID__\(.*\)\.parquet$/\1/p')
for p in $PIDS; do [ -f $O/targets_$p.json ] || python3 -W ignore $C/r3c_matched.py targets $p >> $LOG 2>&1; done
echo "$PIDS" | xargs -S 1024 -P 2 -I{} bash -c "[ -f $O/grid_{}.parquet ] || R5=$ID SI_H=$ID/hmod SLIM_DIR=$ID/slim_atlas $G python3 -W ignore $C/r3c_matched.py grid {}" >> $LOG 2>&1
for p in $PIDS; do [ -f $O/score_$p.csv ] || python3 -W ignore $C/r3c_matched.py score $p >> $LOG 2>&1; done
python3 -W ignore $C/r3c_matched.py summary >> $LOG 2>&1
touch $O/DONE
