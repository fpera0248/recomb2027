#!/bin/bash
# Scaling: pipeline time/memory vs pool size, on one machine. Output: ~/work/jiaqi1011/out/scale_queue/{scale.csv,logs/,DONE|FAILED}. Resumable (re-run skips ok rows).
set -eu
J=${HR_WORK:-$HOME/work}/jiaqi1011; OUT=${OUT:-$J/out/w1_scale}; mkdir -p $OUT $J/logs; rm -f $OUT/DONE $OUT/FAILED
exec >> ${LOGFILE:-$J/logs/w1_scale.log} 2>&1
trap 'rc=$?; [ $rc -ne 0 ] && echo "exit $rc $(date)" > $OUT/FAILED' EXIT
echo "start $(date)"
until [ "$(memory_pressure | sed -n 's/.*free percentage: *\([0-9]*\)%.*/\1/p')" -ge ${MEM_FREE_MIN:-25} ]; do echo "waiting for free memory"; sleep 60; done
export R5=${R5:-$J/scale_data} R5RUN=$OUT/state OUT
mkdir -p $R5
export PATH=$HOME/venv/bin:$PATH
. ${HR_WORK:-$HOME/work}/r5/run/env.sh                       # R5_PY (pinned venv), threads, R5_MODEL, ... with R5 = scale data root
export SI_H=${HR_WORK:-$HOME/work}/r5/code/hmod              # libs from the locked tree; env.sh would point at $R5/hmod
$R5_PY $J/code/w1/scale_pipeline.py run
[ -f $OUT/FAILED ] || echo "ok $(date)" > $OUT/DONE
echo "end $(date)"
