#!/bin/bash
# step 2 scaling + step 3 alternatives, one fresh process per fit, laptop lock per fit
cd ${HR_WORK:-$HOME/work}/jiaqi1011/code/w1; PY=/usr/local/bin/python3
run(){ [ -f ${HR_WORK:-$HOME/work}/jiaqi1011/out/w1/models/$1.json ] && return; until mkdir ${HR_WORK:-$HOME/work}/jiaqi1011/.laptop_lock 2>/dev/null; do sleep 60; done; $PY fit_worker.py "$@" 2>&1 | grep -v Warn | tail -2; rmdir ${HR_WORK:-$HOME/work}/jiaqi1011/.laptop_lock; }
for n in 25 50 100 200 364; do for s in 0 1 2; do run hgb_n${n}_s$s hgb $n $s; done; done
for k in ridge gbr200k mlp et rf; do run ${k}_n364_s0 $k 364 0; done
