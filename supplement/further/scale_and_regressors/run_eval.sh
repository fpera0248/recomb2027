#!/bin/bash
until mkdir ${HR_WORK:-$HOME/work}/jiaqi1011/.laptop_lock 2>/dev/null; do sleep 60; done
OMP_NUM_THREADS=4 /usr/local/bin/python3 eval_models.py > ../../out/w1/eval_log.txt 2>&1; rmdir ${HR_WORK:-$HOME/work}/jiaqi1011/.laptop_lock
