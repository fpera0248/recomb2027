#!/bin/bash
until mkdir ~/work/jiaqi1011/.laptop_lock 2>/dev/null; do sleep 60; done
OMP_NUM_THREADS=4 /usr/local/bin/python3 eval_models.py > ../../out/w1/eval_log.txt 2>&1; rmdir ~/work/jiaqi1011/.laptop_lock
