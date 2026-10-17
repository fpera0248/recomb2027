#!/bin/bash
# Run after AIDA (m27_base) finishes: HLCA first (priority), then lupus.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NJ=2
cd ${HR_WORK:-$HOME/work}/cfgen_natrare/j1004/metrics
python3 one_vs_rest_metrics.py hlca > hlca_run.log 2>&1
echo "=== HLCA DONE, starting lupus ==="
python3 one_vs_rest_metrics.py lupus > lupus_run.log 2>&1
echo "=== LUPUS DONE ==="
