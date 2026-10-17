#!/bin/bash
# bash run_scoring.sh [OUTDIR] : score every arm that has finished pools against PCA on their common pools (3 stages + diagnostics), then tables + figures.
# Laptop: /usr/local/bin/python3 (sklearn 1.5.2 like the mini). Light (HGB on <1M groups); safe without the lock. Works on partial outputs.
set -eu; export W2_OUT=${1:-${HR_WORK:-$HOME/work}/jiaqi1011/out/w2_fm}; C=${HR_WORK:-$HOME/work}/jiaqi1011/code/w2_fm; PY=${PY:-/usr/local/bin/python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
cd "$W2_OUT"
for d in arms/*/; do
  a=$(basename "$d"); [ "$a" = pca ] && continue
  case $a in noise*) [ "${W2_NOISE:-0}" = 1 ] || continue;; esac
  n=$(ls "$d"cands/*.halo.parquet 2>/dev/null | wc -l | tr -d ' '); [ "$n" -ge 1 ] || continue
  echo "== $a ($n pools with candidates)"; $PY $C/score_embeddings.py "$W2_OUT/res_$a.csv" "$W2_OUT/diag_$a.csv" pca "$a" > "$W2_OUT/score_$a.log" 2>&1 || { tail -20 "$W2_OUT/score_$a.log"; echo "SCORING FAILED for $a" >&2; exit 1; }; tail -2 "$W2_OUT/score_$a.log"
done
$PY $C/analyze_embeddings.py "$W2_OUT" > "$W2_OUT/analyze.log" 2>&1 || { tail -20 "$W2_OUT/analyze.log"; exit 1; }; grep -v -i warn "$W2_OUT/analyze.log"
