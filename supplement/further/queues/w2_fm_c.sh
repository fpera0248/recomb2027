#!/bin/bash
# W2 slot C (mini, CPU, 3 threads; start when a slot is free): scGPT blind QA + candidates/features/Leiden alone for variants F, H, P as the slot-A embeddings land.
# Ends when every pool is done or slot A has stopped (DONE_a / FAILED_a) and nothing new appeared. Marker: $W2_OUT/DONE_c | FAILED_c.
set -eu; MARK=c; source "$HOME/work/jiaqi1011/code/w2_fm/w2_common.sh"; rm -f "$O/DONE_c" "$O/FAILED_c"
todo=$(cat "$POOLS")
while [ -n "$todo" ]; do
  prog=0; rest=""
  for pid in $todo; do
    if [ -s "$O/emb/scgpt/$pid.npy" ]; then
      gate; $PY $C/w2_qa.py scgpt $pid; for v in F H P; do gate; $PY $C/w2_cands.py scgpt $v $pid; done; prog=1
    else rest="$rest $pid"; fi
  done
  todo=$(echo $rest)
  [ -z "$todo" ] && break
  if [ $prog = 0 ]; then
    if [ -e "$O/DONE_a" ] || [ -e "$O/FAILED_a" ]; then echo "slot A stopped with embeddings missing: $todo"; [ -e "$O/DONE_a" ] || { fin FAILED; exit 1; }; break; fi
    sleep 120
  fi
done
fin DONE; echo "$(date +%T) done"
