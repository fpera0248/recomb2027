#!/bin/bash
# W2 slot A (mini, MPS): scGPT whole-human zero-shot embeddings of the phase-1 pools (same 2000 HVG as the PCA arm, max_length 2001 = no gene sampling, seed 0).
# expected 13-16 min/pool (laptop probe ~9 min; mini scgpt_if measured 578 s at max_length 1200, scaled to 2001), 29 pools = 6.5-8 h (est_hours 7.5); the first pool's timing.csv row sets the real number. Resumable. Marker: $W2_OUT/DONE_a | FAILED_a.
set -eu; MARK=a; source "$HOME/work/jiaqi1011/code/w2_fm/w2_common.sh"; rm -f "$O/DONE_a" "$O/FAILED_a"
export SCGPT_CKPT=${SCGPT_CKPT:-$HOME/work/r5/tools/scgpt_ckpt} PYTORCH_ENABLE_MPS_FALLBACK=1 KMP_DUPLICATE_LIB_OK=TRUE
test -s "$SCGPT_CKPT/best_model.pt" || { echo "checkpoint missing on the mini: rsync ~/work/cfgen_natrare/j1004/tools/scgpt_ckpt (laptop, md5 of best_model.pt 9922ec94305126e6e4f9c1575cf493ae) to $SCGPT_CKPT" | tee "$O/NEEDS_CKPT"; false; }
md5 -q "$SCGPT_CKPT/best_model.pt" | tee "$O/scgpt_ckpt_md5_mini.txt"      # compare with the laptop copy (9922ec94...) and with the checkpoint used for scgpt_if
rm -f "$O/NEEDS_TOOLIN"
for pid in $(cat "$POOLS"); do
  [ -s "$O/emb/scgpt/$pid.npy" ] && continue
  [ -s "$HOME/work/r5/data/toolin/r/$pid.h5ad" ] || { echo "$pid" >> "$O/NEEDS_TOOLIN"; echo "toolin pool missing, skipping $pid (if rebuilt, write under $W/data/w2_fm, never under r5/data)"; continue; }
  gate; echo "$(date +%T) scgpt $pid"; $PYT $C/w2_embed.py scgpt $pid --device mps --ckpt "$SCGPT_CKPT"
done
fin DONE; echo "$(date +%T) done"
