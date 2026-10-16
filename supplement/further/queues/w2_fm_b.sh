#!/bin/bash
# W2 slot B (mini, CPU, 3 threads; features at 2 threads, m1 at 3, like the locked runs; critical path ~8-9 h): (0) PCA determinism gate, (1) PCA and PCA30 candidates + Leiden alone, (2) pool-major: scVI embedding (400 epochs max, early stopping, 15 min cap,
# 3 threads), blind QA, scVI variants F and H. Features run serially. Resumable. Marker: $W2_OUT/DONE_b | FAILED_b.
set -eu; MARK=b; source "$HOME/work/jiaqi1011/code/w2_fm/w2_common.sh"; rm -f "$O/DONE_b" "$O/FAILED_b"
# scVI: fixed epoch budget from the full-pool plumbing run (out/w2_fm_plumb/scvi_epoch_budget.txt, copy it to $O/scvi_epoch_budget.txt before launch); wall cap 60 min is only a safety net. Share of capped pools is reported.
[ -s "$O/scvi_epoch_budget.txt" ] && export SCVI_MAX_EPOCHS=$(cat "$O/scvi_epoch_budget.txt"); echo "scVI epoch budget: ${SCVI_MAX_EPOCHS:-400 (default, no plumbing budget found)}"
P=$(cat "$POOLS"); set -- $P
gate; $PY $C/w2_cands.py pca F $1 $2 && $PY $C/w2_gate_pca.py $1 $2 || echo "PCA gate not clean (see $O/gate_pca.txt); continuing"
for pid in $P; do gate; $PY $C/w2_cands.py pca F $pid; gate; $PY $C/w2_cands.py pca30 F $pid; done
for pid in $P; do
  gate; $PY $C/w2_embed.py scvi $pid; gate; $PY $C/w2_qa.py scvi $pid
  for v in F H; do gate; $PY $C/w2_cands.py scvi $v $pid; done
done
fin DONE; echo "$(date +%T) done"
