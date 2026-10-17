#!/bin/bash
# (~4 h total: 10 pools x ~16 min embedding + QA, PCA and 3 candidate variants.)
# Embedding queue, Geneformer arm (LAPTOP, MPS; takes the laptop lock itself; launch LAST). Geneformer V1-10M (gf-6L-30M-i2048, gc30M dictionaries, 256-d) from the local official
# checkout ~/geneformer/Geneformer (no install needed: transformers 4.48 + torch MPS are in /usr/local/bin/python3). Everything for this arm runs on the laptop, including its own PCA reference
# (laptop environment differs from the mini's pinned one, so GF is compared with PCA recomputed on the SAME machine, in a separate output tree out/w2_fm_lap). Scope rule fixed in advance:
# p0 of each atlas (10 pools); p1 and p2 are added only if the first pool's embedding takes <= 480 s (measured probe: marginal 0.0475 s/cell = ~16 min/pool, so expect p0 only).
# Caveat to print with any result: V1-10M is the only Geneformer that fits an overnight laptop budget and the foundation-model duplication analysis found it below its quality bar (about 21% of cells hit the 2048-token cap);
# a loss here does not mean foundation models lose. Marker: out/w2_fm_lap/DONE_gf | FAILED_gf.
set -eu
if [ -z "${W2_LOCKED:-}" ]; then export W2_LOCKED=1; exec "${HR_WORK:-$HOME/work}/jiaqi1011/code/w2_fm/laptop_lock_run.sh" "$0" "$@"; fi
export W2_OUT=${HR_WORK:-$HOME/work}/jiaqi1011/out/w2_fm_lap PY=/usr/local/bin/python3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
MARK=gf; source "${HR_WORK:-$HOME/work}/jiaqi1011/code/w2_fm/common.sh"; rm -f "$O/DONE_gf" "$O/FAILED_gf"
P0=$(cat $C/pools_p0.txt); first=$(echo $P0 | cut -d' ' -f1)
$PY $C/embed.py gf $first --device mps
t=$(awk -F, -v p=$first '$1=="embed" && $2=="gf" && $3==p {t=$4} END{print int(t)}' "$O/timing.csv"); echo "first pool embedding: ${t}s"
if [ "$t" -le 480 ]; then P=$(cat $C/pools_phase1.txt); else P=$P0; fi; echo "$P" | tr ' ' '\n' > "$O/gf_pools_used.txt"
for pid in $P; do $PY $C/embed.py gf $pid --device mps; $PY $C/embedding_qa.py gf $pid; done
for pid in $P; do $PY $C/embedding_candidates.py pca F $pid; for v in F H P; do $PY $C/embedding_candidates.py gf $v $pid; done; done
mkdir -p "${HR_WORK:-$HOME/work}/jiaqi1011/figs/w2_fm_lap"; W2_FIG=${HR_WORK:-$HOME/work}/jiaqi1011/figs/w2_fm_lap bash $C/run_scoring.sh "$O"
fin DONE; echo "$(date +%T) done"
