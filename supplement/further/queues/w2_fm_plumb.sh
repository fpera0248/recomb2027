#!/bin/bash
# W2 laptop plumbing + timing on REAL full pools (not part of the experiment; outputs go to out/w2_fm_plumb). Takes the laptop lock itself; ~1 h.
# (a) noise512 (planted 512-d random embedding) through candidates F and P + QA: checks the 512-d code path and its feature timing;
# (b) atl_luad_p0 (a set A pool, toolin pool built from atlas/luad.h5ad by run/adapt_pool.py) with real scGPT (MPS), Geneformer V1-10M (MPS) and scVI (CPU): full-pool embedding time,
#     QA, candidates, and scoring of each space against PCA on this one pool (leiden + frozen; LOAO needs >= 2 atlases and is skipped).
# Marker: out/w2_fm_plumb/DONE_plumb | FAILED_plumb.
set -eu
if [ -z "${W2_LOCKED:-}" ]; then export W2_LOCKED=1; exec "$HOME/work/jiaqi1011/code/w2_fm/laptop_lock_run.sh" "$0" "$@"; fi
export W2_OUT=$HOME/work/jiaqi1011/out/w2_fm_plumb PY=/usr/local/bin/python3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
MARK=plumb; source "$HOME/work/jiaqi1011/code/w2_fm/w2_common.sh"; rm -f "$O/DONE_plumb" "$O/FAILED_plumb"
export PYTORCH_ENABLE_MPS_FALLBACK=1 KMP_DUPLICATE_LIB_OK=TRUE SCGPT_CKPT=$HOME/work/cfgen_natrare/j1004/tools/scgpt_ckpt
P=atl_0bf30b02_p0; $PY $C/w2_cands.py pca F $P; $PY $C/w2_embed.py noise512 $P; $PY $C/w2_qa.py noise512 $P; for v in F P; do $PY $C/w2_cands.py noise512 $v $P; done
L=atl_luad_p0; ls "$W/data/w2_fm/toolin/r/$L.h5ad" >/dev/null || { R5=$W/data/w2_fm SLIM_DIR=$W/data/w2_fm/slim_atlas SI_H=$W/data/w2_fm/hmod $PY $HOME/work/r5/run/adapt_pool.py $L slim_atlas $HOME/work/r5/data/atlas/luad.h5ad; }
$PY $C/w2_cands.py pca F $L
$PYT $C/w2_embed.py scgpt $L --device mps; $PY $C/w2_qa.py scgpt $L; for v in F H P; do $PY $C/w2_cands.py scgpt $v $L; done
$PY $C/w2_embed.py gf $L --device mps; $PY $C/w2_qa.py gf $L; for v in F P; do $PY $C/w2_cands.py gf $v $L; done
$PY $C/w2_embed.py scvi $L; $PY $C/w2_qa.py scvi $L; for v in F H; do $PY $C/w2_cands.py scvi $v $L; done
awk -F, '$1=="embed" && $2=="scvi" && $3=="atl_luad_p0" {n=$0; sub(/.*epochs=/,"",n); sub(/;.*/,"",n); print n}' "$O/timing.csv" | tail -1 > "$O/scvi_epoch_budget.txt"; echo "scVI epochs at early stop on a full pool: $(cat $O/scvi_epoch_budget.txt) (budget for slot B; if equal to 400 the model never stopped early)"
echo $L > "$O/pools_luad.txt"
for a in scgpt_F scgpt_H scgpt_P gf_F gf_P scvi_F scvi_H; do $PY $C/w2_score.py "$O/res_$a.csv" "$O/diag_$a.csv" pca $a --pools "$O/pools_luad.txt"; done
fin DONE
