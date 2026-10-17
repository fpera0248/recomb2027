#!/bin/bash
# chromatin accessibility on every eligible CELLxGENE dataset with fragments. MINI, sequential per dataset, resumable (done markers), at most 2 pipelines at once.
# Per dataset: fragments download (resumable curl) -> pools prep -> gene activity (snapATAC2 venv) -> pipeline roots -> arms via run_unit.py (rna if multiome with counts, ga, gak) -> markers.
# Data: $ATAC_DIR (default /Volumes/Files/atac)/{h5ad,frag,ref,work}. Results: $OUT/<dataset>/{selection.json,pools.json,arms/units/*.csv}, $OUT/selection.csv, $OUT/aggregate/. Markers: $OUT/<dataset>/{DL,PREP,BUILD,ROOTS,ARMS}_DONE|FAILED, $OUT/DONE|FAILED.
# Optional $OUT/pair_override.csv (dataset_id,coarse,fine) replaces the mechanical pair for listed datasets (a manual choice). Kidney regression = dataset 09b518f9 rerun with the pilot's choices (key <id>__pilot).
set -u
J=${HR_WORK:-$HOME/work}/jiaqi1011; A=${ATAC_DIR:-/Volumes/Files/atac}; CODE=$(cd "$(dirname "$0")/../labels_and_atac" && pwd); AA=$CODE/atac; OUT=${OUT:-$J/out/atac_all}; PY=$HOME/venv/bin/python; SNAP=$J/venvs/atac/bin/python
CAND=$J/out/w6_annot_atac/atac21_candidates.csv; GTF=$A/ref/ensdb110_genes_idkey.gtf; KID=09b518f9-da64-44cc-aec8-70a89d55611f; LOG=$J/logs/atac_all.log
mkdir -p $OUT $J/logs $A/h5ad $A/frag $A/work; rm -f $OUT/DONE $OUT/FAILED $OUT/NEEDS_FILES
exec >> $LOG 2>&1; echo "start $(date)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
trap 'rc=$?; [ $rc -ne 0 ] && echo "exit $rc $(date)" > $OUT/FAILED' EXIT
# ---- files that must already be on the mini (rsync from the laptop; none is written by this script) ----
need=""; for f in \
  $CAND $GTF $PY $SNAP ${HR_WORK:-$HOME/work}/r5/refit_dev_nat.pkl ${HR_WORK:-$HOME/work}/r5/used_datasets.csv ${HR_WORK:-$HOME/work}/ablation_donor/out/train_xy.npz \
  ${HR_WORK:-$HOME/work}/r5/code/{sr3,sr3m,sr3h,sr2,split_island_v2,loader,learn14,rescore,score_atl,score_hlca,combo_r2}.py ${HR_WORK:-$HOME/work}/r5/code/hmod/{hlib,methods,rawutil,rprec}.py ${HR_WORK:-$HOME/work}/r5/data/hmod/{hlib,methods,rawutil,rprec}.py \
  $J/code/w1/harness.py $CODE/{run_unit,label_corruption_common,leiden_sub_full}.py $AA/{select_ds,pools,build_ga,aggregate,qhelp}.py; do [ -e "$f" ] || need="$need $f"; done
[ -z "$need" ] || { echo "missing:$need" | tee $OUT/NEEDS_FILES; exit 2; }
$PY -c 'import igraph, leidenalg, scanpy, sklearn, pyarrow, anndata' && $SNAP -c 'import snapatac2' || { echo "missing python imports (igraph leidenalg scanpy sklearn pyarrow anndata | snapatac2)" | tee $OUT/NEEDS_FILES; exit 2; }
[ -f $OUT/not_zero_shot.csv ] || echo "WARN no $OUT/not_zero_shot.csv (kidney study-overlap override)"
free_pct() { memory_pressure | sed -n 's/.*free percentage: *\([0-9]*\)%.*/\1/p'; }
gate() { until [ "$(free_pct)" -ge 25 ] && [ "$(sysctl -n vm.loadavg | awk '{print int($2)}')" -le 10 ]; do sleep 60; done; }
dl() { # dl URL DEST : resumable, rename when complete, needs >= 100 GB free on the volume
  [ -s "$2" ] && return 0; [ "$(df -g $A | awk 'NR==2{print $4}')" -ge 100 ] || { echo "disk low"; return 1; }
  for i in 1 2 3 4 5 6; do curl -sS -L -f -C - --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2" && return 0; sleep 30; done; return 1; }
arm_job() { # arm_job KEY DS ROOTS VARIANT PID
  gate; local R=$A/work/$2/$3
  W6_ROOT=$R/$4 W6_CLEAN=$R/clean W6_OUT=$OUT/$1/arms W6_OOD=1 $PY -W ignore $CODE/run_unit.py $4 $5; }
export -f gate free_pct arm_job; export A OUT PY CODE
# ---- 1. h5ads of the human candidates, then eligibility (metadata only) ----
if [ ! -f $OUT/SELECT_DONE ]; then
  $PY $AA/qhelp.py humans $CAND | while read id url; do gate; dl $url $A/h5ad/$id.h5ad || echo "h5ad download failed $id"; done
  $PY $AA/select_ds.py --cand $CAND --h5ad-dir $A/h5ad --out $OUT && touch $OUT/SELECT_DONE || exit 3
fi
nfail=0
run_ds() { # run_ds KEY DS [pilot]
  local KEY=$1 DS=$2 PILOT=${3:-}; local O=$OUT/$KEY D=$A/work/$DS; local TAG="" RT=roots FL; mkdir -p $O $D; [ -n "$PILOT" ] && { TAG=pilot; RT=roots_pilot; }
  FL=$($PY $AA/qhelp.py flags $OUT $DS); [ -n "$PILOT" ] && FL="--k 2 --coarse Manual_Annotation_L1_RNA_Integration --fine cell_type"
  [ -n "$PILOT" ] && cp $OUT/$DS/selection.json $O/selection.json
  rm -f $O/FAILED; local H5=$A/h5ad/$DS.h5ad; mkdir -p $A/frag/$DS
  if [ ! -f $O/DL_DONE ] && [ -z "$PILOT" ]; then
    local urls; urls=$($PY $AA/qhelp.py urls $CAND $DS | tail -n +2); for u in $urls; do dl $u $A/frag/$DS/$(basename $u) || { touch $O/FAILED; return 1; }; done; touch $O/DL_DONE; fi
  [ -f $O/PREP_DONE ] || { $PY -W ignore $AA/pools.py prep --dataset $DS --h5ad $H5 --sel $O/selection.json --data $D --out $O --tag "$TAG" --roots $RT $FL && touch $O/PREP_DONE || { touch $O/FAILED; return 1; }; }
  if [ ! -f $D/BUILD_DONE ]; then gate; BUILD_JOBS=4 OMP_NUM_THREADS=4 $SNAP -W ignore $AA/build_ga.py --h5ad $H5 --frag $A/frag/$DS/*-fragment.tsv.bgz --gtf $GTF --data $D && touch $D/BUILD_DONE || { touch $O/FAILED; return 1; }; fi
  [ -f $O/ROOTS_DONE ] || { gate; $PY -W ignore $AA/pools.py roots --dataset $DS --h5ad $H5 --sel $O/selection.json --data $D --out $O --roots $RT && touch $O/ROOTS_DONE || { touch $O/FAILED; return 1; }; }
  if [ ! -f $O/ARMS_DONE ]; then
    local jobs; jobs=$($PY - $O/pools.json $KEY $DS $RT <<'P'
import json, sys
i = json.load(open(sys.argv[1]))
for v in i["arms"]:
    for p in i["pools"]: print(sys.argv[2], sys.argv[3], sys.argv[4], v, p)
P
)
    echo "$jobs" | xargs -P 2 -L 1 bash -c 'arm_job "$@"' _   # at most 2 pipelines; unit csv = done marker inside run_unit.py
    local n; n=$(echo "$jobs" | grep -c .); [ "$n" -ge 1 ] && [ "$(ls $O/arms/units/*.csv 2>/dev/null | wc -l)" -ge "$n" ] && touch $O/ARMS_DONE || { touch $O/FAILED; return 1; }
  fi
  echo "$KEY done $(date)"; }
for ds in $($PY $AA/qhelp.py order $OUT $CAND); do
  echo "== $ds $(date)"; run_ds $ds $ds || { nfail=$((nfail + 1)); echo "FAILED $ds"; continue; }
  [ "$ds" = "$KID" ] && { run_ds ${ds}__pilot $ds pilot || { nfail=$((nfail + 1)); echo "FAILED ${ds}__pilot"; }; }
done
$PY -W ignore $AA/aggregate.py --out $OUT || echo "aggregate failed (no units yet?)"; echo "datasets failed: $nfail" > $OUT/DONE; echo "end $(date)"
