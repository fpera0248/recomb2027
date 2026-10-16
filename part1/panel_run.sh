#!/bin/bash
# Usage: panel_run.sh DATASET_ID [pull]. With no argument it runs the next dataset of panel_queue_shuffled.txt
# that has no done, failed or claimed marker in panel_runs/<dataset>/; after an interrupted run, delete that claimed file to requeue it.
ds=$1
if [ -z "$1" ] && [ -f panel_queue_shuffled.txt ]; then
  ds=$(while read q; do [ -e panel_runs/$q/done ] || [ -e panel_runs/$q/failed ] || [ -e panel_runs/$q/claimed ] || { echo $q; break; }; done < panel_queue_shuffled.txt)
  [ -z "$ds" ] && exit 0
fi
R=panel_runs/$ds
D=${PANEL_DATA:?set PANEL_DATA}
mkdir -p $R $D
[ -f $R/done ] && exit 0
[ -z "$1" ] && touch $R/claimed
P=$(python3 -c "import pandas as pd,sys; print(pd.read_csv('panel_eligible.csv').set_index('dataset_id').loc[sys.argv[1],'probe'])" $ds)
step() {
  echo "$(date +%m-%d\ %H:%M:%S) $ds $1" >> panel_runs/progress.log
}
fail() {
  step "FAILED $1: $(grep -hE 'Error|assert' $R/$1.log | tail -1 | cut -c1-200)"
  touch $R/failed
  exit 1
}
if [ ! -f $D/$ds.h5ad ]; then
  python3 panel_pull.py $ds $D/$ds.tmp.h5ad "$P" > $R/pull.log 2>&1 && mv $D/$ds.tmp.h5ad $D/$ds.h5ad || fail pull
  step pulled
fi
[ "$2" = pull ] && exit 0
export H5AD=$D/$ds.h5ad PROBE="$P" SKIP_SD3=1 OUT=$R/gen SAVE_GENE=$R/hl GENE_DIR=$R/hl READOUT_CSV=$R/clf.csv
PREFLIGHT=1 python3 hlca.py > $R/preflight.log 2>&1 || fail preflight
step start
python3 hlca.py > $R/gen.log 2>&1 || fail gen
NJ=2 python3 panel_readout.py run > $R/clf.log 2>&1 || fail clf
python3 noaug.py $R/real.csv > $R/real.log 2>&1 || fail real
touch $R/done
step done
