#!/bin/bash
# One published tool on one adapted pool.
# usage: scripts/run_tool.sh TOOL POOL_ID TOOLIN_DIR OUT_DIR      TOOLIN_DIR/{py,r}/<pool>.h5ad from tools/adapt_pool.py
# Results land in OUT_DIR/tool_out/<method>/<pool>.parquet. Python tools: PYTHON (default python3), TOOLS_SRC, FIRE_BUILD, SCGPT_CKPT;
# R tools: RSCRIPT (default Rscript), RETICULATE_PYTHON, R_EXTRA_LIBS, MCL (see tools/io.R).
set -euo pipefail
tool=$1; pid=$2; in=$3; out=$4
t=$(cd "$(dirname "$0")/../tools" && pwd); py=${PYTHON:-python3}; rs=${RSCRIPT:-Rscript}
w=$out/tool_out/.work/$pid; mkdir -p "$w"
keep() { mkdir -p "$out/tool_out/$1"; mv "$w/$2.parquet" "$out/tool_out/$1/$pid.parquet"; }
case $tool in
  giniclust3) $py $t/run_giniclust3.py $in/py/$pid.h5ad $w/giniclust3.parquet; keep giniclust3 giniclust3;;
  fire)       $py $t/run_fire.py $in/py/$pid.h5ad $w/fire.parquet; keep fire fire; keep fire_lab fire_lab;;
  mars)       $py $t/run_mars.py $in/py/$pid.h5ad $w/mars.parquet; keep mars mars;;
  sccad)      PYTHONHASHSEED=2023 $py $t/run_sccad.py $in/py/$pid.h5ad $w/sccad.parquet; keep sccad sccad; keep sccad_lab sccad_lab;;
  scgpt_if)   PYTORCH_ENABLE_MPS_FALLBACK=1 $py $t/run_scgpt_iforest.py $in/py/$pid.h5ad $w/scgpt_if.parquet --device ${DEVICE:-cpu}; keep scgpt_if scgpt_if;;
  cellsius)   $rs $t/run_cellsius.R $in/r/$pid.h5ad $w/cellsius_label.parquet label; keep cellsius_label cellsius_label;;
  gapclust)   $rs $t/run_gapclust.R $in/r/$pid.h5ad $w/gapclust.parquet; keep gapclust gapclust;;
  scissors)   $rs $t/run_scissors.R $in/r/$pid.h5ad $w/scissors_cluster.parquet cluster; keep scissors_cluster scissors_cluster;;
  *) echo "unknown tool $tool" >&2; exit 2;;
esac
