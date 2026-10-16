#!/bin/bash
# One published tool on one adapted pool.
# usage: scripts/run_tool.sh TOOL POOL_ID TOOLIN_DIR OUT_DIR      TOOLIN_DIR/{py,r}/<pool>.h5ad from tools/adapt_pool.py
# Results land in OUT_DIR/tool_out/<method>/<pool>.parquet. Python tools: TOOLS_PYTHON (else PYTHON, else python3; the tools environment, not the one for run_pool.sh), TOOLS_SRC, FIRE_BUILD, SCGPT_CKPT, DEVICE (scgpt_if only, default cpu);
# raceid, choir and scissors_official read py/ (genes detected in at least 3 cells); TOOLS_CORES (default 4) is the core count of raceid and choir; the per-label arms (raceid_lab, choir_lab, scissors_official_lab) come from tools/run_lab.py.
# choir needs Seurat >= 5 (the paper's CHOIR ran on Seurat 5): CHOIR_RSCRIPT (default RSCRIPT) and CHOIR_R_EXTRA_LIBS (default empty, replaces R_EXTRA_LIBS for choir only).
# R tools: RSCRIPT (default Rscript), RETICULATE_PYTHON, R_EXTRA_LIBS (see tools/io.R), MCL (mcl binary for CellSIUS, default mcl; see tools/run_cellsius.R).
set -euo pipefail
tool=$1
pid=$2
in=$3
out=$4
t=$(cd "$(dirname "$0")/../tools" && pwd)
py=${TOOLS_PYTHON:-${PYTHON:-python3}}
rs=${RSCRIPT:-Rscript}
w=$out/tool_out/.work/$pid
mkdir -p "$w"
keep() {
  mkdir -p "$out/tool_out/$1"
  mv "$w/$2.parquet" "$out/tool_out/$1/$pid.parquet"
}
case $tool in
  giniclust3)
    $py $t/run_giniclust3.py $in/py/$pid.h5ad $w/giniclust3.parquet
    keep giniclust3 giniclust3
    ;;
  fire)
    $py $t/run_fire.py $in/py/$pid.h5ad $w/fire.parquet
    keep fire fire
    keep fire_lab fire_lab
    ;;
  mars)
    $py $t/run_mars.py $in/py/$pid.h5ad $w/mars.parquet
    keep mars mars
    ;;
  sccad)
    PYTHONHASHSEED=2023 $py $t/run_sccad.py $in/py/$pid.h5ad $w/sccad.parquet
    keep sccad sccad
    keep sccad_lab sccad_lab
    ;;
  scgpt_if)
    PYTORCH_ENABLE_MPS_FALLBACK=1 $py $t/run_scgpt_iforest.py $in/py/$pid.h5ad $w/scgpt_if.parquet --device ${DEVICE:-cpu}
    keep scgpt_if scgpt_if
    ;;
  cellsius)
    $rs $t/run_cellsius.R $in/r/$pid.h5ad $w/cellsius_label.parquet label
    keep cellsius_label cellsius_label
    ;;
  gapclust)
    $rs $t/run_gapclust.R $in/r/$pid.h5ad $w/gapclust.parquet
    keep gapclust gapclust
    ;;
  scissors)
    $rs $t/run_scissors.R $in/r/$pid.h5ad $w/scissors_cluster.parquet cluster
    keep scissors_cluster scissors_cluster
    ;;
  giniclust3_readme)
    $py $t/run_giniclust3_readme.py $in/py/$pid.h5ad $w/giniclust3_readme.parquet
    keep giniclust3_readme giniclust3_readme
    keep giniclust3_readme_lab giniclust3_readme_lab
    ;;
  gapclust_official)
    $rs $t/run_gapclust_official.R $in/py/$pid.h5ad $w/gapclust_official.parquet $w/gapclust_official_adapter.parquet
    keep gapclust_official gapclust_official
    keep gapclust_official_adapter gapclust_official_adapter
    ;;
  scissors_official)
    $rs $t/run_scissors_official.R $in/py/$pid.h5ad $w/scissors_official.parquet
    $py $t/run_lab.py $in/py/$pid.h5ad $w/scissors_official.parquet $w/scissors_official_lab.parquet
    keep scissors_official scissors_official
    keep scissors_official_lab scissors_official_lab
    ;;
  raceid | choir)
    if [ ! -s $w/$tool.parquet ]; then   # a failed run_lab.py leaves the R output in .work: resume without the hours of CHOIR
      if [ $tool = choir ]; then
        R_EXTRA_LIBS=${CHOIR_R_EXTRA_LIBS:-} ${CHOIR_RSCRIPT:-$rs} $t/run_choir.R $in/py/$pid.h5ad $w/choir.parquet ${TOOLS_CORES:-4}
      else
        $rs $t/run_raceid.R $in/py/$pid.h5ad $w/raceid.parquet ${TOOLS_CORES:-4}
      fi
    fi
    $py $t/run_lab.py $in/py/$pid.h5ad $w/$tool.parquet $w/${tool}_lab.parquet
    keep $tool $tool
    keep ${tool}_lab ${tool}_lab
    ;;
  *)
    echo "unknown tool $tool" >&2
    exit 2
    ;;
esac
