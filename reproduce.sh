#!/usr/bin/env bash
# Reproduce the numbers of the paper at three depths.
#
#   reproduce.sh paper    download the release assets, recompute the Part II numbers of the 8-atlas test set from the saved predictions, assert the match
#   reproduce.sh rerun    rescore the 69 held-out pools from the frozen candidate tables with this repository's scorer
#   reproduce.sh full     rebuild the 10-atlas test pools from CELLxGENE and rerun the whole pipeline
#
# Env: PYTHON (python3), ASSETS (./assets), OUT (./reproduce_out), TAG and ASSET_SRC (see fetch_data.sh).
# full also: HR_DATA (./work); TOOLS=1 runs the published tools. They need their own Python environment (tools/requirements.txt, scikit-learn 1.9.1,
#   which conflicts with requirements.txt): set TOOLS_PYTHON to it. Also SCGPT_CKPT (scGPT whole-human checkpoint dir), MCL, RSCRIPT, see ENVIRONMENTS.md.
#   A tool that already has an output is skipped; failures are listed in $OUT/tool_failures.txt and the run goes on.
#
# paper    exact. Asserted: Table values, tests, per-atlas differences, null-test rates, Part I figures (see notebooks/). A few minutes, 1.2 GB of downloads.
#          One comparison has a tolerance: the bootstrap interval of the headline difference (0.005 on each end).
# rerun    not exact. Applies models/refit_dev_nat.pkl to the frozen Leiden groups and features and blends with the saved Leiden scores; the saved
#          scores come from a scorer that is not in this repository. Measured on the 69 pools: Spearman of per-cell scores 0.88 on average (0.76 to 0.95),
#          atlas-equal union AP 0.3195 against 0.3213, single pools move by up to 0.2. Asserted: atlas-equal union AP within 0.01, mean Spearman above 0.8.
#          About 2 minutes.
# full     approximate. The cell-level leak check against the earlier datasets (data/leak_check.py, needs the 90 panel files) is not run here.
#           Leiden partitions depend on the numerical stack, so scores differ pool by pool. Tolerance (eval/full_check.py): atlas-equal
#          union AP and the difference to Leiden and to scCAD within 0.03 of the paper, same sign, exact atlas-level p below 0.05. About 30 GB of downloads
#          and 20 GB of pool files; the pipeline ran on a 12-core machine. Without TOOLS=1 only the Leiden comparison (T1) is checked.
set -euo pipefail
mode=${1:?usage: reproduce.sh paper|rerun|full}
here=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}; export ASSETS=$(mkdir -p "${ASSETS:-assets}" && cd "${ASSETS:-assets}" && pwd); OUT=${OUT:-reproduce_out}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
export REPO=$here PYTHONPATH=$here PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4}

case $mode in
  paper)
    "$here/fetch_data.sh" paper "$ASSETS"
    for nb in "$here"/notebooks/0*.ipynb; do
      echo "== $(basename "$nb")"
      $PYTHON -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output-dir "$OUT" "$nb"
    done
    echo "paper: all notebooks ran and every assertion held. Executed notebooks in $OUT" ;;
  rerun)
    "$here/fetch_data.sh" rerun "$ASSETS"
    $PYTHON -m eval.rerun "$ASSETS" "$OUT/rerun" "${@:2}" ;;
  full)
    export HR_DATA=$(mkdir -p "${HR_DATA:-work}" && cd "${HR_DATA:-work}" && pwd)
    "$here/fetch_data.sh" full "$HR_DATA"
    names=$(cut -d, -f1 "$here/data/atlases.csv" | tail -n +2)
    (cd "$here" && $PYTHON -m data.build_pools $names)
    run=$OUT/run; mkdir -p "$run"
    for t in "$HR_DATA"/pilots/atl_*.parquet; do
      pid=$(basename "$t" .parquet)
      [ -s "$run/preds/hiddenrare/$pid.parquet" ] || nice -n 10 "$here/scripts/run_pool.sh" "$HR_DATA/pools/${pid#atl_}.h5ad" "$t" "$run"
    done
    if [ "${TOOLS:-0}" = 1 ]; then
      tp=${TOOLS_PYTHON:-$PYTHON}; mkdir -p "$run/tool_out"
      for t in "$HR_DATA"/pilots/atl_*.parquet; do
        pid=$(basename "$t" .parquet); a=${pid#atl_}; a=${a%_p*}
        $tp "$here/tools/adapt_pool.py" "$HR_DATA/pools/${pid#atl_}.h5ad" "$t" "$HR_DATA/genemap/$a.csv" "$OUT/toolin"
        for tool in giniclust3 fire mars sccad scgpt_if cellsius gapclust scissors; do
          dir=$tool; [ $tool = cellsius ] && dir=cellsius_label; [ $tool = scissors ] && dir=scissors_cluster
          [ -s "$run/tool_out/$dir/$pid.parquet" ] || TOOLS_PYTHON=$tp "$here/scripts/run_tool.sh" $tool "$pid" "$OUT/toolin" "$run" || echo "$tool $pid" >> "$OUT/tool_failures.txt"
        done
      done
    else mkdir -p "$run/tool_out"; echo "TOOLS is not 1: the published tools were not run, T2 (against scCAD) is not evaluated" >&2; fi
    "$here/eval/run_eval.sh" "$HR_DATA" "$run" "$OUT/eval"
    $PYTHON -m eval.full_check "$OUT/eval/result.json" $([ "${TOOLS:-0}" = 1 ] || echo --no-sccad) ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
