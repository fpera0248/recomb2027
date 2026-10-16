#!/usr/bin/env bash
# Reproduce the numbers of the paper at four depths.
#
#   reproduce.sh paper    download the release assets, recompute the Part II numbers of the 8-atlas test set plus the Part I panel and 31-type check, assert the match
#   reproduce.sh rerun    rescore the 69 held-out pools from the frozen candidate tables with this repository's scorer
#   reproduce.sh full     rebuild the 10-atlas test pools from CELLxGENE and rerun our method, Leiden subclustering and (TOOLS=1) the published tools on them (untested)
#   reproduce.sh full8    rebuild the 8-atlas test pools (the set of the headline result) from CELLxGENE and rerun our method, Leiden subclustering and (TOOLS=1) the published tools on them (untested)
#
# Env: PYTHON (python3), ASSETS (./assets), OUT (./reproduce_out; full8: ./reproduce_out8), TAG and ASSET_SRC (see fetch_data.sh).
# full also: HR_DATA (./work; full8: ./work8, so the pools of the two sets never mix); TOOLS=1 runs the published tools. They need their own Python environment (tools/requirements.txt, scikit-learn 1.9.1,
#   which conflicts with requirements.txt): set TOOLS_PYTHON to it. Also SCGPT_CKPT (scGPT whole-human checkpoint dir), MCL, RSCRIPT, see ENVIRONMENTS.md.
#   A tool that already has an output is skipped; failures are listed in $OUT/tool_failures.txt and the run goes on.
#
# paper    exact. Asserted: Table values, tests, per-atlas differences, null-test rates, the Part I numbers of notebook 04 (see notebooks/; they draw no figures). A few minutes, 1.2 GB of downloads.
#          One comparison has a tolerance: the bootstrap interval of the headline difference (0.005 on each end).
# rerun    not exact. Applies models/refit_dev_nat.pkl to the frozen Leiden groups and features and blends with the saved Leiden scores. The pkl
#          is a refit of the same recipe; the fit behind the saved scores was not kept. Measured on the 69 pools: Spearman of per-cell scores 0.878 on average (0.76 to 0.95),
#          atlas-equal union AP 0.3195 against 0.3213, single pools move by up to 0.2. Asserted: atlas-equal union AP within 0.01, mean Spearman above 0.8.
#          About 2 to 3 minutes.
# full     approximate. The cell-level leak check against the earlier datasets (data/leak_check.py, needs the 90 panel files) is not run here.
#          Leiden partitions depend on the numerical stack, so scores differ pool by pool. Tolerance (eval/full_check.py): atlas-equal
#          union AP and the difference to Leiden and to scCAD within 0.03 of the paper, same sign, exact atlas-level p below 0.05. 29.3 GB of downloads;
#          untested end to end, time and disk for the run are not measured. Without TOOLS=1 only the Leiden comparison (T1) is checked; $OUT/eval/result.md then still reads NOT PASSED with T2 among the failed parts, and eval/full_check.py decides the tier.
# full8    approximate, same steps as full, on the 8-atlas set: 22.3 GB of downloads, 70 pools of 20,000 cells built by data/build_pools8.py
#          (the engine of the original run), 69 of them with a target and run (atl_cereb_p5 has none). Before anything runs, eval/check_pools8.py compares each pool
#          (cells, targets, sha1 of the sorted cell ids, donor pack) with eval/expected/test8_pools.tsv, the values of the original pools (pools.tgz of the release). The only saved file used is
#          models/refit_dev_nat.pkl. Paper: ours 0.321, Leiden subclustering 0.220 (+0.101), scCAD with labels 0.231; a rebuild on identical pools gave 0.312 and 0.207.
#          eval/full_check.py --set8 asserts 69 pools in 8 atlases, means and differences within 0.03, same sign, exact atlas-level p below 0.05 for ours minus Leiden.
set -euo pipefail
case ${1:-} in
  -h|--help)
    awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"
    exit 0
    ;;
esac
mode=${1:?usage: reproduce.sh paper|rerun|full|full8}
case $mode in
  paper|rerun|full|full8) ;;
  *)
    echo "unknown mode $mode" >&2
    exit 2
    ;;
esac
here=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
ASSETS=$(mkdir -p "${ASSETS:-assets}" && cd "${ASSETS:-assets}" && pwd)
export ASSETS
case $mode in full8) OUT=${OUT:-reproduce_out8} ;; *) OUT=${OUT:-reproduce_out} ;; esac
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export REPO=$here PYTHONPATH=$here PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4}

case $mode in
  paper)
    "$here/fetch_data.sh" paper "$ASSETS"
    for nb in "$here"/notebooks/0*.ipynb; do
      echo "== $(basename "$nb")"
      $PYTHON -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output-dir "$OUT" "$nb"
    done
    echo "paper: all notebooks ran and every assertion held. Executed notebooks in $OUT"
    ;;
  rerun)
    "$here/fetch_data.sh" rerun "$ASSETS"
    $PYTHON -m eval.rerun "$ASSETS" "$OUT/rerun" "${@:2}"
    ;;
  full|full8)
    if [ "$mode" = full8 ]; then
      builder=data.build_pools8
      export ATLASES_CSV=$here/data/atlases8.csv
      def=work8
    else
      builder=data.build_pools
      export ATLASES_CSV=$here/data/atlases.csv
      def=work
    fi
    HR_DATA=$(mkdir -p "${HR_DATA:-$def}" && cd "${HR_DATA:-$def}" && pwd)
    export HR_DATA
    "$here/fetch_data.sh" "$mode" "$HR_DATA"
    names=$(cut -d, -f1 "$ATLASES_CSV" | tail -n +2)
    (cd "$here" && $PYTHON -m $builder $names)
    [ "$mode" = full8 ] && (cd "$here" && $PYTHON -m eval.check_pools8 "$HR_DATA")
    # full8: atl_cereb_p5 has no target and is not run or scored (69 pools)
    scored() { [ "$mode" = full ] || grep -q "^$1," "$HR_DATA"/targets/targets_*.csv; }
    run=$OUT/run
    mkdir -p "$run"
    for t in "$HR_DATA"/pilots/atl_*.parquet; do
      pid=$(basename "$t" .parquet)
      scored "$pid" || continue
      [ -s "$run/preds/hiddenrare/$pid.parquet" ] || nice -n 10 "$here/scripts/run_pool.sh" "$HR_DATA/pools/${pid#atl_}.h5ad" "$t" "$run"
    done
    if [ "${TOOLS:-0}" = 1 ]; then
      tp=${TOOLS_PYTHON:-$PYTHON}
      mkdir -p "$run/tool_out"
      for t in "$HR_DATA"/pilots/atl_*.parquet; do
        pid=$(basename "$t" .parquet)
        scored "$pid" || continue
        a=${pid#atl_}
        a=${a%_p*}
        $tp "$here/tools/adapt_pool.py" "$HR_DATA/pools/${pid#atl_}.h5ad" "$t" "$HR_DATA/genemap/$a.csv" "$OUT/toolin" || {
          echo "adapt_pool $pid" >> "$OUT/tool_failures.txt"
          continue
        }
        for tool in giniclust3 fire mars sccad scgpt_if cellsius gapclust scissors; do
          dir=$tool
          [ $tool = cellsius ] && dir=cellsius_label
          [ $tool = scissors ] && dir=scissors_cluster
          [ -s "$run/tool_out/$dir/$pid.parquet" ] || TOOLS_PYTHON=$tp "$here/scripts/run_tool.sh" $tool "$pid" "$OUT/toolin" "$run" || echo "$tool $pid" >> "$OUT/tool_failures.txt"
        done
      done
    else
      mkdir -p "$run/tool_out"
      echo "TOOLS is not 1: the published tools were not run, T2 (against scCAD) is not evaluated" >&2
    fi
    "$here/eval/run_eval.sh" "$HR_DATA" "$run" "$OUT/eval"
    $PYTHON -m eval.full_check "$OUT/eval/result.json" $([ "${TOOLS:-0}" = 1 ] || echo --no-sccad) $([ "$mode" = full8 ] && echo --set8)
    ;;
  *)
    echo "unknown mode $mode" >&2
    exit 2
    ;;
esac
