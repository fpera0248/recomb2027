#!/bin/bash
# Score every prediction root and run the analysis.
# usage: eval/run_eval.sh DATA_DIR RUN_DIR OUT_DIR
#   DATA_DIR  $HR_DATA written by data/build_pools.py: pilots/, manifests/, targets/
#   RUN_DIR   preds/<method>/<pool>.parquet from scripts/run_pool.sh, tool_out/<tool>/<pool>.parquet from scripts/run_tool.sh
set -euo pipefail
data=$(cd "$1" && pwd); run=$(cd "$2" && pwd); mkdir -p "$3"; out=$(cd "$3" && pwd); py=${PYTHON:-python3}
root=$(cd "$(dirname "$0")/.." && pwd); export PYTHONPATH=$root
atlases=$(cut -d, -f1 "$root/data/atlases.csv" | tail -n +2)
man=(); tgt=(); for a in $atlases; do man+=("$data/manifests/atlas_manifest_$a.csv"); tgt+=("$data/targets/targets_$a.csv"); done
roots="preds tool_out"
cd "$out"
for r in $roots; do $py -m eval.score_atl "$run/$r" "scores_$r.csv" "$data/pilots" "${man[@]}"; done
for r in $roots; do $py -m eval.analyze scan "$run/$r" "flags_$r.csv"; done
head -1 flags_preds.csv > flags.csv; for r in $roots; do tail -n +2 "flags_$r.csv" >> flags.csv; done
for r in $roots; do $py -m eval.analyze tiefair "$run/$r" "tf_$r.csv" "$data/pilots" "${man[@]}"; done
head -1 tf_preds.csv > tf.csv; for r in $roots; do tail -n +2 "tf_$r.csv" >> tf.csv; done
# null gate, only if the twin pools were run
if ls "$data"/manifests/atlas_manifest_twin_*.csv >/dev/null 2>&1; then
  $py -m eval.export_groups "$run/preds" "$data/pilots" "$data/manifests" groups
  for r in $roots; do $py -m eval.analyze gateprep "$run/$r" groups "gate_$r.csv" "$data/pilots" "${man[@]}" --twin-map groups/twin_map.csv; done
  $py -c "import pandas as p; d=p.read_csv('gate_preds.csv'); d[d.method!='scorer'].to_csv('gate_preds.csv', index=False)"
  head -1 gate_preds.csv > gate.csv; for r in $roots; do tail -n +2 "gate_$r.csv" >> gate.csv; done
  gate=(--gate gate.csv)
else gate=(); fi
drop=(); [ -s "$data/leak_drop.txt" ] && drop=(--drop "$data/leak_drop.txt")
$py -m eval.analyze run --scores scores_*.csv --manifest "${man[@]}" --atlases "$root/data/atlases.csv" --split test ${drop[@]+"${drop[@]}"} \
  --targets "${tgt[@]}" --flags flags.csv --tiefair tf.csv ${gate[@]+"${gate[@]}"} --sccad sccad_lab --out result
$py -m eval.signflip result.json
