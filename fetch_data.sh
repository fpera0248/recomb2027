#!/usr/bin/env bash
# Download and verify what a reproduction tier needs.
#
#   fetch_data.sh paper|rerun|full|all [DEST]
#     paper  predictions, tool predictions, pools, Part I results   -> extracted under DEST (default ./assets)
#     rerun  pools, predictions, rerun inputs                       -> extracted under DEST
#     full   the 10-atlas test files from CELLxGENE (ATLAS_MANIFEST.csv, set test10) -> DEST/atlas/<dataset_id>.h5ad
#     all    paper, rerun and full
#
# Release assets come from $BASE/<asset> with BASE=https://github.com/fpera0248/recomb2027/releases/download/$TAG, and are checked against
# ASSETS.md5 (kept in this repository). ASSET_SRC=DIR takes them from a local directory instead of the network.
# While the repository is private the plain URL fails, and the download falls back to `gh release download` (needs gh and a login with read access).
# CELLxGENE publishes no checksums: every atlas is checked against its size in ATLAS_MANIFEST.csv, and against its md5 when the manifest has one.
set -euo pipefail
case ${1:-} in -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0;; esac
TAG=${TAG:-v0.1.0}
BASE=${BASE:-https://github.com/fpera0248/recomb2027/releases/download/$TAG}
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:?usage: fetch_data.sh paper|rerun|full|all [DEST]}; dest=${2:-assets}
mkdir -p "$dest"

md5of() { if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
sizeof() { wc -c < "$1" | tr -d ' '; }
get() {  # URL OUT
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -C - -o "$2" "$1"; else wget -c -O "$2" "$1"; fi
}

fetch_gh() {  # private repository: unauthenticated URLs return 404, so use the GitHub CLI login
  command -v gh >/dev/null 2>&1 || { echo "download failed; for a private repository install gh and log in, or set ASSET_SRC" >&2; return 1; }
  rm -f "$2"; gh release download "$TAG" -R "${REPO_SLUG:-fpera0248/recomb2027}" -p "$1" -O "$2"
}

asset() {
  local name=$1 f=$dest/$1 want
  want=$(awk -v n="$name" '$2 == n {print $1}' "$here/ASSETS.md5")
  [ -n "$want" ] || { echo "$name is not in ASSETS.md5" >&2; exit 1; }
  if [ ! -s "$f" ]; then
    if [ -n "${ASSET_SRC:-}" ]; then cp "$ASSET_SRC/$name" "$f"; else get "$BASE/$name" "$f" || fetch_gh "$name" "$f"; fi
  fi
  [ "$(md5of "$f")" = "$want" ] || { echo "md5 mismatch: $f (delete it and retry)" >&2; exit 1; }
  echo "ok  $name"
  [ -e "$dest/.extracted_$name" ] || { tar xzf "$f" -C "$dest" && touch "$dest/.extracted_$name"; }
}

atlases() {  # source files of one manifest set
  local set=$1 a=$dest/atlas
  mkdir -p "$a"
  ${PYTHON:-python3} - "$here/ATLAS_MANIFEST.csv" "$set" <<'PY' | while IFS=$'\t' read -r id url size md5; do
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    if r["set"] == sys.argv[2] and r["download_url"]:
        print("\t".join([r["dataset_id"], r["download_url"], r["remote_size"], r["md5"]]))
PY
    f=$a/$id.h5ad
    [ -s "$f" ] || { get "$url" "$f.part" && mv "$f.part" "$f"; }
    [ "$(sizeof "$f")" = "$size" ] || { echo "size mismatch: $f (delete the partial file and rerun)" >&2; exit 1; }
    [ -z "$md5" ] || [ "$(md5of "$f")" = "$md5" ] || { echo "md5 mismatch: $f" >&2; exit 1; }
    echo "ok  $id${md5:+ (md5 checked)}"
  done
}

case $mode in paper|rerun|full|all) ;; *) echo "unknown mode $mode" >&2; exit 2 ;; esac
if [ "$mode" = paper ] || [ "$mode" = all ]; then for n in predictions.tgz tool_predictions.tgz pools.tgz part1_results.tgz; do asset $n; done; fi
if [ "$mode" = rerun ] || [ "$mode" = all ]; then for n in pools.tgz predictions.tgz rerun_inputs.tgz; do asset $n; done; fi
if [ "$mode" = full ] || [ "$mode" = all ]; then atlases test10; fi
