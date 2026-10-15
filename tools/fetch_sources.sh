#!/bin/bash
# Clone the published tools at the commits in sources.txt into $TOOLS_SRC (default ./tools_src).
# usage: tools/fetch_sources.sh        (no arguments)
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: tools/fetch_sources.sh   (env: TOOLS_SRC)" >&2; exit 2; }
dest=${TOOLS_SRC:-tools_src}; mkdir -p "$dest"
grep -v '^#' "$(dirname "$0")/sources.txt" | while read -r name url commit; do
  [ -d "$dest/$name/.git" ] || git clone -q "$url" "$dest/$name"
  git -C "$dest/$name" checkout -q "$commit"
done
md5=$(if command -v md5sum >/dev/null 2>&1; then md5sum "$dest/scCAD/scCAD.py" | cut -d' ' -f1; else md5 -q "$dest/scCAD/scCAD.py"; fi)
[ "$md5" = a0865c37513aaee5eae13551e48da6f2 ] || { echo "scCAD.py md5 is $md5, expected a0865c37513aaee5eae13551e48da6f2" >&2; exit 1; }
