#!/bin/bash
# Clone the published tools at the commits in sources.txt into $TOOLS_SRC (default ./tools_src).
set -euo pipefail
dest=${TOOLS_SRC:-tools_src}; mkdir -p "$dest"
grep -v '^#' "$(dirname "$0")/sources.txt" | while read -r name url commit; do
  [ -d "$dest/$name/.git" ] || git clone -q "$url" "$dest/$name"
  git -C "$dest/$name" checkout -q "$commit"
done
