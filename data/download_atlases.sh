#!/bin/bash
# Download the test atlases listed in atlases.csv (public CELLxGENE h5ad files) into $HR_DATA/atlas. No size or md5 check, so a file left partial by an interrupted download is kept on the next run (delete it first); fetch_data.sh full checks both.
set -euo pipefail
dest=${HR_DATA:-work}/atlas
mkdir -p "$dest"
tail -n +2 "$(dirname "$0")/atlases.csv" | while IFS=, read -r name id coll stratum lab fine url; do
  [ -s "$dest/$id.h5ad" ] || curl -fL -o "$dest/$id.h5ad" "$url"
done
