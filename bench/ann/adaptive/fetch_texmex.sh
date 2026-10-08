#!/usr/bin/env bash
# Downloads and verifies the TEXMEX SIFT1M archive (needed for sift_learn.fvecs), ~161 MB:
#   tools/dev.sh bench/ann/adaptive/fetch_texmex.sh
# Files land in $VECSEARCH_DATA/texmex/; expected hashes are in sources.json.
set -euo pipefail
D=${VECSEARCH_DATA:-/data}/texmex
HERE=$(cd "$(dirname "$0")" && pwd)
want() { python3 -c "import json,sys; print(json.load(open('$HERE/sources.json'))[sys.argv[1]])" "$1"; }
check() { [ "$(sha256sum "$D/$1" | cut -d' ' -f1)" = "$(want "texmex/$1")" ] || { echo "sha256 mismatch: $D/$1" >&2; exit 1; }; }
mkdir -p "$D"
if [ ! -f "$D/sift.tar.gz" ]; then
  curl -sS --fail --max-time 3600 -o "$D/sift.tar.gz.part" ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  mv "$D/sift.tar.gz.part" "$D/sift.tar.gz"
fi
check sift.tar.gz
tar -xzf "$D/sift.tar.gz" -C "$D" sift/sift_base.fvecs sift/sift_learn.fvecs
check sift/sift_base.fvecs
check sift/sift_learn.fvecs
echo "TEXMEX SIFT files verified in $D"
