#!/usr/bin/env bash
# GloVe-100 data preparation (PROTOCOL.md amendment A3), in the dev container:
#   tools/dev.sh "bench/ann/adaptive/prepare_glove_all.sh bench/ann/results/adaptive_search/glove_v1"
set -euo pipefail
RUN=$1
D=${VECSEARCH_DATA:-/data}/adaptive
TOOL=build/bench/bench/ann/adaptive_eval
mkdir -p "$RUN"
cmake --preset bench >/dev/null && cmake --build --preset bench --target adaptive_eval >/dev/null
LOG=$RUN/prepare.log
: >"$LOG"
python3 bench/ann/adaptive/prepare_glove.py "$RUN" | tee -a "$LOG"
$TOOL build "$D/glove_base.fbin" "$D/glove_m16_efc200_s100.hnsw" 16 200 100 cosine | tee -a "$LOG"
echo "{\"snapshot\": \"glove_m16_efc200_s100.hnsw\", \"sha256\": \"$(sha256sum "$D/glove_m16_efc200_s100.hnsw" | cut -d' ' -f1)\"}" | tee -a "$LOG"
for s in learn val test; do
  $TOOL gt "$D/glove_base.fbin" "$D/glove_$s.fbin" 10 "$D/glove_$s.gt" cosine | sed "s/^{/{\"split\": \"$s\", /" | tee -a "$LOG"
done
