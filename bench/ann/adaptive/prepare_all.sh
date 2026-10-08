#!/usr/bin/env bash
# Data preparation for the adaptive-search study, in the dev container:
#   tools/dev.sh "bench/ann/adaptive/prepare_all.sh bench/ann/results/adaptive_search/sift1m_v1"
# Splits, the fixed index snapshot (1 thread, seed 100), a build-determinism check, and exact
# top-10 ground truth for learn / val / test. Logs JSON lines to <run_dir>/prepare.log.
set -euo pipefail
RUN=$1
D=${VECSEARCH_DATA:-/data}/adaptive
TOOL=build/bench/bench/ann/adaptive_eval
mkdir -p "$RUN"
cmake --preset bench >/dev/null && cmake --build --preset bench --target adaptive_eval >/dev/null
LOG=$RUN/prepare.log
: >"$LOG"

python3 bench/ann/adaptive/prepare.py "$RUN" | tee -a "$LOG"

# Determinism: building the first 100k base vectors twice on one thread gives identical files.
python3 - "$D" <<'EOF'
import sys, numpy as np
d = sys.argv[1]
x = np.fromfile(f"{d}/sift_base.fbin", dtype=np.uint32, count=2)
v = np.fromfile(f"{d}/sift_base.fbin", dtype=np.float32, offset=8, count=100000 * int(x[1]))
with open(f"{d}/sift_base100k.fbin", "wb") as f:
    np.array([100000, x[1]], dtype=np.uint32).tofile(f); v.tofile(f)
EOF
$TOOL build "$D/sift_base100k.fbin" "$D/det_a.hnsw" 16 200 100 >/dev/null
$TOOL build "$D/sift_base100k.fbin" "$D/det_b.hnsw" 16 200 100 >/dev/null
a=$(sha256sum "$D/det_a.hnsw" | cut -d' ' -f1); b=$(sha256sum "$D/det_b.hnsw" | cut -d' ' -f1)
echo "{\"determinism_100k\": \"$([ "$a" = "$b" ] && echo identical || echo DIFFERENT)\", \"sha256\": \"$a\"}" | tee -a "$LOG"
rm -f "$D/det_a.hnsw" "$D/det_b.hnsw" "$D/sift_base100k.fbin"

# The snapshot every policy is compared on.
$TOOL build "$D/sift_base.fbin" "$D/sift_m16_efc200_s100.hnsw" 16 200 100 | tee -a "$LOG"
echo "{\"snapshot\": \"sift_m16_efc200_s100.hnsw\", \"sha256\": \"$(sha256sum "$D/sift_m16_efc200_s100.hnsw" | cut -d' ' -f1)\"}" | tee -a "$LOG"

for s in learn val test; do
  $TOOL gt "$D/sift_base.fbin" "$D/sift_$s.fbin" 10 "$D/sift_$s.gt" | sed "s/^{/{\"split\": \"$s\", /" | tee -a "$LOG"
done
