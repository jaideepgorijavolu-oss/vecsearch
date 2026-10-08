#!/usr/bin/env bash
# Reproduces every Phase 2 and Phase 4 number in docs/RESULTS.md (`make bench` runs this).
# Needs: a C++ toolchain, CMake, Python with numpy h5py matplotlib psutil hnswlib faiss-cpu,
# and Linux perf with hardware counters for the profiling step (skipped if unavailable).
# Datasets go to $VECSEARCH_DATA (default ./data), about 1 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
export VECSEARCH_DATA="${VECSEARCH_DATA:-$PWD/data}"

echo "== build"
cmake --preset bench >/dev/null
cmake --build --preset bench
pip install -q .

echo "== Phase 2: distance kernel microbenchmarks"
./build/bench/bench/micro/bench_distance --benchmark_repetitions=3 --benchmark_min_time=0.2s \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=bench/micro/results/distance.json --benchmark_out_format=json >/dev/null
python3 bench/micro/summarize.py bench/micro/results/distance.json > bench/micro/results/distance.md

echo "== Phase 4: datasets"
python3 -c "import sys; sys.path.insert(0, 'bench/ann'); import datasets; [datasets.download(d) for d in ('sift', 'glove')]"

echo "== Phase 4: recall / QPS sweeps (one process per engine and dataset)"
for ds in sift glove; do
  for engine in vecsearch vecsearch-noprefetch hnswlib faiss; do
    python3 bench/ann/run.py --dataset "$ds" --engine "$engine"
  done
done
python3 bench/ann/plot.py > bench/ann/results/tables.md

echo "== Phase 4: profile (perf stat)"
if python3 bench/ann/make_subset.py "$VECSEARCH_DATA/sift-128-euclidean.hdf5" 1000000 "$VECSEARCH_DATA/sift1m" \
   && python3 bench/ann/profile.py --ef 64 --seconds 20 > bench/ann/results/profile.md; then
  cat bench/ann/results/profile.md
else
  echo "perf not available; skipped profiling"
fi
echo "done: see bench/ann/results/"
