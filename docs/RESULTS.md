# Results

Every number in this file was produced by a script in this repo; the command is given next to
each table. Anything not yet measured is marked TBD.

## Hardware and software

| | |
|---|---|
| CPU | AMD Ryzen 9 270 w/ Radeon 780M Graphics (8 cores / 16 threads, Zen 4, AVX2 + FMA + AVX-512) |
| Host | Windows 11 laptop, 16 GB RAM |
| Environment | Docker Desktop, WSL2 VM (Linux 6.18.33.2-microsoft-standard-WSL2), 16 vCPUs, 7.9 GB RAM visible |
| OS image | Ubuntu 24.04 (`docker/dev.Dockerfile`) |
| Compiler | GCC 13.3.0, `-O3` (CMake Release) |
| Power | Windows **Ultimate Performance** power plan, nothing else CPU-heavy running (checked before the run). Phases 2 and 4 were rerun this way on 2026-10-03; the Phase 3 and Phase 6 numbers are from earlier runs in **Eco** mode and are lower than this machine can do. |

Caveats: this is a laptop, so clocks depend on power and temperature (the benchmark library
reports a nominal 3993 MHz). Everything runs inside a WSL2 virtual machine, which adds a little
overhead to memory-intensive work and hides hardware performance counters (see Phase 4).

## Phase 2: Distance kernel microbenchmarks

Command:

```bash
cmake --preset bench && cmake --build --preset bench --target bench_distance
./build/bench/bench/micro/bench_distance --benchmark_repetitions=3 --benchmark_min_time=0.2s \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=bench/micro/results/distance.json --benchmark_out_format=json
python3 bench/micro/summarize.py bench/micro/results/distance.json
```

Raw output: `bench/micro/results/distance.json` (tables: `distance.md`), from the `bench/run_all.sh` run. Medians of 3 repetitions.

Kernels:

- `scalar`: the reference loop in `src/distance/scalar.cpp`, library build flags (`-O3`, baseline x86-64).
- `autovec_O3_native`: the identical loop compiled with `-O3 -march=native` (znver4).
- `autovec_O3_native_fastmath`: the same plus `-ffast-math`.
- `avx2`: the hand-written AVX2 + FMA kernel in `src/distance/avx2.cpp` (what the indexes use on this machine).

Two scenarios:

- **hot**: the same two vectors on every call, so both are in L1. GB/s = 2 × dim × 4 bytes per call.
- **scan**: one query against a 256 MiB database read sequentially (far larger than the cache),
  so every call streams a new vector from DRAM. GB/s = dim × 4 bytes per call.

### hot / l2: ns per call (GB/s)

| dim | scalar | autovec_O3_native | autovec_O3_native_fastmath | avx2 |
|---:|---:|---:|---:|---:|
| 32 | 9.0 (28.5) | 7.4 (34.5) | 1.8 (139.0) | 2.0 (128.2) |
| 64 | 22.4 (22.9) | 19.4 (26.4) | 2.7 (191.8) | 2.6 (194.3) |
| 128 | 47.9 (21.4) | 47.0 (21.8) | 4.4 (230.5) | 4.2 (244.0) |
| 384 | 199.7 (15.4) | 195.6 (15.7) | 12.0 (255.1) | 10.6 (288.7) |
| 768 | 428.0 (14.4) | 426.5 (14.5) | 25.9 (236.8) | 20.8 (298.0) |
| 1536 | 917.5 (13.4) | 876.8 (14.0) | 50.8 (241.9) | 40.9 (300.1) |

### hot / dot: ns per call (GB/s)

| dim | scalar | autovec_O3_native | autovec_O3_native_fastmath | avx2 |
|---:|---:|---:|---:|---:|
| 32 | 8.1 (31.6) | 7.0 (36.6) | 1.6 (155.8) | 1.9 (138.3) |
| 64 | 21.9 (23.3) | 18.7 (27.4) | 2.4 (209.1) | 2.4 (211.1) |
| 128 | 45.3 (22.6) | 45.6 (22.4) | 4.1 (246.9) | 3.8 (272.5) |
| 384 | 193.7 (15.9) | 193.5 (15.9) | 11.8 (260.9) | 10.4 (294.5) |
| 768 | 420.7 (14.6) | 420.1 (14.6) | 25.8 (238.3) | 20.4 (300.6) |
| 1536 | 871.1 (14.1) | 886.2 (13.9) | 52.3 (235.0) | 40.8 (301.1) |

### scan / l2: ns per call (GB/s)

| dim | scalar | autovec_O3_native | autovec_O3_native_fastmath | avx2 |
|---:|---:|---:|---:|---:|
| 32 | 9.6 (13.4) | 8.0 (16.0) | 3.9 (33.1) | 3.9 (32.6) |
| 64 | 23.8 (10.8) | 20.4 (12.5) | 7.6 (33.7) | 7.9 (32.4) |
| 128 | 50.4 (10.2) | 48.9 (10.5) | 17.4 (29.4) | 16.3 (31.4) |
| 384 | 203.0 (7.6) | 201.0 (7.6) | 46.3 (33.2) | 45.3 (33.9) |
| 768 | 435.8 (7.0) | 432.8 (7.1) | 85.2 (36.1) | 87.1 (35.3) |
| 1536 | 894.0 (6.9) | 895.4 (6.9) | 194.5 (31.6) | 196.5 (31.3) |

### scan / dot: ns per call (GB/s)

| dim | scalar | autovec_O3_native | autovec_O3_native_fastmath | avx2 |
|---:|---:|---:|---:|---:|
| 32 | 9.0 (14.3) | 7.5 (17.0) | 3.4 (38.1) | 3.4 (37.2) |
| 64 | 22.9 (11.2) | 20.1 (12.8) | 6.9 (37.2) | 7.0 (36.4) |
| 128 | 47.3 (10.9) | 47.9 (10.7) | 14.8 (34.9) | 15.0 (34.3) |
| 384 | 199.9 (7.7) | 200.5 (7.7) | 53.4 (28.7) | 52.1 (29.5) |
| 768 | 434.4 (7.1) | 440.0 (7.1) | 99.5 (30.9) | 99.7 (30.8) |
| 1536 | 897.2 (6.8) | 901.1 (6.8) | 175.8 (35.1) | 175.4 (35.1) |

### Discussion

- **`-O3 -march=native` alone does nothing.** The plain loop stays scalar (`vaddss` in the
  disassembly) and runs at the scalar speed. Floating-point addition is not associative, so
  without permission to reorder the sum, the compiler cannot split it across vector lanes.
- **With `-ffast-math` the compiler gets close.** It vectorizes with 512-bit AVX-512 registers
  and is 5–25% slower than the hand kernel for dim ≥ 128 when data is in L1, and equal or
  slightly faster at dim 32 and 64, where the hand kernel's main loop runs only once or twice and
  its fixed costs (zeroing four accumulators, the horizontal sum) dominate. The disassembly shows
  why it is not faster despite 2× wider registers: its main loop has a *single* accumulator
  (`vmulps` + `vaddps` into one `zmm`), so every iteration waits for the previous add. The hand
  kernel uses four independent FMA accumulators. `-ffast-math` is not an option for the library
  anyway: it changes NaN/infinity semantics for the whole translation unit.
- **The hand AVX2 kernel is ~20× faster than scalar** when data is in L1 (e.g. dim 768:
  428 → 20.8 ns for L2), reaching ~300 GB/s of L1 load bandwidth.
- **Once data comes from DRAM, every vectorized kernel hits the same wall**: ~29–36 GB/s
  single-thread streaming bandwidth, for all dims. In the scan test the hand AVX2 kernel and the
  fast-math auto-vectorized kernel are within ~7% of each other, and the auto-vectorized one is
  sometimes ahead (e.g. L2 at dim 768: 85.2 vs 87.1 ns); the arithmetic is no longer what
  limits them. The scalar kernel is still compute bound (7–13 GB/s), so SIMD helps 2.5–5×
  there, not 20×.
- **Implication for HNSW**: graph search touches vectors in an unpredictable order, so each
  distance computation is likely a cache miss (latency bound, worse than streaming). SIMD makes
  the arithmetic nearly free; the remaining cost is memory access. Phase 4 measures this.

## Phase 3: HNSW correctness

Command (after `python3 bench/ann/make_subset.py data/sift-128-euclidean.hdf5 100000 data/sift100k`):

```bash
./build/bench/bench/ann/hnsw_eval data/sift100k.base.fbin data/sift100k.query.fbin l2 16 200 0 10,20,40,60,80,120,160
```

Random 100k subset of SIFT1M (seed 0), all 10k queries, ground truth recomputed exactly with
`FlatIndex` on the subset. M = 16, ef_construction = 200, 16 build threads. Raw output:
`bench/ann/results/phase3_sift100k.txt`.

| ef_search | recall@10 |
|---:|---:|
| 10 | 0.7799 |
| 20 | 0.8944 |
| 40 | 0.9632 |
| 60 | 0.9829 |
| 80 | 0.9904 |
| 120 | 0.9957 |
| 160 | 0.9973 |

Build: 3.34 s on 16 threads. Acceptance (recall@10 ≥ 0.95 at a reasonable ef) is met at
ef = 40. Save → load → search returns identical ids and distances. The tool also prints QPS,
but those are single 10k-query runs; proper throughput measurements are in Phase 4.

Other correctness results (from the unit tests and tools, see DESIGN.md):

- Neighbor-selection heuristic vs closest-M (`Hnsw.HeuristicBeatsClosestM`, 20k points in 100
  tight clusters, dim 8, M = 6, ef = 20): recall@10 0.28 vs 0.61.
- Parallel build quality (`bench/ann/parallel_build_quality`, output in
  `bench/ann/results/parallel_build_quality.txt`): the sequential first 1,000 inserts recover
  almost all of the quality lost by a 16-thread build on the earliest nodes.
- ASan + UBSan (`ctest --preset sanitizer`) and TSan (`ctest --preset tsan`) run the full suite,
  including 4- and 8-thread builds and parallel searches, with no reports.

## Phase 4: ANN benchmarks

Command: `make bench` (= `bash bench/run_all.sh`). It builds everything, downloads the datasets,
runs `bench/ann/run.py` once per (dataset, engine) in a separate process, and writes
`bench/ann/results/<dataset>_<engine>.json`, the plots, `tables.md` and the perf profile.
Log of the run used here: `bench/ann/results/run_all.log`.

**Methodology** (identical for every engine; `bench/ann/run.py`, `bench/ann/engines.py`):

- Datasets: ann-benchmarks HDF5 files. SIFT1M: 1,000,000 × 128, L2. GloVe-100: 1,183,514 × 100,
  angular (cosine; Faiss gets normalized vectors and inner product). 10,000 test queries each,
  recall@10 against the files' ground truth.
- Engines: vecsearch (this repo), hnswlib 0.8.0 and Faiss 1.15.1 `IndexHNSWFlat` (both from
  pip). `vecsearch-noprefetch` is this engine with software prefetching turned off: the
  before/after of the profiling-driven optimization below.
- Same parameters for all: M = 16 (32 links on layer 0 in all three), ef_construction = 200,
  16 threads to build and for the "16 threads" runs.
- Throughput: one batch call with all 10,000 queries from NumPy. Best of 7 runs with 16 threads,
  best of 3 with 1 thread.
- Latency: one query per call, one thread, the first 2,000 queries, at the smallest ef where the
  engine reaches recall@10 ≥ 0.95. Includes the Python call overhead (a few µs, the same for
  all).
- Memory: growth of the process RSS during the build (vectors + graph + allocator overhead).
- One run per configuration on a laptop (Ultimate Performance plan, see Hardware). Comparing the full runs made
  during development, run-to-run noise is roughly ±5% for single-thread numbers and up to
  ±10% for 16-thread numbers (a 16-thread batch lasts only 0.1–5 s). Smaller differences are not
  meaningful.

![SIFT1M recall vs QPS](../bench/ann/results/sift_recall_qps.png)
![GloVe-100 recall vs QPS](../bench/ann/results/glove_recall_qps.png)

### SIFT1M (128-d, L2), at recall@10 ≈ 0.95

| engine | build (s) | RSS growth (MiB) | QPS @0.95, 16 threads (ef) | QPS @0.95, 1 thread | QPS @0.99, 1 thread | p50 / p99 latency @0.95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| vecsearch | 63.4 | 726 | 35,149 (64) | 6,488 | 2,609 | 0.158 / 0.234 |
| vecsearch-noprefetch | 74.1 | 726 | 32,174 (64) | 5,086 | 2,099 | 0.193 / 0.308 |
| hnswlib | 67.5 | 753 | 34,255 (64) | 6,293 | 2,483 | 0.169 / 0.279 |
| faiss | 64.7 | 679 | 42,215 (64) | 5,388 | 2,951 | 0.208 / 0.340 |

"@0.95" is the first ef in the sweep with recall ≥ 0.95: ef = 64 for all four (recall 0.9635,
0.9635, 0.9638, 0.9675). "@0.99" is the first ef with recall ≥ 0.99: 192 for vecsearch and
hnswlib, 128 for Faiss, whose recall at equal ef is slightly higher.

### GloVe-100 (100-d, angular), at recall@10 ≈ 0.95

| engine | build (s) | RSS growth (MiB) | QPS @0.95, 16 threads (ef) | QPS @0.95, 1 thread | p50 / p99 latency @0.95 (ms) |
|---|---:|---:|---:|---:|---:|
| vecsearch | 81.3 | 785 | 3,099 (1024) | 530 | 1.99 / 2.79 |
| vecsearch-noprefetch | 81.4 | 785 | 3,021 (1024) | 436 | 2.35 / 3.18 |
| hnswlib | 93.4 | 762 | 2,680 (1024) | 490 | 2.13 / 2.89 |
| faiss | 80.2 | 677 | 2,916 (1024) | 421 | 2.58 / 3.76 |

GloVe is much harder than SIFT: with M = 16 every engine needs ef ≈ 1024 for 0.95 recall@10,
and none reaches 0.99 in the sweep, which stops at ef = 1536 with recall ≈ 0.97. The full sweeps
(every ef: recall, QPS with 16 threads and 1 thread) are in `bench/ann/results/tables.md`.

### Profile of the search hot path

Command: `python3 bench/ann/profile.py --ef 64 --seconds 20` (run by `make bench`; needs Linux
`perf`, and in Docker `--privileged`). `bench/ann/profile_search` searches all SIFT1M queries on
one thread in a loop; the counters of a run that only loads the index are subtracted, and the
rest is divided by the number of queries. Output: `bench/ann/results/profile.md`.

| prefetch | QPS | cycles/query | instructions/query | IPC | L1d misses/query | LLC misses/query | branch misses/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 5,159 | 924,154 | 234,959 | 0.25 | 15,354 | 13,283 | 1,906 |
| on | 6,569 | 742,074 | 307,370 | 0.41 | 16,337 | 14,050 | 2,034 |

`perf record` (sampling, prefetch off, during development): ~51% of cycles in the AVX2 `l2_sq`
kernel and ~43% in the layer-0 search loop (inlined into the search lambda). `perf annotate`
showed:

- inside `l2_sq`, ~55% of its samples on the first loads of the vector (`vsubps` with a memory
  operand): the kernel is waiting for the vector to arrive from DRAM, not computing;
- inside the search loop, ~17% of its samples right after `cmp %dx,(%rax)`, the load of
  `visited.marks[id]`, a random 2-byte read from a 2 MB array: another cache miss per neighbor.

**Reading the profile.** IPC 0.25 on a core that can retire 4+ instructions per cycle means the
search is almost always stalled on memory. ~13,300 last-level cache misses per query at ef = 64
is about one DRAM miss per cache line of every vector visited (a 128-d vector is 8 lines). Each
neighbor was a chain: load its visited mark (miss), then its vector (miss), then compute, and the
next neighbor started only after that. Phase 2 had already shown that the kernel arithmetic is
nearly free; the time is DRAM latency, paid one miss at a time.

### Optimization made because of the profile: batched software prefetching

`search_layer` processes an expanded node's neighbor list in three passes: (1) prefetch the
visited marks of all neighbors; (2) test-and-set them, collect the unvisited ones, and prefetch
every cache line of each unvisited neighbor's vector; (3) compute the distances. It also
prefetches the neighbor block of the next candidate (the new heap top) as soon as the current
one is popped. The misses still happen (LLC misses per query: 13.3k vs 14.1k) but they overlap
instead of queueing: cycles per query −20%, IPC 0.25 → 0.41.

Before/after, same build and graph, `prefetch` toggled at runtime:

| | prefetch off | prefetch on | change |
|---|---:|---:|---:|
| SIFT1M, 1 thread, ef 64 (recall 0.9635): QPS | 5,086 | 6,488 | +28% |
| SIFT1M: p50 / p99 latency (ms) | 0.193 / 0.308 | 0.158 / 0.234 | −18% / −24% |
| GloVe-100, 1 thread, ef 1024 (recall 0.955): QPS | 436 | 530 | +22% |
| GloVe-100: p50 / p99 latency (ms) | 2.35 / 3.18 | 1.99 / 2.79 | −15% / −12% |
| SIFT1M, 16 threads, ef 64: QPS | 32,174 | 35,149 | +9% |

A first, simpler version (prefetch only the *next* neighbor's vector while computing the current
one, the same strategy as hnswlib's search loop) gained only ~5% in an earlier Eco-mode run
(4,338 → 4,555 QPS, `bench/ann/results/profile_v1_next_prefetch.md`): one distance computation
(~20 ns) is too short to hide a ~100 ns DRAM miss. Prefetching a whole neighbor list at once
gives the memory system 10–30 independent misses to work on in parallel.

### Analysis: where vecsearch wins and loses

- **On one thread, vecsearch is the fastest on both datasets, but on SIFT1M only narrowly.**
  At recall ≈ 0.95: SIFT1M 6,488 QPS vs hnswlib 6,293 (+3%, inside run-to-run noise) and Faiss
  5,388 (+20%); GloVe 530 vs 490 (+8%) and 421 (+26%). Tail latency is lowest on both (SIFT p99
  0.23 ms vs 0.28 / 0.34; GloVe 2.79 ms vs 2.89 / 3.76). The graphs are nearly identical (same
  algorithm and parameters; recall at equal ef matches hnswlib to the third decimal), so the
  difference is the search loop. Without prefetching, vecsearch is clearly slower than hnswlib
  (5,086 vs 6,293 on SIFT): the batched prefetch is what makes it competitive.
- **The SIFT lead over hnswlib shrank compared with the earlier Eco-mode run** (+12% there,
  +3% here). In Ultimate mode hnswlib's single-thread QPS rose 21% (5,188 → 6,293) and
  vecsearch's 11% (5,834 → 6,488). Not investigated; a plausible reason is that higher clocks
  help hnswlib's more compute-bound loop more than they help a loop already limited by memory
  parallelism. Only single runs exist per mode, so treat a few percent either way as noise.
- **With 16 threads on SIFT1M, Faiss is the fastest.** vecsearch reaches 35,149 QPS, 83% of
  Faiss (42,215), and is level with hnswlib (34,255, +3%). On GloVe at ef = 1024, vecsearch is
  fastest (3,099 vs 2,680 / 2,916). From 1 to 16 threads on SIFT, vecsearch scales 5.4×,
  hnswlib 5.4×, Faiss 7.8×. All scale sub-linearly: 16 threads share 8 physical cores (SMT) and
  one memory system, and HNSW search is memory-latency bound. **Why Faiss scales better is not
  established by these measurements** (no multi-threaded profile was taken). Hypotheses to test
  next: Faiss computes neighbor distances four at a time, which may suit two SMT threads sharing
  a core better than long prefetch bursts; and vecsearch's bursts of prefetches from two
  hyperthreads may oversubscribe the core's shared miss buffers (which would also explain why
  prefetching gains +28% on 1 thread but only +9% on 16). A `perf stat` of the 16-thread run and
  a run with 8 threads (one per core) would separate these.
- **Faiss has slightly higher recall at equal ef** (SIFT, ef 64: 0.968 vs 0.964), and reaches
  0.99 at a lower ef, so its single-thread QPS at 0.99 is the highest (2,951 vs 2,609 / 2,483).
- **Build time:** within a few percent for vecsearch and Faiss (SIFT 63.4 s vs 64.7; GloVe
  81.3 vs 80.2), hnswlib slower (67.5 / 93.4 s). Construction runs the same `search_layer`, so
  it benefits from prefetching on SIFT (the no-prefetch build takes 74.1 s).
- **Memory:** Faiss uses the least (SIFT: 679 MiB vs 726 for vecsearch and 753 for hnswlib). Of
  vecsearch's 726 MiB, 488 MiB are the vectors (1M × 128 floats) and 126 MiB the layer-0 graph
  (1M × 33 uint32); the rest is labels (8 bytes per node), the label hash map, and per-node
  bookkeeping. Faiss stores no external labels; vecsearch keeps int64 labels and a hash map to
  support upserts and deletes by label.

## Phase 6: Service load test

Setup: `docker compose up --build` (the multi-stage image from `service/Dockerfile`, one Uvicorn
worker), then from the dev container on the same Docker network:

```bash
python3 service/loadtest/run_loadtest.py --url http://vecsearch:8000 --users 1,4,16,64 --duration 30s
```

It loads the SIFT 100k subset over HTTP (20 batches of 5,000 upserts, with a tag per vector),
then runs Locust (`service/loadtest/locustfile.py`, `FastHttpUser`, no think time, so N users =
N requests in flight) for 30 s per level. Each request is a k = 10, ef = 64 search with a random
SIFT query vector as JSON. Raw output: `service/loadtest/results.md`.

Loading 100,000 vectors over HTTP took 12.7 s (~7,900 vectors/s, including JSON parsing and
Pydantic validation of 12.8M floats).

| concurrent users | req/s | failures | client p50 (ms) | client p99 (ms) | handler p50 / p99 (ms) | engine p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 663 | 0 | 1 | 2 | 0.23 / 0.49 | 0.16 / 0.37 |
| 4 | 890 | 0 | 4 | 7 | 0.53 / 1.72 | 0.45 / 1.62 |
| 16 | 994 | 0 | 15 | 29 | 1.11 / 3.26 | 1.02 / 3.14 |
| 64 | 910 | 0 | 68 | 110 | 1.02 / 4.39 | 0.92 / 4.27 |

Columns: *client* = Locust's end-to-end latency (Locust reports whole milliseconds, so these are
coarse); *handler* = time inside the FastAPI handler (`X-Handler-Time-Ms`); *engine* = the C++
search call (`X-Engine-Time-Ms`).

**Where the time goes.** With one user, a request takes ~1 ms end to end, of which the engine
is ~0.16 ms (16%) and the whole handler ~0.23 ms. The other ~0.75 ms is outside the handler:
HTTP parsing, JSON decoding of 128 floats, Pydantic validation, response serialization, the
event-loop/threadpool hand-off, and the Docker network hop. Throughput saturates at about
**1,000 req/s** from 16 users on, while the engine alone does ~5,800 QPS per thread on the much
larger SIFT1M (Phase 4). Past saturation, extra users only queue (p50 68 ms at 64 users).

**Why it saturates:** everything except the C++ search holds Python's GIL, and there is a single
Uvicorn worker process, so the Python request path runs on effectively one core. The engine
times also grow under load (0.16 → ~1 ms) even though the C++ search itself does not slow down
that much: the measured interval ends when the pybind call returns, which requires re-acquiring
the GIL, so under contention it includes waiting for the GIL. Locust runs on the same machine
and takes CPU too.

What would raise throughput (not done; outside this phase's scope): several worker processes
(each would need its own copy of the index, or the index in shared memory), batching
concurrent queries into one engine call, a binary request format instead of JSON float arrays,
or a non-Python front end.

### Filtered search: recall and cost vs. filter selectivity

Command: `python3 bench/ann/filter_recall.py --ef 64` (and `--ef 256`). SIFT 100k subset,
1,000 queries, 1 thread; a random fraction of ids is allowed; ground truth = exact top-10 among
the allowed vectors. Raw output: `bench/ann/results/filter_recall.md`.

| selectivity | allowed vectors | recall@10 (ef 64) | filtered HNSW QPS (ef 64) | brute force over allowed QPS |
|---:|---:|---:|---:|---:|
| 100% (no filter) | 100,000 | 0.9836 | 8,481 | 470 |
| 50% | 50,000 | 0.9934 | 5,177 | 1,123 |
| 10% | 10,000 | 0.9995 | 1,519 | 8,051 |
| 1% | 1,000 | 1.0000 | 254 | 57,521 |
| 0.1% | 100 | 1.0000 | 43 | 421,321 |

Recall does not drop with these (random, uncorrelated) filters; it rises, because the search
keeps expanding until it has ef = 64 *allowed* results and so explores far more of the graph.
The cost is latency: 33× fewer QPS at 1% selectivity. Brute force over the allowed set wins
somewhere between 50% and 10% selectivity, so a real system should switch strategies per query
(see DESIGN.md, Filtering). Correlated filters were not measured.

## Resume bullets (numbers from this file)

- Implemented an HNSW approximate nearest neighbor index from scratch in C++20 with
  hand-written AVX2/NEON distance kernels and multithreaded search, reaching 6,488 QPS on one
  thread at 0.96 recall@10 on SIFT1M (on par with hnswlib, 20% above Faiss) and the lowest p99
  latency of the three (0.23 ms).
- Profiled the search hot path with `perf` (IPC 0.25, ~13k LLC misses per query), identified
  serialized DRAM misses on vector and visited-list loads, and added batched software
  prefetching, improving single-thread throughput by 28% and cutting p99 latency by 24%.
- Exposed the engine through zero-copy pybind11 bindings (GIL released) and a Dockerized
  FastAPI service with metadata-filtered search and strict input validation, sustaining ~990
  req/s with p99 29 ms at 16 concurrent clients (Eco-mode measurement).
