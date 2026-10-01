# Learning log

One entry per phase: the key ideas, the design decisions, and questions an interviewer might ask.

## Phase 0: Scaffold and CI

**What was built**

- A CMake project (C++20, `-Wall -Wextra -Wpedantic`) with presets for `release`, `debug`,
  `sanitizer` (AddressSanitizer + UndefinedBehaviorSanitizer) and `tsan` (ThreadSanitizer).
- GoogleTest is pulled in with `FetchContent`, so a fresh clone needs nothing but CMake and a
  compiler. `gtest_discover_tests` registers each `TEST` with CTest individually.
- `.clang-format` (Google style, 100 columns) and a CI job that fails on unformatted code.
- GitHub Actions: release build + tests on Ubuntu x86-64 and macOS ARM (so both the AVX2 and NEON
  code paths get exercised later), plus ASan/UBSan and TSan jobs on Ubuntu.
- `tools/dev.sh` runs any command in a Docker image (`docker/dev.Dockerfile`, Ubuntu 24.04 with
  GCC 13, Clang 18, CMake, Python, hnswlib, Faiss). The dev machine is Windows, and the spec targets
  GCC/Clang, so all building, testing and benchmarking happens in that Linux container.

**Design decisions**

- *Custom build types instead of ad-hoc flags.* `CMAKE_BUILD_TYPE=Sanitizer` makes "run the tests
  under ASan" a single preset, the same command locally and in CI.
- *ASan and TSan are separate builds.* They both replace the allocator and shadow memory and
  cannot be linked into the same binary.
- *`-O1` for sanitizer builds.* `-O0` is very slow under ASan; `-O1` keeps stack traces usable.
- *No `-march=native` by default.* A library binary must run on any x86-64 CPU. SIMD code is
  enabled per file and chosen at runtime (Phase 2). `VECSEARCH_NATIVE=ON` exists only for the
  "what does the auto-vectorizer do with native flags" benchmark.

**Interview questions**

1. Why can't you combine AddressSanitizer and ThreadSanitizer in one build?
2. What does UBSan catch that ASan does not? Give an example (signed overflow, misaligned load,
   invalid shift).
3. Why fetch GoogleTest with `FetchContent` rather than requiring a system install? What are the
   downsides (build time, network access in CI)?
4. Why do you test on both x86-64 and ARM runners?
5. What does `CMAKE_POSITION_INDEPENDENT_CODE` do, and why will the Python extension need it?

## Phase 1: Storage and exact search

**What was built**

- `AlignedAllocator<T, 64>` and `AlignedVector<T>`: `std::vector` storage from aligned
  `operator new`. Vectors are stored contiguously, one row per vector, with the row length
  padded to a multiple of 16 floats (`padded_dim`), so every vector starts on a 64-byte
  cache-line boundary. The padding is zero.
- Metrics: squared L2, inner product (distance `1 - <a,b>`, so smaller is always better), and
  cosine (vectors and queries normalized to unit length, then inner product).
- Scalar `l2_sq` and `dot` kernels, written as plain loops with one accumulator.
- `TopK<Id>`: a bounded max-heap of size k. The root is the worst kept result, so rejecting a
  candidate costs one comparison. Ties are broken by id so results are deterministic.
- `FlatIndex`: `add` appends rows, `search` scans every row per query, in parallel across
  queries with `parallel_for` (dynamic chunked scheduling over an atomic counter).
- Binary save/load with an 8-byte magic string and a format version.
- A NumPy fixture (`tools/make_flat_fixture.py`): random data plus NumPy's float64 brute-force
  top-10 ids for all three metrics. The C++ test requires identical ids.

**Design decisions**

- *Why contiguous storage?* A brute-force scan reads memory sequentially, which the hardware
  prefetcher handles perfectly. `std::vector<std::vector<float>>` would put each vector in its
  own heap allocation: a pointer chase per vector and no prefetching.
- *Why 64-byte alignment?* A misaligned 128-float (512 B) vector touches 9 cache lines instead
  of 8, and SIMD loads that cross a cache line are slower. Alignment also lets kernels use
  aligned loads if they want to (ours use unaligned loads, which cost the same on aligned data
  on modern x86 and are safe for user-provided query pointers).
- *Why `1 - dot` for inner product?* One "smaller is closer" convention lets every index and
  heap be metric-agnostic. It is the same convention as hnswlib.
- *Why normalize cosine at insert?* Then cosine is just a dot product: no per-comparison norm
  computation, and the same SIMD kernel serves both metrics.
- *Why a max-heap for top-k?* O(n log k) instead of O(n log n) for a full sort, and O(k) memory.
- *Ground truth as a fixture.* The C++ tests have no Python dependency at test time, but the
  expected ids really come from NumPy (computed in float64, stable sort for ties).

**Interview questions**

1. Why is a max-heap (not a min-heap) the right structure for keeping the k *smallest* items?
2. What is the cost of a 64-byte-aligned allocation vs. a normal one, and when does alignment
   actually matter for performance?
3. Why does `-O3` not vectorize the scalar `l2_sq` loop, and what flag would let it?
4. Cosine similarity and inner product give the same ranking when? What breaks if a user adds a
   zero vector?
5. Your parallel brute-force search writes into one shared result array from many threads.
   Why is that not a data race?

## Phase 2: SIMD kernels

**What was built**

- `src/distance/avx2.cpp`: AVX2 + FMA `l2_sq` and `dot`. Main loop of 32 floats with four
  independent 8-lane accumulators, then an 8-wide loop, then a scalar tail; horizontal sum at
  the end. Only this file is compiled with `-mavx2 -mfma`.
- `src/distance/neon.cpp`: the same structure with 4-lane NEON registers (`vfmaq_f32`,
  `vaddvq_f32`), built on ARM (tested by the macOS CI runner).
- `src/distance/dispatch.cpp`: `active_kernels()` runs `__builtin_cpu_supports("avx2")` and
  `("fma")` once (a function-local static, so initialization is thread-safe) and returns a
  `Kernels` struct of function pointers. `VECSEARCH_KERNELS=scalar` forces the fallback.
- Tests: every supported kernel against scalar for every dim 1..200 plus 384/768/1536/1537,
  from deliberately misaligned pointers, within 1e-4 relative error.
- `bench/micro/bench_distance`: Google Benchmark, scalar vs `-O3 -march=native` vs
  `-O3 -march=native -ffast-math` vs hand SIMD, hot (L1) and scan (DRAM) scenarios.

**Design decisions**

- *Per-file target flags, not `-mavx2` globally.* If the whole library were built with AVX2,
  the compiler could emit AVX2 instructions anywhere (including in code that runs before
  dispatch) and the binary would crash with SIGILL on a CPU without AVX2.
- *Function pointers chosen once.* The feature check costs a few instructions but is done at
  most once. The indirect call per distance costs ~1 ns and is predicted perfectly (always the
  same target). The alternative, templating the whole index on the kernel, removes the
  indirect call but multiplies compile time and code size.
- *Four accumulators.* An FMA has a latency of ~4 cycles but two can start per cycle, so a
  single accumulator chain uses 1/8 of the available FMA throughput. Four chains is a common
  sweet spot that also keeps register pressure low.
- *Scalar tail instead of masked loads.* Simple and obviously correct; for dims like 128 the
  tail never runs. AVX2 masked loads (`_mm256_maskload_ps`) are an option for odd dims.
- *Unaligned loads (`loadu`).* Queries come from users (NumPy arrays) and may not be aligned;
  on modern x86 `loadu` on aligned data is as fast as `load`.

**What the measurements showed** (see RESULTS.md): the compiler does not vectorize the plain
loop at all without `-ffast-math`; with it, it gets within 8–35% using a single accumulator.
Once vectors come from DRAM, all vectorized kernels run at the same ~30 GB/s: the kernel is
memory-bound, not compute-bound.

**Interview questions**

1. Why can't the compiler vectorize `sum += a[i] * b[i]` without `-ffast-math`? What exactly
   does `-fassociative-math` allow?
2. Why does the hand kernel use four accumulators? How would you pick the number?
3. How does runtime dispatch avoid SIGILL on old CPUs? What has to be true about the files
   that are *not* compiled with `-mavx2`?
4. At what point does a distance kernel become memory-bandwidth bound, and how did you show
   it? What would AVX-512 change in that regime?
5. Your SIMD result differs from scalar in the last bits. Why, and why is 1e-4 relative error
   an acceptable tolerance for nearest-neighbor search?

## Phase 3: HNSW index

**What was built** (`include/vecsearch/hnsw_index.hpp`, `src/index/hnsw_index.cpp`)

- HNSW following Malkov & Yashunin (2018): random levels with `mL = 1/ln(M)`, greedy descent
  through the upper layers, beam search (`search_layer`, Algorithm 2) on the target layer,
  neighbor selection heuristic (Algorithm 4) when linking and when pruning a full neighbor
  list. `M0 = 2M` links on layer 0.
- Compact storage: fixed-size layer-0 neighbor blocks in one array (`[count, ids...]`), small
  per-node arrays for the few nodes on upper layers.
- `VisitedList` with an epoch counter (no per-query clearing).
- Parallel batch search (per-thread `Scratch` buffers from a pool) and parallel construction
  (per-node 1-byte spinlocks, global lock only for the entry point).
- Soft deletion (tombstones), upsert by label, filtered search via an `accept` predicate,
  save/load with a versioned header and validation.
- Tools: `hnsw_eval` (recall/QPS sweep with exact ground truth) and `parallel_build_quality`.

**Design decisions**

- *Two heaps in search_layer.* Candidates (min-heap: what to expand next) and results
  (max-heap: the worst kept result is the stopping bound). Stop when the closest candidate is
  worse than the worst result: nothing left can improve the beam.
- *Why the heuristic matters.* "Closest M" on clustered data links every node only inside its
  own cluster, and the graph falls apart into islands. The heuristic skips a candidate that is
  closer to an already-chosen neighbor than to the base node, so it spends links on new
  directions. Measured: recall 0.28 → 0.61 on clustered data at the same ef.
- *Tombstones and filters share one mechanism*: rejected nodes are expanded (they keep the
  graph connected) but never enter the results heap.
- *Copy neighbors under the lock, compute outside it.* Keeps lock hold times tiny.
- *Never hold two node locks.* Deadlock-free by construction, no lock ordering needed.
- *Sequential prefix.* Found by measuring: with 16 threads, nodes inserted while the graph was
  tiny ended up badly linked. Inserting the first 1,000 nodes on one thread fixes most of it at
  no measurable cost.
- *Levels drawn before the parallel phase.* Deterministic for a given seed, and a node's level
  and storage never change while other threads might read them.

**Interview questions**

1. Walk through `search_layer`. Why two heaps, and what is the exact stopping condition? What
   changes when some nodes are filtered out?
2. Why `mL = 1/ln(M)`? What is the expected number of layers for n = 1M and M = 16?
3. Explain Algorithm 4 with a picture. When does it help, and when is it no better than
   closest-M?
4. How does parallel construction avoid deadlocks and data races? What would break if you
   read a neighbor list without the lock during construction?
5. Why does deleting nodes by tombstone hurt search over time, and what would you do about it?

## Phase 5: Python package (done before Phase 4)

*Order change:* the Phase 4 benchmark harness drives this engine, hnswlib and Faiss from the
same Python script, so that all three are timed the same way (batch calls from NumPy into C++).
That needs the bindings, so Phase 5 was done first.

**What was built**

- `python/bindings.cpp` (pybind11 module `vecsearch._core`): `FlatIndex` and `HNSWIndex` with
  `add`, `search`, `delete`, `save`, `load`, `ef_search` / `prefetch` setters, `len()`, `in`.
- Zero-copy input: arguments are taken as `py::handle`, checked to be `numpy.ndarray`, dtype
  float32, C-contiguous, 1-D or 2-D with the right dimension, and then the C++ code reads the
  NumPy buffer directly. Wrong inputs raise `TypeError` / `ValueError` that say how to fix them
  (e.g. "use arr.astype(np.float32)"). Ids and filters (small) are converted to int64.
- Zero-copy output: the `SearchResult` is moved to the heap and the two NumPy result arrays
  point into it, owned by a `py::capsule` that frees it when both arrays are gone.
- The GIL is released (`py::gil_scoped_release`) during add, search and save.
- Packaging: `pyproject.toml` with scikit-build-core; `pip install .` builds the extension with
  CMake. pytest suite (26 tests): exact match with NumPy (Flat), recall vs NumPy (HNSW), k > n,
  empty index, wrong dtype / dim / shape / contiguity, ids mismatch, delete, filter,
  save/load, concurrent searches from Python threads, results outliving the index.
- `tools/run_readme_example.py` runs the README's Python block, in CI too.

**Design decisions**

- *Refuse rather than convert.* `py::array_t<float, forcecast>` would silently copy a float64
  array: a 4 GB copy for a large dataset, and "zero copy" would be false. An error is better.
- *Why releasing the GIL matters:* without it, a web server's worker threads would serialize
  on every search even though the C++ search is thread-safe.
- *Lifetime while the GIL is released*: the `Matrix` struct holds a reference to the array, so
  Python cannot free the buffer while C++ is reading it.

**Interview questions**

1. What is the buffer protocol, and what exactly does "zero copy" mean for input and output
   here?
2. What can go wrong if you release the GIL and the NumPy array is freed or resized by
   another Python thread?
3. Why reject float64 input instead of converting it?
4. How does the capsule keep the result memory alive, and when is it freed?
5. How does scikit-build-core turn a CMake project into a wheel?

## Phase 4: Benchmarks and profiling

**What was built**

- `bench/ann/run.py` + `engines.py`: one harness for vecsearch, hnswlib and Faiss with identical
  parameters (M = 16, ef_construction = 200, 16 threads), one process per (dataset, engine) so
  RSS measurements do not mix. ef sweep → recall@10, QPS (16 threads and 1 thread), build time,
  memory, p50/p99 single-query latency at recall ≈ 0.95.
- `bench/ann/plot.py`: recall-vs-QPS plots and the tables in RESULTS.md.
- `bench/ann/profile_search.cpp` + `profile.py`: a single-threaded search loop under
  `perf stat`, with a load-only baseline subtracted, giving counters *per query*.
- `make bench` / `bench/run_all.sh` regenerates every Phase 2 and Phase 4 number.

**What the profile showed and what changed**

- IPC 0.3 and ~13.5k LLC misses per query: the search is DRAM-latency bound. `perf annotate`
  put the stalls on the first load of each vector and on the visited-list load.
- Fix: process a neighbor list in passes (prefetch all visited marks; collect unvisited and
  prefetch their vectors; then compute), plus prefetch the next candidate's neighbor block. The
  number of misses is the same, but they overlap: cycles/query −24%, QPS +21% (SIFT, 1 thread).
- Prefetching only the *next* neighbor (hnswlib's approach) gained just ~5%: one distance
  computation (~20 ns) cannot hide a ~100 ns miss.

**Honest results:** fastest single-thread on both datasets, fastest build; Faiss is ~11% ahead
with 16 threads on SIFT and the cause is not yet measured. Laptop in Eco mode; one run per
configuration with ~±5–10% noise.

**Interview questions**

1. What does IPC 0.3 tell you, and which counters would you look at next?
2. Explain memory-level parallelism. Why does prefetching a whole neighbor list beat
   prefetching one neighbor ahead?
3. The prefetch didn't reduce cache misses. How can it still make the search 21% faster?
4. Why does the gain shrink from +21% on 1 thread to +8% on 16 threads? How would you test
   your hypothesis?
5. How do you make sure a benchmark against hnswlib/Faiss is fair (parameters, threads, recall
   at equal ef vs equal QPS, process isolation, noise)?

## Phase 6: Thin service layer

**What was built**

- `service/app`: FastAPI with `POST /collections`, batch upsert, search, delete, `/health`.
  Pydantic models enforce static bounds (k in 1..1000, dim, name pattern, batch size); the
  dimension check needs the collection, so the handler does it and returns 422.
- `RWLock` (writer-preferring) per collection: many concurrent searches (the C++ search
  releases the GIL), or one writer.
- Tags: `{key: int|str}` per vector, an inverted index `(key, value) -> ids`, filters are AND of
  equalities; the matching ids go to the C++ search as `filter`, so filtering happens during
  graph traversal.
- `service/Dockerfile` (multi-stage: compiler stage builds the wheel, slim runtime stage,
  non-root user, healthcheck), `docker-compose.yml`.
- Tests with FastAPI's `TestClient` (10 tests: correctness vs brute force, filters, upsert of
  tags, delete, validation errors, concurrent reads + writes, the RW lock itself).
- Locust load test with server-side timing headers to split latency into engine vs everything
  else; a filter-selectivity benchmark.

**What the measurements showed:** the service saturates at ~1,000 req/s on one Uvicorn worker
because everything except the engine holds the GIL; at one user, the engine is ~16% of the
latency. Restrictive filters do not hurt recall in this design (the search keeps going until
it has ef allowed results), they hurt latency: below ~10% selectivity, brute force over the
allowed ids is faster.

**Interview questions**

1. Why a reader-writer lock and not a plain mutex? Why writer-preferring?
2. Your "engine time" grows from 0.16 ms to 1 ms under load. Is the C++ code slower? (GIL
   re-acquisition is inside the measured interval.)
3. How would you get past 1,000 req/s? Compare multiple processes, query batching and a
   different wire format.
4. Filters: why traverse filtered-out nodes instead of skipping them? When should a query
   planner switch to brute force?
5. What does each Dockerfile stage contain, and why is the compiler not in the final image?
