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
