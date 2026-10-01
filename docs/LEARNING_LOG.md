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
