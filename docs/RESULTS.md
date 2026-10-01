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

Raw output: `bench/micro/results/distance.json`. Medians of 3 repetitions.

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
| 32 | 13.4 (19.1) | 11.8 (21.7) | 2.9 (89.5) | 3.2 (81.3) |
| 64 | 34.1 (15.0) | 29.8 (17.2) | 4.1 (126.1) | 5.0 (102.4) |
| 128 | 72.9 (14.0) | 75.1 (13.6) | 7.5 (137.2) | 6.9 (149.4) |
| 384 | 303.5 (10.2) | 311.7 (9.9) | 19.8 (155.3) | 16.8 (182.7) |
| 768 | 647.4 (9.5) | 652.8 (9.4) | 43.1 (142.7) | 32.5 (189.2) |
| 1536 | 1378.7 (9.0) | 1351.5 (9.1) | 83.8 (146.7) | 64.8 (189.8) |

### hot / dot: ns per call (GB/s)

| dim | scalar | autovec_O3_native | autovec_O3_native_fastmath | avx2 |
|---:|---:|---:|---:|---:|
| 32 | 12.3 (20.8) | 10.9 (23.6) | 2.8 (92.4) | 2.9 (89.4) |
| 64 | 32.7 (15.7) | 29.2 (17.5) | 4.4 (117.8) | 3.8 (134.5) |
| 128 | 72.7 (14.1) | 71.6 (14.3) | 6.7 (152.5) | 5.5 (185.1) |
| 384 | 299.2 (10.3) | 296.3 (10.4) | 19.1 (160.5) | 16.3 (189.0) |
| 768 | 691.1 (8.9) | 650.1 (9.5) | 41.3 (150.1) | 31.1 (197.7) |
| 1536 | 1334.2 (9.2) | 1340.2 (9.2) | 84.0 (146.3) | 62.5 (199.0) |

### scan / l2: ns per call (GB/s)

| dim | scalar | autovec_O3_native | autovec_O3_native_fastmath | avx2 |
|---:|---:|---:|---:|---:|
| 32 | 14.7 (8.7) | 12.4 (10.3) | 4.8 (26.8) | 4.8 (26.6) |
| 64 | 36.3 (7.1) | 31.0 (8.3) | 9.4 (27.2) | 9.0 (28.5) |
| 128 | 76.4 (6.7) | 73.8 (6.9) | 19.7 (26.1) | 18.5 (27.6) |
| 384 | 308.3 (5.0) | 309.4 (5.0) | 55.9 (27.8) | 48.4 (31.7) |
| 768 | 661.3 (4.6) | 662.5 (4.6) | 96.5 (31.9) | 108.5 (28.3) |
| 1536 | 1381.3 (4.4) | 1365.1 (4.5) | 228.8 (27.3) | 224.2 (27.4) |

### scan / dot: ns per call (GB/s)

| dim | scalar | autovec_O3_native | autovec_O3_native_fastmath | avx2 |
|---:|---:|---:|---:|---:|
| 32 | 13.5 (9.5) | 11.4 (11.2) | 4.1 (31.4) | 4.1 (31.4) |
| 64 | 35.0 (7.3) | 30.5 (8.4) | 8.2 (31.2) | 7.7 (33.6) |
| 128 | 76.9 (6.7) | 74.5 (6.9) | 19.7 (26.0) | 16.6 (30.8) |
| 384 | 303.6 (5.1) | 312.2 (4.9) | 64.8 (23.7) | 60.8 (25.3) |
| 768 | 649.5 (4.7) | 667.0 (4.6) | 121.4 (25.3) | 115.1 (26.7) |
| 1536 | 1357.0 (4.5) | 1418.7 (4.3) | 190.4 (32.3) | 192.1 (32.0) |

### Discussion

- **`-O3 -march=native` alone does nothing.** The plain loop stays scalar (`vaddss` in the
  disassembly) and runs at the scalar speed. Floating-point addition is not associative, so
  without permission to reorder the sum, the compiler cannot split it across vector lanes.
- **With `-ffast-math` the compiler gets close.** It vectorizes with 512-bit AVX-512 registers
  and is 8–35% slower than the hand kernel for dim ≥ 128. It is slightly faster for L2 at
  dim 32 and 64, where the hand kernel's main loop runs only once or twice and its fixed costs
  (zeroing four accumulators, the horizontal sum) dominate. The disassembly shows why it is not faster despite 2× wider registers: its main loop
  has a *single* accumulator (`vmulps` + `vaddps` into one `zmm`), so every iteration waits for
  the previous add. The hand kernel uses four independent FMA accumulators. `-ffast-math` is
  not an option for the library anyway: it changes NaN/infinity semantics for the whole
  translation unit.
- **The hand AVX2 kernel is ~20× faster than scalar** when data is in L1 (e.g. dim 768:
  647 → 32.5 ns for L2), reaching ~190 GB/s of L1 load bandwidth.
- **Once data comes from DRAM, every vectorized kernel hits the same wall**: ~26–32 GB/s
  single-thread streaming bandwidth, for all dims. In the scan test the AVX2 kernel and the
  fast-math auto-vectorized kernel are equal within noise. The scalar kernel is still compute
  bound (4–9 GB/s), so SIMD helps 3–7× there, not 20×.
- **Implication for HNSW**: graph search touches vectors in an unpredictable order, so each
  distance computation is likely a cache miss (latency bound, worse than streaming). SIMD makes
  the arithmetic nearly free; the remaining cost is memory access. Phase 4 measures this.

## Phase 3: HNSW correctness

TBD

## Phase 4: ANN benchmarks

TBD

## Phase 6: Service load test

TBD
