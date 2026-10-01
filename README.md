# vecsearch

A from-scratch approximate nearest neighbor (ANN) search engine in C++20: an HNSW graph index
with hand-written AVX2/NEON distance kernels and runtime CPU dispatch, multithreaded build and
search, soft deletes and filtered search, zero-copy Python bindings, and a small FastAPI
service. It is benchmarked against hnswlib and Faiss on SIFT1M and GloVe-100 with
recall-vs-throughput curves.

![SIFT1M recall vs QPS](bench/ann/results/sift_recall_qps.png)

Headline numbers (SIFT1M, M = 16, ef_construction = 200, recall@10 ≈ 0.96, laptop Ryzen 9 270
in Eco power mode; full methodology and caveats in [docs/RESULTS.md](docs/RESULTS.md)):

| | vecsearch | hnswlib | Faiss HNSW |
|---|---:|---:|---:|
| QPS, 1 thread | **5,834** | 5,188 | 4,740 |
| QPS, 16 threads | 35,437 | 32,252 | **39,998** |
| p99 latency (ms) | **0.32** | 0.45 | 0.39 |
| build (s) | **64.5** | 75.0 | 75.4 |

vecsearch is fastest on one thread on both SIFT1M and GloVe-100, thanks to a batched
software-prefetching search loop found by profiling (+21% QPS). Faiss scales better across 16
threads; RESULTS.md discusses why that is not yet explained.

## Quickstart

### Python

```bash
pip install .
```

```python
import numpy as np
import vecsearch

rng = np.random.default_rng(0)
data = rng.standard_normal((10_000, 128)).astype(np.float32)

index = vecsearch.HNSWIndex(dim=128, metric="l2", M=16, ef_construction=200)
index.add(data)                                   # ids 0..9999 (or pass ids=...)
ids, distances = index.search(data[:5], k=10, ef=64)
print(ids[:, 0])                                  # each vector finds itself: [0 1 2 3 4]
```

### C++

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
```

```cpp
#include "vecsearch/hnsw_index.hpp"

vecsearch::HnswIndex index(128, vecsearch::Metric::L2, {.M = 16, .ef_construction = 200});
index.add(data, n);                                       // row-major n x 128 floats
vecsearch::SearchResult r = index.search(queries, nq, /*k=*/10, /*ef=*/64);
// r.ids[q * 10 + j], r.distances[q * 10 + j]
```

### Docker

TBD (Phase 6).

## Development

The spec targets GCC/Clang on Linux/macOS. On Windows everything runs in a Docker image:

```bash
docker build -t vecsearch-dev -f docker/dev.Dockerfile docker
tools/dev.sh "cmake --preset release && cmake --build --preset release && ctest --preset release"
```

Presets: `release`, `debug`, `sanitizer` (ASan + UBSan), `tsan`, `bench` (adds Google
Benchmark and the ANN tools).

## Docs

- [docs/DESIGN.md](docs/DESIGN.md): architecture, memory layout, SIMD dispatch, HNSW,
  concurrency, deletion and filtering.
- [docs/RESULTS.md](docs/RESULTS.md): hardware, methodology, all benchmark numbers.
- [docs/LEARNING_LOG.md](docs/LEARNING_LOG.md): per-phase notes and interview questions.
