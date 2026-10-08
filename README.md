# vecsearch

[![CI](https://github.com/jaideepgorijavolu-oss/vecsearch/actions/workflows/ci.yml/badge.svg)](https://github.com/jaideepgorijavolu-oss/vecsearch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A from-scratch approximate nearest neighbor (ANN) search engine in C++20: an HNSW graph index
with hand-written AVX2/NEON distance kernels and runtime CPU dispatch, multithreaded build and
search, soft deletes and filtered search, zero-copy Python bindings, and a small FastAPI
service. It is benchmarked against hnswlib and Faiss on SIFT1M and GloVe-100 with
recall-vs-throughput curves.

![SIFT1M recall vs QPS](bench/ann/results/sift_recall_qps.png)

Headline numbers (SIFT1M, M = 16, ef_construction = 200, recall@10 ≈ 0.96, laptop Ryzen 9 270,
Ultimate Performance power plan; full methodology and caveats in [docs/RESULTS.md](docs/RESULTS.md)):

| | vecsearch | hnswlib | Faiss HNSW |
|---|---:|---:|---:|
| QPS, 1 thread | **6,488** | 6,293 | 5,388 |
| QPS, 16 threads | 35,149 | 34,255 | **42,215** |
| p99 latency (ms) | **0.23** | 0.28 | 0.34 |
| build (s) | **63.4** | 67.5 | 64.7 |

vecsearch has the highest single-thread throughput and lowest tail latency on both SIFT1M and
GloVe-100 (narrowly ahead of hnswlib on SIFT), thanks to a batched software-prefetching search
loop found by profiling (+28% QPS). Faiss scales better across 16 threads; RESULTS.md discusses
why that is not yet explained.

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

### Docker (HTTP service)

```bash
docker compose up --build        # serves on http://localhost:8000
```

```bash
curl -X POST localhost:8000/collections -H 'Content-Type: application/json'   -d '{"name": "docs", "dim": 3, "metric": "cosine"}'
curl -X POST localhost:8000/collections/docs/vectors -H 'Content-Type: application/json'   -d '{"vectors": [{"id": 1, "vector": [1, 0, 0], "tags": {"lang": "en"}},
                   {"id": 2, "vector": [0, 1, 0], "tags": {"lang": "fr"}}]}'
curl -X POST localhost:8000/collections/docs/search -H 'Content-Type: application/json'   -d '{"vector": [0.9, 0.1, 0], "k": 2, "filter": {"lang": "en"}}'
curl -X DELETE localhost:8000/collections/docs/vectors/2
```

Endpoints: `POST /collections`, `POST /collections/{name}/vectors` (batch upsert with tags),
`POST /collections/{name}/search` (optional tag filter), `DELETE /collections/{name}/vectors/{id}`,
`GET /health`. Interactive docs at `/docs`.

The service is **in-memory and process-local**: collections live in one Uvicorn process and
are lost on restart (no persistence, no sharing across workers or replicas). Vector values must
be finite with |x| ≤ 1e15 (so float32 distances cannot overflow); ids are non-negative int64.

## Development

The spec targets GCC/Clang on Linux/macOS. On Windows everything runs in a Docker image:

```bash
docker build -t vecsearch-dev -f docker/dev.Dockerfile docker
tools/dev.sh "cmake --preset release && cmake --build --preset release && ctest --preset release"
```

Presets: `release`, `debug`, `sanitizer` (ASan + UBSan), `tsan`, `bench` (adds Google
Benchmark and the ANN tools). In Docker, TSan needs ASLR off:
`DEV_DOCKER_FLAGS="--security-opt seccomp=unconfined" tools/dev.sh "setarch -R bash -c 'cmake --build --preset tsan && ctest --preset tsan'"`.
`make bench` reproduces every benchmark number (about 1.5 hours; perf profiling needs
`DEV_DOCKER_FLAGS=--privileged`).

## License

MIT, see [LICENSE](LICENSE).

## Docs

- [docs/DESIGN.md](docs/DESIGN.md): architecture, memory layout, SIMD dispatch, HNSW,
  concurrency, deletion and filtering.
- [docs/RESULTS.md](docs/RESULTS.md): hardware, methodology, all benchmark numbers.
- [docs/LEARNING_LOG.md](docs/LEARNING_LOG.md): per-phase notes and interview questions.
- [docs/adaptive_search/RESULTS.md](docs/adaptive_search/RESULTS.md): learned early termination
  experiment (SIFT1M: no latency win; GloVe-100: 21% faster than fixed ef at ~0.947 recall).
