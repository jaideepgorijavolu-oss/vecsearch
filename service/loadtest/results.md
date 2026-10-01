service: {'status': 'ok', 'collections': 0, 'kernels': 'avx2'}
loaded 100000 vectors over HTTP in 12.7 s
users 1: 663 req/s, p50 1 ms, p99 2 ms
users 4: 890 req/s, p50 4 ms, p99 7 ms
users 16: 994 req/s, p50 15 ms, p99 29 ms
users 64: 910 req/s, p50 68 ms, p99 110 ms

SIFT 100k subset, k = 10, ef = 64, 30s per level

| concurrent users | req/s | failures | client p50 (ms) | client p99 (ms) | handler p50 / p99 (ms) | engine p50 / p99 (ms) | engine share of client p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 663 | 0 | 1 | 2 | 0.23 / 0.49 | 0.16 / 0.37 | 16% |
| 4 | 890 | 0 | 4 | 7 | 0.53 / 1.72 | 0.45 / 1.62 | 11% |
| 16 | 994 | 0 | 15 | 29 | 1.11 / 3.26 | 1.02 / 3.14 | 7% |
| 64 | 910 | 0 | 68 | 110 | 1.02 / 4.39 | 0.92 / 4.27 | 1% |
