SIFT 100k subset, M=16, ef_construction=200, ef_search=64, k=10, 1000 queries, 1 thread

| selectivity | allowed vectors | recall@10 | filtered HNSW QPS | brute force over allowed QPS |
|---:|---:|---:|---:|---:|
| 1 | 100,000 | 0.9836 | 8,481 | 470 |
| 0.5 | 50,000 | 0.9934 | 5,177 | 1,123 |
| 0.1 | 10,000 | 0.9995 | 1,519 | 8,051 |
| 0.01 | 1,000 | 1.0000 | 254 | 57,521 |
| 0.001 | 100 | 1.0000 | 43 | 421,321 |

Same with ef_search=256:

| selectivity | allowed vectors | recall@10 | filtered HNSW QPS | brute force over allowed QPS |
|---:|---:|---:|---:|---:|
| 1 | 100,000 | 0.9987 | 2,555 | 475 |
| 0.5 | 50,000 | 0.9991 | 1,721 | 1,084 |
| 0.1 | 10,000 | 0.9992 | 522 | 9,092 |
| 0.01 | 1,000 | 0.9998 | 92 | 49,487 |
| 0.001 | 100 | 1.0000 | 30 | 404,959 |
