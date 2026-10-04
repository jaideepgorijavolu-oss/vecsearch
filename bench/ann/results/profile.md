SIFT1M, M=16, ef_construction=200, ef_search=64, 1 thread, 20 s loop

| prefetch | QPS | cycles/query | instructions/query | IPC | L1d misses/query | cache misses/query (LLC) | branch misses/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 5159 | 924154 | 234959 | 0.25 | 15354 | 13283 | 1906 |
| on | 6569 | 742074 | 307370 | 0.41 | 16337 | 14050 | 2034 |
