SIFT1M, M=16, ef_construction=200, ef_search=64, 1 thread, 20 s loop

| prefetch | QPS | cycles/query | instructions/query | IPC | L1d misses/query | cache misses/query (LLC) | branch misses/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 4780 | 738142 | 233448 | 0.32 | 15491 | 13462 | 1875 |
| on | 4805 | 559941 | 306559 | 0.55 | 16497 | 14095 | 2031 |
