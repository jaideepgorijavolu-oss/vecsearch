<!-- Output of `python3 bench/ann/profile.py --ef 64 --seconds 20` at the first Phase 4 profiling
     run, when "prefetch on" meant: prefetch the next neighbor's vector while computing the
     current one's distance (the search_layer loop before the batched rewrite). -->
SIFT1M, M=16, ef_construction=200, ef_search=64, 1 thread, 20 s loop

| prefetch | QPS | cycles/query | instructions/query | IPC | L1d misses/query | cache misses/query (LLC) | branch misses/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 4338 | 771458 | 231625 | 0.30 | 17747 | 13863 | 1821 |
| on | 4555 | 723452 | 305328 | 0.42 | 19263 | 15372 | 1863 |
