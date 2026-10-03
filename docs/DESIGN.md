# Design

## Architecture

```
            ┌───────────────────────── service/ (FastAPI, Docker) ─────────────────────────┐
            │  /collections  /vectors  /search      per-collection reader-writer lock       │
            │  tag index (key,value) -> labels      filter -> allowed labels                │
            └──────────────────────────────────────┬────────────────────────────────────────┘
                                                   │ Python calls (GIL released in C++)
            ┌──────────────────────── python/ (pybind11, vecsearch._core) ─────────────────┐
            │  zero-copy NumPy input (C-contiguous float32 checked, never converted)        │
            └──────────────────────────────────────┬────────────────────────────────────────┘
                                                   │
 ┌──────────────────────────────────────── C++ library (libvecsearch) ─────────────────────────────┐
 │  FlatIndex (exact)          HnswIndex (graph)                parallel_for (per-call threads)    │
 │       │                       │   │   │                                                         │
 │       │       visited list ───┘   │   └── per-node SpinLocks, global entry-point mutex          │
 │       ▼                           ▼                                                             │
 │  AlignedVector<float> rows   level-0 blocks + upper-layer blocks                                │
 │       │                           │                                                             │
 │       └────────── Distance{fn, is_ip} ── active_kernels(): scalar | avx2 | neon                 │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Memory layout

**Vectors.** One contiguous `AlignedVector<float>` (64-byte-aligned allocation). Row `i`
starts at `i * stride`, with `stride = dim` rounded up to a multiple of 16 floats, so every row
starts on a cache-line boundary. Padding floats are zero. For SIFT (dim 128) the stride equals
the dim, so there is no waste; for dim 100 (GloVe) rows are 112 floats (12% padding).

**HNSW graph.**

- Layer 0 (every node): one flat `AlignedVector<uint32_t>` of fixed-size blocks,
  `block(i) = [count, n_0 … n_{M0-1}]` with `M0 = 2M`. Fixed size means node `i`'s block is at
  `i * (1 + M0)`, with no pointer to follow: one address computation, then one or two cache
  lines (for M = 16, a block is 132 bytes).
- Upper layers (~1/M of nodes): `upper_[i]` is a per-node `std::vector<uint32_t>` holding
  `level(i)` blocks of `1 + M`. These are touched only during greedy descent (a handful of
  nodes per query), so their scattered allocation does not matter.
- Per node: `labels_` (int64 user id), `levels_` (uint8), `deleted_` (uint8 tombstone),
  one 1-byte `SpinLock`. `label_to_id_` (hash map) holds live labels only.

Vectors and links are stored separately (hnswlib interleaves them in one block per node). This
is analyzed in RESULTS.md, Phase 4.

## SIMD dispatch

`src/distance/avx2.cpp` is the only file compiled with `-mavx2 -mfma`; `neon.cpp` is compiled
on AArch64. `active_kernels()` checks the CPU once (function-local static) and returns a
`Kernels {l2_sq, dot, name}` struct of function pointers. Indexes hold a `Distance {fn, is_ip}`
built from it. `VECSEARCH_KERNELS=scalar` forces the fallback. Kernels use four accumulators to
hide FMA latency, unaligned loads (queries come from user memory), and a scalar tail for dims
that are not a multiple of the vector width.

## HNSW

Parameters: `M` (links per node on upper layers, `M0 = 2M` on layer 0), `ef_construction`
(beam width while building), `ef_search` (beam width while searching, per call or default),
level multiplier `mL = 1/ln(M)`.

- **Level assignment**: `level = floor(-ln(U) * mL)`, `U ~ Uniform(0, 1]`. A node reaches level
  `l` with probability `M^-l`. Levels are drawn sequentially before any insert thread starts,
  so the level sequence is deterministic for a given seed.
- **Insert** (Algorithm 1): greedy descent (beam 1) from the entry point down to `level + 1`,
  then on each layer `min(level, max_level) … 0`: `search_layer` with `ef_construction`,
  `select_neighbors` (heuristic) picks `M`, write the new node's block, add the reverse edge to
  each neighbor; a neighbor that is full re-runs the heuristic over its old links plus the new
  node and keeps at most `Mmax` (M0 on layer 0).
- **search_layer** (Algorithm 2): a min-heap of candidates and a max-heap of the best `ef`
  results; stop when the closest candidate is further than the worst result and the results are
  full. A candidate is expanded only if it could enter the results.
- **Neighbor selection** (Algorithm 4, without the optional `extendCandidates` and
  `keepPrunedConnections`): scan candidates closest-first and keep `c` only if it is closer to
  the base node than to every already-kept neighbor. On clustered data this keeps the
  long-range links between clusters. Test `Hnsw.HeuristicBeatsClosestM`: recall@10 at ef = 20
  on 20k points in 100 tight clusters (dim 8, M = 6): 0.28 with closest-M, 0.61 with the
  heuristic.
- **Visited set**: `VisitedList`, a `uint16_t` per node and an epoch counter. A new query bumps
  the epoch; "visited" means `marks[i] == epoch`. Clearing happens only when the epoch wraps
  (every 65,535 queries), instead of an O(n) clear per query.
- **Search**: greedy descent through the upper layers, then `search_layer` on layer 0 with
  `max(ef, k)`, then the best `k` accepted results.

## Concurrency model

**Batch search** uses `parallel_for`: threads pull chunks of query indexes from an atomic
counter. Each worker borrows its own `Scratch` (visited list, both heaps, a neighbor buffer,
the normalized query) from a mutex-protected pool for the duration of the call. During search,
the graph is read-only and nothing shared is written, so no locks are taken and the results
array is written at disjoint rows.

**Parallel construction** (`add`):

1. *Sequential setup.* Copy (and normalize) the vectors, assign labels, draw levels, allocate
   upper-layer blocks, and grow storage. After this point each node's vector, level and block
   storage never change during the parallel phase, so they can be read without locks.
2. *Sequential prefix.* While the index has fewer than 1,000 nodes, inserts run on one thread.
   With T threads, every insert misses the up to T-1 nodes being inserted at the same moment,
   which matters a lot in a graph of a few hundred nodes. Measured with
   `bench/ann/parallel_build_quality` (100k random 16-d vectors, self-hits at ef = 50 among the
   first 2,000 inserted): 1,999 with one thread; with 16 threads 1,973 without the prefix and
   1,995 with it. The last 2,000 inserted are 2,000/2,000 in every configuration.
3. *Parallel inserts.* Neighbor lists are the only shared mutable state. Every node has a
   1-byte `SpinLock`. A reader copies the node's block under its lock, then computes distances
   without it. A writer (setting the new node's own list, appending or re-pruning a neighbor's
   list) holds that node's lock. **A thread never holds two node locks at once**, so there is no
   lock order and no deadlock.
4. *Entry point.* `entry_point_` and `max_level_` are read under `global_mu_`. An insert whose
   level exceeds the current maximum keeps `global_mu_` for its whole insertion, then publishes
   itself as the new entry point; this happens about `log_M(n)` times per build, so the
   serialization is negligible.

Why spinlocks: locks are held for well under a microsecond (copying ≤ 33 ints, or one
re-prune), contention on a single node is rare, and 1 byte per node beats 40 bytes for
`std::mutex` (40 MB for 1M vectors). They spin on a relaxed load and only attempt the exchange
when the lock looks free (test-and-test-and-set), yielding after 64 spins in case the holder was
preempted.

**Not supported**: `add()` concurrent with `search()` or `remove()` on the same index. The
service layer enforces this with a reader-writer lock per collection (many searches, or one
writer). ThreadSanitizer runs the whole test suite, including 4- and 8-thread builds and
parallel searches (`ctest --preset tsan`).

## Deletion

`remove(label)` sets a tombstone and drops the label from `label_to_id_`. The node stays in the
graph: searches still traverse it (so the graph stays connected), but `search_layer` never puts
it in the results. Adding an existing label is an upsert: the old node becomes a tombstone and
a new node is inserted. Tradeoff: tombstones still cost memory and traversal time, and a graph
with many tombstones spends effort on nodes it cannot return. A real system would compact
(rebuild) once the tombstone fraction passes a threshold; this one does not.

## Filtering

`search(..., allowed_labels)` turns the allowed labels into a per-node byte mask and passes an
`accept(id)` predicate to `search_layer`. Like tombstones, filtered-out nodes are traversed but
never returned, which keeps the graph navigable. The search keeps going until it holds `ef`
*accepted* results (or runs out of candidates), so a restrictive filter makes the beam walk
through many rejected nodes.

Measured (`bench/ann/filter_recall.py`, SIFT 100k, random filters, ef = 64; RESULTS.md,
Phase 6): recall stays ≥ 0.98 at every selectivity down to 0.1%, but QPS falls from 8,481
(no filter) to 254 at 1% and 43 at 0.1%, while brute force over just the allowed vectors runs
at 57k and 421k QPS. So with this design the cost of a restrictive filter is **latency, not
recall**, and below roughly 10% selectivity, scanning the allowed set exactly is both faster
and exact. A production system would pick per query: brute force when the filter is small,
filtered graph search otherwise. Not implemented here (out of scope); the service always uses
the graph. Caveat: these filters are uncorrelated with the vectors. A filter whose allowed
vectors sit far from the query in the graph (correlated filters) can also cost recall; that
was not measured.

## Persistence

Binary file: 8-byte magic (`VSHNSW01` / `VSFLAT01`) and a uint32 version, parameters, then raw
arrays (labels, levels, tombstones, layer-0 blocks, upper blocks, vectors). Loading treats every
field as untrusted: it checks header ranges, that the exact file size matches the header
(overflow-safe, before allocating), that max level equals the highest node level and the entry
point is on it, neighbor counts against capacity, that every layer-l edge targets an existing
node on layer l, deleted flags, and that live labels are unique and not -1. Any violation throws
(`RuntimeError` in Python) instead of crashing a later search; `tests/cpp/test_corrupt.cpp`
covers each case, every truncation, and random byte flips under ASan. Host byte order (no cross-endian portability).

## What was intentionally left out

- Distribution, sharding, replication, WAL or crash safety: out of scope (spec non-goals).
- Concurrent add + search inside the C++ index: would need atomic link counts and memory
  ordering for every neighbor read in the search hot path; the service's per-collection RW lock
  is much simpler for a single-node service.
- Graph compaction after deletes, quantization, other index types: not needed for the spec's
  benchmarks; would be the next steps.
- A persistent thread pool: per-call threads cost tens of microseconds, which is negligible for
  batch calls, and single-query calls with `num_threads = 1` run inline.
