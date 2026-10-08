# Adaptive HNSW search: audit and protocol (draft, awaiting approval)

Status: DRAFT v0, 2026-10-07. Becomes binding when committed before any final-test evaluation.
Amendments go at the bottom with a date and reason.

## 1. Audit (2026-10-07)

Inspected (read only): `src/index/hnsw_index.cpp`, `include/vecsearch/hnsw_index.hpp`,
`bench/ann/{run,engines,datasets}.py`, `bench/ann/hnsw_eval.cpp`, `CMakePresets.json`,
`.github/workflows/ci.yml`, `docker/dev.Dockerfile`, `docs/RESULTS.md`, result JSONs.

Ran: `tools/dev.sh "cmake --preset release && cmake --build --preset release && ctest --preset release"`
→ 56/56 passed (GCC 13.3.0, Ubuntu 24.04 container, Docker 29.8.0 on WSL2, 16 vCPU, 7.4 GiB).
Not yet run: pytest, sanitizer/tsan presets. Nothing recomputed, no policy trained.

Baseline: `main` @ 89f3333 (results last regenerated in 64c3745, 2026-10-03).
Machine: AMD Ryzen 9 270, 8C/16T, ~15 GB RAM, Windows 11, all builds inside the dev container.

| Artifact (bench/ann/results/) | Metric | Build | Operating point recorded |
|---|---|---|---|
| sift_vecsearch.json | l2 | M16 efc200, 16 thr, 63.4 s | ef 64: R@10 0.9635, 6,488 QPS 1-thr, p50 0.158 / p99 0.234 ms |
| sift_vecsearch-noprefetch.json | l2 | 74.1 s | ef 64: 0.9635, 5,086 QPS |
| sift_hnswlib.json / sift_faiss.json | l2 | 67.5 / 64.7 s | ef 64: 0.9638 / 0.9675 |
| glove_vecsearch(-noprefetch), glove_hnswlib, glove_faiss .json | cosine | 81.3 s (vecsearch) | ef 1024: 0.9553, p50 1.99 ms |

Search API: `HnswIndex::search(queries, nq, k, ef, threads, allowed_labels)`; standard
Algorithm 2 beam search on layer 0 after greedy descent. `save`/`load` exist and are validated.
Threading is `std::thread` (no OpenMP), so TSan works (existing `tsan` preset, needs ASLR off).
SIMD: runtime dispatch scalar/AVX2/NEON; library not built with `-march=native` by default, so
AVX-512 is not used. Datasets (`vecsearch-data` volume): ann-benchmarks HDF5 files
(sha256 sift dd6f0a6e…, glove 544af1d5…).

### Discrepancies with the brief

1. ann-benchmarks HDF5 has no learn set. `sift_learn.fvecs` needs the TEXMEX `sift.tar.gz`
   (~161 MB), and I must verify `sift_base.fvecs` equals HDF5 `train` before reusing its ground truth.
2. Legacy JSONs record no commit, CPU model, flags, seed, or per-query data; timings are
   best-of-N batch calls via Python. They stay untouched and are not compared with new timings.
3. Legacy indexes were built on 16 threads, so the graph is not deterministic. The new study
   uses its own snapshot (built on one thread) and re-measures fixed-ef on it.
4. `data/sift1m.hnsw` in the volume has no provenance: not used.
5. The engine has no distance-evaluation or step counters and no budget/termination hook.
6. Brief mentions p95; legacy reports p99. New runs report p50/p95/p99.
7. Order and publishing: user said "go, use your own judgement" (2026-10-07) and did not
   confirm the sensor checkpoint; recorded as user-directed order. Publishing is undefined, so
   nothing is pushed until the final handoff is approved.

## 2. Question and prior work

Can a small learned policy reduce single-thread SIFT1M query cost at matched held-out recall@10,
including feature and inference overhead, versus fixed-ef HNSW and a tuned patience heuristic?
This reimplements the HNSW variant of Li, Zhang, Andersen, He, "Improving Approximate Nearest
Neighbor Search through Learned Adaptive Early Termination", SIGMOD 2020 (code:
efficient/faiss-learned-termination). HNSW: Malkov & Yashunin, TPAMI 2018. No novelty claimed.
Exact feature list and checkpoint rule of the paper to be re-read and cited precisely in the report.

## 3. Design (follows Li et al.)

Run layer 0 with a large `ef_max`; stop when the count of layer-0 distance evaluations reaches
a budget B. No restart, so no repeated work. Implemented as a new templated termination policy
in `search_layer` and a new `search_adaptive(...)` entry point with per-query stats
(distance evals, expansions, stop reason). `search()` is unchanged in behavior; its
instantiation uses a no-op policy.

- Fixed-budget cap (non-adaptive): same B for every query at `ef_max`.
- Patience heuristic: stop when the top-k set has not changed for N consecutive expansions
  (tracked with a k-sized heap; its cost is counted). Both N and the ef it runs at are tuned on validation.
- Learned: at checkpoint c distance evals, compute features, predict log T̂, set
  B = c + m·T̂ (m tuned on validation per recall target). Queries that finish before c finish normally.
- Oracle (diagnostic, not deployable): per-query budgets from traces.

**Label:** T_q = number of layer-0 distance evals after which the running top-k already contains
every exact top-k neighbor the full `ef_max` run will find. One `ef_max` trace per query gives
recall at every budget exactly, because a budgeted run is a prefix of the unbudgeted run. This
is checked by a test.

**Runtime features** (search state only): d_ep (distance to the layer-0 entry), d_1 and d_k of
the results so far, ratios d_1/d_ep, d_k/d_ep, d_k/d_1, closest candidate distance / d_k, evals
since d_1 last improved, upper-layer evals. Ablation: plus the raw query vector (as in the paper).
Never features: ground truth, eventual recall, query ids, future search outcomes.
Non-finite features (for example d_ep = 0) fall back to the fixed-ef path.

**Models:** ridge on log features vs. a small GBDT (sklearn, ≤100 trees, depth ≤4), exported to a
versioned flat file and evaluated in C++ (no ML runtime in the library). Choose by validation.

## 4. Data and splits (SIFT1M only)

- Index: HDF5 `train` (1M, L2). One snapshot: M16, efc200, seed 100, built on 1 thread; sha256 recorded.
  Determinism is checked by building the 100k subset twice.
- Train: `sift_learn.fvecs` (100k, disjoint from base by construction). Exact top-100 computed
  with FlatIndex. Its cost is reported. Learn vectors that exactly equal a val/test query, or
  duplicate another learn vector, are dropped (counts reported).
- Val/test: the 10k HDF5 test queries, split by permutation seed 0 into 5,000 val and 5,000 test. Manifests saved.
- Recall@10 by id (legacy-compatible) plus tie-aware recall (returned dist ≤ d_10 + 1e-6·d_10).
  I check zero-distance self-matches (expected none) and the frequency of ties.

## 5. Selection and evaluation

- Recall targets: proposed {0.90, 0.95, 0.99}, fixed after a validation-only feasibility check.
- ef_max ∈ {128, 256, 512}, checkpoint c ∈ {2 values}, model ∈ {ridge, GBDT} × {±query vector}:
  all selected on validation (simulated evals, then timed confirmation).
- Fixed ef: dense sweep; for each target, the smallest ef reaching it on validation.
- Final test runs once with the frozen choices. A miss is reported as a miss.
- Timing: C++ tool, one query per call, 1 thread pinned (`taskset`), Release, no sanitizers,
  1 warmup + 5 timed passes with policy order rotated per pass. Per-query latency, dist evals,
  expansions, recall saved. Report p50/p95/p99, mean, QPS = 1/mean, spread across passes,
  low-recall fraction (R@10 < 0.8) and recall quantiles. Policy overhead is also microbenchmarked.
- Success, predeclared: at a target, learned beats both fixed-ef and patience if test mean
  latency is lower by more than the pass-to-pass spread, with test recall not more than
  0.002 below the other method's. Anything else is reported as a tie or a loss.
- Stress condition: a second snapshot with graph seed 101 and no retraining.

## 6. Checks

Existing ctest and pytest; `search()` outputs byte-identical to the 89f3333 binary on SIFT test
queries at several ef; `search_adaptive` with no budget equals `search` at the same ef; the trace
prefix property; feature edge cases (k > results, zero distances, non-finite → fallback); policy
file version/corruption errors; concurrent `search_adaptive` under TSan; sanitizer and tsan presets.
The policy is validated only for unfiltered search on a static index; filtered or deleted
indexes use the fixed path.

## 7. Resources

Download ~161 MB. Disk +~1.5 GB (docker volume; traces and GT not committed, hashes recorded).
Compute ≈ 1–1.5 h total: snapshot ~8 min, learn GT ~15 min, traces ~10 min, timing ~15 min,
training minutes. Out of scope for this MVS: GloVe, multithread runs, graph-variance repeats.
