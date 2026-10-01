"""Benchmark one engine on one dataset; writes bench/ann/results/<dataset>_<engine>.json.

Methodology (identical for every engine):
  * build with M, ef_construction and `threads` threads; time it; RSS growth = index memory
  * for each ef_search: batch-search all test queries with all threads (best of 7 runs) and
    with one thread (best of 3); recall@10 against the dataset's ground truth
  * at the smallest ef with recall@10 >= 0.95: per-query latency (one query per call,
    one thread) over the first `latency_queries` queries -> p50 / p99

Usage: python3 bench/ann/run.py --dataset sift --engine vecsearch [--threads 16]
Each (dataset, engine) should run in its own process so RSS measurements don't mix.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import time

import numpy as np
import psutil

from datasets import load
from engines import ENGINES

EF_SWEEP = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
# GloVe-100 is much harder: with M=16 it needs ef > 512 to reach 0.95 recall@10.
EF_SWEEP_EXTRA = {"glove": [768, 1024, 1536]}
K = 10
RESULTS = pathlib.Path(__file__).parent / "results"


def recall_at_k(ids: np.ndarray, truth: np.ndarray, k: int = K) -> float:
    hits = 0
    for got, want in zip(ids[:, :k], truth[:, :k]):
        hits += len(set(got.tolist()) & set(want.tolist()))
    return hits / (len(truth) * k)


def best_time(fn, repeats: int) -> tuple[float, np.ndarray]:
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="runs per 1-thread measurement")
    ap.add_argument("--repeats-all", type=int, default=7,
                    help="runs per all-thread measurement (short runs, so noisier)")
    ap.add_argument("--latency-queries", type=int, default=2000)
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()

    train, test, neighbors, metric = load(args.dataset)
    engine = ENGINES[args.engine]()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    t0 = time.perf_counter()
    engine.build(train, metric, args.M, args.ef_construction, args.threads)
    build_s = time.perf_counter() - t0
    rss_after = proc.memory_info().rss
    print(f"[{args.dataset}/{engine.name}] build {build_s:.1f} s, "
          f"RSS +{(rss_after - rss_before) / 2**20:.0f} MiB")

    sweep = []
    for ef in EF_SWEEP + EF_SWEEP_EXTRA.get(args.dataset, []):
        engine.set_ef(ef)
        t_all, ids = best_time(lambda: engine.search(test, K, args.threads), args.repeats_all)
        t_one, ids1 = best_time(lambda: engine.search(test, K, 1), args.repeats)
        rec = recall_at_k(ids, neighbors)
        sweep.append({
            "ef": ef,
            "recall": rec,
            "recall_1thread": recall_at_k(ids1, neighbors),
            "qps_all_threads": len(test) / t_all,
            "qps_1thread": len(test) / t_one,
        })
        print(f"  ef {ef:4d}  recall {rec:.4f}  QPS all {len(test) / t_all:9.0f}  "
              f"1-thread {len(test) / t_one:8.0f}")

    latency = None
    target = next((s for s in sweep if s["recall"] >= args.target_recall), None)
    if target is not None:
        engine.set_ef(target["ef"])
        lat = []
        for q in test[: args.latency_queries]:
            q = q[None, :]
            t0 = time.perf_counter_ns()
            engine.search(q, K, 1)
            lat.append((time.perf_counter_ns() - t0) / 1e6)
        lat = np.array(lat)
        latency = {
            "ef": target["ef"],
            "recall": target["recall"],
            "p50_ms": float(np.percentile(lat, 50)),
            "p99_ms": float(np.percentile(lat, 99)),
            "mean_ms": float(lat.mean()),
            "queries": len(lat),
        }
        print(f"  latency at ef {target['ef']}: p50 {latency['p50_ms']:.3f} ms, "
              f"p99 {latency['p99_ms']:.3f} ms")

    out = {
        "dataset": args.dataset,
        "engine": engine.name,
        "metric": metric,
        "n": int(train.shape[0]),
        "dim": int(train.shape[1]),
        "queries": int(test.shape[0]),
        "M": args.M,
        "ef_construction": args.ef_construction,
        "threads": args.threads,
        "build_seconds": build_s,
        "rss_growth_bytes": rss_after - rss_before,
        "reported_memory_bytes": engine.memory_bytes(),
        "sweep": sweep,
        "latency": latency,
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "python": platform.python_version()},
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.dataset}_{engine.name}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
