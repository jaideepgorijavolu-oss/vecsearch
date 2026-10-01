"""Recall and speed of filtered HNSW search as the filter gets more restrictive.

SIFT 100k subset (bench/ann/make_subset.py). For each selectivity s, a random fraction s of the
ids is allowed. Ground truth: exact top-10 among the allowed vectors (FlatIndex over them).
Reports recall@10 and single-thread QPS of filtered HNSW search, and the QPS of simply
brute-forcing the allowed vectors, which is what a planner should pick for tiny filters.

Usage: python3 bench/ann/filter_recall.py [--ef 64]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import vecsearch


def read_fbin(path):
    raw = np.fromfile(path, dtype=np.float32)
    n, dim = raw[:2].view(np.uint32)
    return raw[2:].reshape(int(n), int(dim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("VECSEARCH_DATA", "data"))
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--queries", type=int, default=1000)
    args = ap.parse_args()

    base = read_fbin(f"{args.data}/sift100k.base.fbin")
    queries = read_fbin(f"{args.data}/sift100k.query.fbin")[: args.queries]
    index = vecsearch.HNSWIndex(base.shape[1], "l2", M=16, ef_construction=200)
    index.add(base)
    rng = np.random.default_rng(0)

    print(f"SIFT 100k subset, M=16, ef_construction=200, ef_search={args.ef}, k=10, "
          f"{len(queries)} queries, 1 thread\n")
    print("| selectivity | allowed vectors | recall@10 | filtered HNSW QPS | brute force over allowed QPS |")
    print("|---:|---:|---:|---:|---:|")
    for sel in (1.0, 0.5, 0.1, 0.01, 0.001):
        allowed = np.sort(rng.choice(len(base), size=max(10, int(len(base) * sel)), replace=False))
        flat = vecsearch.FlatIndex(base.shape[1], "l2")
        flat.add(np.ascontiguousarray(base[allowed]))

        t0 = time.perf_counter()
        exact, _ = flat.search(queries, k=10, num_threads=1)
        flat_qps = len(queries) / (time.perf_counter() - t0)
        truth = allowed[exact]

        flt = None if sel == 1.0 else allowed
        t0 = time.perf_counter()
        ids, _ = index.search(queries, k=10, ef=args.ef, num_threads=1, filter=flt)
        hnsw_qps = len(queries) / (time.perf_counter() - t0)
        recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
        print(f"| {sel:g} | {len(allowed):,} | {recall:.4f} | {hnsw_qps:,.0f} | {flat_qps:,.0f} |")


if __name__ == "__main__":
    main()
