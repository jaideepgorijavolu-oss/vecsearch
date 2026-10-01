"""Uniform adapters for the three engines benchmarked: vecsearch, hnswlib and Faiss.

Every adapter builds with the same M / ef_construction and thread count, and answers batch
queries with one call into C++ (so Python overhead is the same small constant for all three).
"""

from __future__ import annotations

import numpy as np


class Engine:
    name = ""

    def build(self, data: np.ndarray, metric: str, M: int, ef_construction: int, threads: int):
        raise NotImplementedError

    def set_ef(self, ef: int):
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int, threads: int) -> np.ndarray:
        raise NotImplementedError

    def memory_bytes(self) -> int | None:
        return None


class VecSearch(Engine):
    def __init__(self, prefetch: bool = True):
        self.prefetch = prefetch
        self.name = "vecsearch" if prefetch else "vecsearch-noprefetch"

    def build(self, data, metric, M, ef_construction, threads):
        import vecsearch

        self.index = vecsearch.HNSWIndex(data.shape[1], metric, M=M, ef_construction=ef_construction)
        self.index.prefetch = self.prefetch
        self.index.add(data, num_threads=threads)
        self.ef = 10

    def set_ef(self, ef):
        self.ef = ef

    def search(self, queries, k, threads):
        return self.index.search(queries, k=k, ef=self.ef, num_threads=threads)[0]

    def memory_bytes(self):
        return self.index.memory_bytes


class Hnswlib(Engine):
    name = "hnswlib"

    def build(self, data, metric, M, ef_construction, threads):
        import hnswlib

        space = {"l2": "l2", "cosine": "cosine"}[metric]
        self.index = hnswlib.Index(space=space, dim=data.shape[1])
        self.index.init_index(max_elements=len(data), M=M, ef_construction=ef_construction,
                              random_seed=100)
        self.index.add_items(data, num_threads=threads)

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def search(self, queries, k, threads):
        return self.index.knn_query(queries, k=k, num_threads=threads)[0]


class Faiss(Engine):
    name = "faiss"

    def build(self, data, metric, M, ef_construction, threads):
        import faiss

        self.faiss = faiss
        self.cosine = metric == "cosine"
        if self.cosine:  # cosine = inner product on unit vectors
            data = data / np.linalg.norm(data, axis=1, keepdims=True)
            self.index = faiss.IndexHNSWFlat(data.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(data.shape[1], M)
        self.index.hnsw.efConstruction = ef_construction
        faiss.omp_set_num_threads(threads)
        self.index.add(np.ascontiguousarray(data, dtype=np.float32))

    def set_ef(self, ef):
        self.index.hnsw.efSearch = ef

    def search(self, queries, k, threads):
        if self.cosine:
            queries = np.ascontiguousarray(
                queries / np.linalg.norm(queries, axis=1, keepdims=True), dtype=np.float32)
        self.faiss.omp_set_num_threads(threads)
        return self.index.search(queries, k)[1]


ENGINES = {
    "vecsearch": lambda: VecSearch(prefetch=True),
    "vecsearch-noprefetch": lambda: VecSearch(prefetch=False),
    "hnswlib": Hnswlib,
    "faiss": Faiss,
}
