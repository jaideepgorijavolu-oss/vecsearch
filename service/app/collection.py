"""One named collection: an HNSW index plus per-vector tags, guarded by a reader-writer lock."""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

import vecsearch

from .rwlock import RWLock

TagValue = int | str


class Collection:
    def __init__(self, name: str, dim: int, metric: str, M: int, ef_construction: int):
        self.name = name
        self.dim = dim
        self.metric = metric
        self.index = vecsearch.HNSWIndex(dim, metric, M=M, ef_construction=ef_construction)
        self.lock = RWLock()
        self.tags: dict[int, dict[str, TagValue]] = {}
        # Inverted index for filters: (key, value) -> ids having that tag.
        self.postings: dict[tuple[str, TagValue], set[int]] = defaultdict(set)

    # ---- writes (caller holds the write lock) ----
    def _drop_tags(self, vid: int) -> None:
        for key, value in self.tags.pop(vid, {}).items():
            ids = self.postings[(key, value)]
            ids.discard(vid)
            if not ids:
                del self.postings[(key, value)]

    def upsert(self, ids: np.ndarray, vectors: np.ndarray, tags: list[dict[str, TagValue]]) -> None:
        with self.lock.write():
            self.index.add(vectors, ids=ids)
            for vid, t in zip(ids.tolist(), tags):
                self._drop_tags(vid)
                if t:
                    self.tags[vid] = dict(t)
                    for key, value in t.items():
                        self.postings[(key, value)].add(vid)

    def delete(self, vid: int) -> bool:
        with self.lock.write():
            if not self.index.delete(vid):
                return False
            self._drop_tags(vid)
            return True

    # ---- reads ----
    def _allowed(self, flt: dict[str, TagValue]) -> np.ndarray:
        """Ids matching every key=value pair (AND), smallest posting list first."""
        sets = sorted((self.postings.get((k, v), set()) for k, v in flt.items()), key=len)
        allowed = set(sets[0])
        for s in sets[1:]:
            allowed &= s
        return np.fromiter(allowed, dtype=np.int64, count=len(allowed))

    def search(self, vector: np.ndarray, k: int, ef: int | None, flt: dict[str, TagValue] | None):
        """Returns (ids, distances, engine_seconds). engine_seconds covers only the C++ call."""
        with self.lock.read():
            allowed = self._allowed(flt) if flt else None
            if allowed is not None and len(allowed) == 0:
                return [], [], 0.0
            t0 = time.perf_counter()
            ids, dists = self.index.search(vector, k=k, ef=ef, num_threads=1, filter=allowed)
            engine_s = time.perf_counter() - t0
        keep = ids[0] >= 0
        return ids[0][keep].tolist(), dists[0][keep].tolist(), engine_s

    def __len__(self) -> int:
        return len(self.index)
