"""vecsearch: HNSW approximate nearest neighbor search, written from scratch in C++20.

>>> import numpy as np, vecsearch
>>> index = vecsearch.HNSWIndex(dim=128, metric="l2")
>>> index.add(np.random.rand(1000, 128).astype(np.float32))
>>> ids, distances = index.search(np.random.rand(5, 128).astype(np.float32), k=10)
"""

from ._core import FlatIndex, HNSWIndex, __version__, active_kernels

__all__ = ["FlatIndex", "HNSWIndex", "active_kernels", "__version__"]
