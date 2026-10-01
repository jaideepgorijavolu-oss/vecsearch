"""Write a random subset of an ann-benchmarks HDF5 dataset as .fbin files for the C++ tools.

.fbin layout: uint32 n, uint32 dim, then n*dim float32 (row-major).

Usage: python3 bench/ann/make_subset.py data/sift-128-euclidean.hdf5 100000 data/sift100k
  -> data/sift100k.base.fbin (random 100k of the 1M base vectors, seed 0)
     data/sift100k.query.fbin (all 10k queries)
"""

import sys

import h5py
import numpy as np


def write_fbin(path: str, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(x.shape, dtype="<u4").tofile(f)
        x.astype("<f4").tofile(f)


def main() -> None:
    src, n, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with h5py.File(src, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(train), size=min(n, len(train)), replace=False))
    write_fbin(prefix + ".base.fbin", train[idx])
    write_fbin(prefix + ".query.fbin", test)
    print(f"{prefix}: base {len(idx)}x{train.shape[1]}, queries {len(test)}")


if __name__ == "__main__":
    main()
