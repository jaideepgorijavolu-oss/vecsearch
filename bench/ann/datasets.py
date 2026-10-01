"""ann-benchmarks datasets (HDF5: train, test, neighbors, distances)."""

from __future__ import annotations

import os
import pathlib
import urllib.request

import h5py
import numpy as np

DATASETS = {
    # name: (file, metric used by our engines)
    "sift": ("sift-128-euclidean.hdf5", "l2"),
    "glove": ("glove-100-angular.hdf5", "cosine"),
}
BASE_URL = "http://ann-benchmarks.com/"


def data_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("VECSEARCH_DATA", pathlib.Path(__file__).parents[2] / "data"))


def download(name: str) -> pathlib.Path:
    fname = DATASETS[name][0]
    path = data_dir() / fname
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BASE_URL + fname} -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(BASE_URL + fname, tmp)
        tmp.rename(path)
    return path


def load(name: str):
    """Returns (train, test, neighbors, metric). neighbors are the true top-100 ids."""
    with h5py.File(download(name), "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        neighbors = f["neighbors"][:]
    return train, test, neighbors, DATASETS[name][1]
