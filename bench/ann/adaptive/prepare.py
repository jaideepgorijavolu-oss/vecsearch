"""Prepare SIFT1M data for the adaptive-search study (docs/adaptive_search/PROTOCOL.md).

Writes .fbin files to $VECSEARCH_DATA/adaptive/ and a split manifest to the run directory:
  base   = ann-benchmarks train (1M), verified identical to TEXMEX sift_base.fvecs
  learn  = TEXMEX sift_learn.fvecs (100k) minus exact duplicates (within learn, or of any
           val/test query) -> policy training
  val    = 5,000 of the 10,000 ann-benchmarks test queries (permutation seed 0)
  test   = the other 5,000 (final evaluation only)

Usage (in the dev container): python3 bench/ann/adaptive/prepare.py <run_dir>
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from datasets import data_dir, load  # noqa: E402

SPLIT_SEED = 0


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fvecs(path: pathlib.Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    rows = raw.reshape(-1, dim + 1)
    assert (rows[:, 0] == dim).all(), f"{path}: inconsistent dimensions"
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def write_fbin(path: pathlib.Path, x: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        np.ascontiguousarray(x, dtype=np.float32).tofile(f)


def row_keys(x: np.ndarray) -> list[bytes]:
    return [r.tobytes() for r in np.ascontiguousarray(x)]


def verify_sources(names: list[str]) -> None:
    """Inputs must be byte-identical to the ones the published runs used (sources.json)."""
    want = json.loads((pathlib.Path(__file__).parent / "sources.json").read_text())
    for name in names:
        got = sha256(data_dir() / name)
        if got != want[name]:
            raise SystemExit(f"{data_dir() / name}: sha256 {got} != expected {want[name]}")


def main() -> None:
    run_dir = pathlib.Path(sys.argv[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir() / "adaptive"
    out.mkdir(exist_ok=True)
    texmex = data_dir() / "texmex" / "sift"

    train, test, _, metric = load("sift")  # downloads the HDF5 file if missing
    verify_sources(["sift-128-euclidean.hdf5", "texmex/sift/sift_base.fvecs",
                    "texmex/sift/sift_learn.fvecs"])
    assert metric == "l2"
    base_fvecs = read_fvecs(texmex / "sift_base.fvecs")
    assert np.array_equal(train, base_fvecs), "HDF5 train differs from TEXMEX sift_base"
    learn = read_fvecs(texmex / "sift_learn.fvecs")

    perm = np.random.default_rng(SPLIT_SEED).permutation(len(test))
    val_idx, test_idx = np.sort(perm[:5000]), np.sort(perm[5000:])

    # Exact-duplicate removal. Learn vectors are drawn separately from base in TEXMEX, but we
    # check rather than assume.
    query_keys = set(row_keys(test))
    seen: set[bytes] = set()
    keep, dup_within, dup_query = [], 0, 0
    for i, key in enumerate(row_keys(learn)):
        if key in query_keys:
            dup_query += 1
        elif key in seen:
            dup_within += 1
        else:
            seen.add(key)
            keep.append(i)
    keep = np.array(keep)
    base_keys = set(row_keys(train))
    learn_in_base = sum(k in base_keys for k in row_keys(learn[keep]))
    query_in_base = sum(k in base_keys for k in row_keys(test))
    test_dups = len(test) - len(set(row_keys(test)))

    files = {"base": train, "learn": learn[keep], "val": test[val_idx], "test": test[test_idx]}
    hashes = {}
    for name, x in files.items():
        p = out / f"sift_{name}.fbin"
        write_fbin(p, x)
        hashes[f"sift_{name}.fbin"] = sha256(p)

    manifest = {
        "dataset": "SIFT1M",
        "metric": "l2 (squared Euclidean)",
        "sources": {
            "sift-128-euclidean.hdf5": sha256(data_dir() / "sift-128-euclidean.hdf5"),
            "texmex/sift/sift_base.fvecs": sha256(texmex / "sift_base.fvecs"),
            "texmex/sift/sift_learn.fvecs": sha256(texmex / "sift_learn.fvecs"),
        },
        "base_equals_texmex_base": True,
        "split_seed": SPLIT_SEED,
        "counts": {
            "base": len(train), "learn_raw": len(learn), "learn_kept": int(len(keep)),
            "learn_dropped_duplicate_within": dup_within,
            "learn_dropped_equal_to_query": dup_query,
            "learn_kept_identical_to_a_base_vector": int(learn_in_base),
            "test_queries_identical_to_a_base_vector": int(query_in_base),
            "duplicate_rows_within_10k_queries": int(test_dups),
            "val": len(val_idx), "test": len(test_idx),
        },
        "val_query_ids": val_idx.tolist(),   # rows of the ann-benchmarks test set
        "test_query_ids": test_idx.tolist(),
        "learn_dropped_ids": sorted(set(range(len(learn))) - set(keep.tolist())),
        "outputs_sha256": hashes,
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["counts"], indent=1))


if __name__ == "__main__":
    main()
