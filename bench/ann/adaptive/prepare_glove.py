"""Prepare GloVe-100 (angular) for the adaptive-search study (PROTOCOL.md, amendment A3).

GloVe has no separate learn set. Sampling rule: 100,000 rows of the ann-benchmarks `train` set
are drawn without replacement (seed 0) as policy-training queries and REMOVED from the indexed
database, so a training query never finds itself (no zero-distance self-matches). The remaining
1,083,514 rows are the base. Training rows that duplicate a val/test query, another training row,
or a base row are dropped. The 10k ann-benchmarks test queries split 5,000 val / 5,000 test
(seed 0). Metric: cosine (vecsearch normalizes vectors on insert and per query), which ranks
exactly like the angular metric of the ann-benchmarks ground truth. Ground truth is recomputed
against the reduced base (exact, FlatIndex cosine).

Usage (in the dev container): python3 bench/ann/adaptive/prepare_glove.py <run_dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from datasets import data_dir, load  # noqa: E402
from prepare import row_keys, sha256, write_fbin  # noqa: E402

SEED = 0
N_LEARN = 100_000


def main() -> None:
    run_dir = pathlib.Path(sys.argv[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir() / "adaptive"
    out.mkdir(exist_ok=True)

    train, test, _, metric = load("glove")
    assert metric == "cosine"
    rng = np.random.default_rng(SEED)
    learn_rows = np.sort(rng.choice(len(train), N_LEARN, replace=False))
    in_learn = np.zeros(len(train), dtype=bool)
    in_learn[learn_rows] = True
    base, learn = train[~in_learn], train[learn_rows]

    perm = np.random.default_rng(SEED).permutation(len(test))
    val_idx, test_idx = np.sort(perm[:5000]), np.sort(perm[5000:])

    query_keys, base_keys = set(row_keys(test)), set(row_keys(base))
    seen: set[bytes] = set()
    keep, dup_query, dup_base, dup_within = [], 0, 0, 0
    for i, key in enumerate(row_keys(learn)):
        if key in query_keys:
            dup_query += 1
        elif key in base_keys:
            dup_base += 1
        elif key in seen:
            dup_within += 1
        else:
            seen.add(key)
            keep.append(i)
    keep = np.array(keep)

    norms = {n: np.linalg.norm(x, axis=1) for n, x in [("base", base), ("learn", learn), ("test", test)]}
    files = {"base": base, "learn": learn[keep], "val": test[val_idx], "test": test[test_idx]}
    hashes = {}
    for name, x in files.items():
        p = out / f"glove_{name}.fbin"
        write_fbin(p, x)
        hashes[p.name] = sha256(p)

    manifest = {
        "dataset": "GloVe-100 (ann-benchmarks glove-100-angular)",
        "metric": "cosine (vectors normalized by vecsearch); same ranking as angular",
        "sources": {"glove-100-angular.hdf5": sha256(data_dir() / "glove-100-angular.hdf5")},
        "seed": SEED,
        "counts": {
            "train_rows": len(train), "base": len(base), "learn_sampled": N_LEARN,
            "learn_kept": int(len(keep)), "learn_dropped_equal_to_query": dup_query,
            "learn_dropped_equal_to_base": dup_base, "learn_dropped_duplicate_within": dup_within,
            "queries_identical_to_a_base_row": int(sum(k in base_keys for k in row_keys(test))),
            "duplicate_rows_within_10k_queries": int(len(test) - len(set(row_keys(test)))),
            "zero_norm_rows": {n: int((v == 0).sum()) for n, v in norms.items()},
            "val": len(val_idx), "test": len(test_idx),
        },
        "norm_range": {n: [float(v.min()), float(v.max())] for n, v in norms.items()},
        "learn_rows_of_train": learn_rows[keep].tolist(),
        "val_query_ids": val_idx.tolist(),
        "test_query_ids": test_idx.tolist(),
        "outputs_sha256": hashes,
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest))
    print(json.dumps(manifest["counts"], indent=1))


if __name__ == "__main__":
    main()
