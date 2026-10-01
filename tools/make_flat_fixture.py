"""Generate tests/cpp/fixtures/flat_numpy.bin: random data plus NumPy brute-force top-k ids.

The C++ FlatIndex test loads this file and requires identical ids for every metric.
Run from the repo root:  python3 tools/make_flat_fixture.py

File layout (little-endian):
  int32 n, dim, nq, k
  float32 base[n*dim], queries[nq*dim]
  int64 ids_l2[nq*k], ids_ip[nq*k], ids_cosine[nq*k]
"""

import pathlib

import numpy as np

N, DIM, NQ, K = 2000, 37, 50, 10  # dim 37: not a multiple of any SIMD width


def topk(dist: np.ndarray, k: int) -> np.ndarray:
    # Stable sort: equal distances are ordered by id, matching the C++ tie-break.
    return np.argsort(dist, axis=1, kind="stable")[:, :k].astype(np.int64)


def main() -> None:
    rng = np.random.default_rng(1234)
    base = rng.standard_normal((N, DIM)).astype(np.float32)
    queries = rng.standard_normal((NQ, DIM)).astype(np.float32)

    b64, q64 = base.astype(np.float64), queries.astype(np.float64)
    l2 = ((q64[:, None, :] - b64[None, :, :]) ** 2).sum(-1)
    ip = 1.0 - q64 @ b64.T
    bn = b64 / np.linalg.norm(b64, axis=1, keepdims=True)
    qn = q64 / np.linalg.norm(q64, axis=1, keepdims=True)
    cos = 1.0 - qn @ bn.T

    out = pathlib.Path(__file__).resolve().parent.parent / "tests/cpp/fixtures/flat_numpy.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        np.array([N, DIM, NQ, K], dtype="<i4").tofile(f)
        base.astype("<f4").tofile(f)
        queries.astype("<f4").tofile(f)
        for d in (l2, ip, cos):
            topk(d, K).astype("<i8").tofile(f)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
