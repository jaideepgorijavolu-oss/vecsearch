import threading

import numpy as np
import pytest

import vecsearch


def numpy_knn(base, queries, k, metric):
    b, q = base.astype(np.float64), queries.astype(np.float64)
    if metric == "l2":
        d = ((q[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    elif metric == "ip":
        d = 1.0 - q @ b.T
    else:
        bn = b / np.linalg.norm(b, axis=1, keepdims=True)
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        d = 1.0 - qn @ bn.T
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def recall(got, truth):
    k = truth.shape[1]
    return np.mean([len(set(g) & set(t)) / k for g, t in zip(got, truth)])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3000, 40)).astype(np.float32)
    queries = rng.standard_normal((100, 40)).astype(np.float32)
    return base, queries


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_flat_matches_numpy(data, metric):
    base, queries = data
    index = vecsearch.FlatIndex(40, metric)
    index.add(base)
    ids, dists = index.search(queries, k=10)
    assert ids.shape == (100, 10) and ids.dtype == np.int64
    assert dists.shape == (100, 10) and dists.dtype == np.float32
    np.testing.assert_array_equal(ids, numpy_knn(base, queries, 10, metric))
    assert np.all(np.diff(dists, axis=1) >= 0)  # sorted best first


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_hnsw_recall_vs_numpy(data, metric):
    base, queries = data
    index = vecsearch.HNSWIndex(40, metric, M=16, ef_construction=200)
    index.add(base)
    ids, _ = index.search(queries, k=10, ef=200)
    assert recall(ids, numpy_knn(base, queries, 10, metric)) >= 0.95


def test_hnsw_distances_match_numpy(data):
    base, queries = data
    index = vecsearch.HNSWIndex(40)
    index.add(base)
    ids, dists = index.search(queries, k=5, ef=100)
    expected = ((queries[:, None, :] - base[ids]) ** 2).sum(-1)
    np.testing.assert_allclose(dists, expected, rtol=1e-4)


def test_single_vector_query_and_custom_ids(data):
    base, _ = data
    index = vecsearch.HNSWIndex(40)
    labels = np.arange(len(base), dtype=np.int64) * 10 + 7
    index.add(base, ids=labels)
    ids, _ = index.search(base[5], k=1)  # 1-D query
    assert ids.shape == (1, 1) and ids[0, 0] == 57
    assert 57 in index and 58 not in index


def test_k_larger_than_index():
    index = vecsearch.HNSWIndex(4)
    index.add(np.eye(4, dtype=np.float32)[:3])
    ids, dists = index.search(np.zeros(4, dtype=np.float32), k=5)
    assert list(ids[0, 3:]) == [-1, -1]
    assert np.all(np.isinf(dists[0, 3:]))
    assert set(ids[0, :3]) == {0, 1, 2}


@pytest.mark.parametrize("cls", [vecsearch.FlatIndex, vecsearch.HNSWIndex])
def test_empty_index(cls):
    index = cls(8)
    assert len(index) == 0
    ids, dists = index.search(np.zeros((2, 8), dtype=np.float32), k=3)
    assert ids.shape == (2, 3) and np.all(ids == -1) and np.all(np.isinf(dists))


@pytest.mark.parametrize("cls", [vecsearch.FlatIndex, vecsearch.HNSWIndex])
def test_wrong_dtype(cls):
    index = cls(8)
    with pytest.raises(TypeError, match="float32"):
        index.add(np.zeros((2, 8), dtype=np.float64))
    with pytest.raises(TypeError, match="numpy.ndarray"):
        index.add([[0.0] * 8])


@pytest.mark.parametrize("cls", [vecsearch.FlatIndex, vecsearch.HNSWIndex])
def test_wrong_dim_and_shape(cls):
    index = cls(8)
    with pytest.raises(ValueError, match="dimension 7"):
        index.add(np.zeros((2, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="3-D"):
        index.search(np.zeros((1, 2, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="k must be"):
        index.search(np.zeros(8, dtype=np.float32), k=0)


def test_non_contiguous_rejected():
    index = vecsearch.HNSWIndex(8)
    x = np.zeros((10, 16), dtype=np.float32)[:, ::2]  # strided view
    with pytest.raises(ValueError, match="C-contiguous"):
        index.add(x)


def test_ids_length_mismatch():
    index = vecsearch.HNSWIndex(4)
    with pytest.raises(ValueError, match="ids has 1"):
        index.add(np.zeros((2, 4), dtype=np.float32), ids=np.array([1]))


def test_bad_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        vecsearch.HNSWIndex(4, "hamming")


def test_delete(data):
    base, queries = data
    index = vecsearch.HNSWIndex(40)
    index.add(base)
    ids, _ = index.search(queries, k=10, ef=100)
    for i in np.unique(ids[:, 0]):
        assert index.delete(int(i))
    assert not index.delete(10**9)
    assert len(index) == len(base) - len(np.unique(ids[:, 0]))
    ids2, _ = index.search(queries, k=10, ef=100)
    assert not set(ids2.ravel()) & set(ids[:, 0])

    flat = vecsearch.FlatIndex(40)
    flat.add(base)
    assert flat.delete(3) and not flat.delete(3)
    assert len(flat) == len(base) - 1


def test_filter(data):
    base, queries = data
    index = vecsearch.HNSWIndex(40)
    index.add(base)
    allowed = np.arange(0, len(base), 3)
    ids, _ = index.search(queries, k=10, ef=100, filter=allowed)
    assert np.all(ids % 3 == 0)
    truth = allowed[numpy_knn(base[allowed], queries, 10, "l2")]
    assert recall(ids, truth) >= 0.9


@pytest.mark.parametrize("cls", ["flat", "hnsw"])
def test_save_load(tmp_path, data, cls):
    base, queries = data
    if cls == "flat":
        index = vecsearch.FlatIndex(40, "cosine")
    else:
        index = vecsearch.HNSWIndex(40, "cosine", M=8, ef_search=30)
    index.add(base)
    index.delete(11)
    path = str(tmp_path / "index.bin")
    index.save(path)
    loaded = type(index).load(path)
    assert len(loaded) == len(index)
    a, b = index.search(queries, k=10), loaded.search(queries, k=10)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    if cls == "hnsw":
        assert loaded.ef_search == 30 and loaded.M == 8 and loaded.metric == "cosine"


def test_load_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="cannot open"):
        vecsearch.HNSWIndex.load(str(tmp_path / "nope.bin"))


def test_parameters():
    index = vecsearch.HNSWIndex(16, M=12, ef_construction=64)
    assert (index.M, index.ef_construction, index.dim) == (12, 64, 16)
    index.ef_search = 99
    assert index.ef_search == 99
    index.prefetch = False
    assert index.prefetch is False
    assert vecsearch.active_kernels() in ("avx2", "neon", "scalar")


def test_releases_gil(data):
    """Searches from several Python threads run concurrently and give the same answers."""
    base, queries = data
    index = vecsearch.HNSWIndex(40)
    index.add(base)
    expected = index.search(queries, k=10, num_threads=1)[0]
    results = [None] * 4

    def work(i):
        results[i] = index.search(queries, k=10, num_threads=1)[0]

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        np.testing.assert_array_equal(r, expected)


def test_result_arrays_outlive_index(data):
    base, queries = data
    index = vecsearch.HNSWIndex(40)
    index.add(base)
    ids, dists = index.search(queries, k=3)
    del index
    assert ids.sum() > 0 and np.isfinite(dists).all()
