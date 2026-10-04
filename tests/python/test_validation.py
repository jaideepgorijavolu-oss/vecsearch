"""Regression tests: corrupt index files raise, and ids must be real int64 values (not -1)."""

import struct

import numpy as np
import pytest

import vecsearch

MAX_LEVEL_OFFSET = 62  # see HnswIndex::save: header fields before max_level


# ---- corrupt files ----

def test_one_node_max_level_raised_raises(tmp_path):
    """Used to load fine and then kill Python with SIGSEGV on search."""
    index = vecsearch.HNSWIndex(4)
    index.add(np.ones((1, 4), dtype=np.float32))
    path = tmp_path / "one.bin"
    index.save(str(path))
    raw = bytearray(path.read_bytes())
    level = struct.unpack_from("<i", raw, MAX_LEVEL_OFFSET)[0]
    struct.pack_into("<i", raw, MAX_LEVEL_OFFSET, level + 1)  # a layer the node doesn't have
    path.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="corrupt"):
        loaded = vecsearch.HNSWIndex.load(str(path))
        loaded.search(np.ones(4, dtype=np.float32), k=1)


@pytest.mark.parametrize("cls", [vecsearch.HNSWIndex, vecsearch.FlatIndex])
def test_truncated_files_raise(tmp_path, cls):
    index = cls(8)
    index.add(np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32))
    path = tmp_path / "idx.bin"
    index.save(str(path))
    raw = path.read_bytes()
    for cut in (0, 5, 20, 70, len(raw) // 2, len(raw) - 1):
        path.write_bytes(raw[:cut])
        with pytest.raises(RuntimeError):
            cls.load(str(path))
    path.write_bytes(raw + b"\0")
    with pytest.raises(RuntimeError):
        cls.load(str(path))


# ---- ids ----

@pytest.mark.parametrize("ids", [
    [1.9, 2.9],
    np.array([1.9, 2.9]),
    np.array([1.0, 2.0]),
    [True, False],
    np.array([True, False]),
    [1, 2**63],
    np.array([1, 2**63], dtype=np.uint64),
    ["1", "2"],
])
def test_add_rejects_non_int64_ids(ids):
    index = vecsearch.HNSWIndex(4)
    index.add(np.zeros((2, 4), dtype=np.float32), ids=[1, 2])
    with pytest.raises((TypeError, ValueError)):
        index.add(np.ones((2, 4), dtype=np.float32), ids=ids)
    assert len(index) == 2 and index.element_count == 2  # nothing was inserted or replaced


@pytest.mark.parametrize("ids", [[-1, 5], np.array([3, -1], dtype=np.int64)])
def test_add_rejects_reserved_minus_one(ids):
    index = vecsearch.HNSWIndex(4)
    with pytest.raises(ValueError, match="-1"):
        index.add(np.zeros((2, 4), dtype=np.float32), ids=ids)
    assert index.element_count == 0


@pytest.mark.parametrize("ids", [
    [0, 1],
    np.array([0, 1], dtype=np.int32),
    np.array([0, 1], dtype=np.uint8),
    [-5, 2**63 - 1],
])
def test_add_accepts_integer_ids(ids):
    index = vecsearch.HNSWIndex(4)
    index.add(np.eye(4, dtype=np.float32)[:2], ids=ids)
    assert len(index) == 2
    got = index.search(np.eye(4, dtype=np.float32)[:2], k=1)[0][:, 0]
    assert got.tolist() == [int(i) for i in ids]


@pytest.mark.parametrize("flt", [[1.5], np.array([1.0]), [True], [-1], [2**64]])
def test_filter_rejects_bad_ids(flt):
    index = vecsearch.HNSWIndex(4)
    index.add(np.eye(4, dtype=np.float32))
    with pytest.raises((TypeError, ValueError)):
        index.search(np.ones(4, dtype=np.float32), k=2, filter=flt)


def test_filter_accepts_empty_and_int_lists():
    index = vecsearch.HNSWIndex(4)
    index.add(np.eye(4, dtype=np.float32))
    ids, _ = index.search(np.ones(4, dtype=np.float32), k=2, filter=[])
    assert ids.tolist() == [[-1, -1]]
    ids, _ = index.search(np.ones(4, dtype=np.float32), k=4, filter=np.array([2], dtype=np.int16))
    assert ids[0, 0] == 2


@pytest.mark.parametrize("cls", [vecsearch.HNSWIndex, vecsearch.FlatIndex])
@pytest.mark.parametrize("bad", [1.0, True, -1, 2**63, "1"])
def test_delete_rejects_bad_ids(cls, bad):
    index = cls(4)
    index.add(np.eye(4, dtype=np.float32))
    with pytest.raises((TypeError, ValueError)):
        index.delete(bad)
    assert len(index) == 4
    assert index.delete(np.int64(1)) and len(index) == 3


# ---- bools hidden inside mixed lists (np.asarray([True, 2]) silently becomes [1, 2]) ----

MIXED_BOOL_IDS = [[True, 2], [1, False], [np.True_, 2], (2, True)]


@pytest.mark.parametrize("ids", MIXED_BOOL_IDS + [np.array([True, False])])
def test_add_rejects_bools_in_mixed_lists(ids):
    index = vecsearch.HNSWIndex(2)
    index.add(np.array([[0, 0]], dtype=np.float32), ids=[1])
    with pytest.raises(TypeError):
        index.add(np.array([[9, 9], [20, 20]], dtype=np.float32), ids=ids)
    ids_out, dists = index.search(np.array([0, 0], dtype=np.float32), k=1)
    assert ids_out[0, 0] == 1 and dists[0, 0] == 0.0  # vector 1 was not replaced
    assert index.element_count == 1


@pytest.mark.parametrize("flt", MIXED_BOOL_IDS + [np.array([True, False])])
def test_filter_rejects_bools_in_mixed_lists(flt):
    index = vecsearch.HNSWIndex(2)
    index.add(np.eye(2, dtype=np.float32), ids=[1, 2])
    with pytest.raises(TypeError):
        index.search(np.ones(2, dtype=np.float32), k=1, filter=flt)


@pytest.mark.parametrize("ids", [[1, 2], (1, 2), [np.int64(1), np.int32(2)],
                                 np.array([1, 2], dtype=np.int64)])
def test_integer_lists_and_arrays_still_work(ids):
    index = vecsearch.HNSWIndex(2)
    index.add(np.eye(2, dtype=np.float32), ids=ids)
    assert index.search(np.eye(2, dtype=np.float32), k=1)[0][:, 0].tolist() == [1, 2]
    got, _ = index.search(np.eye(2, dtype=np.float32)[0], k=2, filter=ids)
    assert sorted(got[0].tolist()) == [1, 2]
