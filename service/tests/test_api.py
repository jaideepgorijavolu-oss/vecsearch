import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.rwlock import RWLock


@pytest.fixture()
def client():
    main._collections.clear()
    return TestClient(main.app)


def make_collection(client, name="docs", dim=8, metric="l2", n=200, seed=0, tags=True):
    assert client.post("/collections", json={"name": name, "dim": dim, "metric": metric}).status_code == 201
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    items = []
    for i in range(n):
        item = {"id": i, "vector": vecs[i].tolist()}
        if tags:
            item["tags"] = {"parity": "even" if i % 2 == 0 else "odd", "bucket": i % 10}
        items.append(item)
    r = client.post(f"/collections/{name}/vectors", json={"vectors": items})
    assert r.status_code == 200 and r.json() == {"upserted": n, "size": n}
    return vecs


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_search_finds_self_and_reports_timings(client):
    vecs = make_collection(client)
    r = client.post("/collections/docs/search", json={"vector": vecs[17].tolist(), "k": 5})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 5 and hits[0]["id"] == 17 and hits[0]["distance"] == pytest.approx(0, abs=1e-5)
    assert float(r.headers["X-Engine-Time-Ms"]) > 0
    assert float(r.headers["X-Handler-Time-Ms"]) >= float(r.headers["X-Engine-Time-Ms"])


def test_search_matches_brute_force(client):
    vecs = make_collection(client, n=300, tags=False)
    q = np.random.default_rng(1).standard_normal(8).astype(np.float32)
    r = client.post("/collections/docs/search", json={"vector": q.tolist(), "k": 10, "ef": 200})
    expected = np.argsort(((vecs - q) ** 2).sum(1))[:10]
    assert [h["id"] for h in r.json()["hits"]] == expected.tolist()


def test_filter(client):
    vecs = make_collection(client)
    r = client.post("/collections/docs/search",
                    json={"vector": vecs[3].tolist(), "k": 10, "filter": {"parity": "even"}})
    ids = [h["id"] for h in r.json()["hits"]]
    assert len(ids) == 10 and all(i % 2 == 0 for i in ids)
    r = client.post("/collections/docs/search",
                    json={"vector": vecs[3].tolist(), "k": 50, "filter": {"parity": "odd", "bucket": 3}})
    ids = [h["id"] for h in r.json()["hits"]]
    assert ids[0] == 3 and set(ids) == set(range(3, 200, 10))  # all 20 matching vectors
    r = client.post("/collections/docs/search",
                    json={"vector": vecs[3].tolist(), "filter": {"parity": "nope"}})
    assert r.json()["hits"] == []


def test_upsert_replaces_vector_and_tags(client):
    make_collection(client)
    far = [100.0] * 8
    r = client.post("/collections/docs/vectors",
                    json={"vectors": [{"id": 4, "vector": far, "tags": {"parity": "odd"}}]})
    assert r.json() == {"upserted": 1, "size": 200}
    r = client.post("/collections/docs/search", json={"vector": far, "k": 1, "filter": {"parity": "odd"}})
    assert r.json()["hits"][0]["id"] == 4
    r = client.post("/collections/docs/search",
                    json={"vector": far, "k": 100, "filter": {"parity": "even"}})
    assert 4 not in [h["id"] for h in r.json()["hits"]]


def test_delete(client):
    vecs = make_collection(client)
    assert client.delete("/collections/docs/vectors/17").json() == {"deleted": True, "size": 199}
    assert client.delete("/collections/docs/vectors/17").status_code == 404
    r = client.post("/collections/docs/search", json={"vector": vecs[17].tolist(), "k": 5})
    assert 17 not in [h["id"] for h in r.json()["hits"]]
    r = client.post("/collections/docs/search",
                    json={"vector": vecs[17].tolist(), "k": 100, "filter": {"bucket": 7}})
    assert 17 not in [h["id"] for h in r.json()["hits"]]


def test_validation_errors(client):
    make_collection(client)
    assert client.post("/collections", json={"name": "docs", "dim": 8}).status_code == 409
    assert client.post("/collections", json={"name": "bad name!", "dim": 8}).status_code == 422
    assert client.post("/collections", json={"name": "x", "dim": 0}).status_code == 422
    assert client.post("/collections", json={"name": "x", "dim": 8, "metric": "hamming"}).status_code == 422
    r = client.post("/collections/docs/search", json={"vector": [1.0] * 7})
    assert r.status_code == 422 and "dimension 7" in r.json()["detail"]
    assert client.post("/collections/docs/search", json={"vector": [1.0] * 8, "k": 0}).status_code == 422
    assert client.post("/collections/docs/search", json={"vector": [1.0] * 8, "k": 5000}).status_code == 422
    assert client.post("/collections/nope/search", json={"vector": [1.0] * 8}).status_code == 404
    r = client.post("/collections/docs/vectors", json={"vectors": [{"id": 1, "vector": [1.0] * 9}]})
    assert r.status_code == 422
    r = client.post("/collections/docs/vectors",
                    json={"vectors": [{"id": 1, "vector": [1.0] * 8}, {"id": 1, "vector": [2.0] * 8}]})
    assert r.status_code == 422 and "duplicate" in r.json()["detail"]
    assert client.post("/collections/docs/vectors", json={"vectors": []}).status_code == 422


def test_cosine_collection(client):
    vecs = make_collection(client, name="cos", metric="cosine", tags=False)
    r = client.post("/collections/cos/search", json={"vector": (vecs[9] * 3).tolist(), "k": 1})
    assert r.json()["hits"][0]["id"] == 9


def test_concurrent_reads_and_writes(client):
    """Searches and upserts from many threads at once: no errors, and the final size is right."""
    vecs = make_collection(client, n=500, tags=False)
    errors = []

    def reader():
        for i in range(30):
            r = client.post("/collections/docs/search", json={"vector": vecs[i].tolist(), "k": 3})
            if r.status_code != 200:
                errors.append(r.status_code)

    def writer(base):
        for i in range(10):
            vid = 10_000 + base * 100 + i
            r = client.post("/collections/docs/vectors",
                            json={"vectors": [{"id": vid, "vector": vecs[i].tolist()}]})
            if r.status_code != 200:
                errors.append(r.status_code)

    threads = [threading.Thread(target=reader) for _ in range(6)]
    threads += [threading.Thread(target=writer, args=(b,)) for b in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert client.delete("/collections/docs/vectors/0").json()["size"] == 500 + 30 - 1


def test_rwlock_excludes_writer_from_readers():
    lock = RWLock()
    state = {"readers": 0, "max_readers": 0, "violations": 0}
    guard = threading.Lock()

    def read():
        for _ in range(200):
            with lock.read():
                with guard:
                    state["readers"] += 1
                    state["max_readers"] = max(state["max_readers"], state["readers"])
                with guard:
                    state["readers"] -= 1

    def write():
        for _ in range(200):
            with lock.write():
                with guard:
                    if state["readers"]:
                        state["violations"] += 1

    threads = [threading.Thread(target=read) for _ in range(4)] + [threading.Thread(target=write)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["violations"] == 0


# ---- invalid numbers and ids must be 422 and leave the collection unchanged ----

BAD_NUMBERS = [float("nan"), float("inf"), float("-inf"), 1e100, -1e100, 1e16]


def post_raw(client, url, body):
    """httpx refuses to encode NaN/Infinity, but Python clients (json.dumps) send them happily."""
    return client.post(url, content=json.dumps(body), headers={"Content-Type": "application/json"})


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_upsert_rejects_bad_numbers_atomically(client, bad):
    vecs = make_collection(client, n=20)
    good = {"id": 100, "vector": [0.5] * 8, "tags": {"parity": "new"}}
    bad_item = {"id": 101, "vector": [0.1] * 7 + [bad], "tags": {"parity": "new"}}
    r = post_raw(client, "/collections/docs/vectors", {"vectors": [good, bad_item]})
    assert r.status_code == 422, r.text
    # Nothing from the batch went in: not the valid vector, not its tags.
    r = client.post("/collections/docs/search",
                    json={"vector": [0.5] * 8, "k": 5, "filter": {"parity": "new"}})
    assert r.status_code == 200 and r.json()["hits"] == []
    assert client.delete("/collections/docs/vectors/100").status_code == 404
    # And the collection still searches fine.
    r = client.post("/collections/docs/search", json={"vector": vecs[0].tolist(), "k": 3})
    assert r.status_code == 200 and r.json()["hits"][0]["id"] == 0


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_search_rejects_bad_numbers(client, bad):
    make_collection(client, n=20)
    r = post_raw(client, "/collections/docs/search", {"vector": [0.1] * 7 + [bad], "k": 3})
    assert r.status_code == 422, r.text


def test_largest_allowed_values_still_search(client):
    """At the documented bound (|x| <= 1e15) squared distances stay finite."""
    make_collection(client, n=5, tags=False)
    big = [1e15, -1e15] * 4
    assert client.post("/collections/docs/vectors",
                       json={"vectors": [{"id": 50, "vector": big}]}).status_code == 200
    r = client.post("/collections/docs/search", json={"vector": [-x for x in big], "k": 6})
    assert r.status_code == 200
    assert len(r.json()["hits"]) == 6


@pytest.mark.parametrize("bad_id", [1.9, 2.0, True, -1, 2**63, "3"])
def test_upsert_rejects_bad_ids(client, bad_id):
    make_collection(client, n=5, tags=False)
    r = client.post("/collections/docs/vectors", json={"vectors": [{"id": bad_id, "vector": [0.0] * 8}]})
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("bad_tag", [1.5, True, None, [1]])
def test_tags_must_be_int_or_string(client, bad_tag):
    make_collection(client, n=5, tags=False)
    item = {"id": 9, "vector": [0.0] * 8, "tags": {"t": bad_tag}}
    assert client.post("/collections/docs/vectors", json={"vectors": [item]}).status_code == 422
    r = client.post("/collections/docs/search", json={"vector": [0.0] * 8, "filter": {"t": bad_tag}})
    assert r.status_code == 422


def test_delete_negative_id_is_422(client):
    make_collection(client, n=5, tags=False)
    assert client.delete("/collections/docs/vectors/-1").status_code == 422
