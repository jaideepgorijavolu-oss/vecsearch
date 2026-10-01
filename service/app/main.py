"""Thin HTTP service around vecsearch.

Endpoints:
  POST   /collections                         create a collection
  POST   /collections/{name}/vectors          batch upsert (with optional tags)
  POST   /collections/{name}/search           k nearest neighbors, optional tag filter
  DELETE /collections/{name}/vectors/{id}     soft delete
  GET    /health

Handlers are plain `def` functions, so FastAPI runs them in its thread pool; the C++ search
releases the GIL, so concurrent searches run in parallel. Each collection has a reader-writer
lock: many searches at once, or one writer.

Every search response carries the time spent in the engine (X-Engine-Time-Ms) and in the whole
handler (X-Handler-Time-Ms), so a load test can split latency into engine, Python/validation
and HTTP time.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Response, status

import vecsearch

from .collection import Collection
from .models import (
    CreateCollection,
    DeleteResult,
    Health,
    Hit,
    SearchRequest,
    SearchResponse,
    UpsertRequest,
    UpsertResult,
)

app = FastAPI(title="vecsearch", version=vecsearch.__version__)
_collections: dict[str, Collection] = {}
_registry_lock = threading.Lock()


def _get(name: str) -> Collection:
    col = _collections.get(name)
    if col is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"collection '{name}' not found")
    return col


def _check_dim(col: Collection, n: int) -> None:
    if n != col.dim:
        raise HTTPException(422,
                            f"vector has dimension {n}, collection '{col.name}' has {col.dim}")


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", collections=len(_collections), kernels=vecsearch.active_kernels())


@app.post("/collections", status_code=status.HTTP_201_CREATED)
def create_collection(req: CreateCollection) -> dict:
    with _registry_lock:
        if req.name in _collections:
            raise HTTPException(status.HTTP_409_CONFLICT, f"collection '{req.name}' already exists")
        _collections[req.name] = Collection(req.name, req.dim, req.metric, req.M,
                                            req.ef_construction)
    return {"name": req.name, "dim": req.dim, "metric": req.metric}


@app.post("/collections/{name}/vectors", response_model=UpsertResult)
def upsert(name: str, req: UpsertRequest) -> UpsertResult:
    col = _get(name)
    for item in req.vectors:
        _check_dim(col, len(item.vector))
    ids = np.array([v.id for v in req.vectors], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise HTTPException(422, "duplicate ids in one batch")
    vectors = np.array([v.vector for v in req.vectors], dtype=np.float32)
    col.upsert(ids, vectors, [v.tags for v in req.vectors])
    return UpsertResult(upserted=len(ids), size=len(col))


@app.post("/collections/{name}/search", response_model=SearchResponse)
def search(name: str, req: SearchRequest, response: Response) -> SearchResponse:
    t0 = time.perf_counter()
    col = _get(name)
    _check_dim(col, len(req.vector))
    query = np.asarray(req.vector, dtype=np.float32)
    ids, dists, engine_s = col.search(query, req.k, req.ef, req.filter)
    result = SearchResponse(hits=[Hit(id=i, distance=d) for i, d in zip(ids, dists)])
    response.headers["X-Engine-Time-Ms"] = f"{engine_s * 1e3:.4f}"
    response.headers["X-Handler-Time-Ms"] = f"{(time.perf_counter() - t0) * 1e3:.4f}"
    return result


@app.delete("/collections/{name}/vectors/{vid}", response_model=DeleteResult)
def delete_vector(name: str, vid: int) -> DeleteResult:
    col = _get(name)
    if not col.delete(vid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"id {vid} not found in '{name}'")
    return DeleteResult(deleted=True, size=len(col))
