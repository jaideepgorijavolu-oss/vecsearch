"""Request / response schemas. Static bounds live here; the dimension check needs the
collection, so it happens in the handlers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TagValue = int | str
Tags = dict[str, TagValue]

MAX_DIM = 4096
MAX_BATCH = 10_000
MAX_K = 1000


class CreateCollection(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    dim: int = Field(ge=1, le=MAX_DIM)
    metric: Literal["l2", "ip", "cosine"] = "l2"
    M: int = Field(16, ge=2, le=128)
    ef_construction: int = Field(200, ge=1, le=4096)


class VectorItem(BaseModel):
    id: int = Field(ge=0, le=2**63 - 1)
    vector: list[float] = Field(min_length=1, max_length=MAX_DIM)
    tags: Tags = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    vectors: list[VectorItem] = Field(min_length=1, max_length=MAX_BATCH)


class UpsertResult(BaseModel):
    upserted: int
    size: int


class SearchRequest(BaseModel):
    vector: list[float] = Field(min_length=1, max_length=MAX_DIM)
    k: int = Field(10, ge=1, le=MAX_K)
    ef: int | None = Field(None, ge=1, le=10_000)
    filter: Tags | None = Field(None, description="Tags that must all match (AND)")


class Hit(BaseModel):
    id: int
    distance: float


class SearchResponse(BaseModel):
    hits: list[Hit]


class DeleteResult(BaseModel):
    deleted: bool
    size: int


class Health(BaseModel):
    status: str
    collections: int
    kernels: str
