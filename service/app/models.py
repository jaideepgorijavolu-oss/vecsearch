"""Request / response schemas. Static bounds live here; the dimension check needs the
collection, so it happens in the handlers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictInt, StrictStr

# Strict types: JSON 1.5, true or "3" are rejected rather than coerced to an int.
TagValue = StrictInt | StrictStr
Tags = dict[str, TagValue]

# Vector components must be finite and |x| <= MAX_ABS. The bound keeps every squared L2
# distance and dot product finite in float32 for any dim <= MAX_DIM: (2e15)^2 * 4096 = 1.6e34,
# far below float32's max of 3.4e38. Without it, NaN/inf (or 1e100, which overflows float32)
# get stored and every later search returns non-JSON-encodable distances (HTTP 500).
MAX_ABS = 1e15
Component = Annotated[float, Field(allow_inf_nan=False, ge=-MAX_ABS, le=MAX_ABS)]

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
    id: StrictInt = Field(ge=0, le=2**63 - 1)
    vector: list[Component] = Field(min_length=1, max_length=MAX_DIM)
    tags: Tags = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    vectors: list[VectorItem] = Field(min_length=1, max_length=MAX_BATCH)


class UpsertResult(BaseModel):
    upserted: int
    size: int


class SearchRequest(BaseModel):
    vector: list[Component] = Field(min_length=1, max_length=MAX_DIM)
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
