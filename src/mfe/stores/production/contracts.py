"""Strict backend-neutral contracts for production storage."""

from __future__ import annotations

import math
import re
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EMBEDDING_DIM = 512
MAX_VIDEO_ID_BYTES = 255
MAX_INT64 = 2**63 - 1
SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def _clean_string(value: str, name: str, *, max_bytes: int) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} contains a control character")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return value


class StorageRecord(BaseModel):
    """One logical keyframe row; at least one backend payload is required."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    video_id: str
    frame_id: int = Field(ge=0, le=MAX_INT64)
    embedding: tuple[float, ...] | None = None
    text: str | None = None
    source: str | None = None
    sv: int | None = Field(default=None, ge=0, le=MAX_INT64)
    ev: int | None = Field(default=None, ge=0, le=MAX_INT64)

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        return _clean_string(value, "video_id", max_bytes=MAX_VIDEO_ID_BYTES)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is None:
            return None
        if len(value) != EMBEDDING_DIM:
            raise ValueError(f"embedding must contain exactly {EMBEDDING_DIM} values")
        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding contains NaN or infinity")
        if not any(item != 0.0 for item in value):
            raise ValueError("embedding must not be a zero vector")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("text must be non-empty when present")
        if "\x00" in value:
            raise ValueError("text contains a NUL character")
        return value

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _clean_string(value, "source", max_bytes=32)
        if not SOURCE_PATTERN.fullmatch(value):
            raise ValueError("source must match [a-z][a-z0-9_-]{0,31}")
        return value

    @model_validator(mode="after")
    def validate_payload_and_bounds(self) -> Self:
        if self.embedding is None and self.text is None:
            raise ValueError("record must contain an embedding or text")
        if (self.text is None) != (self.source is None):
            raise ValueError("text and source must be present together")
        if (self.sv is None) != (self.ev is None):
            raise ValueError("sv and ev must be present together")
        if self.sv is not None and not self.sv <= self.frame_id <= self.ev:  # type: ignore[operator]
            raise ValueError("frame_id must satisfy sv <= frame_id <= ev")
        return self

    @classmethod
    def from_values(cls, **values: Any) -> Self:
        """Accept array-like embeddings and explicitly upcast float16 to float32."""
        embedding = values.get("embedding")
        if embedding is not None:
            array = np.asarray(embedding, dtype=np.float32)
            if array.ndim != 1:
                raise ValueError("embedding must be one-dimensional")
            values["embedding"] = tuple(float(item) for item in array)
        return cls.model_validate(values)


class StorageHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    video_id: str
    frame_id: int = Field(ge=0)
    score: float
    source: str | None = None
    text: str | None = None


class IngestError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_key: tuple[str, int] | None
    backend: str
    message: str
    retryable: bool = False


class IngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    attempted: int = Field(ge=0)
    inserted: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: tuple[IngestError, ...] = ()

    @property
    def succeeded(self) -> int:
        return self.inserted + self.updated


class HealthStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    connection_ok: bool
    schema_ready: bool
    index_ready: bool
    analyzer_status: str | None = None
    message: str = ""
