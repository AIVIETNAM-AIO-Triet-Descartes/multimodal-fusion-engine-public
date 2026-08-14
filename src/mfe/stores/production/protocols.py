"""Minimal protocols for storage orchestration without global clients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from .contracts import HealthStatus, IngestReport, StorageHit, StorageRecord


class VectorStore(Protocol):
    def ensure_schema(self) -> None: ...

    def ingest(self, records: Iterable[StorageRecord]) -> IngestReport: ...

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 10,
        video_id: str | None = None,
    ) -> list[StorageHit]: ...

    def count(self) -> int: ...

    def health(self) -> HealthStatus: ...


class TextStore(Protocol):
    def ensure_schema(self) -> None: ...

    def ingest(self, records: Iterable[StorageRecord]) -> IngestReport: ...

    def search(
        self,
        text: str,
        *,
        top_k: int = 10,
        video_id: str | None = None,
        source: str | None = None,
        phrase: bool = False,
    ) -> list[StorageHit]: ...

    def count(self) -> int: ...

    def health(self) -> HealthStatus: ...
