"""Generic validation-first fan-out ingestion for both storage backends."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from .contracts import IngestReport, StorageRecord
from .protocols import TextStore, VectorStore


class DualIngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_records: int
    milvus: IngestReport
    elasticsearch: IngestReport


class DualStoreIngestor:
    """Fans validated rows out; cross-backend atomicity is intentionally not promised."""

    def __init__(self, vector_store: VectorStore, text_store: TextStore) -> None:
        self.vector_store = vector_store
        self.text_store = text_store

    def ingest(self, records: Iterable[StorageRecord]) -> DualIngestReport:
        rows = list(records)
        vector_keys = [
            (row.video_id, row.frame_id) for row in rows if row.embedding is not None
        ]
        text_keys = [
            (row.video_id, row.frame_id, row.source) for row in rows if row.text is not None
        ]
        if len(set(vector_keys)) != len(vector_keys):
            raise ValueError("duplicate vector logical key in one generic batch")
        if len(set(text_keys)) != len(text_keys):
            raise ValueError("duplicate text logical key and source in one generic batch")
        milvus = self.vector_store.ingest(rows)
        elasticsearch = self.text_store.ingest(rows)
        return DualIngestReport(
            logical_records=len(rows), milvus=milvus, elasticsearch=elasticsearch
        )

    def ingest_batches(
        self, records: Iterable[StorageRecord], *, batch_size: int = 256
    ) -> DualIngestReport:
        if not 1 <= batch_size <= 10_000:
            raise ValueError("batch_size must be between 1 and 10000")
        aggregate_milvus = IngestReport(backend="milvus", attempted=0)
        aggregate_elasticsearch = IngestReport(backend="elasticsearch", attempted=0)
        logical_records = 0
        batch: list[StorageRecord] = []

        def combine(left: IngestReport, right: IngestReport) -> IngestReport:
            return IngestReport(
                backend=left.backend,
                attempted=left.attempted + right.attempted,
                inserted=left.inserted + right.inserted,
                updated=left.updated + right.updated,
                failed=left.failed + right.failed,
                errors=left.errors + right.errors,
            )

        for record in records:
            batch.append(record)
            if len(batch) < batch_size:
                continue
            report = self.ingest(batch)
            logical_records += len(batch)
            aggregate_milvus = combine(aggregate_milvus, report.milvus)
            aggregate_elasticsearch = combine(aggregate_elasticsearch, report.elasticsearch)
            batch = []
        if batch:
            report = self.ingest(batch)
            logical_records += len(batch)
            aggregate_milvus = combine(aggregate_milvus, report.milvus)
            aggregate_elasticsearch = combine(aggregate_elasticsearch, report.elasticsearch)
        return DualIngestReport(
            logical_records=logical_records,
            milvus=aggregate_milvus,
            elasticsearch=aggregate_elasticsearch,
        )
