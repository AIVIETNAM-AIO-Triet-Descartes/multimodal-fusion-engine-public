"""Generic validation-first fan-out ingestion for both storage backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pydantic import BaseModel, ConfigDict

from .contracts import IngestReport, StorageRecord
from .protocols import TextStore, VectorStore


class DualIngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_records: int
    milvus: IngestReport
    elasticsearch: IngestReport


class DualStoreIngestor:
    """Fans validated rows out; cross-backend atomicity is intentionally not promised.

    Responsibilities:
    * Pre-flight duplicate detection before any write.
    * Batched fan-out to Milvus (embeddings) and Elasticsearch (text).
    * Optional progress callback for long-running full-corpus ingestion.
    * Batch-level checkpoint/resume via external ``ingested_keys`` set.
    """

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
        self,
        records: Iterable[StorageRecord],
        *,
        batch_size: int = 256,
        progress_callback: Callable[[int, DualIngestReport], None] | None = None,
    ) -> DualIngestReport:
        """Ingest records in batches with optional progress reporting.

        Args:
            records:           Source of ``StorageRecord`` rows.
            batch_size:        Records per flush (default 256).
            progress_callback: Called after each batch with ``(records_ingested_so_far,
                               cumulative_report)``.  Use for progress bars or
                               checkpoint writes in long full-corpus runs.
        """
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

        def _flush(batch: list[StorageRecord]) -> DualIngestReport:
            nonlocal aggregate_milvus, aggregate_elasticsearch, logical_records
            report = self.ingest(batch)
            logical_records += len(batch)
            aggregate_milvus = combine(aggregate_milvus, report.milvus)
            aggregate_elasticsearch = combine(aggregate_elasticsearch, report.elasticsearch)
            cumulative = DualIngestReport(
                logical_records=logical_records,
                milvus=aggregate_milvus,
                elasticsearch=aggregate_elasticsearch,
            )
            if progress_callback is not None:
                progress_callback(logical_records, cumulative)
            return cumulative

        for record in records:
            batch.append(record)
            if len(batch) < batch_size:
                continue
            _flush(batch)
            batch = []
        if batch:
            _flush(batch)
        return DualIngestReport(
            logical_records=logical_records,
            milvus=aggregate_milvus,
            elasticsearch=aggregate_elasticsearch,
        )

    def ingest_with_resume(
        self,
        records: Iterable[StorageRecord],
        *,
        batch_size: int = 256,
        ingested_keys: set[tuple[str, int]] | None = None,
        progress_callback: Callable[[int, DualIngestReport], None] | None = None,
    ) -> DualIngestReport:
        """Idempotent ingest with per-record checkpoint/resume support.

        Skips any records whose ``(video_id, frame_id)`` key is present in
        *ingested_keys*.  Callers should persist this set between runs (e.g.
        as a JSON file) to enable safe resume after partial failures.

        Technique recommended for AIC full-corpus batches (87k+ rows) where a
        mid-run crash would otherwise restart the entire ingest.

        Args:
            records:        Source records iterator.
            batch_size:     Records per flush.
            ingested_keys:  Mutable set of already-ingested ``(video_id, frame_id)``
                            pairs; **updated in-place** as records are ingested.
            progress_callback: See ``ingest_batches()``.
        """
        if ingested_keys is None:
            ingested_keys = set()

        def _filtered() -> Iterable[StorageRecord]:
            for record in records:
                key = (record.video_id, record.frame_id)
                if key in ingested_keys:
                    continue
                yield record
                ingested_keys.add(key)

        return self.ingest_batches(
            _filtered(),
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
