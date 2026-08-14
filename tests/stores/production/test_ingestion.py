from __future__ import annotations

from mfe.stores.production.contracts import IngestReport, StorageRecord
from mfe.stores.production.ingestion import DualStoreIngestor


class RecordingStore:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.batches: list[list[StorageRecord]] = []

    def ingest(self, records: list[StorageRecord]) -> IngestReport:
        self.batches.append(records)
        relevant = sum(
            record.embedding is not None if self.backend == "milvus" else record.text is not None
            for record in records
        )
        return IngestReport(backend=self.backend, attempted=relevant, inserted=relevant)


def test_multiple_text_sources_can_share_one_logical_frame() -> None:
    vector = RecordingStore("milvus")
    text = RecordingStore("elasticsearch")
    rows = [
        StorageRecord.from_values(
            video_id="v", frame_id=1, embedding=[1.0] + [0.0] * 511
        ),
        StorageRecord(video_id="v", frame_id=1, text="ocr", source="ocr"),
        StorageRecord(video_id="v", frame_id=1, text="caption", source="caption"),
    ]
    report = DualStoreIngestor(vector, text).ingest(rows)
    assert report.milvus.inserted == 1
    assert report.elasticsearch.inserted == 2


def test_batched_ingestion_aggregates_backend_reports() -> None:
    vector = RecordingStore("milvus")
    text = RecordingStore("elasticsearch")
    rows = [
        StorageRecord(video_id="v", frame_id=frame, text=str(frame), source="ocr")
        for frame in range(5)
    ]
    report = DualStoreIngestor(vector, text).ingest_batches(rows, batch_size=2)
    assert [len(batch) for batch in text.batches] == [2, 2, 1]
    assert report.logical_records == report.elasticsearch.inserted == 5
