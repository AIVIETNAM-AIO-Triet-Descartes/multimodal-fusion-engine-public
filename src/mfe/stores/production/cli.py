"""Small operational CLI for schema, ingestion, query, health, and count checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .config import ElasticsearchConfig, MilvusConfig
from .contracts import StorageRecord
from .elasticsearch import ElasticsearchTextStore
from .ingestion import DualStoreIngestor
from .milvus import MilvusVectorStore


def _jsonl_records(path: Path) -> Iterable[StorageRecord]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield StorageRecord.from_values(**payload)
            except Exception as exc:
                raise ValueError(f"invalid JSONL row {line_number}: {exc}") from exc


def _validate_jsonl(path: Path) -> int:
    """First pass: validate every row/key before any backend mutation."""
    vector_keys: set[tuple[str, int]] = set()
    text_keys: set[tuple[str, int, str | None]] = set()
    count = 0
    for record in _jsonl_records(path):
        count += 1
        if record.embedding is not None:
            key = (record.video_id, record.frame_id)
            if key in vector_keys:
                raise ValueError(f"duplicate vector key before ingestion: {key}")
            vector_keys.add(key)
        if record.text is not None:
            text_key = (record.video_id, record.frame_id, record.source)
            if text_key in text_keys:
                raise ValueError(f"duplicate text key before ingestion: {text_key}")
            text_keys.add(text_key)
    return count


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ensure-schema")
    ingest = subparsers.add_parser("ingest-jsonl")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--batch-size", type=int, default=256)
    vector = subparsers.add_parser("search-vector")
    vector.add_argument("embedding_json", type=Path)
    vector.add_argument("--top-k", type=int, default=10)
    vector.add_argument("--video-id")
    text = subparsers.add_parser("search-text")
    text.add_argument("query")
    text.add_argument("--top-k", type=int, default=10)
    text.add_argument("--video-id")
    text.add_argument("--source")
    text.add_argument("--phrase", action="store_true")
    subparsers.add_parser("health")
    subparsers.add_parser("counts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    milvus = MilvusVectorStore(MilvusConfig.from_env())
    elasticsearch = ElasticsearchTextStore(ElasticsearchConfig.from_env())
    if args.command == "ensure-schema":
        milvus.ensure_schema()
        elasticsearch.ensure_schema()
        _write({"milvus": "ready", "elasticsearch": "ready"})
    elif args.command == "ingest-jsonl":
        before_checksum = _file_sha256(args.path)
        validated_count = _validate_jsonl(args.path)
        report = DualStoreIngestor(milvus, elasticsearch).ingest_batches(
            _jsonl_records(args.path), batch_size=args.batch_size
        )
        after_checksum = _file_sha256(args.path)
        if report.logical_records != validated_count or before_checksum != after_checksum:
            raise RuntimeError("JSONL changed between validation and ingestion passes")
        _write(report)
        return int(report.milvus.failed > 0 or report.elasticsearch.failed > 0)
    elif args.command == "search-vector":
        embedding = json.loads(args.embedding_json.read_text(encoding="utf-8"))
        _write(
            [
                hit.model_dump(mode="json")
                for hit in milvus.search(
                    embedding, top_k=args.top_k, video_id=args.video_id
                )
            ]
        )
    elif args.command == "search-text":
        _write(
            [
                hit.model_dump(mode="json")
                for hit in elasticsearch.search(
                    args.query,
                    top_k=args.top_k,
                    video_id=args.video_id,
                    source=args.source,
                    phrase=args.phrase,
                )
            ]
        )
    elif args.command == "health":
        _write(
            {
                "milvus": milvus.health().model_dump(mode="json"),
                "elasticsearch": elasticsearch.health().model_dump(mode="json"),
            }
        )
    elif args.command == "counts":
        _write({"milvus": milvus.count(), "elasticsearch": elasticsearch.count()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
