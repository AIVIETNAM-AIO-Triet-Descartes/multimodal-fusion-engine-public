"""Reversible, collision-free physical IDs derived from logical storage keys."""

from __future__ import annotations

import base64
import json


def _encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True).decode("utf-8")


def milvus_primary_key(video_id: str, frame_id: int) -> str:
    return f"v1:{_encode(video_id)}:{frame_id}"


def parse_milvus_primary_key(value: str) -> tuple[str, int]:
    version, encoded_video, frame = value.split(":", 2)
    if version != "v1":
        raise ValueError("unsupported Milvus primary-key version")
    return _decode(encoded_video), int(frame)


def elasticsearch_document_id(video_id: str, frame_id: int, source: str) -> str:
    return f"t1:{_encode(video_id)}:{frame_id}:{_encode(source)}"


def parse_elasticsearch_document_id(value: str) -> tuple[str, int, str]:
    version, encoded_video, frame, encoded_source = value.split(":", 3)
    if version != "t1":
        raise ValueError("unsupported Elasticsearch document-ID version")
    return _decode(encoded_video), int(frame), _decode(encoded_source)


def milvus_string_literal(value: str) -> str:
    """Return a JSON-compatible quoted literal for a Milvus filter expression."""
    return json.dumps(value, ensure_ascii=False)
