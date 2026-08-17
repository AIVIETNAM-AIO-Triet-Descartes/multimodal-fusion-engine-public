from __future__ import annotations

import unicodedata

import numpy as np
import pytest
from pydantic import ValidationError

from mfe.stores.production.contracts import StorageRecord
from mfe.stores.production.ids import (
    elasticsearch_document_id,
    milvus_primary_key,
    parse_elasticsearch_document_id,
    parse_milvus_primary_key,
)


def vector(value: float = 1.0) -> np.ndarray:
    return np.full(512, value, dtype=np.float16)


def test_float16_embedding_is_upcast_and_validated() -> None:
    record = StorageRecord.from_values(video_id="L01_V001", frame_id=7, embedding=vector())
    assert len(record.embedding or ()) == 512
    assert isinstance(record.embedding[0], float)  # type: ignore[index]


def test_text_is_normalized_to_nfc_before_ingestion() -> None:
    nfd = unicodedata.normalize("NFD", "Thành phố")
    record = StorageRecord.from_values(
        video_id="L01_V001", frame_id=7, text=nfd, source="ocr"
    )
    assert record.text == "Thành phố"
    assert unicodedata.is_normalized("NFC", record.text)


@pytest.mark.parametrize(
    "values",
    [
        {"video_id": "x", "frame_id": 0},
        {"video_id": "x", "frame_id": 0, "text": "hello"},
        {"video_id": "x", "frame_id": 0, "text": " ", "source": "ocr"},
        {"video_id": "x", "frame_id": 2, "embedding": tuple([1.0] * 511)},
        {"video_id": "x", "frame_id": 2, "embedding": tuple([0.0] * 512)},
        {"video_id": "x", "frame_id": 2, "embedding": tuple([float("nan")] * 512)},
        {"video_id": "x", "frame_id": 2, "embedding": tuple([1.0] * 512), "sv": 3, "ev": 9},
    ],
)
def test_invalid_rows_fail_before_backend(values: dict[str, object]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        StorageRecord.model_validate(values)


def test_ids_round_trip_unicode_and_delimiters_without_collision() -> None:
    logical = ("video:đặc/biệt", 123)
    milvus_id = milvus_primary_key(*logical)
    text_id = elasticsearch_document_id(*logical, "ocr")
    assert parse_milvus_primary_key(milvus_id) == logical
    assert parse_elasticsearch_document_id(text_id) == (*logical, "ocr")
    assert milvus_primary_key("ab", 12) != milvus_primary_key("ab1", 2)
