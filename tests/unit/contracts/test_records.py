"""Locks the 4-store record schemas — docs/00 §4.

The point of these models is to make risk #1 in docs/00 §8 ("Lệch schema giữa
các người") fail loudly at the boundary. These tests check the failure actually
fires.
"""

import math

import pytest
from pydantic import ValidationError

from mfe.contracts.records import (
    AssetRecord,
    AssetType,
    InvertedRecord,
    KeyframeRow,
    VectorRecord,
    VideoRow,
)


def _unit_vector(dim: int = 4) -> list[float]:
    return [1 / math.sqrt(dim)] * dim


class TestVideoRow:
    def test_accepts_docs_example(self):
        # docs/05 §1.1(c)
        row = VideoRow(
            video_id="L21_V001",
            source_path="data/raw/L21_V001.mp4",
            fps=29.97,
            duration_ms=600000,
            total_frames=17982,
            width=1280,
            height=720,
        )
        assert row.fps == 29.97

    def test_rejects_zero_fps(self):
        """A zero fps means probing failed; letting it through corrupts every
        derived timestamp (docs/00 §7 "Không hardcode FPS")."""
        with pytest.raises(ValidationError):
            VideoRow(
                video_id="L21_V001",
                source_path="data/raw/L21_V001.mp4",
                fps=0,
                duration_ms=600000,
                total_frames=17982,
                width=1280,
                height=720,
            )


def _keyframe(**overrides):
    """A valid KeyframeRow (docs/00 §3 example), with per-test overrides."""
    fields = dict(
        keyframe_id="L21_V001#000090",
        video_id="L21_V001",
        keyframe_index=2,
        frame_idx=90,
        pts_time=3.0,
        timestamp_ms=3000,
    )
    fields.update(overrides)
    return KeyframeRow(**fields)


class TestKeyframeRow:
    def test_accepts_docs_example(self):
        # docs/00 §3
        row = _keyframe(shot_id="L21_V001#s02")
        assert row.frame_idx == 90
        assert row.pts_time == 3.0
        assert row.shot_id == "L21_V001#s02"

    def test_shot_id_is_optional_in_sprint_1(self):
        assert _keyframe().shot_id is None

    def test_rejects_id_disagreeing_with_video_id(self):
        """docs/02 §4: "sai một chỗ là cả pipeline lệch"."""
        with pytest.raises(ValidationError, match="encodes video_id"):
            _keyframe(video_id="L21_V002")

    def test_rejects_id_disagreeing_with_frame_idx(self):
        with pytest.raises(ValidationError, match="encodes frame_idx"):
            _keyframe(frame_idx=91)

    def test_rejects_malformed_keyframe_id(self):
        with pytest.raises(ValidationError):
            _keyframe(keyframe_id="L21_V001#90")  # under-padded

    def test_ordinal_does_not_have_to_match_the_id(self):
        """The ordinal is a column now, not an identity.

        Row n=2 of L21_V001 is frame 90, and the id spells the frame. Changing
        the ordinal alone must not be rejected — a re-extraction renumbers every
        ordinal while the frames stay put, and that is the whole reason the key
        moved off the ordinal.
        """
        row = _keyframe(keyframe_index=7)
        assert row.keyframe_index == 7
        assert row.frame_idx == 90
        assert row.keyframe_id == "L21_V001#000090"


class TestVectorRecord:
    def test_accepts_normalized_embedding(self):
        record = VectorRecord(
            internal_id=0,
            keyframe_id="L21_V001#000000",
            video_id="L21_V001",
            embedding=_unit_vector(),
        )
        assert record.internal_id == 0

    def test_rejects_unnormalized_embedding(self):
        """docs/00 §4.1: "Vector CLIP đã chuẩn hóa L2". An unnormalized vector
        makes cosine scores incomparable and silently degrades every search."""
        with pytest.raises(ValidationError, match="L2-normalized"):
            VectorRecord(
                internal_id=0,
                keyframe_id="L21_V001#000000",
                video_id="L21_V001",
                embedding=[1.0, 1.0, 1.0, 1.0],  # norm 2.0
            )

    def test_rejects_empty_embedding(self):
        with pytest.raises(ValidationError):
            VectorRecord(
                internal_id=0,
                keyframe_id="L21_V001#000000",
                video_id="L21_V001",
                embedding=[],
            )


class TestInvertedRecord:
    def test_text_fields_default_empty(self):
        """OCR off => empty ocr_text, not a missing field (docs/03 §2.1)."""
        record = InvertedRecord(keyframe_id="L21_V002#000118", video_id="L21_V002")
        assert record.ocr_text == ""
        assert record.asr_text == ""
        assert record.object_labels == []

    def test_accepts_docs_example(self):
        # docs/03 §2.1(c)
        record = InvertedRecord(
            keyframe_id="L21_V001#000025",
            video_id="L21_V001",
            object_labels=["person", "handbag", "car"],
            ocr_text="BẢN TIN 18H",
            metadata_text="Thời sự tối 12/03...",
        )
        assert "handbag" in record.object_labels


class TestAssetRecord:
    def test_accepts_both_docs_examples(self):
        # docs/03 §3(c): one keyframe, two asset rows.
        image = AssetRecord(
            keyframe_id="L21_V001#000025",
            asset_type=AssetType.IMAGE,
            asset_path="data/keyframes/L21_V001/025.jpg",
            width=1280,
            height=720,
            byte_size=148213,
        )
        segment = AssetRecord(
            keyframe_id="L21_V001#000025",
            asset_type=AssetType.VIDEO_SEGMENT,
            asset_path="data/segments/L21_V001/025.mp4",
            width=1280,
            height=720,
            byte_size=1043998,
        )
        assert image.asset_type != segment.asset_type

    def test_rejects_unknown_asset_type(self):
        with pytest.raises(ValidationError):
            AssetRecord(
                keyframe_id="L21_V001#000025",
                asset_type="thumbnail",
                asset_path="x.jpg",
                width=1,
                height=1,
                byte_size=0,
            )


def test_records_are_frozen():
    """Records travel across workstream boundaries; mutation in transit would
    be untraceable."""
    row = _keyframe()
    with pytest.raises(ValidationError):
        row.frame_idx = 999


def test_records_reject_unknown_fields():
    """extra="forbid" turns a typo'd field name into an error rather than a
    silently dropped column."""
    with pytest.raises(ValidationError):
        _keyframe(frameidx=90)  # typo of frame_idx
