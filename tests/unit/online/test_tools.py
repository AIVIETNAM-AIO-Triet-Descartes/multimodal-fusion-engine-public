"""Unit tests for online retrieval tool contracts (docs/04 §2)."""

import pytest

from mfe.contracts.errors import ErrorCode
from mfe.online.tools import registry
from mfe.online.tools.build_segment import build_segment
from mfe.online.tools.evidence import create_evidence_id
from mfe.online.tools.filter_metadata import filter_metadata
from mfe.online.tools.get_neighbors import get_neighbors
from mfe.online.tools.get_video_segment import get_video_segment
from mfe.online.tools.rerank import rerank
from mfe.online.tools.search_image import search_image
from mfe.online.tools.search_text import search_text


def test_evidence_id_format():
    ev_id = create_evidence_id("search_text", "run123", 1)
    assert ev_id == "search_text:run123:0001"


def test_search_text_success():
    res = search_text("red car driving on street", top_k=10, run_id="test_run")
    assert res.ok
    assert res.error is None
    assert len(res.items) > 0
    assert res.items[0].evidence_id.startswith("search_text:test_run:")


def test_search_text_topk_constraint_exceeded():
    res = search_text("query", top_k=600)
    assert not res.ok
    assert res.error is not None
    assert res.error.code == ErrorCode.INVALID_ARGUMENT
    assert "exceeds maximum constraint" in res.error.message


def test_search_text_empty_query():
    res = search_text("   ", top_k=10)
    assert not res.ok
    assert res.error is not None
    assert res.error.code == ErrorCode.INVALID_ARGUMENT


def test_search_image_validation():
    # Both provided -> error
    res_both = search_image(image_bytes=b"123", keyframe_id="L21_V001#001")
    assert not res_both.ok
    assert res_both.error.code == ErrorCode.INVALID_ARGUMENT

    # Neither provided -> error
    res_neither = search_image()
    assert not res_neither.ok
    assert res_neither.error.code == ErrorCode.INVALID_ARGUMENT

    # Keyframe provided -> ok
    res_ok = search_image(keyframe_id="L21_V001#001")
    assert res_ok.ok


def test_build_segment_single_keyframe():
    res = build_segment(["L21_V001#005"], lead_pad_ms=500, trail_pad_ms=500)
    assert res.ok
    assert len(res.items) == 1
    seg = res.items[0]
    assert seg["video_id"] == "L21_V001"
    assert seg["start_ms"] == 4500  # 5000 - 500
    assert seg["end_ms"] == 5500  # 5000 + 500


def test_build_segment_multi_keyframe():
    res = build_segment(["L21_V001#002", "L21_V001#005"], lead_pad_ms=1000, trail_pad_ms=1000)
    assert res.ok
    seg = res.items[0]
    assert seg["video_id"] == "L21_V001"
    assert seg["start_ms"] == 1000  # min(2000, 5000) - 1000
    assert seg["end_ms"] == 6000  # max(2000, 5000) + 1000


def test_build_segment_mixed_videos_error():
    res = build_segment(["L21_V001#001", "L21_V002#001"])
    assert not res.ok
    assert res.error.code == ErrorCode.INVALID_ARGUMENT


def test_get_video_segment():
    res = get_video_segment(["L21_V001#001"])
    assert res.ok
    assert "video_path" in res.items[0]
    assert "start_ms" in res.items[0]
    assert "end_ms" in res.items[0]


def test_get_neighbors():
    res = get_neighbors("L21_V001#005", window=2)
    assert res.ok
    assert len(res.items) == 4  # idx 3, 4, 6, 7
    kf_ids = [n.keyframe_id for n in res.items]
    assert "L21_V001#003" in kf_ids
    assert "L21_V001#007" in kf_ids


def test_filter_metadata():
    index = {
        "L21_V001#001": {"objects": ["person", "car"]},
        "L21_V001#002": {"objects": ["person", "bag"]},
    }
    res = filter_metadata({"object": "bag"}, metadata_index=index)
    assert res.ok
    assert res.items == ["L21_V001#002"]


def test_tool_registry():
    spec = registry.get("search_text")
    assert spec.name == "search_text"
    specs = registry.list_specs()
    assert len(specs) >= 5
