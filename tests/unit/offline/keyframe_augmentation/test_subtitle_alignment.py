from __future__ import annotations

import pytest

from mfe.offline.keyframe_augmentation.subtitle_alignment import (
    SubtitleAlignmentStatus,
    align_supplemental_subtitles,
    build_btc_subtitle_windows,
)


def _mapping() -> list[dict[str, object]]:
    return [
        {"n": "1", "frame_idx": "10", "pts_time": "0.4", "fps": "25"},
        {"n": "2", "frame_idx": "20", "pts_time": "0.8", "fps": "25"},
        {"n": "3", "frame_idx": "40", "pts_time": "1.6", "fps": "25"},
    ]


def test_aligns_by_literal_btc_frame_window() -> None:
    windows = build_btc_subtitle_windows("L21_V001", _mapping())
    result = align_supplemental_subtitles(
        "L21_V001",
        [
            {"keyframe_id": "L21_V001#000015", "frame_idx": 15},
            {"keyframe_id": "L21_V001#000039", "frame_idx": 39},
            {"keyframe_id": "L21_V001#000041", "frame_idx": 41},
        ],
        windows,
        {
            "L21_V001#000010": " first window ",
            "L21_V001#000020": "second window",
            "L21_V001#000040": "last window",
        },
    )

    assert result.subtitles == {
        "L21_V001#000015": "first window",
        "L21_V001#000039": "second window",
        "L21_V001#000041": "last window",
    }
    assert result.audit_rows[1]["source_window_end_frame_idx_exclusive"] == 40
    assert {row["status"] for row in result.audit_rows} == {
        SubtitleAlignmentStatus.ALIGNED.value
    }


def test_records_before_first_and_silent_windows_without_fake_text() -> None:
    windows = build_btc_subtitle_windows("L21_V001", _mapping())
    result = align_supplemental_subtitles(
        "L21_V001",
        [
            {"keyframe_id": "L21_V001#000005", "frame_idx": 5},
            {"keyframe_id": "L21_V001#000025", "frame_idx": 25},
        ],
        windows,
        {"L21_V001#000010": "speech"},
    )

    assert result.subtitles == {}
    assert [row["status"] for row in result.audit_rows] == [
        SubtitleAlignmentStatus.BEFORE_FIRST_BTC.value,
        SubtitleAlignmentStatus.SOURCE_WINDOW_SILENT.value,
    ]


@pytest.mark.parametrize(
    "mapping",
    [
        [{"n": 2, "frame_idx": 10}],
        [{"n": 1, "frame_idx": 20}, {"n": 2, "frame_idx": 10}],
        [{"n": 1, "frame_idx": 10}, {"n": 2, "frame_idx": 10}],
    ],
)
def test_rejects_invalid_btc_mapping(mapping: list[dict[str, int]]) -> None:
    with pytest.raises(ValueError):
        build_btc_subtitle_windows("L21_V001", mapping)


def test_rejects_bad_or_duplicate_supplemental_identity() -> None:
    windows = build_btc_subtitle_windows("L21_V001", _mapping())
    with pytest.raises(ValueError, match="keyframe_id"):
        align_supplemental_subtitles(
            "L21_V001",
            [{"keyframe_id": "L21_V001#000016", "frame_idx": 15}],
            windows,
            {},
        )
    with pytest.raises(ValueError, match="overlaps BTC"):
        align_supplemental_subtitles(
            "L21_V001",
            [{"keyframe_id": "L21_V001#000020", "frame_idx": 20}],
            windows,
            {},
        )
