"""Unit tests for the custom-keyframe audit validator."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import (
    build_keyframe_id,
    frame_idx_from_pts,
    pts_time_to_timestamp_ms,
)
from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.keyframe_audit import (
    FRAME_ERROR_TOLERANCE,
    MIN_INTERPOLATION_GAP_FRAMES,
    audit_custom_keyframe,
)
from mfe.mapping.relational import RelationalMap


def _keyframe(
    video_id: str,
    keyframe_index: int,
    frame_idx: int,
    pts_time: float,
) -> KeyframeRow:
    return KeyframeRow(
        keyframe_id=build_keyframe_id(video_id, frame_idx),
        video_id=video_id,
        keyframe_index=keyframe_index,
        frame_idx=frame_idx,
        pts_time=pts_time,
        timestamp_ms=pts_time_to_timestamp_ms(pts_time),
    )


@pytest.fixture
def audit_db() -> Iterator[RelationalMap]:
    """Map with real 29.97-fps rows and a clearly labelled synthetic edge case."""
    database = RelationalMap(":memory:")
    database.init_schema()

    database.upsert_video(
        VideoRow(
            video_id="L25_V052",
            source_path="fake/L25_V052.mp4",
            fps=29.97,
            duration_ms=1_800_000,
            total_frames=53_946,
            width=1280,
            height=720,
        )
    )
    database.upsert_keyframes(
        [
            # Literal rows from data/sample_01/map-keyframes/L25_V052.csv.
            _keyframe("L25_V052", 3, 82, 2.73607),
            _keyframe("L25_V052", 4, 149, 5.005),
            _keyframe("L25_V052", 291, 29_999, 1001.0),
            _keyframe("L25_V052", 292, 30_149, 1006.0),
        ]
    )

    # This video is deliberately synthetic.  Every provided BTC mapping starts
    # at pts_time=0 and no measured adjacent gap exceeds 300 frames, so the
    # missing-boundary and sparse-evidence branches cannot be represented by a
    # literal official pair.  A synthetic ID avoids attributing invented rows to
    # a real video such as L21_V001.
    database.upsert_video(
        VideoRow(
            video_id="EDGE_CASE",
            source_path="fake/edge-case.mp4",
            fps=25.0,
            duration_ms=100_000,
            total_frames=2500,
            width=1280,
            height=720,
        )
    )
    database.upsert_keyframes(
        [
            _keyframe("EDGE_CASE", 1, 250, 10.0),
            _keyframe("EDGE_CASE", 2, 750, 30.0),
        ]
    )

    yield database
    database.close()


def test_fps_mismatch_inside_bracket_triggers_interpolation_warning(
    audit_db: RelationalMap,
) -> None:
    """30.0 instead of 29.97 stays in a real bracket but is wrong by 31 frames."""
    pts_time = (1001.0 + 1006.0) / 2.0
    result = audit_custom_keyframe(
        relational_map=audit_db,
        video_id="L25_V052",
        pts_time=pts_time,
        candidate_frame_idx=frame_idx_from_pts(pts_time, 30.0),
    )

    assert result.valid
    assert not result.confident
    assert result.failures == []
    assert len(result.warnings) == 1
    assert result.warnings[0].code == ErrorCode.FRAME_INCONSISTENT
    assert result.warnings[0].rule == "Tỉ lệ nội suy"
    assert "frame_error 31.000" in result.warnings[0].message


def test_correct_fps_has_no_false_positive(audit_db: RelationalMap) -> None:
    """The correct 29.97 FPS produces no warning inside the same real bracket."""
    result = audit_custom_keyframe(audit_db, "L25_V052", pts_time=1003.5)

    assert result.valid
    assert result.confident
    assert result.warnings == []


class _FixedBracketMap:
    """Small read-only map for brackets SQLite intentionally cannot store."""

    def __init__(
        self,
        video: VideoRow,
        previous: KeyframeRow | None,
        following: KeyframeRow | None,
    ) -> None:
        self.video = video
        self.previous = previous
        self.following = following

    def get_video(self, video_id: str) -> VideoRow:
        assert video_id == self.video.video_id
        return self.video

    def find_bracketing_keyframes(
        self, video_id: str, pts_time: float
    ) -> tuple[KeyframeRow | None, KeyframeRow | None]:
        assert video_id == self.video.video_id
        return self.previous, self.following


class _BrokenVideoMap:
    """Reader that exposes a non-NOT_FOUND storage failure unchanged."""

    def get_video(self, video_id: str) -> VideoRow:
        raise ContractError(ErrorCode.STORE_UNAVAILABLE, f"cannot read {video_id}")

    def find_bracketing_keyframes(
        self, video_id: str, pts_time: float
    ) -> tuple[KeyframeRow | None, KeyframeRow | None]:
        raise AssertionError("bracket lookup must not run when video lookup fails")


def test_duplicate_adjacent_frame_indices_do_not_divide_by_zero() -> None:
    """Reproduce one of 614 raw adjacent pairs sharing frame_idx zero."""
    video = VideoRow(
        video_id="L21_V006",
        source_path="fake/L21_V006.mp4",
        fps=30.0,
        duration_ms=60_000,
        total_frames=1800,
        width=1280,
        height=720,
    )
    previous = _keyframe("L21_V006", 1, 0, 0.0)
    following = _keyframe("L21_V006", 2, 0, 0.0333333)
    relational_map = _FixedBracketMap(video, previous, following)

    result = audit_custom_keyframe(relational_map, "L21_V006", pts_time=0.016)

    assert result.valid
    assert result.confident
    assert result.warnings == []


def test_non_not_found_video_error_is_not_hidden() -> None:
    with pytest.raises(ContractError) as raised:
        audit_custom_keyframe(_BrokenVideoMap(), "UNAVAILABLE", pts_time=1.0)

    assert raised.value.code == ErrorCode.STORE_UNAVAILABLE


def test_one_frame_gap_falls_back_to_inclusive_bounds() -> None:
    """A tiny denominator adds no information beyond the range check."""
    assert MIN_INTERPOLATION_GAP_FRAMES == 1
    video = VideoRow(
        video_id="DENSE",
        source_path="fake/dense.mp4",
        fps=30.0,
        duration_ms=1000,
        total_frames=30,
        width=640,
        height=360,
    )
    relational_map = _FixedBracketMap(
        video,
        _keyframe("DENSE", 1, 0, 0.0),
        _keyframe("DENSE", 2, 1, 0.034),
    )

    result = audit_custom_keyframe(relational_map, "DENSE", pts_time=0.02)

    assert result.valid
    assert result.confident
    assert result.warnings == []


def test_one_frame_gap_honors_a_stricter_custom_tolerance() -> None:
    """The small-gap shortcut must not silently override a caller's tolerance."""
    video = VideoRow(
        video_id="DENSE_STRICT",
        source_path="fake/dense-strict.mp4",
        fps=30.0,
        duration_ms=1000,
        total_frames=30,
        width=640,
        height=360,
    )
    relational_map = _FixedBracketMap(
        video,
        _keyframe("DENSE_STRICT", 1, 0, 0.0),
        _keyframe("DENSE_STRICT", 2, 1, 0.034),
    )

    result = audit_custom_keyframe(
        relational_map,
        "DENSE_STRICT",
        pts_time=0.02,
        frame_error_tolerance=0.5,
    )

    assert result.valid
    assert not result.confident
    assert any("interpolation frame_error" in warning.message for warning in result.warnings)


def test_missing_previous_boundary_is_a_warning(audit_db: RelationalMap) -> None:
    result = audit_custom_keyframe(audit_db, "EDGE_CASE", pts_time=5.0)

    assert result.valid
    assert not result.confident
    assert any("before first keyframe" in warning.message for warning in result.warnings)


def test_missing_following_boundary_is_a_warning(audit_db: RelationalMap) -> None:
    result = audit_custom_keyframe(audit_db, "EDGE_CASE", pts_time=35.0)

    assert result.valid
    assert not result.confident
    assert any("after last keyframe" in warning.message for warning in result.warnings)


def test_sparse_bracket_is_a_warning(audit_db: RelationalMap) -> None:
    result = audit_custom_keyframe(audit_db, "EDGE_CASE", pts_time=20.0)

    assert result.valid
    assert not result.confident
    assert any("large gap between bracketing keyframes" in w.message for w in result.warnings)


def test_exact_official_keyframe_match_is_confident(audit_db: RelationalMap) -> None:
    result = audit_custom_keyframe(audit_db, "EDGE_CASE", pts_time=10.0)

    assert result.valid
    assert result.confident
    assert result.warnings == []


def test_exact_official_keyframe_mismatch_is_blocked(audit_db: RelationalMap) -> None:
    result = audit_custom_keyframe(
        audit_db,
        "EDGE_CASE",
        pts_time=10.0,
        candidate_frame_idx=251,
    )

    assert not result.valid
    assert result.blocked
    assert result.failures[0].code == ErrorCode.FRAME_INCONSISTENT


@pytest.mark.parametrize("candidate", [249, 751])
def test_candidate_outside_official_bracket_is_blocked(
    audit_db: RelationalMap,
    candidate: int,
) -> None:
    result = audit_custom_keyframe(
        audit_db,
        "EDGE_CASE",
        pts_time=20.0,
        candidate_frame_idx=candidate,
    )

    assert not result.valid
    assert result.failures[0].code == ErrorCode.FRAME_INCONSISTENT
    assert not any(warning.rule == "Tỉ lệ nội suy" for warning in result.warnings)


def test_video_and_candidate_bounds_are_blocking(audit_db: RelationalMap) -> None:
    missing = audit_custom_keyframe(audit_db, "MISSING_VIDEO", pts_time=10.0)
    bad_timestamp = audit_custom_keyframe(audit_db, "EDGE_CASE", pts_time=200.0)
    bad_frame = audit_custom_keyframe(
        audit_db,
        "EDGE_CASE",
        pts_time=20.0,
        candidate_frame_idx=2500,
    )

    assert missing.failures[0].code == ErrorCode.VIDEO_NOT_FOUND
    assert bad_timestamp.failures[0].code == ErrorCode.TIMESTAMP_OUT_OF_BOUNDS
    assert bad_frame.failures[0].code == ErrorCode.FRAME_OUT_OF_BOUNDS
    assert missing.blocked and bad_timestamp.blocked and bad_frame.blocked


def test_tolerance_is_inclusive(audit_db: RelationalMap) -> None:
    """A real midpoint with exactly 0.5 frame error is inside the tolerance."""
    pts_time = (2.73607 + 5.005) / 2.0
    result = audit_custom_keyframe(
        audit_db,
        "L25_V052",
        pts_time=pts_time,
        frame_error_tolerance=0.5,
    )

    assert result.confident
    assert FRAME_ERROR_TOLERANCE == 1.0


@pytest.mark.parametrize(
    "tolerance", [-0.1, float("nan"), float("inf"), True, "1.0"]
)
def test_invalid_tolerance_raises_contract_error(
    audit_db: RelationalMap,
    tolerance: object,
) -> None:
    with pytest.raises(ContractError) as raised:
        audit_custom_keyframe(
            audit_db,
            "EDGE_CASE",
            pts_time=20.0,
            frame_error_tolerance=tolerance,
        )

    assert raised.value.code == ErrorCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    "pts_time", [-0.001, float("nan"), float("inf"), True, "1.0"]
)
def test_invalid_timestamp_is_blocked_without_deriving_a_frame(
    audit_db: RelationalMap,
    pts_time: object,
) -> None:
    if isinstance(pts_time, (bool, str)):
        with pytest.raises(ContractError) as raised:
            audit_custom_keyframe(audit_db, "EDGE_CASE", pts_time=pts_time)
        assert raised.value.code == ErrorCode.INVALID_ARGUMENT
        return

    result = audit_custom_keyframe(audit_db, "EDGE_CASE", pts_time=pts_time)

    assert result.blocked
    assert result.failures[0].code == ErrorCode.TIMESTAMP_OUT_OF_BOUNDS


@pytest.mark.parametrize("candidate", [500.5, True, "500"])
def test_non_integer_candidate_frame_is_rejected(
    audit_db: RelationalMap,
    candidate: object,
) -> None:
    with pytest.raises(ContractError) as raised:
        audit_custom_keyframe(
            audit_db,
            "EDGE_CASE",
            pts_time=20.0,
            candidate_frame_idx=candidate,
        )

    assert raised.value.code == ErrorCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    ("pts_time", "candidate"),
    [(5.0, 300), (35.0, 700)],
)
def test_open_boundary_still_blocks_violation_against_known_side(
    audit_db: RelationalMap,
    pts_time: float,
    candidate: int,
) -> None:
    result = audit_custom_keyframe(
        audit_db,
        "EDGE_CASE",
        pts_time=pts_time,
        candidate_frame_idx=candidate,
    )

    assert result.blocked
    assert result.failures[0].code == ErrorCode.FRAME_INCONSISTENT


def test_non_positive_timestamp_gap_warns_instead_of_dividing() -> None:
    video = VideoRow(
        video_id="BAD_BRACKET",
        source_path="fake/bad-bracket.mp4",
        fps=25.0,
        duration_ms=100_000,
        total_frames=2500,
        width=640,
        height=360,
    )
    relational_map = _FixedBracketMap(
        video,
        _keyframe("BAD_BRACKET", 1, 250, 20.0),
        _keyframe("BAD_BRACKET", 2, 750, 10.0),
    )

    result = audit_custom_keyframe(
        relational_map,
        "BAD_BRACKET",
        pts_time=15.0,
        candidate_frame_idx=500,
    )

    assert result.valid
    assert not result.confident
    assert any("non-positive timestamp gap" in warning.message for warning in result.warnings)
