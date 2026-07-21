"""Locks the ID model — docs/00 §3, docs/adr/0002-id-format.md.

This is the P0 test. docs/00 §1 requires "Mapping ID<->frame<->timestamp đúng
100%", and docs/context/buoi-1.md §8 puts it first among the places teams lose
points: "tìm đúng hình nhưng submit sai ID/frame là lỗi trắng".
"""

import pytest

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import (
    build_keyframe_id,
    build_shot_id,
    parse_keyframe_id,
    parse_shot_id,
    pts_time_to_timestamp_ms,
)


class TestBuildKeyframeId:
    @pytest.mark.parametrize(
        ("video_id", "keyframe_index", "expected"),
        [
            ("L21_V001", 1, "L21_V001#001"),  # docs/00 §3
            ("L21_V001", 250, "L21_V001#250"),
            ("L21_V001", 0, "L21_V001#000"),  # boundary
            ("L30_V099", 91, "L30_V099#091"),
        ],
    )
    def test_matches_docs_format(self, video_id, keyframe_index, expected):
        assert build_keyframe_id(video_id, keyframe_index) == expected

    def test_index_over_999_uses_more_digits(self):
        """3 is a minimum pad width, not a ceiling — a long video keeps working."""
        assert build_keyframe_id("L21_V001", 1234) == "L21_V001#1234"

    def test_rejects_negative_index(self):
        with pytest.raises(ContractError) as exc:
            build_keyframe_id("L21_V001", -1)
        assert exc.value.code is ErrorCode.INVALID_ID

    @pytest.mark.parametrize("bad_video_id", ["", "L21 V001", "L21_V001#x"])
    def test_rejects_unusable_video_id(self, bad_video_id):
        """docs/00 §7: keyframe_id never contains whitespace; a '#' would make
        the id ambiguous to parse."""
        with pytest.raises(ContractError) as exc:
            build_keyframe_id(bad_video_id, 1)
        assert exc.value.code is ErrorCode.INVALID_ID


class TestParseKeyframeId:
    @pytest.mark.parametrize(
        ("keyframe_id", "expected"),
        [
            ("L21_V001#001", ("L21_V001", 1)),
            ("L21_V001#000", ("L21_V001", 0)),
            ("L21_V001#250", ("L21_V001", 250)),
        ],
    )
    def test_matches_docs_examples(self, keyframe_id, expected):
        assert parse_keyframe_id(keyframe_id) == expected

    def test_strips_zero_padding_to_int(self):
        _, index = parse_keyframe_id("L21_V001#001")
        assert index == 1
        assert isinstance(index, int)

    @pytest.mark.parametrize(
        "malformed",
        [
            "L21_V001",  # no index part
            "L21_V001#1",  # under-padded (< 3 digits)
            "L21_V001#12",  # under-padded
            "L21_V001#abc",  # non-numeric
            "L21_V001#s07",  # that's a shot_id, not a keyframe_id
            "#001",  # no video_id
        ],
    )
    def test_rejects_malformed(self, malformed):
        with pytest.raises(ContractError) as exc:
            parse_keyframe_id(malformed)
        assert exc.value.code is ErrorCode.INVALID_ID

    @pytest.mark.parametrize("keyframe_index", [0, 1, 91, 250, 999, 1234])
    def test_round_trips(self, keyframe_index):
        video_id = "L21_V001"
        assert parse_keyframe_id(build_keyframe_id(video_id, keyframe_index)) == (
            video_id,
            keyframe_index,
        )


class TestShotId:
    def test_matches_docs_example(self):
        # docs/00 §3: L21_V001#s07
        assert build_shot_id("L21_V001", 7) == "L21_V001#s07"

    def test_round_trips(self):
        assert parse_shot_id(build_shot_id("L21_V002", 5)) == ("L21_V002", 5)

    def test_shot_id_is_not_a_keyframe_id(self):
        """The two schemes share a '#' separator; they must not be confusable."""
        with pytest.raises(ContractError):
            parse_keyframe_id(build_shot_id("L21_V001", 7))


class TestPtsTimeToTimestampMs:
    """The one sanctioned time conversion: seconds -> milliseconds.

    There is deliberately NO frame<->time derivation. docs/adr/0002-id-format.md:
    `round(pts_time * fps)` was measured wrong on 83-93% of 29.97fps rows, so
    frame_idx and pts_time are read from the CSV, never computed.
    """

    @pytest.mark.parametrize(
        ("pts_time", "expected"),
        [
            (0.0, 0),
            (3.0, 3000),  # docs/00 §3 example
            (41.174, 41174),
            (47.2, 47200),
        ],
    )
    def test_converts(self, pts_time, expected):
        assert pts_time_to_timestamp_ms(pts_time) == expected

    def test_rejects_negative(self):
        with pytest.raises(ContractError) as exc:
            pts_time_to_timestamp_ms(-1.0)
        assert exc.value.code is ErrorCode.INVALID_ID
