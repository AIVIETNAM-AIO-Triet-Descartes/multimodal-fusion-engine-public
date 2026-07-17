"""Locks the ID model — docs/00 §3.

This is the P0 test. docs/00 §1 requires "Mapping ID<->frame<->timestamp đúng
100%", and docs/context/buoi-1.md §8 puts it first among the places teams lose
points: "tìm đúng hình nhưng submit sai ID/frame là lỗi trắng".
"""

import pytest

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import (
    MAX_FRAME_NUMBER,
    build_keyframe_id,
    build_shot_id,
    frame_to_timestamp_ms,
    parse_keyframe_id,
    parse_shot_id,
    timestamp_ms_to_frame,
)


class TestBuildKeyframeId:
    @pytest.mark.parametrize(
        ("video_id", "frame_number", "expected"),
        [
            # Every example that appears in the docs.
            ("L01_V001", 1234, "L01_V001#001234"),  # docs/00 §3
            ("L01_V001", 250, "L01_V001#000250"),  # docs/03 §1(c)
            ("L01_V001", 0, "L01_V001#000000"),  # docs/03 §1(c) — boundary
            ("L01_V002", 1180, "L01_V002#001180"),  # docs/03 §1(c)
        ],
    )
    def test_matches_docs_examples(self, video_id, frame_number, expected):
        assert build_keyframe_id(video_id, frame_number) == expected

    def test_rejects_negative_frame(self):
        with pytest.raises(ContractError) as exc:
            build_keyframe_id("L01_V001", -1)
        assert exc.value.code is ErrorCode.INVALID_ID

    def test_rejects_frame_beyond_padding_width(self):
        """Silently widening the field would re-key every store."""
        with pytest.raises(ContractError) as exc:
            build_keyframe_id("L01_V001", MAX_FRAME_NUMBER + 1)
        assert exc.value.code is ErrorCode.INVALID_ID

    @pytest.mark.parametrize("bad_video_id", ["", "L01 V001", "L01_V001#x"])
    def test_rejects_unusable_video_id(self, bad_video_id):
        """docs/00 §7: keyframe_id never contains whitespace. A '#' would make
        the id ambiguous to parse."""
        with pytest.raises(ContractError) as exc:
            build_keyframe_id(bad_video_id, 1)
        assert exc.value.code is ErrorCode.INVALID_ID


class TestParseKeyframeId:
    @pytest.mark.parametrize(
        ("keyframe_id", "expected"),
        [
            ("L01_V001#001234", ("L01_V001", 1234)),
            ("L01_V001#000000", ("L01_V001", 0)),
            ("L01_V002#001180", ("L01_V002", 1180)),
        ],
    )
    def test_matches_docs_examples(self, keyframe_id, expected):
        assert parse_keyframe_id(keyframe_id) == expected

    def test_strips_zero_padding_to_int(self):
        """'000250' must come back as 250, not '000250' and not 250.0."""
        _, frame_number = parse_keyframe_id("L01_V001#000250")
        assert frame_number == 250
        assert isinstance(frame_number, int)

    @pytest.mark.parametrize(
        "malformed",
        [
            "L01_V001",  # no frame part
            "L01_V001#1234",  # unpadded
            "L01_V001#0001234",  # over-padded
            "L01_V001#abcdef",  # non-numeric
            "L01_V001#s07",  # that's a shot_id, not a keyframe_id
            "#001234",  # no video_id
        ],
    )
    def test_rejects_malformed(self, malformed):
        with pytest.raises(ContractError) as exc:
            parse_keyframe_id(malformed)
        assert exc.value.code is ErrorCode.INVALID_ID

    @pytest.mark.parametrize("frame_number", [0, 1, 250, 999, 1234, MAX_FRAME_NUMBER])
    def test_round_trips(self, frame_number):
        video_id = "L01_V001"
        assert parse_keyframe_id(build_keyframe_id(video_id, frame_number)) == (
            video_id,
            frame_number,
        )


class TestShotId:
    def test_matches_docs_example(self):
        # docs/00 §3 and docs/05 §1.2(c): L01_V001#s07
        assert build_shot_id("L01_V001", 7) == "L01_V001#s07"

    def test_round_trips(self):
        assert parse_shot_id(build_shot_id("L01_V002", 5)) == ("L01_V002", 5)

    def test_shot_id_is_not_a_keyframe_id(self):
        """The two schemes share a '#' separator; they must not be confusable."""
        with pytest.raises(ContractError):
            parse_keyframe_id(build_shot_id("L01_V001", 7))


class TestFrameToTimestamp:
    """The normative formula is docs/00 §3: round(frame_number / fps * 1000).

    Two worked examples in the docs disagree with it. See
    docs/adr/0002-id-format.md — the formula is normative, the examples are
    illustrative, and the examples are the ones being corrected.
    """

    @pytest.mark.parametrize(
        ("frame_number", "fps", "expected"),
        [
            # Agrees with docs/05 §1.2(c).
            (250, 29.97, 8342),
            (1180, 25.00, 47200),
            # Boundary.
            (0, 29.97, 0),
        ],
    )
    def test_agrees_with_docs_examples(self, frame_number, fps, expected):
        assert frame_to_timestamp_ms(frame_number, fps) == expected

    def test_disagrees_with_two_stale_docs_examples(self):
        """1234 @ 29.97fps = 41174.508ms -> round() -> 41175.

        docs/05 §1.2(c) says 41174 (that is floor, and it contradicts the 8342
        on the row above it, which is round). docs/00 §3 says 41400, which no
        formula produces. This test pins the formula; the docs are corrected to
        match it, not the other way round.
        """
        assert frame_to_timestamp_ms(1234, 29.97) == 41175

    def test_rejects_zero_or_negative_fps(self):
        """fps comes from per-video metadata and is never assumed
        (docs/00 §7, docs/05 §1.1). A zero here means probing failed."""
        for bad_fps in (0.0, -29.97):
            with pytest.raises(ContractError) as exc:
                frame_to_timestamp_ms(100, bad_fps)
            assert exc.value.code is ErrorCode.INVALID_ID

    def test_rejects_negative_frame(self):
        with pytest.raises(ContractError):
            frame_to_timestamp_ms(-1, 29.97)


class TestTimestampToFrame:
    def test_inverts_on_exact_fps(self):
        assert timestamp_ms_to_frame(47200, 25.0) == 1180

    def test_is_lossy_on_fractional_fps(self):
        """Documents *why* docs/00 §3 says to store both columns rather than
        derive one at submit time: the round trip does not always close."""
        fps = 29.97
        frame_number = 1234
        timestamp_ms = frame_to_timestamp_ms(frame_number, fps)
        recovered = timestamp_ms_to_frame(timestamp_ms, fps)
        # It happens to close here, but only within +/-1 in general — which is
        # exactly the margin that turns a correct hit into a wrong submission.
        assert abs(recovered - frame_number) <= 1
