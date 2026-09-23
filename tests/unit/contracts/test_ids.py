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
    frame_idx_from_pts,
    parse_keyframe_id,
    parse_shot_id,
    pts_time_to_timestamp_ms,
    video_group,
)


class TestBuildKeyframeId:
    @pytest.mark.parametrize(
        ("video_id", "frame_id", "expected"),
        [
            ("L21_V001", 90, "L21_V001#000090"),  # row n=2 of the real CSV
            ("L21_V001", 31124, "L21_V001#031124"),  # row n=250
            ("L21_V001", 0, "L21_V001#000000"),  # boundary
            ("L30_V099", 91, "L30_V099#000091"),
        ],
    )
    def test_matches_docs_format(self, video_id, frame_id, expected):
        assert build_keyframe_id(video_id, frame_id) == expected

    def test_frame_over_999999_uses_more_digits(self):
        """6 is a minimum pad width, not a ceiling — a long video keeps working."""
        assert build_keyframe_id("L21_V001", 1234567) == "L21_V001#1234567"

    def test_ordinal_is_not_the_key(self):
        """The regression this format exists to prevent.

        Keyframe n=250 of L21_V001 is frame 31124. Under the old scheme its id
        was `#250`, which reads like a frame number and is not one — submitting
        it would name a frame 17 minutes early.
        """
        assert build_keyframe_id("L21_V001", 31124) != "L21_V001#000250"

    def test_rejects_negative_frame_id(self):
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
            ("L21_V001#000090", ("L21_V001", 90)),
            ("L21_V001#000000", ("L21_V001", 0)),
            ("L21_V001#031124", ("L21_V001", 31124)),
        ],
    )
    def test_matches_docs_examples(self, keyframe_id, expected):
        assert parse_keyframe_id(keyframe_id) == expected

    def test_strips_zero_padding_to_int(self):
        _, frame_id = parse_keyframe_id("L21_V001#000090")
        assert frame_id == 90
        assert isinstance(frame_id, int)

    @pytest.mark.parametrize(
        "malformed",
        [
            "L21_V001",  # no frame part
            "L21_V001#1",  # under-padded (< 6 digits)
            "L21_V001#001",  # the OLD ordinal-based format — must not parse
            "L21_V001#31124",  # right number, missing a digit of padding
            "L21_V001#abcdef",  # non-numeric
            "L21_V001#s07",  # that's a shot_id, not a keyframe_id
            "#000001",  # no video_id
        ],
    )
    def test_rejects_malformed(self, malformed):
        with pytest.raises(ContractError) as exc:
            parse_keyframe_id(malformed)
        assert exc.value.code is ErrorCode.INVALID_ID

    @pytest.mark.parametrize("frame_id", [0, 1, 91, 250, 999, 31124, 1234567])
    def test_round_trips(self, frame_id):
        video_id = "L21_V001"
        assert parse_keyframe_id(build_keyframe_id(video_id, frame_id)) == (
            video_id,
            frame_id,
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


class TestFrameIdxFromPts:
    @pytest.mark.parametrize(
        ("pts_time", "fps", "expected"),
        [
            (0.0333333, 30.0, 0),
            (4.03333, 30.0, 120),
            (17.3333, 30.0, 519),
            (1893.66, 29.97, 56752),
        ],
    )
    def test_uses_dataset_floor_rule(self, pts_time, fps, expected):
        assert frame_idx_from_pts(pts_time, fps) == expected

    def test_does_not_round_to_nearest_frame(self):
        pts_time, fps = 0.0333333, 30.0
        assert frame_idx_from_pts(pts_time, fps) == 0
        assert round(pts_time * fps) == 1

    @pytest.mark.parametrize(("pts_time", "fps"), [(-0.01, 30.0), (1.0, 0.0)])
    def test_rejects_invalid_inputs(self, pts_time, fps):
        with pytest.raises(ContractError) as exc:
            frame_idx_from_pts(pts_time, fps)
        assert exc.value.code is ErrorCode.INVALID_ID


class TestPtsTimeToTimestampMs:
    """The sanctioned conversion for seconds -> milliseconds.

    This is separate from custom-frame derivation, which is locked above to
    `floor(pts_time * fps)`. When BTC's CSV already supplies `frame_idx`, the
    pipeline still copies that source value instead of recomputing it.
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


class TestVideoGroup:
    """BTC dùng HAI dấu nối; `split("_")` hỏng im lặng với một nửa batch 2.

    Đọc từ thư mục trung tâm của zip gốc (23-09-2026):
        Videos_M01.zip      -> videos/M01_V001.mp4     gạch DƯỚI
        Video_N001-N010.zip -> videos/N001-V001.mov    gạch NGANG
    """

    @pytest.mark.parametrize(
        ("video_id", "expected"),
        [
            ("L21_V001", "L21"),        # batch 1
            ("M01_V001", "M01"),        # batch 2, tin tức — gạch dưới
            ("M06_V021", "M06"),
            ("N001-V001", "N001"),      # batch 2, giao thông — gạch ngang
            ("N081-V001", "N081"),
            ("S01-V008", "S01"),        # batch 2, đua xe đạp
        ],
    )
    def test_tach_dung_ca_hai_dau_noi(self, video_id, expected):
        assert video_group(video_id) == expected

    def test_split_gach_duoi_HONG_voi_nhom_N(self):
        """Phép tách cũ hỏng ở đâu — giữ lại để không ai quay về nó."""
        assert "N001-V001".split("_")[0] == "N001-V001"      # trả NGUYÊN chuỗi
        assert video_group("N001-V001") == "N001"

    def test_split_gach_ngang_HONG_voi_nhom_M(self):
        assert "M01_V001".split("-")[0] == "M01_V001"
        assert video_group("M01_V001") == "M01"

    def test_video_id_khong_hop_le_thi_nem(self):
        with pytest.raises(ContractError) as exc:
            video_group("co khoang trang")
        assert exc.value.code is ErrorCode.INVALID_ID
