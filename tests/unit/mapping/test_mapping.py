import pytest

from mfe.contracts.errors import ErrorCode
from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.relational import RelationalMap
from mfe.mapping.submit import build_submission
from mfe.mapping.validator import Validator


@pytest.fixture
def db():
    db = RelationalMap(":memory:")
    db.init_schema()
    
    video1 = VideoRow(
        video_id="L21_V001",
        source_path="fake/path.mp4",
        fps=25.0,
        duration_ms=10000,
        total_frames=250,
        width=1280,
        height=720
    )
    db.upsert_video(video1)
    
    kf1 = KeyframeRow(
        keyframe_id="L21_V001#000010",
        video_id="L21_V001",
        keyframe_index=1,
        frame_idx=10,
        pts_time=0.4,
        timestamp_ms=400,
        shot_id="shot_1"
    )
    
    kf2 = KeyframeRow(
        keyframe_id="L21_V001#000260",
        video_id="L21_V001",
        keyframe_index=2,
        frame_idx=260, # intentionally out of bounds
        pts_time=10.4,
        timestamp_ms=10400, # intentionally out of bounds
        shot_id="shot_1"
    )

    db.upsert_keyframes([kf1, kf2])
    return db


def test_relational_map(db):
    assert db.count_keyframes() == 2


def test_get_video_keyframes_is_temporally_ordered(db):
    rows = db.get_video_keyframes("L21_V001")
    assert [row.keyframe_index for row in rows] == [1, 2]
    assert db.get_video_keyframes("missing") == []


def test_verify_mapping(db):
    problems = db.verify_mapping()
    assert len(problems) >= 2 # out of bounds frame, out of bounds timestamp


def test_adapters(db):
    
    rec1 = build_submission("L21_V001#000010", "by_frame", db)
    assert rec1 == {"video_id": "L21_V001", "frame_id": 10}  # BTC goi la frame_id

    rec2 = build_submission("L21_V001#000010", "by_timestamp", db)
    assert rec2 == {"video_id": "L21_V001", "timestamp_ms": 400}
    
    rec3 = build_submission("L21_V001#000010", "raw_keyframe", db)
    assert rec3 == {"keyframe_id": "L21_V001#000010"}


def test_validator(db):
    val = Validator(db)
    
    # Valid
    res_valid = val.validate(
        "L21_V001#000010", evidence_id="tool:run_1:1", schema_format="by_frame"
    )
    assert res_valid.valid

    # Invalid evidence
    res_no_ev = val.validate(
        "L21_V001#000010", evidence_id="invalid evidence", schema_format="by_frame"
    )
    assert not res_no_ev.valid
    assert res_no_ev.failures[0].code == ErrorCode.MISSING_EVIDENCE

    # Out of bounds
    res_oob = val.validate(
        "L21_V001#000260", evidence_id="tool:run_1:1", schema_format="by_frame"
    )
    assert not res_oob.valid
    
    # Should have frame and timestamp out of bounds
    codes = [f.code for f in res_oob.failures]
    assert ErrorCode.FRAME_OUT_OF_BOUNDS in codes
    assert ErrorCode.TIMESTAMP_OUT_OF_BOUNDS in codes


# --- luật "Nhất quán frame" + find_bracketing_keyframes -----------------------
# Các case dưới đây bắt nguồn từ mapping THẬT của batch 1 (873 file, 177.321
# dòng), không phải ví dụ tự nghĩ ra. Bằng chứng đo được nằm trong docstring
# `RelationalMap.find_bracketing_keyframes` và `Validator._check_frame_consistency`:
# 614 cặp trùng frame_idx trên 192/873 video, 0 cặp thật sự giảm dần, và phép
# kiểm tỉ lệ nội suy bắt được 98,1% lỗi fps với 0% báo nhầm.


#: Ba dòng đầu của L21_V006 batch 1. Hai dòng đầu cách nhau chưa đủ 1 frame nên
#: `floor()` cho cùng frame_idx = 0 — có 614 cặp như vậy trên 192/873 video.
L21_V006_RAW_ROWS = [
    # (n, pts_time, frame_idx)
    (1, 0.0, 0),
    (2, 0.0333333, 0),  # cùng frame_idx với n=1
    (3, 3.0, 90),
]


@pytest.fixture
def dup_frame_db():
    """L21_V006 SAU khi ingest đã gộp cặp trùng frame_idx.

    Từ khi `keyframe_id` khoá theo `frame_idx`, hai dòng cùng frame_idx sẽ đúc
    ra cùng một id — nên chúng không thể cùng tồn tại trong map nữa. Ingest gộp
    chúng (giữ n nhỏ hơn), và `UNIQUE(video_id, frame_idx)` chặn nếu một
    extractor tương lai quên gộp.
    """
    db = RelationalMap(":memory:")
    db.init_schema()
    db.upsert_video(
        VideoRow(
            video_id="L21_V006",
            source_path="fake/L21_V006.mp4",
            fps=30.0,
            duration_ms=60000,
            total_frames=1800,
            width=1280,
            height=720,
        )
    )
    db.upsert_keyframes(
        [
            KeyframeRow(
                keyframe_id="L21_V006#000000",
                video_id="L21_V006",
                keyframe_index=1,
                frame_idx=0,
                pts_time=0.0,
                timestamp_ms=0,
            ),
            KeyframeRow(
                keyframe_id="L21_V006#000090",
                video_id="L21_V006",
                keyframe_index=3,  # n=2 đã bị gộp vào n=1
                frame_idx=90,
                pts_time=3.0,
                timestamp_ms=3000,
            ),
        ]
    )
    yield db
    db.close()


def test_equal_frame_idx_rows_collapse_at_ingest():
    """Cặp trùng frame_idx được gộp, giữ n nhỏ hơn, và có báo cáo lại.

    Trước đây case này được xử ở validator (nới `>=` thành `>` để khỏi chặn
    nhầm 1.228 keyframe thật). Từ khi id khoá theo frame_idx, nó được xử sớm
    hơn một bước — ở ingest — vì hai dòng như vậy sẽ đúc ra cùng một id.
    """
    from mfe.offline.keyframes import collapse_duplicate_frames
    from mfe.offline.provided import MapRow

    rows = [
        MapRow(n=n, pts_time=pts, fps=30.0, frame_idx=frame)
        for n, pts, frame in L21_V006_RAW_ROWS
    ]
    kept, collapsed = collapse_duplicate_frames(rows)

    assert [r.n for r in kept] == [1, 3]
    assert collapsed == [(2, 1)]  # n=2 bị bỏ, nhường cho n=1


def test_collapsed_rows_are_all_valid(dup_frame_db):
    """Regression: bản trước dùng >= / <= và chặn nhầm 1.228 keyframe thật."""
    val = Validator(dup_frame_db)
    for kf_id in ("L21_V006#000000", "L21_V006#000090"):
        res = val.validate(kf_id, evidence_id="search_text:t:0", schema_format="by_frame")
        assert res.valid, f"{kf_id} bị chặn nhầm: {[f.message for f in res.failures]}"


def test_wrong_fps_is_caught_by_the_formula_rule(dup_frame_db):
    """frame_idx không khớp floor(pts*fps) -> chặn, với mã lỗi riêng."""
    dup_frame_db.upsert_keyframes(
        [
            KeyframeRow(
                keyframe_id="L21_V006#000120",
                video_id="L21_V006",
                keyframe_index=4,
                frame_idx=120,  # đúng phải là floor(5.0*30) = 150
                pts_time=5.0,
                timestamp_ms=5000,
            )
        ]
    )
    res = Validator(dup_frame_db).validate(
        "L21_V006#000120", evidence_id="search_text:t:0", schema_format="by_frame"
    )
    assert not res.valid
    assert ErrorCode.FRAME_INCONSISTENT in [f.code for f in res.failures]


def test_find_bracketing_keyframes(dup_frame_db):
    prev, nxt = dup_frame_db.find_bracketing_keyframes("L21_V006", 1.5)
    assert prev.keyframe_id == "L21_V006#000000"
    assert nxt.keyframe_id == "L21_V006#000090"


def test_find_bracketing_exact_hit_lands_in_prev(dup_frame_db):
    """pts trùng đúng keyframe chính thức -> nằm ở prev, không phải next."""
    prev, nxt = dup_frame_db.find_bracketing_keyframes("L21_V006", 3.0)
    assert prev.keyframe_id == "L21_V006#000090"
    assert prev.pts_time == 3.0
    assert nxt is None


def test_find_bracketing_before_first_and_after_last(dup_frame_db):
    prev, nxt = dup_frame_db.find_bracketing_keyframes("L21_V006", -1.0)
    assert prev is None and nxt.keyframe_id == "L21_V006#000000"

    prev, nxt = dup_frame_db.find_bracketing_keyframes("L21_V006", 999.0)
    assert prev.keyframe_id == "L21_V006#000090" and nxt is None


def test_validation_result_carries_warnings_separately():
    """warnings không được làm submit bị chặn."""
    from mfe.mapping.validator import ValidationResult, ValidationWarning

    res = ValidationResult(
        valid=True,
        warnings=[
            ValidationWarning(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Nhất quán frame",
                message="khoảng (prev, next) quá rộng",
            )
        ],
    )
    assert res.valid and not res.blocked
    assert not res.confident
