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
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        keyframe_index=1,
        frame_idx=10,
        pts_time=0.4,
        timestamp_ms=400,
        shot_id="shot_1"
    )
    
    kf2 = KeyframeRow(
        keyframe_id="L21_V001#002",
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


def test_verify_mapping(db):
    problems = db.verify_mapping()
    assert len(problems) >= 2 # out of bounds frame, out of bounds timestamp


def test_adapters(db):
    
    rec1 = build_submission("L21_V001#001", "by_frame", db)
    assert rec1 == {"video_id": "L21_V001", "frame_idx": 10}

    rec2 = build_submission("L21_V001#001", "by_timestamp", db)
    assert rec2 == {"video_id": "L21_V001", "timestamp_ms": 400}
    
    rec3 = build_submission("L21_V001#001", "raw_keyframe", db)
    assert rec3 == {"keyframe_id": "L21_V001#001"}


def test_validator(db):
    val = Validator(db)
    
    # Valid
    res_valid = val.validate(
        "L21_V001#001", evidence_id="tool:run_1:1", schema_format="by_frame"
    )
    assert res_valid.valid

    # Invalid evidence
    res_no_ev = val.validate(
        "L21_V001#001", evidence_id="invalid evidence", schema_format="by_frame"
    )
    assert not res_no_ev.valid
    assert res_no_ev.failures[0].code == ErrorCode.MISSING_EVIDENCE

    # Out of bounds
    res_oob = val.validate(
        "L21_V001#002", evidence_id="tool:run_1:1", schema_format="by_frame"
    )
    assert not res_oob.valid
    
    # Should have frame and timestamp out of bounds
    codes = [f.code for f in res_oob.failures]
    assert ErrorCode.FRAME_OUT_OF_BOUNDS in codes
    assert ErrorCode.TIMESTAMP_OUT_OF_BOUNDS in codes
