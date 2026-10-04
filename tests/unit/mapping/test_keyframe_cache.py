from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.relational import RelationalMap


def catalog(capacity):
    db = RelationalMap(":memory:", keyframe_cache_max=capacity)
    db.init_schema()
    db.upsert_video(VideoRow(video_id="L21_V001", source_path="fixture.mp4", fps=25,
                             duration_ms=1000, total_frames=25, width=32, height=32))
    rows = [KeyframeRow(keyframe_id=f"L21_V001#{i:06d}", video_id="L21_V001",
                         frame_idx=i, keyframe_index=i, pts_time=i/25, timestamp_ms=i*40)
            for i in range(1, 7)]
    db.upsert_keyframes(rows)
    return db, rows


def test_bounded_cache_keeps_coordinates_correct_after_eviction():
    db, rows = catalog(2)
    for row in rows:
        assert db.get_keyframe(row.keyframe_id) == row
        assert db.keyframe_cache_stats()["keyframe_entries"] <= 2
    assert db.get_keyframe(rows[0].keyframe_id) == rows[0]
    assert db.keyframe_cache_stats() == {"keyframe_entries": 2, "keyframe_capacity": 2}
    db.close()
    assert db.keyframe_cache_stats()["keyframe_entries"] == 0


def test_upsert_invalidates_cached_coordinates():
    db, rows = catalog(2)
    row = rows[0]
    assert db.get_keyframe(row.keyframe_id) == row
    updated = row.model_copy(update={"shot_id": "updated"})
    db.upsert_keyframes([updated])
    assert db.get_keyframe(row.keyframe_id).shot_id == "updated"


def test_cache_can_be_disabled_without_changing_lookup():
    db, rows = catalog(0)
    for row in rows:
        assert db.get_keyframe(row.keyframe_id) == row
    assert db.keyframe_cache_stats()["keyframe_entries"] == 0
