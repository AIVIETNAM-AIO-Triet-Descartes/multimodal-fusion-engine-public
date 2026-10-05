from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mfe.stores.production.audit import audit_indexing_core


def test_core_audit_reports_blank_duplicate_and_count_anomalies(tmp_path: Path) -> None:
    mapping = tmp_path / "map-keyframes" / "L01_V001.csv"
    mapping.parent.mkdir(parents=True)
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 0, "fps": 29.97, "frame_idx": 0})
        writer.writerow({"n": 2, "pts_time": 0.01, "fps": 29.97, "frame_idx": ""})
    feature = tmp_path / "clip-features" / "L01_V001.npy"
    feature.parent.mkdir(parents=True)
    np.save(feature, np.ones((2, 512), dtype=np.float16))
    for tree, suffix in (("keyframes", ".jpg"), ("objects", ".json")):
        directory = tmp_path / tree / "L01_V001"
        directory.mkdir(parents=True)
        (directory / f"001{suffix}").write_bytes(b"{}")

    result = audit_indexing_core(tmp_path)
    assert result["mapping_video_count"] == 1
    assert result["totals"]["blank_frame_idx_rows"] == 1
    assert result["totals"]["computed_frame_duplicates"] == 1
    assert result["totals"]["keyframe_count_mismatches"] == 1
    assert result["totals"]["object_count_mismatches"] == 1
    assert result["anomalous_video_count"] == 1
    assert result["videos"][0]["duplicate_computed_frame_ids"] == [0]
