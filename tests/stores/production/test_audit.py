from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mfe.stores.production.audit import audit_sample


def fixture(root: Path, *, fps: float = 29.97, bad_frame: bool = False) -> None:
    mapping = root / "map-keyframes" / "L01_V001.csv"
    mapping.parent.mkdir(parents=True)
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 0.0, "fps": fps, "frame_idx": 1 if bad_frame else 0})
        writer.writerow({"n": 2, "pts_time": 10.0, "fps": fps, "frame_idx": 299})
    features = root / "clip-features" / "L01_V001.npy"
    features.parent.mkdir(parents=True)
    np.save(features, np.full((2, 512), 1 / np.sqrt(512), dtype=np.float16))
    video = root / "videos" / "L01_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"test video fixture")
    media_info = root / "media-info" / "L01_V001.json"
    media_info.parent.mkdir(parents=True)
    media_info.write_text(json.dumps({"fps": fps}), encoding="utf-8")
    for tree, suffix, payload in (
        ("keyframes", ".jpg", b"image"),
        ("objects", ".json", b"{}"),
    ):
        directory = root / tree / "L01_V001"
        directory.mkdir(parents=True)
        for index in (1, 2):
            (directory / f"{index:03d}{suffix}").write_bytes(payload)


def test_audit_checks_float_formula_shape_counts_and_checksums(tmp_path: Path) -> None:
    fixture(tmp_path)
    result = audit_sample(tmp_path)
    assert result["video_count"] == 1
    assert result["vector_rows"] == 2
    assert result["frame_formula_mismatch_count"] == 0
    assert len(result["videos"][0]["embedding_sha256"]) == 64


def test_audit_exposes_frame_formula_conflict(tmp_path: Path) -> None:
    fixture(tmp_path, bad_frame=True)
    result = audit_sample(tmp_path)
    assert result["frame_formula_mismatch_count"] == 1


def test_audit_rejects_wrong_embedding_shape(tmp_path: Path) -> None:
    fixture(tmp_path)
    np.save(tmp_path / "clip-features" / "L01_V001.npy", np.ones((2, 511), dtype=np.float16))
    with pytest.raises(ValueError, match="shape"):
        audit_sample(tmp_path)
