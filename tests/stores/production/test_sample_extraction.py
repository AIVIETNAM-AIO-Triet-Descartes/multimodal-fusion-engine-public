from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_script() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "extract_sample_2026.py"
    spec = importlib.util.spec_from_file_location("extract_sample_2026", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(root: Path, module: ModuleType) -> None:
    assert set(module.REQUIRED_TREES) == {
        "videos",
        "keyframes",
        "clip-features",
        "map-keyframes",
        "objects",
        "media-info",
    }
    video = root / "videos" / "batch-1" / "L01_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    mapping = root / "map-keyframes" / "batch-1" / "L01_V001.csv"
    mapping.parent.mkdir(parents=True)
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 0, "fps": 25, "frame_idx": 0})
        writer.writerow({"n": 2, "pts_time": 1, "fps": 25, "frame_idx": 25})
    features = root / "clip-features" / "batch-1" / "L01_V001.npy"
    features.parent.mkdir(parents=True)
    np.save(features, np.ones((2, 512), dtype=np.float16))
    media_info = root / "media-info" / "batch-1" / "L01_V001.json"
    media_info.parent.mkdir(parents=True)
    media_info.write_text(json.dumps({"fps": 25}), encoding="utf-8")
    for tree, suffix, payload in (
        ("keyframes", ".jpg", b"image"),
        ("objects", ".json", b"{}"),
    ):
        directory = root / tree / "batch-1" / "L01_V001"
        directory.mkdir(parents=True)
        for index in (1, 2):
            (directory / f"{index:03d}{suffix}").write_bytes(payload)


def test_dry_run_does_not_write_and_manifest_has_audit_fields(tmp_path: Path) -> None:
    module = load_script()
    source = tmp_path / "source"
    make_source(source, module)
    destination = tmp_path / "sample_2026"
    manifest = module.extract(
        source, destination, ["L01_V001"], batch="batch-1", execute=False
    )
    assert not destination.exists()
    assert manifest["competition_year"] == 2026
    assert manifest["file_count"] == 8
    assert manifest["schema_anomalies"] == ["PENDING_POST_COPY_AUDIT"]


def test_execute_preserves_layout_and_refuses_overwrite(tmp_path: Path) -> None:
    module = load_script()
    source = tmp_path / "source"
    make_source(source, module)
    destination = tmp_path / "sample_2026"
    manifest = module.extract(
        source, destination, ["L01_V001"], batch="batch-1", execute=True
    )
    assert (destination / "manifest.json").is_file()
    assert all((destination / tree).is_dir() for tree in module.REQUIRED_TREES)
    assert manifest["schema_anomalies"] == []
    assert manifest["audit"]["vector_rows"] == 2
    with pytest.raises(FileExistsError):
        module.extract(source, destination, ["L01_V001"], batch="batch-1", execute=True)


def test_wrong_destination_and_partial_artifacts_fail(tmp_path: Path) -> None:
    module = load_script()
    source = tmp_path / "source"
    make_source(source, module)
    with pytest.raises(ValueError, match="sample_2026"):
        module.extract(source, tmp_path / "sample-new", ["L01_V001"], batch="b", execute=False)
    next(path for path in (source / module.REQUIRED_TREES[0]).rglob("*") if path.is_file()).unlink()
    with pytest.raises(FileNotFoundError):
        module.extract(source, tmp_path / "sample_2026", ["L01_V001"], batch="b", execute=False)
