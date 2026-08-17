from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_script() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "build_sample_2026_records.py"
    spec = importlib.util.spec_from_file_location("build_sample_2026_records", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_sample(root: Path, *, duplicate: bool = False) -> None:
    for tree in ("map-keyframes", "clip-features", "objects"):
        (root / tree).mkdir(parents=True)
    mapping = root / "map-keyframes" / "L01_V001.csv"
    second_time = 0.01 if duplicate else 1.0
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 0, "fps": 25, "frame_idx": 0})
        writer.writerow(
            {
                "n": 2,
                "pts_time": second_time,
                "fps": 25,
                "frame_idx": 0 if duplicate else 25,
            }
        )
    embeddings = np.zeros((2, 512), dtype=np.float16)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    np.save(root / "clip-features" / "L01_V001.npy", embeddings)
    object_dir = root / "objects" / "L01_V001"
    object_dir.mkdir()
    labels = [
        unicodedata.normalize("NFD", "Thành phố"),
        "Car",
    ]
    for index, label in enumerate(labels, start=1):
        (object_dir / f"{index:03d}.json").write_text(
            json.dumps(
                {
                    "detection_class_entities": [label, "low confidence noise"],
                    "detection_scores": ["0.9", "0.1"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def test_builder_filters_labels_normalizes_text_and_records_checksums(tmp_path: Path) -> None:
    module = load_script()
    root = tmp_path / "sample_2026"
    output = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "records.manifest.json"
    make_sample(root)
    manifest = module.build_records(root, output, manifest_path)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["text"] for row in rows] == ["Thành phố", "Car"]
    assert all(unicodedata.is_normalized("NFC", row["text"]) for row in rows)
    assert manifest["records_written"] == 2
    assert manifest["source_fingerprint"]["file_count"] == 4
    assert manifest["text"]["selected_labels_median"] == 1
    assert manifest["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(manifest["builder_sha256"]) == 64
    assert set(manifest["repository"]) == {"commit", "dirty"}
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["output_sha256"] == manifest[
        "output_sha256"
    ]
    second_output = tmp_path / "records-second.jsonl"
    second_manifest_path = tmp_path / "records-second.manifest.json"
    second = module.build_records(root, second_output, second_manifest_path)
    assert second["output_sha256"] == manifest["output_sha256"]
    with pytest.raises(FileExistsError):
        module.build_records(root, output, manifest_path)


def test_duplicate_keys_fail_closed_unless_quarantine_is_explicit(tmp_path: Path) -> None:
    module = load_script()
    root = tmp_path / "sample_2026"
    output = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "records.manifest.json"
    make_sample(root, duplicate=True)

    with pytest.raises(ValueError, match="duplicate"):
        module.build_records(root, output, manifest_path)
    assert not output.exists()
    manifest = module.build_records(
        root,
        output,
        manifest_path,
        duplicate_policy="quarantine-first",
    )
    assert manifest["records_written"] == 1
    assert manifest["quarantined_duplicate_rows"] == 1
    assert manifest["duplicates"][0]["embedding_cosine"] == pytest.approx(0.0)
