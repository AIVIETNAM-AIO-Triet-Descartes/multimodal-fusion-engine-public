from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json
from mfe.offline.keyframe_augmentation.delta_siglip import (
    image_manifest,
    partition_feature_worklist,
    validate_siglip_baseline_manifest,
    write_deterministic_image_tar,
)


def _row(video_id: str, frame_idx: int) -> dict:
    row = {
        "keyframe_id": f"{video_id}#{frame_idx:06d}",
        "video_id": video_id,
        "frame_idx": frame_idx,
        "pts_time": (frame_idx + 0.5) / 25.0,
        "fps": 25.0,
        "arms": ["B3"],
    }
    row["work_item_hash"] = sha256_json(row)
    return row


def test_partition_is_disjoint_and_validates_work_item_hash() -> None:
    rows = [_row("V1", 1), _row("V2", 2)]
    result = partition_feature_worklist(rows, {"quang1": ["V1"], "trong1": ["V2"]})

    assert [row["video_id"] for row in result["quang1"]] == ["V1"]
    assert [row["video_id"] for row in result["trong1"]] == ["V2"]
    broken = json.loads(json.dumps(rows))
    broken[0]["frame_idx"] = 9
    with pytest.raises(ValueError, match="invalid feature work item hash"):
        partition_feature_worklist(broken, {"quang1": ["V1", "V2"]})


def test_tar_and_image_manifest_are_deterministic(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    first = images / "000000001.jpg"
    second = images / "000000002.jpg"
    first.write_bytes(b"jpeg-one")
    second.write_bytes(b"jpeg-two")
    paths = {first.name: first, second.name: second}

    tar_a = write_deterministic_image_tar(paths, tmp_path / "a.tar")
    tar_b = write_deterministic_image_tar(paths, tmp_path / "b.tar")
    assert sha256_file(tar_a) == sha256_file(tar_b)
    with tarfile.open(tar_a) as archive:
        assert archive.getnames() == [first.name, second.name]

    manifest = image_manifest(
        video_id="V", rows=[_row("V", 1), _row("V", 2)], image_dir=images
    )
    unsigned = dict(manifest)
    payload_hash = unsigned.pop("manifest_payload_hash")
    assert payload_hash == sha256_json(unsigned)
    assert manifest["frame_count"] == 2


def test_legacy_siglip_manifest_uses_artifact_contract_without_inventing_env() -> None:
    manifest = {
        "artifacts": {
            "siglip_vectors": {
                "sha256": "abc",
                "model": "google/siglip-so400m-patch14-384",
                "dim": 1152,
                "params": {"tower": "image", "batch_size": 32},
            }
        },
        "env": {"transformers": "5.0.0"},
        "stages": {},
    }

    report = validate_siglip_baseline_manifest(
        manifest,
        vector_sha256="abc",
        checkpoint="google/siglip-so400m-patch14-384",
        dimension=1152,
        batch_size=32,
    )

    assert report["artifact_contract_status"] == "PASS"
    assert report["stage_environment_status"] == "UNVERIFIED_LEGACY_MANIFEST"
    assert report["stage_environment"] == {}


def test_siglip_manifest_rejects_artifact_hash_drift() -> None:
    manifest = {
        "artifacts": {
            "siglip_vectors": {
                "sha256": "expected",
                "model": "model",
                "dim": 1152,
                "params": {"tower": "image", "batch_size": 32},
            }
        },
        "stages": {"n4_siglip": {"env": {"torch": "2.0"}}},
    }

    report = validate_siglip_baseline_manifest(
        manifest,
        vector_sha256="actual",
        checkpoint="model",
        dimension=1152,
        batch_size=32,
    )

    assert report["artifact_contract_status"] == "BLOCKED"
    assert report["artifact_contract_checks"]["sha256"] is False
    assert report["stage_environment_status"] == "AVAILABLE"
