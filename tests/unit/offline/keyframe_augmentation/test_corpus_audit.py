from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json
from mfe.offline.keyframe_augmentation.corpus_audit import (
    audit_video,
    build_summary,
    byte_balanced_slices,
    validate_corpus_index,
)
from mfe.offline.keyframe_augmentation.models import AugmentationConfig
from mfe.offline.source import VideoEntry


def _config() -> AugmentationConfig:
    return AugmentationConfig(
        dataset_version="test",
        split_version="no-split-read",
        target_gap_seconds=4.0,
        k_max=6,
    )


def test_byte_balanced_slices_matches_greedy_contract() -> None:
    catalogue = {
        name: VideoEntry(name, "x.zip", "file:x.zip", 0, size)
        for name, size in {"v1": 10, "v2": 9, "v3": 8, "v4": 7}.items()
    }
    assert byte_balanced_slices(["v1", "v2", "v3", "v4"], 2, catalogue) == [
        ["v1", "v4"],
        ["v2", "v3"],
    ]


def test_validate_corpus_index_rejects_incomplete_manifest() -> None:
    producer = {"run_id": "run"}
    manifest = {
        "schema_version": 1,
        "run_id": "run",
        "member": "quang",
        "worker": "quang1",
        "repo_commit": "abc",
        "config_sha256": "cfg",
        "assigned_count": 1,
        "verified_complete_count": 0,
        "verified": [],
        "failures_this_session": [],
        "created_at_utc": "now",
    }
    manifest["manifest_payload_hash"] = sha256_json(manifest)
    with pytest.raises(ValueError, match="incomplete final manifest"):
        validate_corpus_index(
            producer_config=producer,
            expected={"quang1": ["L21_V001"]},
            manifests={"quang1": manifest},
            markers={"L21_V001": {}},
            producer_config_sha256="cfg",
        )


def test_audit_video_validates_hashes_and_classifies_c0_c1(tmp_path: Path) -> None:
    shots_path = tmp_path / "shots.jsonl"
    shots = [
        {
            "video_id": "L21_V001",
            "shot_id": "s0",
            "start_frame": 0,
            "end_frame": 99,
            "fps": 10.0,
            "detector": "transnetv2",
            "detector_fingerprint": "fp",
            "confidence": 0.9,
            "audit_status": "auto",
        },
        {
            "video_id": "L21_V001",
            "shot_id": "s1",
            "start_frame": 100,
            "end_frame": 149,
            "fps": 10.0,
            "detector": "transnetv2",
            "detector_fingerprint": "fp",
            "confidence": 0.9,
            "audit_status": "auto",
        },
    ]
    shots_path.write_text("".join(json.dumps(row) + "\n" for row in shots), encoding="utf-8")
    mapping_path = tmp_path / "mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 0, "pts_time": 1.0, "fps": 10.0, "frame_idx": 10})
    marker = {
        "shots_sha256": sha256_file(shots_path),
        "btc_mapping_sha256": sha256_file(mapping_path),
        "shot_count": 2,
        "literal_fps": 10.0,
    }

    per_video, audits = audit_video(
        video_id="L21_V001",
        shots_path=shots_path,
        mapping_path=mapping_path,
        marker=marker,
        config=_config(),
    )

    assert [row["coverage_class"] for row in audits] == ["C1", "C0"]
    assert per_video["eligible_shots"] == 2
    summary = build_summary(per_video_rows=[per_video], config=_config())
    assert summary["decision_authority"] == "diagnostic_only"
    assert summary["oracle_status"]["O1-S"].startswith("BLOCKED")


def test_audit_video_rejects_timeline_gap(tmp_path: Path) -> None:
    shots_path = tmp_path / "shots.jsonl"
    rows = [
        {"video_id": "v", "shot_id": "a", "start_frame": 0, "end_frame": 9,
         "fps": 10.0, "detector_fingerprint": "fp", "confidence": 1.0},
        {"video_id": "v", "shot_id": "b", "start_frame": 11, "end_frame": 19,
         "fps": 10.0, "detector_fingerprint": "fp", "confidence": 1.0},
    ]
    shots_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text("n,pts_time,fps,frame_idx\n0,0.1,10,1\n", encoding="utf-8")
    marker = {
        "shots_sha256": sha256_file(shots_path),
        "btc_mapping_sha256": sha256_file(mapping_path),
        "shot_count": 2,
        "literal_fps": 10.0,
    }
    with pytest.raises(ValueError, match="gap or overlap"):
        audit_video(
            video_id="v",
            shots_path=shots_path,
            mapping_path=mapping_path,
            marker=marker,
            config=_config(),
        )
