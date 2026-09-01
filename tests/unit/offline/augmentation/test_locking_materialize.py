from __future__ import annotations

import hashlib
import json

import pytest

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.relational import RelationalMap
from mfe.offline.augmentation.cli import _enforce_heldout_grid
from mfe.offline.augmentation.config import AugmentationConfig, load_augmentation_config
from mfe.offline.augmentation.locking import build_gate0_lock
from mfe.offline.augmentation.materialize import assign_augmented_ordinals
from mfe.offline.augmentation.models import (
    BtcFrame,
    CandidateRow,
    CoverageClass,
    Selector,
    VideoInventoryRow,
)
from mfe.offline.augmentation.snapshot import build_relational_snapshot
from mfe.offline.augmentation.store import ExperimentStore
from mfe.offline.transport import LocalDirTransport


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_committed_v1_config_is_strictly_loadable():
    config = load_augmentation_config("config/keyframe_augmentation_runtime_v1.yaml")

    assert config.experiment.id == "btc_aug_cov_v1"
    assert config.detector.checkpoint_sha256 == "0" * 64


def inventory(video_id: str) -> VideoInventoryRow:
    return VideoInventoryRow(
        video_id=video_id,
        source_path=f"/canonical/videos/{video_id}.mp4",
        sha256=digest(video_id.encode()),
        byte_size=10,
        fps=29.97,
        duration_seconds=100.0,
        btc_keyframes=10,
        map_path=f"/canonical/maps/{video_id}.csv",
        map_sha256=digest((video_id + "-map").encode()),
    )


def test_gate0_locks_heldout_hashes_but_never_adds_heldout_to_pilot(tmp_path, monkeypatch):
    config = AugmentationConfig.model_validate(
        {
            "experiment": {"id": "test-v1", "parent_bundle": "btc"},
            "detector": {
                "checkpoint": "model.bin",
                "checkpoint_sha256": "0" * 64,
            },
            "dataset": {
                "dev_queryset": "eval/dev.jsonl",
                "heldout_queryset": "eval/heldout.jsonl",
                "video_inventory": "inventory.jsonl",
                "extra_pilot_videos": 1,
            },
        }
    )
    config._repo_root = tmp_path
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "dev.jsonl").write_text(
        json.dumps({"gt_video_id": "L21_V001"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "eval" / "heldout.jsonl").write_text(
        json.dumps({"gt_video_id": "L21_V002"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("mfe.offline.augmentation.locking._git_commit", lambda _: "abc123")

    lock = build_gate0_lock(
        config,
        [inventory("L21_V001"), inventory("L21_V002"), inventory("L21_V003")],
        parent_manifest_sha256="f" * 64,
    )

    assert "L21_V002" not in lock["cohort"]
    assert lock["heldout_videos_excluded"] == ["L21_V002"]
    assert {row["video_id"] for row in lock["inventory"]} == {
        "L21_V001",
        "L21_V002",
        "L21_V003",
    }


def test_heldout_grid_must_match_frozen_dev_policy(tmp_path):
    config = AugmentationConfig.model_validate(
        {
            "experiment": {"id": "test-v1", "parent_bundle": "btc"},
            "detector": {"checkpoint": "model.bin", "checkpoint_sha256": "0" * 64},
        }
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "dev_go": True,
                "config_hash": config.canonical_hash(),
                "threshold": 0.5,
                "target_gap_seconds": 6.0,
                "k_max": 3,
            }
        ),
        encoding="utf-8",
    )

    _enforce_heldout_grid(config, "heldout", str(freeze), 0.5, 6.0, 3)
    with pytest.raises(RuntimeError, match="frozen grid"):
        _enforce_heldout_grid(config, "heldout", str(freeze), 0.6, 6.0, 3)


def test_augmented_ordinals_are_temporal_but_identity_remains_source_frame():
    btc = [
        BtcFrame(
            keyframe_id="L21_V001#000010",
            video_id="L21_V001",
            frame_idx=10,
            pts_time=1.0,
        ),
        BtcFrame(
            keyframe_id="L21_V001#000030",
            video_id="L21_V001",
            frame_idx=30,
            pts_time=3.0,
        ),
    ]
    delta = [
        CandidateRow(
            keyframe_id="L21_V001#000020",
            video_id="L21_V001",
            frame_idx=20,
            literal_fps=10.0,
            pts_time=2.0,
            decoder_pts_time=2.0,
            shot_id="L21_V001#s00",
            selector=Selector.FARTHEST,
            coverage_class=CoverageClass.C1,
            selection_round=1,
            distance_to_nearest_btc=1.0,
            detector_confidence=0.9,
            decoder_fingerprint="decoder-v1",
            config_hash="config-v1",
        )
    ]

    assert assign_augmented_ordinals(btc, delta) == {
        "L21_V001#000010": 0,
        "L21_V001#000020": 1,
        "L21_V001#000030": 2,
    }


def test_relational_snapshot_clones_base_and_rebuilds_temporal_ordinals(tmp_path):
    base = tmp_path / "base.db"
    output = tmp_path / "augmented.db"
    with RelationalMap(str(base)) as relational:
        relational.init_schema()
        relational.upsert_video(
            VideoRow(
                video_id="L21_V001",
                source_path="video.mp4",
                fps=10.0,
                duration_ms=5000,
                total_frames=50,
                width=1920,
                height=1080,
            )
        )
        relational.upsert_keyframes(
            [
                KeyframeRow(
                    keyframe_id="L21_V001#000010",
                    video_id="L21_V001",
                    keyframe_index=0,
                    frame_idx=10,
                    pts_time=1.0,
                    timestamp_ms=1000,
                ),
                KeyframeRow(
                    keyframe_id="L21_V001#000030",
                    video_id="L21_V001",
                    keyframe_index=1,
                    frame_idx=30,
                    pts_time=3.0,
                    timestamp_ms=3000,
                ),
            ]
        )
    delta = CandidateRow(
        keyframe_id="L21_V001#000020",
        video_id="L21_V001",
        frame_idx=20,
        literal_fps=10.0,
        pts_time=2.0,
        decoder_pts_time=2.0,
        shot_id="L21_V001#s00",
        selector=Selector.FARTHEST,
        coverage_class=CoverageClass.C1,
        selection_round=1,
        distance_to_nearest_btc=1.0,
        detector_confidence=0.9,
        decoder_fingerprint="decoder-v1",
        config_hash="config-v1",
    )

    report = build_relational_snapshot(base, output, [delta])

    assert report["delta_rows"] == 1
    with RelationalMap(str(output)) as relational:
        assert relational.get_keyframe("L21_V001#000010").keyframe_index == 0
        assert relational.get_keyframe("L21_V001#000020").keyframe_index == 1
        assert relational.get_keyframe("L21_V001#000030").keyframe_index == 2
        assert relational.verify_mapping() == []


def test_relational_snapshot_can_preserve_but_not_hide_frozen_base_problems(tmp_path):
    base = tmp_path / "base-with-known-drift.db"
    strict_output = tmp_path / "strict.db"
    compatible_output = tmp_path / "compatible.db"
    with RelationalMap(str(base)) as relational:
        relational.init_schema()
        relational.upsert_video(
            VideoRow(
                video_id="L21_V001",
                source_path="video.mp4",
                fps=10.0,
                duration_ms=5000,
                total_frames=50,
                width=1920,
                height=1080,
            )
        )
        relational.upsert_keyframes(
            [
                KeyframeRow(
                    keyframe_id="L21_V001#000010",
                    video_id="L21_V001",
                    keyframe_index=0,
                    frame_idx=10,
                    pts_time=1.0,
                    timestamp_ms=999,
                )
            ]
        )

    with pytest.raises(RuntimeError, match="base mapping audit failed"):
        build_relational_snapshot(base, strict_output, [])

    report = build_relational_snapshot(
        base,
        compatible_output,
        [],
        allow_preexisting_mapping_problems=True,
    )

    assert report["baseline_mapping_problem_count"] == 1
    assert report["snapshot_mapping_problem_count"] == 1
    assert report["introduced_mapping_problem_count"] == 0
    assert report["removed_mapping_problem_count"] == 0


def test_experiment_store_restores_prefix_atomically_and_verifies_hash(tmp_path):
    transport = LocalDirTransport(tmp_path / "remote")
    source = tmp_path / "source.txt"
    source.write_text("durable\n", encoding="utf-8")
    store = ExperimentStore(transport, "exp", tmp_path / "work")
    # ``publish`` also writes a long hash marker.  On Windows the pytest temp
    # root can approach MAX_PATH, so use the marker-free batch primitive here;
    # this test is focused on restore atomicity and hash verification.
    store.publish_many([(source, "gate1/dev/raw/video.npz")])

    destination = tmp_path / "fresh"
    restored = store.restore_prefix("gate1/dev", destination)

    assert restored == [destination / "gate1" / "dev" / "raw" / "video.npz"]
    assert restored[0].read_text(encoding="utf-8") == "durable\n"
    assert not list(destination.rglob("*.download"))

    remote_file = (
        tmp_path
        / "remote"
        / "experiments"
        / "keyframe-augmentation"
        / "exp"
        / "gate1"
        / "dev"
        / "raw"
        / "video.npz"
    )
    remote_file.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="restore hash mismatch"):
        store.restore_prefix("gate1/dev", tmp_path / "corrupt")


def test_experiment_store_rejects_prefix_escape(tmp_path):
    store = ExperimentStore(LocalDirTransport(tmp_path / "remote"), "exp", tmp_path / "work")

    with pytest.raises(ValueError, match="escape"):
        store.restore_prefix("../other", tmp_path / "fresh")
