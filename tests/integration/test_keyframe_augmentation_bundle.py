from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from mfe.offline.keyframe_augmentation.bundle import (
    activate_snapshot,
    materialize_bundle,
    rollback_snapshot,
)
from mfe.offline.keyframe_augmentation.models import (
    AugmentationConfig,
    BtcFrame,
    CandidateFrame,
)


def _write_video_and_btc(root: Path) -> tuple[Path, Path]:
    videos = root / "videos"
    keyframes = root / "keyframes" / "V"
    videos.mkdir(parents=True)
    keyframes.mkdir(parents=True)
    video_path = videos / "V.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    decoded = []
    for index in range(10):
        image = np.full((24, 32, 3), index * 20, dtype=np.uint8)
        writer.write(image)
        decoded.append(image)
    writer.release()
    cv2.imwrite(str(keyframes / "001.jpg"), decoded[0])
    cv2.imwrite(str(keyframes / "002.jpg"), decoded[9])
    return videos, root / "keyframes"


def _core_config(root: Path, videos: Path, keyframes: Path) -> Path:
    config_dir = root / "config"
    config_dir.mkdir()
    paths = {
        "videos": str(videos),
        "keyframes": str(keyframes),
        "clip_features": str(root / "clip"),
        "map_keyframes": str(root / "maps"),
        "objects": str(root / "objects"),
        "media_info": str(root / "media"),
        "segments": str(root / "segments"),
        "vector_store": str(root / "base-vector"),
        "inverted_store": str(root / "base-inverted"),
        "asset_store": str(root / "base-asset"),
        "relational_db": str(root / "base.db"),
        "eval_results": str(root / "eval"),
        "throughput": str(root / "throughput.json"),
        "event_log": str(root / "events.jsonl"),
    }
    config = {
        "paths": paths,
        "retrieval": {
            "embed_dim": 2,
            "coarse_topk": 10,
            "rerank_topk": 10,
            "rerank_budget_ms": 100,
        },
        "eval": {"queryset": "eval.jsonl"},
    }
    path = config_dir / "core.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_delta_bundle_provenance_dedup_determinism_and_rollback(tmp_path: Path) -> None:
    videos, keyframes = _write_video_and_btc(tmp_path)
    core = _core_config(tmp_path, videos, keyframes)
    mappings = {
        "V": [
            BtcFrame.build(video_id="V", ordinal=1, frame_idx=0, pts_time=0.0, fps=10.0),
            BtcFrame.build(video_id="V", ordinal=2, frame_idx=9, pts_time=0.9, fps=10.0),
        ]
    }
    config = AugmentationConfig(
        dataset_version="test",
        split_version="test",
        guardrail_ratio=1.0,
        bootstrap_samples=100,
    )
    common = {
        "arm": "B3",
        "video_id": "V",
        "shot_id": "V#s00",
        "pts_time": 0.5,
        "fps": 10.0,
        "selector": "dynamic_density",
        "coverage_class": "C1",
        "priority": (1.0, 1.0, 1.0, "V", 5),
        "detector_fingerprint": "transnetv2:test",
        "audit_status": "confirmed",
    }
    candidates = [
        CandidateFrame(frame_idx=0, **{**common, "pts_time": 0.0}),
        CandidateFrame(frame_idx=5, **common),
    ]

    manifest = materialize_bundle(
        candidates=candidates,
        mappings=mappings,
        config=config,
        core_config_path=core,
        output_dir=tmp_path / "bundle",
        parent_bundle="btc-test",
        run_id="run-test",
        dataset_hash="d" * 64,
    )

    assert manifest["btc_base_immutable"] is True
    assert manifest["origin_counts"] == {"btc": 2, "supplemental": 1}
    assert manifest["dedup_dropped"] == 1
    assert manifest["readiness"]["index_ready"] is False
    assert len(manifest["artifact_hashes"]["augmentation_manifest"]) == 64
    provenance = [
        json.loads(line)
        for line in (tmp_path / "bundle" / "augmentation_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["ordinal"] for row in provenance] == [1, 2, 3]
    delta = next(row for row in provenance if row["origin"] == "supplemental")
    assert delta["keyframe_id"] == "V#000005"
    assert delta["selector"] == "dynamic_density"
    assert delta["frame_idx_source"] == "decoder_counter"
    assert delta["pts_time_source"] == "decoder_pts"
    assert delta["decoder_pts_time"] is not None
    assert delta["mapping_audit_status"] == "PASS"
    assert delta["interpolation_frame_error"] <= 1.0
    assert len(delta["image_sha256"]) == 64

    pointer = tmp_path / "active-snapshot.json"
    base = activate_snapshot(pointer, core)
    augmented = activate_snapshot(pointer, tmp_path / "bundle" / "snapshot-config.yaml")
    rolled_back = rollback_snapshot(pointer)
    assert base["active_config"] == str(core.resolve())
    assert augmented["previous_config"] == str(core.resolve())
    assert rolled_back["active_config"] == str(core.resolve())
