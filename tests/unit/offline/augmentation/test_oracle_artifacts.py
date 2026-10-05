from __future__ import annotations

import math

import pytest

from mfe.offline.augmentation.io import canonical_hash
from mfe.offline.augmentation.oracle_artifacts import (
    OracleArtifactCandidate,
    OracleArtifactTarget,
    oracle_candidate_for_snapshot,
    validate_oracle_completion_marker,
)


def candidate_payload() -> dict:
    return {
        "audit_status": "auto_accepted",
        "config_hash": "0" * 64,
        "coverage_class": "C0",
        "decoder_fingerprint": "decoder-v1",
        "detector_confidence": 0.9,
        "distance_to_nearest_btc": 2.0,
        "frame_idx": 2416,
        "keyframe_id": "L21_V002#002416",
        "pts_time": 80.53333333333333,
        "pts_time_source": "decoder_pts",
        "selection_pts_time": 80.55,
        "selection_round": 1,
        "selector": "oracle_target",
        "shot_id": "L21_V002#s0025",
        "video_id": "L21_V002",
    }


def test_oracle_artifact_contract_accepts_decoder_pts_and_inclusive_interval() -> None:
    candidate = OracleArtifactCandidate.model_validate(candidate_payload())
    target = OracleArtifactTarget.model_validate(
        {
            "query_id": "q1",
            "event_index": 0,
            "video_id": candidate.video_id,
            "frame_range_start": candidate.frame_idx,
            "frame_range_end": candidate.frame_idx,
            "target": candidate,
        }
    )

    assert target.target.pts_time_source == "decoder_pts"


def test_oracle_artifact_contract_rejects_identity_drift() -> None:
    payload = candidate_payload()
    payload["keyframe_id"] = "L21_V002#002417"

    with pytest.raises(ValueError, match="keyframe_id"):
        OracleArtifactCandidate.model_validate(payload)


@pytest.mark.parametrize(
    ("frame_idx", "decoder_pts_time", "literal_fps"),
    [
        (1983, 79.32, 25.0),
        (3074, 102.56913333333334, 29.97),
    ],
)
def test_oracle_snapshot_adapter_preserves_decoder_pts_and_floor_roundtrips(
    frame_idx: int,
    decoder_pts_time: float,
    literal_fps: float,
) -> None:
    payload = candidate_payload()
    payload.update(
        {
            "frame_idx": frame_idx,
            "keyframe_id": f"L21_V002#{frame_idx:06d}",
            "pts_time": decoder_pts_time,
        }
    )
    candidate = OracleArtifactCandidate.model_validate(payload)

    snapshot_candidate = oracle_candidate_for_snapshot(
        candidate,
        literal_fps=literal_fps,
    )

    assert snapshot_candidate.decoder_pts_time == decoder_pts_time
    assert snapshot_candidate.pts_time_source == "synthetic_literal_fps_interval"
    assert math.floor(snapshot_candidate.pts_time * literal_fps) == frame_idx
    assert snapshot_candidate.frame_idx == frame_idx


def test_oracle_marker_requires_matching_remote_hashes() -> None:
    marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "run_id": "oracle-materialize-v1",
        "worker": "oracle1",
        "video_id": "L21_V002",
        "config_hash": "0" * 64,
        "oracle_candidates_sha256": "1" * 64,
        "source_video_sha256": "2" * 64,
        "artifacts": {"/research/frame.jpg": "3" * 64},
        "btc_mutated": False,
    }
    marker["marker_payload_hash"] = canonical_hash(marker)

    common = {
        "run_id": "oracle-materialize-v1",
        "worker": "oracle1",
        "video_id": "L21_V002",
        "config_hash": "0" * 64,
        "oracle_candidates_sha256": "1" * 64,
    }
    assert validate_oracle_completion_marker(
        marker,
        observed_remote_hashes={"/research/frame.jpg": "3" * 64},
        **common,
    )
    assert not validate_oracle_completion_marker(
        marker,
        observed_remote_hashes={"/research/frame.jpg": "f" * 64},
        **common,
    )
