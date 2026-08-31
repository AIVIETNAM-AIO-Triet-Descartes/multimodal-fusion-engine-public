from __future__ import annotations

import pytest

from mfe.offline.augmentation.acceptance import (
    ARMS,
    build_blocked_oracle_evidence,
    validate_feature_parity_acceptance,
    validate_primary_equal_budget,
    validate_video_assignment,
    validate_worker_assignment,
)
from mfe.offline.keyframe_augmentation.diagnostics import FEATURE_PARITY_FIELDS


def _rows(count: int) -> list[dict[str, str]]:
    return [{"keyframe_id": f"V#00{index:04d}"} for index in range(count)]


def test_worker_assignment_matches_team_shards() -> None:
    validate_worker_assignment("quang", "quang4")
    validate_worker_assignment("trong", "trong2")
    with pytest.raises(ValueError, match="not assigned"):
        validate_worker_assignment("trong", "quang1")


def test_video_assignment_matches_canonical_zip_book() -> None:
    assigned = validate_video_assignment(
        "config/zips_batch1.yaml", "quang", "L21_V007"
    )
    assert "Videos_L21_a.zip" in assigned
    with pytest.raises(ValueError, match="not assigned"):
        validate_video_assignment("config/zips_batch1.yaml", "trong", "L21_V007")


def test_primary_equal_budget_uses_floor_and_passes_only_exact_budget() -> None:
    report = validate_primary_equal_budget(
        base_keyframes=101,
        cap_ratio=0.03,
        arm_rows={arm: _rows(3) for arm in ARMS},
    )

    assert report["locked_hard_delta_budget"] == 3
    assert report["primary_equal_budget_pass"] is True
    assert report["B0_delta_count"] == 0


def test_primary_equal_budget_does_not_hide_underfill() -> None:
    arms = {arm: _rows(3) for arm in ARMS}
    arms["B1"] = _rows(2)

    with pytest.raises(RuntimeError, match="BLOCKED_PRIMARY_EQUAL_BUDGET"):
        validate_primary_equal_budget(
            base_keyframes=101,
            cap_ratio=0.03,
            arm_rows=arms,
        )


def test_blocked_o1s_evidence_is_explicit_and_has_no_guessed_scores() -> None:
    rows = build_blocked_oracle_evidence(
        [
            {
                "query_id": "q1",
                "query_type": "QA",
                "video_id": "L21_V007",
                "start_frame": 10,
                "end_frame": 20,
            }
        ],
        provenance_hashes={"gate0_dataset_hash": "d" * 64},
    )

    assert rows[0]["evidence_status"] == "BLOCKED_MISSING_STAGE_ARTIFACT"
    assert rows[0]["failure_stage"] == "representation"
    assert rows[0]["gt_interval_inclusive"] == [10, 20]
    assert rows[0]["best_source_frame_idx"] is None
    assert rows[0]["best_source_score_by_stage"] == {}


def test_feature_parity_requires_numeric_canary_not_metadata_only() -> None:
    fingerprint = {field: f"locked-{field}" for field in FEATURE_PARITY_FIELDS}
    with pytest.raises(RuntimeError, match="not MEASURED"):
        validate_feature_parity_acceptance(fingerprint, dict(fingerprint))

    report = validate_feature_parity_acceptance(
        fingerprint,
        dict(fingerprint),
        canary_evidence={
            "sample_count": 16,
            "btc_artifact_sha256": "b" * 64,
            "delta_redecode_artifact_sha256": "d" * 64,
            "max_abs_diff": 1e-7,
            "tolerance": 1e-6,
        },
    )
    assert report["status"] == "MEASURED"
