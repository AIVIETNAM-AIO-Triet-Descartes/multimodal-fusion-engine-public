from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mfe.offline.keyframe_augmentation.diagnostics import (
    FEATURE_PARITY_FIELDS,
    compare_feature_parity,
    crowding_report,
)
from mfe.offline.keyframe_augmentation.oracle import (
    OracleEvidenceRow,
    blocked_oracle_row,
    validate_oracle_rows,
)
from mfe.offline.stages import _quad_to_box

ROOT = Path(__file__).parents[4]
CONFIG = ROOT / "config"


def _load(name: str) -> dict:
    with (CONFIG / name).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert isinstance(payload, dict)
    return payload


def _parity_fingerprint() -> dict[str, object]:
    return {field: f"locked-{field}" for field in FEATURE_PARITY_FIELDS}


def test_safety_manifest_captures_before_identity_and_policy() -> None:
    payload = _load("keyframe_augmentation_safety_manifest_2026-08-30.yaml")

    assert payload["workspace"]["absolute_path"] == r"D:\baitap\aic\multimodal-fusion-engine"
    assert payload["workspace"]["gpu_available"] is False
    assert payload["workspace"]["agents_md"] == "absent"
    assert payload["workspace"]["package_lock_or_pyproject_sha256"]
    assert payload["baseline_provenance"]["baseline_captured_before_first_corrective_write"]
    assert payload["before"]["dirty_paths_before"]
    assert payload["before"]["untracked_paths_before"]
    assert "mutate BTC" in " ".join(payload["prohibited_actions"])


def test_source_ledger_has_authority_grades_team_cards_search_and_conflicts() -> None:
    payload = _load("keyframe_augmentation_source_ledger_2026-08-30.yaml")
    sources = payload["sources"]

    required = {
        "source_id",
        "title",
        "url_or_path",
        "source_type",
        "publisher_or_team",
        "authors",
        "published_at",
        "accessed_at",
        "version_commit_tag",
        "archive_or_content_sha256",
        "primary",
        "authority_grade",
        "claims_supported",
        "claims_contradicted",
        "reproducible_artifact",
        "dataset_and_metric",
        "transfer_risk_to_AIC",
        "notes",
    }
    assert all(required <= set(source) for source in sources)
    assert {source["authority_grade"] for source in sources} >= {"A0", "A1", "A2", "B", "C", "D"}
    allowed_types = {
        "official_rule",
        "organizer_data",
        "peer_reviewed_paper",
        "preprint",
        "official_code",
        "official_issue",
        "proceedings",
        "leaderboard",
        "team_report",
        "forum",
        "blog",
        "social",
        "internal_note",
    }
    assert {source["source_type"] for source in sources} <= allowed_types
    assert all(isinstance(source["authors"], list) for source in sources)
    assert all(
        claim.startswith("C-")
        for source in sources
        for claim in source["claims_supported"] + source["claims_contradicted"]
    )
    assert len(payload["team_cards"]) == 4
    team_fields = {
        "team_id",
        "team_name",
        "strength_evidence",
        "system_components",
        "keyframe_strategy",
        "retrieval_fusion_reranker",
        "reported_ablation",
        "public_code",
        "applicable_to_AIC_BTC",
        "non_transferable_parts",
        "hypotheses_generated",
    }
    assert all(team_fields <= set(card) for card in payload["team_cards"])
    assert len(payload["search_log"]["clusters"]) == 6
    assert all(cluster["rounds"] == 3 for cluster in payload["search_log"]["clusters"])
    assert payload["conflict_ledger"]
    assert {row["resolution"] for row in payload["conflict_ledger"]} <= {
        "accepted_source",
        "experiment_needed",
        "unresolved",
    }


def test_math_ledger_is_machine_readable_and_has_required_fps_properties() -> None:
    payload = _load("keyframe_augmentation_math_call_site_ledger_2026-08-30.yaml")
    operations = payload["required_operations"]

    assert operations["frame_idx_from_pts"].startswith("floor")
    assert operations["hard_delta_budget"].startswith("floor")
    assert operations["k_target_and_min_gap"] == "ceil"
    assert operations["timestamp_ms"] == "round(pts_time * 1000)"
    assert set(payload["property_tests"]["fps_values"]) == {25.0, 30.0, 29.97, 26.44}
    assert {row["status"] for row in payload["call_sites"]} <= {
        "correct",
        "bug",
        "ambiguous",
        "dead_code",
    }
    assert len(payload["call_sites"]) >= 20


def test_hypothesis_backlog_has_falsification_and_priority_for_every_item() -> None:
    payload = _load("keyframe_augmentation_hypothesis_backlog_2026-08-30.yaml")

    assert payload["decision_authority"] == "diagnostic_only_until_official_ab"
    for hypothesis in payload["hypotheses"]:
        assert hypothesis["proposal_id"].startswith("P-")
        assert hypothesis["mechanism"]
        assert hypothesis["target_failure_slice"]
        assert hypothesis["control_arm"]
        assert hypothesis["treatment_arm"]
        assert hypothesis["falsification_rule"]
        assert set(hypothesis["priority_score"]) == {
            "expected_gain",
            "evidence_strength",
            "implementation_cost",
            "regression_risk",
        }


def test_experiment_matrix_separates_primary_underfill_and_factorized_effects() -> None:
    payload = _load("keyframe_augmentation_experiment_matrix_2026-08-30.yaml")

    fairness = payload["fairness"]
    assert fairness["locked_hard_delta_budget"].startswith("floor")
    assert "invalid" in fairness["underfill_policy"]["primary_ab"]
    assert "separately labeled" in fairness["underfill_policy"]["matched_underfill_secondary"]
    assert payload["budget_proof"]["underfill_does_not_satisfy_primary_equal_budget_proof"]
    variables = {
        row["id"]: row["variable"] for row in payload["factorized_ablations"]
    }
    assert variables == {
        "E-S0": "selector",
        "E-X1": "candidate_exposure",
        "E-C1": "origin_aware_score_calibration",
        "E-XC": "interaction_term",
    }


def test_oracle_schema_supports_query_types_and_exact_failure_stages() -> None:
    payload = _load("keyframe_augmentation_oracle_schema_2026-08-30.yaml")

    assert payload["status"] == "VALIDATOR_READY_EXECUTION_BLOCKED"
    assert set(payload["failure_stage_enum"]) == {
        "temporal",
        "representation",
        "retrieval",
        "fusion",
        "reranker",
        "scorer",
        "none",
    }
    assert payload["oracle_definitions"]["O1-T"]["forbidden_inference"]
    assert payload["oracle_definitions"]["O1-S"]["lower_bound_rule"]
    assert len(payload["oracle_definitions"]["O1-R"]["ceilings"]) == 3


def test_blocked_oracle_row_does_not_infer_stage_scores() -> None:
    row = blocked_oracle_row(
        {
            "query_id": "q1",
            "query_type": "QA",
            "video_id": "V1",
            "start_frame": 10,
            "end_frame": 10,
        },
        provenance_hashes={"gate0_dataset_hash": "d" * 64},
    )

    assert row.interval == (10, 10)
    assert row.evidence_status == "BLOCKED_MISSING_STAGE_ARTIFACT"
    assert row.failure_stage == "representation"
    assert row.best_source_frame_idx is None
    assert row.best_source_stage_scores == {}
    assert row.schema_version == 2
    assert "gt_interval_inclusive" in row.model_dump()
    assert "interval" not in row.model_dump()


def test_oracle_interval_is_inclusive() -> None:
    row = OracleEvidenceRow(
        query_id="q1",
        query_type="TRAKE",
        video_id="V1",
        interval=(7, 7),
        failure_stage="temporal",
        evidence_status="BLOCKED_MISSING_STAGE_ARTIFACT",
        provenance_hashes={"gate0_dataset_hash": "d" * 64},
    )

    assert row.interval == (7, 7)
    with pytest.raises(ValueError, match="inclusive"):
        OracleEvidenceRow(
            query_id="q1",
            query_type="TRAKE",
            video_id="V1",
            interval=(8, 7),
            failure_stage="temporal",
            evidence_status="BLOCKED_MISSING_STAGE_ARTIFACT",
            provenance_hashes={"gate0_dataset_hash": "d" * 64},
        )


def test_oracle_v2_canonical_fields_and_v1_aliases_are_compatible() -> None:
    legacy = OracleEvidenceRow(
        schema_version=1,
        query_id="q-legacy",
        query_type="KIS",
        video_id="V1",
        interval=(3, 4),
        best_btc_stage_scores={"retrieval": 0.25},
        best_source_stage_scores={"retrieval": 0.5},
        failure_stage="retrieval",
        evidence_status="BLOCKED_MISSING_STAGE_ARTIFACT",
        provenance_hashes={"legacy_artifact": "a" * 64},
    )

    dumped = legacy.model_dump()
    assert dumped["gt_interval_inclusive"] == (3, 4)
    assert dumped["best_btc_score_by_stage"] == {"retrieval": 0.25}
    assert dumped["best_source_score_by_stage"] == {"retrieval": 0.5}
    assert "best_btc_stage_scores" not in dumped


def test_measured_oracle_requires_source_frame_rank_and_stage_scores() -> None:
    common = {
        "query_id": "q1",
        "query_type": "KIS",
        "video_id": "V1",
        "interval": (10, 20),
        "failure_stage": "none",
        "evidence_status": "MEASURED",
        "provenance_hashes": {"source_run_hash": "s" * 64},
    }
    with pytest.raises(ValueError, match="best source frame"):
        OracleEvidenceRow(**common)

    row = OracleEvidenceRow(
        **common,
        best_source_frame_idx=12,
        best_source_rank=1,
        best_source_stage_scores={"retrieval": 0.9, "reranker": 0.8},
        o1_s_delta=0.1,
        o1_r_delta=0.2,
    )
    assert validate_oracle_rows([row.model_dump()])[0] == row


def test_oracle_rows_reject_duplicate_query_event_identity() -> None:
    row = {
        "query_id": "q1",
        "query_type": "KIS",
        "video_id": "V1",
        "interval": (1, 2),
        "failure_stage": "temporal",
        "evidence_status": "BLOCKED_MISSING_STAGE_ARTIFACT",
        "provenance_hashes": {"gate0": "d" * 64},
    }

    with pytest.raises(ValueError, match="duplicate"):
        validate_oracle_rows([row, row])


def test_feature_parity_exact_metadata_match_without_canary_is_unverified() -> None:
    fingerprint = _parity_fingerprint()

    report = compare_feature_parity(fingerprint, dict(fingerprint))

    assert report["status"] == "UNVERIFIED"
    assert report["metadata_status"] == "MATCH"
    assert report["missing_fields"] == []
    assert report["mismatched_fields"] == []
    assert report["decision_authority"] == "diagnostic_only"


def test_feature_parity_requires_passing_frozen_numeric_canary_for_measured() -> None:
    fingerprint = _parity_fingerprint()

    report = compare_feature_parity(
        fingerprint,
        dict(fingerprint),
        canary_evidence={
            "sample_count": 32,
            "btc_artifact_sha256": "b" * 64,
            "delta_redecode_artifact_sha256": "d" * 64,
            "max_abs_diff": 1e-7,
            "tolerance": 1e-6,
        },
    )

    assert report["status"] == "MEASURED"
    assert report["canary_evidence"]["status"] == "PASS"


def test_feature_parity_mismatch_blocks_even_when_embed_dim_matches() -> None:
    btc = _parity_fingerprint()
    supplemental = dict(btc)
    supplemental["color_order"] = "BGR"
    supplemental["embed_dim"] = btc["embed_dim"]

    report = compare_feature_parity(btc, supplemental)

    assert report["status"] == "BLOCKED"
    assert report["mismatched_fields"] == ["color_order"]


def test_crowding_reports_btc_displacement() -> None:
    btc_ids = {"V1#000001", "V1#000002", "V1#000003"}
    ranked = ["V2#000010", "V2#000010", "V2#000011", "V1#000001", "V2#000012"]

    report = crowding_report(
        ranked,
        btc_keyframe_ids=btc_ids,
        top_k=5,
        baseline_btc_top_k=["V1#000001", "V1#000002", "V1#000003"],
        near_duplicate_cluster_by_id={
            "V2#000010": "cluster-a",
            "V2#000011": "cluster-a",
            "V1#000001": "cluster-b",
            "V2#000012": "cluster-c",
        },
        scores_by_id={
            "V2#000010": 0.9,
            "V2#000011": 0.8,
            "V1#000001": 0.7,
            "V2#000012": 0.6,
        },
        stage_rankings={"retrieval": ranked, "reranker": reversed(ranked)},
    )

    assert report["supplemental_rows_top_k"] == 4
    assert report["duplicate_rows_top_k"] == 1
    assert report["btc_displaced_from_baseline_top_k"] == 2
    assert report["best_btc_rank_delta"] == 3
    assert report["max_near_duplicate_cluster_size_top_k"] == 3
    assert report["score_distribution_by_origin"]["supplemental"]["count"] == 4
    assert report["per_origin_exposure_by_stage"]["retrieval"]["btc"] == 1
    assert report["evidence_status"] == "MEASURED"
    assert report["decision_authority"] == "diagnostic_only"


def test_ocr_box_floor_ceil_contains_polygon() -> None:
    box = _quad_to_box([1.2, 2.8, 9.1, 2.9, 9.1, 11.2, 1.2, 11.2])

    assert box == (1, 2, 10, 12)


def test_adversarial_manifest_has_all_fifteen_cases_and_no_unlabeled_case() -> None:
    payload = _load("keyframe_augmentation_adversarial_cases_2026-08-30.yaml")

    cases = payload["cases"]
    assert len(cases) == 15
    assert {case["id"] for case in cases} == {f"ADV-{index:02d}" for index in range(1, 16)}
    assert all(case["test_id"] and case["status"] and case["command"] for case in cases)
