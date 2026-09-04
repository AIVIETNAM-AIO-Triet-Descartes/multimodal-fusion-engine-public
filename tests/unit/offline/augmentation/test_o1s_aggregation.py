from __future__ import annotations

import numpy as np
import pytest

from mfe.offline.augmentation.io import canonical_hash
from mfe.offline.augmentation.o1s_aggregation import (
    O1SEventTrace,
    adversarial_counterexample_summary,
    classify_siglip_failure_stage,
    cosine_scores,
    deterministic_best,
    deterministic_injected_rank,
    interval_candidate_indices,
    normalize_feature_rows,
    o1s_math_call_site_ledger,
    validate_o1s_worker_manifest,
)


def test_normalize_and_cosine_are_finite_and_exact_for_basis_vectors() -> None:
    matrix = normalize_feature_rows(np.asarray([[3.0, 0.0], [0.0, 2.0]]))
    query = normalize_feature_rows(np.asarray([[4.0, 0.0]]))[0]
    assert cosine_scores(matrix, query).tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_feature_rows(np.zeros((1, 2), dtype=np.float32))


def test_best_and_injected_rank_use_identity_tie_break() -> None:
    ids = ["L21_V001#000002", "L21_V001#000001", "L21_V001#000003"]
    scores = [0.5, 0.5, 0.4]
    assert deterministic_best(ids, scores) == ("L21_V001#000001", 0.5, 1)
    assert deterministic_injected_rank(
        ids,
        scores,
        candidate_id="L21_V001#000000",
        candidate_score=0.5,
    ) == 1
    assert deterministic_injected_rank(
        ids,
        scores,
        candidate_id="L21_V001#000004",
        candidate_score=0.5,
    ) == 3


def test_interval_positions_use_canonical_frame_identity() -> None:
    ids = [
        "L21_V001#000010",
        "L21_V002#000011",
        "L21_V001#000012",
    ]
    assert interval_candidate_indices(
        ids, video_id="L21_V001", start_frame=10, end_frame=11
    ).tolist() == [0]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "has_btc_in_interval": False,
                "source_signal_delta": None,
                "best_source_rank": 1,
                "best_btc_rank": None,
            },
            "temporal",
        ),
        (
            {
                "has_btc_in_interval": True,
                "source_signal_delta": 0.1,
                "best_source_rank": 20,
                "best_btc_rank": 40,
            },
            "representation",
        ),
        (
            {
                "has_btc_in_interval": True,
                "source_signal_delta": 0.1,
                "best_source_rank": 101,
                "best_btc_rank": 130,
            },
            "retrieval",
        ),
        (
            {
                "has_btc_in_interval": True,
                "source_signal_delta": 0.0,
                "best_source_rank": 10,
                "best_btc_rank": 10,
            },
            "none",
        ),
    ],
)
def test_failure_stage_is_limited_to_observed_siglip_boundary(kwargs: dict, expected: str) -> None:
    assert classify_siglip_failure_stage(
        **kwargs,
        exposure_depth=100,
        improvement_epsilon=1e-6,
    ) == expected


def test_event_schema_rejects_temporal_claim_with_interval_btc() -> None:
    payload = {
        "query_id": "q",
        "query_type": "KIS",
        "event_index": 0,
        "video_id": "L21_V001",
        "gt_interval_inclusive": (10, 20),
        "sampling_policy": "inclusive_uniform_integer_lb64",
        "full_interval_scan": True,
        "best_btc_frame_idx": 11,
        "best_btc_rank": 2,
        "best_btc_score_by_stage": {
            "siglip_dense": 0.5,
            "fusion": None,
            "reranker": None,
            "scorer": None,
        },
        "top_btc_outside_interval_id": "L21_V002#000001",
        "top_btc_outside_interval_rank": 1,
        "top_btc_outside_interval_score": 0.7,
        "best_source_frame_idx": 12,
        "best_source_rank": 1,
        "best_source_score_by_stage": {
            "siglip_dense": 0.8,
            "fusion": None,
            "reranker": None,
            "scorer": None,
        },
        "o1_s_delta": 0.3,
        "failure_stage": "temporal",
        "failure_stage_status": "MEASURED_SIGLIP_BOUNDARY_ONLY",
        "provenance_hashes": {"x": "a" * 64},
    }
    with pytest.raises(ValueError, match="cannot be classified temporal"):
        O1SEventTrace.model_validate(payload)


def test_worker_manifest_validation_fails_closed_on_assignment_or_hash_drift() -> None:
    assigned = ["L21_V001", "L21_V002"]
    manifest = {
        "status": "COMPLETE",
        "decision_authority": "diagnostic_only",
        "runner": "quang",
        "runners": ["quang", "trong"],
        "run_id": "o1s-lb64-scan-v1",
        "scan_plan_hash": "a" * 64,
        "sampling_policy": "inclusive_uniform_integer_lb64",
        "max_samples_per_event": 64,
        "full_o1s_status": "BLOCKED_SAMPLED_INTERVALS_ARE_LOWER_BOUND",
        "notebook18_report_sha256": "c" * 64,
        "targets_sha256": "d" * 64,
        "assigned": assigned,
        "assigned_count": 2,
        "assigned_frame_count": 10,
        "verified_complete": assigned,
        "verified_complete_count": 2,
        "failures_this_session": [],
        "repo_commit": "b" * 40,
        "siglip_canary_report_hash": "e" * 64,
        "heldout_accessed": False,
        "btc_mutated": False,
    }
    manifest["manifest_payload_hash"] = canonical_hash(manifest)
    kwargs = {
        "runner": "quang",
        "run_id": "o1s-lb64-scan-v1",
        "scan_plan_hash": "a" * 64,
        "expected_assigned": assigned,
        "expected_frame_count": 10,
        "expected_repo_commit": "b" * 40,
        "expected_notebook18_report_sha256": "c" * 64,
        "expected_targets_sha256": "d" * 64,
        "expected_siglip_canary_report_hash": "e" * 64,
    }
    assert validate_o1s_worker_manifest(manifest, **kwargs)
    assert not validate_o1s_worker_manifest(
        {**manifest, "verified_complete": assigned[:1]}, **kwargs
    )


def test_math_ledger_is_unique_and_names_every_decision_formula() -> None:
    ledger = o1s_math_call_site_ledger()
    assert [row["formula_id"] for row in ledger] == [
        "M-O1S-001",
        "M-O1S-002",
        "M-O1S-003",
        "M-O1S-004",
        "M-O1S-005",
        "M-O1S-006",
        "M-O1S-007",
    ]
    assert len({row["symbol"] for row in ledger}) == len(ledger)


def _trace(**updates) -> O1SEventTrace:
    payload = {
        "query_id": "q",
        "query_type": "KIS",
        "event_index": 0,
        "video_id": "L21_V001",
        "gt_interval_inclusive": (10, 100),
        "sampling_policy": "inclusive_uniform_integer_lb64",
        "full_interval_scan": False,
        "best_btc_frame_idx": 20,
        "best_btc_rank": 120,
        "best_btc_score_by_stage": {
            "siglip_dense": 0.5,
            "fusion": None,
            "reranker": None,
            "scorer": None,
        },
        "top_btc_outside_interval_id": "L21_V001#000101",
        "top_btc_outside_interval_rank": 1,
        "top_btc_outside_interval_score": 0.9,
        "best_source_frame_idx": 30,
        "best_source_rank": 101,
        "best_source_score_by_stage": {
            "siglip_dense": 0.7,
            "fusion": None,
            "reranker": None,
            "scorer": None,
        },
        "o1_s_delta": 0.2,
        "o1_r_delta": 0.0,
        "failure_stage": "retrieval",
        "failure_stage_status": "MEASURED_SIGLIP_BOUNDARY_ONLY_DOWNSTREAM_BLOCKED",
        "provenance_hashes": {"input": "f" * 64},
    }
    payload.update(updates)
    return O1SEventTrace.model_validate(payload)


def test_event_schema_checks_lb64_delta_stage_nulls_and_outside_identity() -> None:
    _trace()
    with pytest.raises(ValueError, match="full_interval_scan"):
        _trace(full_interval_scan=True)
    with pytest.raises(ValueError, match="O1-S delta"):
        _trace(o1_s_delta=0.1)
    with pytest.raises(ValueError, match="downstream source"):
        _trace(
            best_source_score_by_stage={
                "siglip_dense": 0.7,
                "fusion": 0.6,
                "reranker": None,
                "scorer": None,
            }
        )
    with pytest.raises(ValueError, match="lies inside"):
        _trace(top_btc_outside_interval_id="L21_V001#000050")


def test_event_schema_accepts_exact_full_scan_for_long_interval() -> None:
    trace = _trace(
        sampling_policy="inclusive_full_interval_exact",
        full_interval_scan=True,
    )
    assert trace.full_interval_scan is True
    assert trace.sampling_policy == "inclusive_full_interval_exact"


def test_adversarial_counterexamples_block_common_overclaims() -> None:
    summary = adversarial_counterexample_summary(
        [_trace()], exposure_depth=100, improvement_epsilon=1e-6
    )
    assert summary == {
        "sampled_intervals_where_no_gain_is_not_full_o1s_evidence": 1,
        "source_signal_gain_but_still_outside_exposure": 1,
        "source_signal_gain_but_nonpositive_o1r_score_delta": 1,
        "positive_o1r_score_delta_without_sampled_source_gain": 0,
        "temporal_missing_events": 0,
    }
