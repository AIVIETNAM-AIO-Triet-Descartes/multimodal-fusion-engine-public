from __future__ import annotations

from mfe.offline.keyframe_augmentation.gate2 import run_gate2, run_oracle_diagnostic
from mfe.offline.keyframe_augmentation.models import AugmentationConfig


def _config() -> AugmentationConfig:
    return AugmentationConfig(
        dataset_version="test",
        split_version="test",
        bootstrap_samples=100,
        non_inferiority_margin=0.1,
    )


def _run(arm: str, scores: dict[str, float], *, delta: int, latency: float) -> dict:
    return {
        "arm": arm,
        "split": "dev",
        "dataset_hash": "d" * 64,
        "retrieval_fingerprint": "frozen-retrieval",
        "delta_count": delta,
        "scores": scores,
        "mean_final_score": sum(scores.values()) / len(scores),
        "score_details": {
            query_id: {
                "query_type": "KIS",
                "r_at_k": {k: score for k in (1, 5, 20, 50, 100)},
                "first_hit_rank": 1 if score > 0 else None,
            }
            for query_id, score in scores.items()
        },
        "p95_latency_ms": latency,
        "storage_bytes": 100,
        "preprocess_seconds": 1.0,
        "cost_status": "MEASURED",
        "feature_parity_status": "MEASURED",
        "o1_r_headroom_status": "MEASURED",
        "o1_r_headroom_verified": False,
    }


def test_gate2_uses_official_score_equal_budget_noninferiority_and_latency() -> None:
    baseline = {"q1": 0.0, "q2": 1.0}
    runs = [_run("B0", baseline, delta=0, latency=100.0)]
    for arm in ("B1", "B2", "B4"):
        runs.append(_run(arm, baseline, delta=2, latency=105.0))
    runs.append(_run("B3", {"q1": 1.0, "q2": 1.0}, delta=2, latency=105.0))

    report = run_gate2(
        runs=runs,
        config=_config(),
        dataset_hash="d" * 64,
        query_to_video={"q1": "V1", "q2": "V2"},
        coverage_gap_query_ids={"q1"},
    )
    repeated = run_gate2(
        runs=runs,
        config=_config(),
        dataset_hash="d" * 64,
        query_to_video={"q1": "V1", "q2": "V2"},
        coverage_gap_query_ids={"q1"},
    )

    assert report["primary_metric"] == "official Final Score end-to-end"
    assert report["equal_delta_budget"] == 2
    assert report["comparisons"]["B3"]["coverage_gap_delta"] == 1.0
    assert report["comparisons"]["B3"]["non_inferiority_pass"]
    assert report["comparisons"]["B3"]["latency_pass"]
    assert report["comparisons"]["B3"]["r_at_k_delta_vs_b0"][1] == 0.5
    assert report["comparisons"]["B3"]["by_query_type_delta_vs_b0"]["KIS"] == 0.5
    assert report["comparisons"]["B3"]["global_delta_interval_95_two_sided"]
    assert report["comparisons"]["B3"]["winning_query_ids"] == ["q1"]
    assert report["comparisons"]["B3"]["tied_query_ids"] == ["q2"]
    assert report["comparisons"]["B3"]["multiple_comparison_risk"] is True
    assert report["run_hash"] == repeated["run_hash"]
    assert report["decision"] == "GO_TO_FROZEN_HELDOUT"


def test_gate2_does_not_issue_global_no_go_when_oracle_is_unverified() -> None:
    baseline = {"q1": 0.0}
    runs = [_run("B0", baseline, delta=0, latency=100.0)]
    for arm in ("B1", "B2", "B3", "B4"):
        run = _run(arm, baseline, delta=1, latency=100.0)
        run["o1_r_headroom_status"] = "UNVERIFIED"
        runs.append(run)

    report = run_gate2(
        runs=runs,
        config=_config(),
        dataset_hash="d" * 64,
        query_to_video={"q1": "V1"},
        coverage_gap_query_ids=set(),
    )

    assert report["decision"] == "BLOCKED_ORACLE_UNVERIFIED_KEEP_BTC_ONLY"
    assert report["dake_v2_allowed"] is False


def test_gate2_can_go_on_retrieval_headroom_when_temporal_gap_slice_is_empty() -> None:
    baseline = {"q1": 0.0, "q2": 1.0}
    runs = [_run("B0", baseline, delta=0, latency=100.0)]
    for arm in ("B1", "B2", "B4"):
        runs.append(_run(arm, baseline, delta=2, latency=100.0))
    runs.append(_run("B3", {"q1": 1.0, "q2": 1.0}, delta=2, latency=100.0))

    report = run_gate2(
        runs=runs,
        config=_config(),
        dataset_hash="d" * 64,
        query_to_video={"q1": "V1", "q2": "V2"},
        coverage_gap_query_ids=set(),
    )

    assert report["comparisons"]["B3"]["target_slice"] == "baseline_retrieval_headroom"
    assert report["comparisons"]["B3"]["target_slice_delta"] == 1.0
    assert report["decision"] == "GO_TO_FROZEN_HELDOUT"


def test_gate2_rejects_unequal_delta_budget() -> None:
    import pytest

    baseline = {"q1": 0.0}
    runs = [_run("B0", baseline, delta=0, latency=100.0)]
    for index, arm in enumerate(("B1", "B2", "B3", "B4"), 1):
        runs.append(_run(arm, baseline, delta=index, latency=100.0))

    with pytest.raises(ValueError, match="same"):
        run_gate2(
            runs=runs,
            config=_config(),
            dataset_hash="d" * 64,
            query_to_video={"q1": "V1"},
            coverage_gap_query_ids=set(),
        )


def test_gate2_rejects_mismatched_dataset_or_downstream_fingerprint() -> None:
    import pytest

    baseline = {"q1": 0.0}
    runs = [_run("B0", baseline, delta=0, latency=100.0)]
    for arm in ("B1", "B2", "B3", "B4"):
        runs.append(_run(arm, baseline, delta=1, latency=100.0))

    runs[1]["dataset_hash"] = "x" * 64
    with pytest.raises(ValueError, match="dataset hashes"):
        run_gate2(
            runs=runs,
            config=_config(),
            dataset_hash="d" * 64,
            query_to_video={"q1": "V1"},
            coverage_gap_query_ids=set(),
        )

    runs[1]["dataset_hash"] = "d" * 64
    runs[1]["retrieval_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        run_gate2(
            runs=runs,
            config=_config(),
            dataset_hash="d" * 64,
            query_to_video={"q1": "V1"},
            coverage_gap_query_ids=set(),
        )


def test_oracle_diagnostic_measures_semantic_and_rank_headroom_with_provenance() -> None:
    baseline = _run("B0", {"q1": 0.0, "q2": 1.0}, delta=0, latency=100.0)
    source = _run("B1", {"q1": 0.8, "q2": 1.0}, delta=2, latency=100.0)
    source.update(
        {
            "oracle": "O1-S",
            "oracle_contract": "best_source_frame_inclusive_gt_interval",
            "candidate_manifest_hash": "c" * 64,
            "selection_artifact_sha256": "s" * 64,
        }
    )
    force = _run("B1", {"q1": 1.0, "q2": 1.0}, delta=2, latency=100.0)
    force.update({"oracle": "O1-R", "oracle_contract": "force_rank_ground_truth"})

    report = run_oracle_diagnostic(
        baseline=baseline,
        source_oracle=source,
        force_rank_oracle=force,
        dataset_hash="d" * 64,
    )

    assert report["o1_s_headroom_queries"] == 1
    assert report["o1_r_headroom_queries"] == 1
    assert report["per_query"][0]["btc_first_hit_rank"] is None
    assert report["evidence"]["feature_parity"] == "MEASURED"
