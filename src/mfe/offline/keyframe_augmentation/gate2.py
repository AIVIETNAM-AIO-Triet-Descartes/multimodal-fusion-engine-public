"""Gate 2: equal-budget B0-B4 official-score and systems comparison."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from mfe.offline.keyframe_augmentation.common import sha256_json
from mfe.offline.keyframe_augmentation.models import AugmentationConfig

ARMS = ("B0", "B1", "B2", "B3", "B4")


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _load_official_scores(
    path: Path,
) -> tuple[dict[str, float], float, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_query")
    if not isinstance(rows, list):
        raise ValueError(
            f"{path}: official held-out aggregate or incomplete output cannot be tuned/audited"
        )
    scores: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = row.get("query_id")
        wide = row.get("wide")
        if not isinstance(query_id, str) or not isinstance(wide, dict):
            raise ValueError(f"{path}: malformed official per_query row")
        if query_id in scores:
            raise ValueError(f"{path}: duplicate official query_id {query_id}")
        scores[query_id] = float(wide["final_score"])
        per_k = wide.get("r_at_k")
        if not isinstance(per_k, dict):
            raise ValueError(f"{path}: query {query_id} is missing official R@k values")
        normalized_per_k = {int(k): float(value) for k, value in per_k.items()}
        if set(normalized_per_k) != {1, 5, 20, 50, 100}:
            raise ValueError(f"{path}: query {query_id} has non-official R@k thresholds")
        query_type = wide.get("query_type")
        if not isinstance(query_type, str):
            raise ValueError(f"{path}: query {query_id} is missing query_type")
        details[query_id] = {
            "query_type": query_type,
            "r_at_k": normalized_per_k,
            "first_hit_rank": next(
                (
                    rank
                    for rank, value in enumerate(wide.get("r_scores", []), 1)
                    if float(value) > 0.0
                ),
                None,
            ),
        }
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    reported = payload.get("mean_final_score")
    if reported is not None and not math.isclose(mean, float(reported), abs_tol=1e-12):
        raise ValueError(f"{path}: per-query mean disagrees with official aggregate")
    return scores, mean, details


def _load_score_run(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    run = json.loads(source.read_text(encoding="utf-8"))
    official_path = Path(run["official_score_path"])
    if not official_path.is_absolute():
        official_path = (source.parent / official_path).resolve()
    scores, mean, details = _load_official_scores(official_path)
    run["source_path"] = str(source)
    run["official_score_path"] = str(official_path)
    run["scores"] = scores
    run["mean_final_score"] = mean
    run["score_details"] = details
    latencies = [float(value) for value in run.get("latency_ms", [])]
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError(f"{path}: latency values must be finite and non-negative")
    run["p50_latency_ms"] = _percentile(latencies, 0.50)
    run["p95_latency_ms"] = _percentile(latencies, 0.95)
    run["p99_latency_ms"] = _percentile(latencies, 0.99)
    return run


def load_arm_run(path: str | Path) -> dict[str, Any]:
    run = _load_score_run(path)
    if run.get("arm") not in ARMS:
        raise ValueError(f"{path}: arm must be one of {ARMS}")
    return run


def load_oracle_run(path: str | Path) -> dict[str, Any]:
    run = _load_score_run(path)
    if run.get("oracle") not in {"O1-S", "O1-R"}:
        raise ValueError(f"{path}: oracle must be O1-S or O1-R")
    return run


def run_oracle_diagnostic(
    *,
    baseline: dict[str, Any],
    source_oracle: dict[str, Any],
    force_rank_oracle: dict[str, Any],
    dataset_hash: str,
) -> dict[str, Any]:
    """Measure semantic/representative and force-rank ceilings with real scores."""
    if baseline.get("arm") != "B0" or int(baseline.get("delta_count", -1)) != 0:
        raise ValueError("oracle diagnostic baseline must be BTC-only B0")
    if source_oracle.get("oracle") != "O1-S":
        raise ValueError("source oracle must declare oracle=O1-S")
    if force_rank_oracle.get("oracle") != "O1-R":
        raise ValueError("force-rank oracle must declare oracle=O1-R")
    if any(
        run.get("dataset_hash") != dataset_hash
        for run in (baseline, source_oracle, force_rank_oracle)
    ):
        raise ValueError("oracle run dataset hashes do not match Gate 0")
    if source_oracle.get("retrieval_fingerprint") != baseline.get("retrieval_fingerprint"):
        raise ValueError("O1-S must use the frozen B0 retrieval/fusion/reranker pipeline")
    if source_oracle.get("feature_parity_status") != "MEASURED":
        raise ValueError("O1-S requires measured delta feature parity")
    if source_oracle.get("oracle_contract") != "best_source_frame_inclusive_gt_interval":
        raise ValueError("O1-S contract must evaluate the full inclusive GT interval")
    if force_rank_oracle.get("oracle_contract") != "force_rank_ground_truth":
        raise ValueError("O1-R contract must force-rank ground truth")
    required_provenance = ("candidate_manifest_hash", "selection_artifact_sha256")
    if any(not source_oracle.get(key) for key in required_provenance):
        raise ValueError("O1-S is missing candidate/selection provenance")

    query_ids = set(baseline["scores"])
    if set(source_oracle["scores"]) != query_ids or set(force_rank_oracle["scores"]) != query_ids:
        raise ValueError("B0/O1-S/O1-R must score identical query IDs")
    rows: list[dict[str, Any]] = []
    for query_id in sorted(query_ids):
        detail = baseline["score_details"][query_id]
        baseline_score = baseline["scores"][query_id]
        source_score = source_oracle["scores"][query_id]
        force_score = force_rank_oracle["scores"][query_id]
        rows.append(
            {
                "query_id": query_id,
                "query_type": detail["query_type"],
                "btc_final_score": baseline_score,
                "btc_first_hit_rank": detail["first_hit_rank"],
                "o1_s_final_score": source_score,
                "o1_s_delta": source_score - baseline_score,
                "o1_r_final_score": force_score,
                "o1_r_delta": force_score - baseline_score,
            }
        )
    query_types = sorted({row["query_type"] for row in rows})
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_hash": dataset_hash,
        "queries": len(rows),
        "baseline_mean_final_score": baseline["mean_final_score"],
        "o1_s_mean_final_score": source_oracle["mean_final_score"],
        "o1_r_mean_final_score": force_rank_oracle["mean_final_score"],
        "o1_s_headroom_queries": sum(row["o1_s_delta"] > 0 for row in rows),
        "o1_r_headroom_queries": sum(row["o1_r_delta"] > 0 for row in rows),
        "by_query_type": {
            query_type: {
                "queries": sum(row["query_type"] == query_type for row in rows),
                "o1_s_mean_delta": sum(
                    row["o1_s_delta"] for row in rows if row["query_type"] == query_type
                )
                / sum(row["query_type"] == query_type for row in rows),
                "o1_r_mean_delta": sum(
                    row["o1_r_delta"] for row in rows if row["query_type"] == query_type
                )
                / sum(row["query_type"] == query_type for row in rows),
            }
            for query_type in query_types
        },
        "per_query": rows,
        "provenance": {
            "retrieval_fingerprint": baseline.get("retrieval_fingerprint"),
            "candidate_manifest_hash": source_oracle["candidate_manifest_hash"],
            "selection_artifact_sha256": source_oracle["selection_artifact_sha256"],
        },
        "evidence": {"official_scores": "MEASURED", "feature_parity": "MEASURED"},
    }
    report["run_hash"] = sha256_json(report)
    return report


def _cluster_bootstrap_interval(
    deltas: dict[str, float],
    query_to_video: dict[str, str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | list[float]] | None:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    by_video: dict[str, list[float]] = defaultdict(list)
    for query_id, delta in deltas.items():
        if query_id not in query_to_video:
            raise ValueError(f"query {query_id} is missing from the locked DEV map")
        by_video[query_to_video[query_id]].append(delta)
    videos = sorted(by_video)
    if not videos:
        return None
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(videos) for _ in videos]
        values = [value for video_id in selected for value in by_video[video_id]]
        bootstrapped.append(sum(values) / len(values))
    lower_one_sided = _percentile(bootstrapped, 0.05)
    lower_two_sided = _percentile(bootstrapped, 0.025)
    upper_two_sided = _percentile(bootstrapped, 0.975)
    assert lower_one_sided is not None
    assert lower_two_sided is not None
    assert upper_two_sided is not None
    return {
        "lcb_95_one_sided": lower_one_sided,
        "interval_95_two_sided": [lower_two_sided, upper_two_sided],
    }


def _cluster_bootstrap_lcb(
    deltas: dict[str, float],
    query_to_video: dict[str, str],
    *,
    samples: int,
    seed: int,
) -> float | None:
    """Compatibility wrapper for callers that only consume the one-sided LCB."""

    interval = _cluster_bootstrap_interval(
        deltas, query_to_video, samples=samples, seed=seed
    )
    return None if interval is None else float(interval["lcb_95_one_sided"])


def run_gate2(
    *,
    runs: list[dict[str, Any]],
    config: AugmentationConfig,
    dataset_hash: str,
    query_to_video: dict[str, str],
    coverage_gap_query_ids: set[str],
) -> dict[str, Any]:
    by_arm = {run["arm"]: run for run in runs}
    if len(by_arm) != len(runs):
        raise ValueError("Gate 2 received duplicate arm run(s)")
    missing = sorted(set(ARMS) - set(by_arm))
    if missing:
        raise ValueError(f"Gate 2 requires B0-B4; missing {missing}")
    if any(run.get("split") != "dev" for run in runs):
        raise ValueError("Gate 2 tuning/comparison accepts DEV runs only")
    if any(run.get("dataset_hash") != dataset_hash for run in runs):
        raise ValueError("Gate 2 run dataset hashes do not match Gate 0")
    fingerprints = {str(run.get("retrieval_fingerprint")) for run in runs}
    if len(fingerprints) != 1 or "None" in fingerprints:
        raise ValueError("B0-B4 must use one explicit retrieval/fusion/reranker fingerprint")

    budgets = {int(by_arm[arm].get("delta_count", -1)) for arm in ARMS[1:]}
    if len(budgets) != 1 or next(iter(budgets)) < 0:
        raise ValueError("B1-B4 must declare the same non-negative delta_count")
    if int(by_arm["B0"].get("delta_count", -1)) != 0:
        raise ValueError("B0 must be BTC-only with delta_count=0")
    common_budget = next(iter(budgets))

    baseline_scores = by_arm["B0"]["scores"]
    query_ids = set(baseline_scores)
    if any(set(run["scores"]) != query_ids for run in runs):
        raise ValueError("B0-B4 official outputs must score the identical query IDs")
    baseline_p95 = by_arm["B0"]["p95_latency_ms"]
    baseline_details = by_arm["B0"]["score_details"]
    comparisons: dict[str, Any] = {}
    for arm in ARMS[1:]:
        run = by_arm[arm]
        deltas = {
            query_id: run["scores"][query_id] - baseline_scores[query_id]
            for query_id in sorted(query_ids)
        }
        gap_values = [
            delta for query_id, delta in deltas.items() if query_id in coverage_gap_query_ids
        ]
        global_delta = sum(deltas.values()) / len(deltas) if deltas else 0.0
        gap_delta = sum(gap_values) / len(gap_values) if gap_values else None
        bootstrap = _cluster_bootstrap_interval(
            deltas,
            query_to_video,
            samples=config.bootstrap_samples,
            seed=config.seed,
        )
        lcb = None if bootstrap is None else float(bootstrap["lcb_95_one_sided"])
        p95 = run["p95_latency_ms"]
        retrieval_headroom_ids = {
            query_id for query_id, score in baseline_scores.items() if score < 1.0
        }
        retrieval_headroom_values = [
            delta for query_id, delta in deltas.items() if query_id in retrieval_headroom_ids
        ]
        r_at_k_delta: dict[int, float] = {}
        for k in (1, 5, 20, 50, 100):
            values = [
                run["score_details"][query_id]["r_at_k"][k]
                - baseline_details[query_id]["r_at_k"][k]
                for query_id in sorted(query_ids)
            ]
            r_at_k_delta[k] = sum(values) / len(values) if values else 0.0
        by_type_delta: dict[str, float] = {}
        query_types = sorted({row["query_type"] for row in baseline_details.values()})
        for query_type in query_types:
            values = [
                delta
                for query_id, delta in deltas.items()
                if baseline_details[query_id]["query_type"] == query_type
            ]
            by_type_delta[query_type] = sum(values) / len(values) if values else 0.0
        latency_ratio = (
            (p95 - baseline_p95) / baseline_p95
            if p95 is not None and baseline_p95 not in {None, 0.0}
            else None
        )
        comparisons[arm] = {
            "official_mean_final_score": run["mean_final_score"],
            "official_delta_vs_b0": global_delta,
            "coverage_gap_queries": len(gap_values),
            "coverage_gap_delta": gap_delta,
            "baseline_retrieval_headroom_queries": len(retrieval_headroom_values),
            "retrieval_headroom_delta": (
                sum(retrieval_headroom_values) / len(retrieval_headroom_values)
                if retrieval_headroom_values
                else None
            ),
            "r_at_k_delta_vs_b0": r_at_k_delta,
            "by_query_type_delta_vs_b0": by_type_delta,
            "global_delta_lcb_95_one_sided": lcb,
            "global_delta_interval_95_two_sided": (
                None if bootstrap is None else bootstrap["interval_95_two_sided"]
            ),
            "paired_query_deltas": deltas,
            "winning_query_ids": sorted(
                query_id for query_id, delta in deltas.items() if delta > 0
            ),
            "losing_query_ids": sorted(
                query_id for query_id, delta in deltas.items() if delta < 0
            ),
            "tied_query_ids": sorted(
                query_id for query_id, delta in deltas.items() if delta == 0
            ),
            "rescued_queries": sum(delta > 0 for delta in deltas.values()),
            "regressed_queries": sum(delta < 0 for delta in deltas.values()),
            "net_rescued_queries": sum(delta > 0 for delta in deltas.values())
            - sum(delta < 0 for delta in deltas.values()),
            "p50_latency_ms": run.get("p50_latency_ms"),
            "p95_latency_ms": p95,
            "p99_latency_ms": run.get("p99_latency_ms"),
            "p95_latency_increase_ratio": latency_ratio,
            "storage_bytes": run.get("storage_bytes"),
            "preprocess_seconds": run.get("preprocess_seconds"),
            "cost_status": run.get("cost_status", "UNVERIFIED"),
            "feature_parity_status": run.get("feature_parity_status", "UNVERIFIED"),
            "duplicate_rate": run.get("duplicate_rate"),
            "crowding": run.get("crowding"),
            "non_inferiority_pass": lcb is not None and lcb >= -config.non_inferiority_margin,
            "latency_pass": latency_ratio is not None
            and latency_ratio <= config.max_p95_latency_increase_ratio,
            "multiple_comparison_risk": True,
            "decision_authority": "official_dev_gate",
        }
        if gap_values:
            comparisons[arm]["target_slice"] = "temporal_coverage_gap"
            comparisons[arm]["target_slice_queries"] = len(gap_values)
            comparisons[arm]["target_slice_delta"] = gap_delta
        else:
            comparisons[arm]["target_slice"] = "baseline_retrieval_headroom"
            comparisons[arm]["target_slice_queries"] = len(retrieval_headroom_values)
            comparisons[arm]["target_slice_delta"] = comparisons[arm][
                "retrieval_headroom_delta"
            ]

    b3 = comparisons["B3"]
    all_feature_parity_measured = all(
        run.get("feature_parity_status") == "MEASURED" for run in runs
    )
    go = (
        b3["target_slice_delta"] is not None
        and b3["target_slice_delta"] > 0
        and b3["non_inferiority_pass"]
        and b3["latency_pass"]
        and b3["cost_status"] == "MEASURED"
        and all_feature_parity_measured
    )
    o1_r_status = str(by_arm["B3"].get("o1_r_headroom_status", "UNVERIFIED"))
    o1_r_headroom = bool(by_arm["B3"].get("o1_r_headroom_verified", False))
    if go:
        decision = "GO_TO_FROZEN_HELDOUT"
    elif o1_r_status != "MEASURED":
        decision = "BLOCKED_ORACLE_UNVERIFIED_KEEP_BTC_ONLY"
    elif o1_r_headroom:
        decision = "NO_GO_B3_ON_DEV_ALLOW_DAKE_V2"
    else:
        decision = "NO_GO_KEYFRAME_RESEARCH_ON_MEASURED_DEV"
    report = {
        "schema_version": 2,
        "dataset_hash": dataset_hash,
        "split": "dev",
        "primary_metric": "official Final Score end-to-end",
        "retrieval_fingerprint": next(iter(fingerprints)),
        "equal_delta_budget": common_budget,
        "baseline": {
            "official_mean_final_score": by_arm["B0"]["mean_final_score"],
            "p50_latency_ms": by_arm["B0"].get("p50_latency_ms"),
            "p95_latency_ms": baseline_p95,
            "p99_latency_ms": by_arm["B0"].get("p99_latency_ms"),
        },
        "comparisons": comparisons,
        "decision": decision,
        "dake_v2_allowed": not go and o1_r_status == "MEASURED" and o1_r_headroom,
        "heldout_used_for_tuning": False,
        "all_feature_parity_measured": all_feature_parity_measured,
        "evidence": {
            "official_scores": "MEASURED",
            "latency": "MEASURED" if baseline_p95 is not None else "UNVERIFIED",
            "cost": "MEASURED"
            if all(run.get("cost_status") == "MEASURED" for run in runs)
            else "UNVERIFIED",
        },
    }
    report["run_hash"] = sha256_json(report)
    return report
