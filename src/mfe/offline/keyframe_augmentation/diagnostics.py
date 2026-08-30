"""Local diagnostics for feature parity and rank crowding.

These functions compare metadata and ranked IDs only.  They never write a
feature/index/live-store artifact and return ``BLOCKED`` when a parity field is
missing or differs, so equal vector dimensions cannot masquerade as parity.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

FEATURE_PARITY_FIELDS = (
    "model",
    "weights_sha256",
    "decode_library",
    "color_order",
    "resize",
    "interpolation",
    "normalization",
    "dtype",
    "device",
    "batch_size",
    "embed_dim",
    "embedding_norm",
    "faiss",
    "bm25",
    "fusion",
    "reranker",
)


def compare_feature_parity(
    btc: Mapping[str, Any],
    supplemental: Mapping[str, Any],
    *,
    canary_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare fingerprints and require a numeric canary before ``MEASURED``.

    Exact metadata is necessary but not sufficient: without a frozen canary
    artifact, parity remains ``UNVERIFIED`` rather than being promoted from
    configuration equality alone.
    """

    missing = sorted(
        field
        for field in FEATURE_PARITY_FIELDS
        if field not in btc or field not in supplemental
    )
    mismatches = sorted(
        field
        for field in FEATURE_PARITY_FIELDS
        if field in btc and field in supplemental and btc[field] != supplemental[field]
    )
    metadata_status = "MATCH" if not missing and not mismatches else "MISMATCH"
    canary: dict[str, Any] | None = None
    if canary_evidence is not None:
        required = {
            "sample_count",
            "btc_artifact_sha256",
            "delta_redecode_artifact_sha256",
            "max_abs_diff",
            "tolerance",
        }
        absent = sorted(required - set(canary_evidence))
        if absent:
            raise ValueError(f"canary evidence is missing fields: {absent}")
        sample_count = int(canary_evidence["sample_count"])
        max_abs_diff = float(canary_evidence["max_abs_diff"])
        tolerance = float(canary_evidence["tolerance"])
        if sample_count < 1 or not math.isfinite(max_abs_diff) or max_abs_diff < 0:
            raise ValueError("canary sample_count/difference must be positive and finite")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("canary tolerance must be finite and non-negative")
        canary = dict(canary_evidence)
        canary["status"] = "PASS" if max_abs_diff <= tolerance else "FAIL"
    if metadata_status == "MISMATCH":
        status = "BLOCKED"
    elif canary is None:
        status = "UNVERIFIED"
    else:
        status = "MEASURED" if canary["status"] == "PASS" else "BLOCKED"
    return {
        "schema_version": 2,
        "status": status,
        "metadata_status": metadata_status,
        "canary_evidence": canary,
        "required_fields": list(FEATURE_PARITY_FIELDS),
        "missing_fields": missing,
        "mismatched_fields": mismatches,
        "decision_authority": "diagnostic_only",
        "interpretation": (
            "metadata and frozen numeric canary establish feature parity"
            if status == "MEASURED"
            else (
                "metadata matches, but a frozen BTC-versus-redecode canary is required"
                if status == "UNVERIFIED"
                else "supplemental run cannot be compared or activated"
            )
        ),
    }


def crowding_report(
    ranked_keyframe_ids: Iterable[str],
    *,
    btc_keyframe_ids: set[str],
    top_k: int,
    baseline_btc_top_k: Iterable[str] | None = None,
    near_duplicate_cluster_by_id: Mapping[str, str] | None = None,
    scores_by_id: Mapping[str, float] | None = None,
    stage_rankings: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Summarize supplemental crowding and optional BTC displacement for one query."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = list(ranked_keyframe_ids)
    top = ranked[:top_k]
    duplicate_rows = len(top) - len(set(top))
    supplemental_top = [keyframe_id for keyframe_id in top if keyframe_id not in btc_keyframe_ids]
    btc_top = [keyframe_id for keyframe_id in top if keyframe_id in btc_keyframe_ids]
    btc_displaced = None
    best_btc_rank_baseline = None
    best_btc_rank_candidate = next(
        (rank for rank, keyframe_id in enumerate(top, 1) if keyframe_id in btc_keyframe_ids),
        None,
    )
    if baseline_btc_top_k is not None:
        baseline_list = list(baseline_btc_top_k)[:top_k]
        baseline = set(baseline_list)
        btc_displaced = len(baseline - set(btc_top))
        best_btc_rank_baseline = next(
            (
                rank
                for rank, keyframe_id in enumerate(baseline_list, 1)
                if keyframe_id in btc_keyframe_ids
            ),
            None,
        )

    cluster_sizes = None
    if near_duplicate_cluster_by_id is not None:
        cluster_sizes = dict(
            sorted(
                Counter(
                    near_duplicate_cluster_by_id[keyframe_id]
                    for keyframe_id in top
                    if keyframe_id in near_duplicate_cluster_by_id
                ).items()
            )
        )

    score_distribution_by_origin = None
    if scores_by_id is not None:
        distributions: dict[str, list[float]] = {"btc": [], "supplemental": []}
        for keyframe_id in top:
            if keyframe_id not in scores_by_id:
                continue
            score = float(scores_by_id[keyframe_id])
            if not math.isfinite(score):
                raise ValueError("rank scores must be finite")
            origin = "btc" if keyframe_id in btc_keyframe_ids else "supplemental"
            distributions[origin].append(score)
        score_distribution_by_origin = {
            origin: {
                "count": len(values),
                "min": min(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
                "max": max(values) if values else None,
            }
            for origin, values in distributions.items()
        }

    per_origin_exposure_by_stage = None
    if stage_rankings is not None:
        per_origin_exposure_by_stage = {}
        for stage, ranking in sorted(stage_rankings.items()):
            stage_top = list(ranking)[:top_k]
            per_origin_exposure_by_stage[stage] = {
                "btc": sum(item in btc_keyframe_ids for item in stage_top),
                "supplemental": sum(item not in btc_keyframe_ids for item in stage_top),
                "rows": len(stage_top),
            }

    optional_complete = all(
        value is not None
        for value in (
            baseline_btc_top_k,
            near_duplicate_cluster_by_id,
            scores_by_id,
            stage_rankings,
        )
    )
    return {
        "schema_version": 2,
        "top_k": top_k,
        "ranked_rows": len(ranked),
        "unique_top_k": len(set(top)),
        "duplicate_rows_top_k": duplicate_rows,
        "btc_rows_top_k": len(btc_top),
        "supplemental_rows_top_k": len(supplemental_top),
        "supplemental_share_top_k": len(supplemental_top) / len(top) if top else 0.0,
        "btc_displaced_from_baseline_top_k": btc_displaced,
        "best_btc_rank_baseline": best_btc_rank_baseline,
        "best_btc_rank_candidate": best_btc_rank_candidate,
        "best_btc_rank_delta": (
            best_btc_rank_candidate - best_btc_rank_baseline
            if best_btc_rank_candidate is not None and best_btc_rank_baseline is not None
            else None
        ),
        "near_duplicate_cluster_sizes_top_k": cluster_sizes,
        "max_near_duplicate_cluster_size_top_k": (
            max(cluster_sizes.values()) if cluster_sizes else None
        ),
        "score_distribution_by_origin": score_distribution_by_origin,
        "per_origin_exposure_by_stage": per_origin_exposure_by_stage,
        "duplicate_keyframe_ids_top_k": sorted(
            keyframe_id for keyframe_id, count in Counter(top).items() if count > 1
        ),
        "decision_authority": "diagnostic_only",
        "evidence_status": "MEASURED" if optional_complete else "PARTIAL_UNVERIFIED",
    }


__all__ = [
    "FEATURE_PARITY_FIELDS",
    "compare_feature_parity",
    "crowding_report",
]
