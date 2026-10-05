"""Fail-closed math and schemas for the DEV O1-S-LB64 aggregate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfe.contracts.ids import parse_keyframe_id
from mfe.offline.augmentation.io import canonical_hash

FailureStage = Literal[
    "temporal", "representation", "retrieval", "fusion", "reranker", "scorer", "none"
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

_STAGES = ("siglip_dense", "fusion", "reranker", "scorer")


class O1SEventTrace(BaseModel):
    """One DEV query/event result at the measured SigLIP stage."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query_id: str = Field(min_length=1)
    query_type: Literal["KIS", "QA", "TRAKE"]
    event_index: int = Field(ge=0)
    video_id: str = Field(min_length=1)
    gt_interval_inclusive: tuple[int, int]
    sampling_policy: Literal[
        "inclusive_uniform_integer_lb64",
        "inclusive_full_interval_exact",
    ]
    full_interval_scan: bool
    best_btc_frame_idx: int | None = Field(default=None, ge=0)
    best_btc_rank: int | None = Field(default=None, ge=1)
    best_btc_score_by_stage: dict[str, float | None]
    top_btc_outside_interval_id: str
    top_btc_outside_interval_rank: int = Field(ge=1)
    top_btc_outside_interval_score: float
    best_source_frame_idx: int = Field(ge=0)
    best_source_rank: int = Field(ge=1)
    best_source_score_by_stage: dict[str, float | None]
    o1_t_delta: float | None = None
    o1_s_delta: float | None = None
    o1_r_delta: float | None = None
    failure_stage: FailureStage
    failure_stage_status: str = Field(min_length=1)
    provenance_hashes: dict[str, Sha256] = Field(min_length=1)

    @model_validator(mode="after")
    def _event_contract_is_consistent(self) -> O1SEventTrace:
        start, end = self.gt_interval_inclusive
        if start < 0 or end < start:
            raise ValueError("O1-S interval must be inclusive and non-empty")
        if not start <= self.best_source_frame_idx <= end:
            raise ValueError("best source lies outside its DEV interval")
        if self.best_btc_frame_idx is not None and not start <= self.best_btc_frame_idx <= end:
            raise ValueError("best BTC lies outside its DEV interval")
        if (self.best_btc_frame_idx is None) != (self.best_btc_rank is None):
            raise ValueError("best BTC frame/rank must both be present or both be absent")
        expected_full_scan = (
            end - start + 1 <= 64
            if self.sampling_policy == "inclusive_uniform_integer_lb64"
            else True
        )
        if self.full_interval_scan != expected_full_scan:
            raise ValueError("full_interval_scan disagrees with the sampling policy")
        if set(self.best_source_score_by_stage) != set(_STAGES):
            raise ValueError("best source stage map is incomplete or contains an unknown stage")
        if set(self.best_btc_score_by_stage) != set(_STAGES):
            raise ValueError("best BTC stage map is incomplete or contains an unknown stage")
        source_siglip = self.best_source_score_by_stage["siglip_dense"]
        btc_siglip = self.best_btc_score_by_stage["siglip_dense"]
        if source_siglip is None or not math.isfinite(source_siglip):
            raise ValueError("best source SigLIP score must be finite")
        if not math.isfinite(self.top_btc_outside_interval_score):
            raise ValueError("top outside BTC score must be finite")
        if any(
            value is not None and not math.isfinite(value)
            for value in (self.o1_t_delta, self.o1_s_delta, self.o1_r_delta)
        ):
            raise ValueError("oracle deltas must be finite when present")
        if any(self.best_source_score_by_stage[stage] is not None for stage in _STAGES[1:]):
            raise ValueError("unmeasured downstream source stages must remain null")
        if any(self.best_btc_score_by_stage[stage] is not None for stage in _STAGES[1:]):
            raise ValueError("unmeasured downstream BTC stages must remain null")
        outside_video, outside_frame = parse_keyframe_id(self.top_btc_outside_interval_id)
        if outside_video == self.video_id and start <= outside_frame <= end:
            raise ValueError("top outside BTC identity lies inside the target interval")
        if self.best_btc_frame_idx is None:
            if btc_siglip is not None or self.o1_s_delta is not None:
                raise ValueError("temporal-missing events cannot have BTC score or O1-S delta")
            if self.failure_stage != "temporal":
                raise ValueError("an interval without BTC must be classified temporal")
        else:
            if btc_siglip is None or not math.isfinite(btc_siglip):
                raise ValueError("represented events require a finite BTC SigLIP score")
            expected_delta = source_siglip - btc_siglip
            if self.o1_s_delta is None or not math.isclose(
                self.o1_s_delta, expected_delta, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("O1-S delta disagrees with source minus BTC SigLIP score")
            if self.failure_stage == "temporal":
                raise ValueError("represented intervals cannot be classified temporal")
        return self


def normalize_feature_rows(matrix: np.ndarray) -> np.ndarray:
    """Return finite float32 L2-normalized rows; reject zero vectors."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("feature matrix must be non-empty and two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("feature matrix contains non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("feature matrix contains a zero-norm row")
    return values / norms


def cosine_scores(normalized_matrix: np.ndarray, normalized_query: np.ndarray) -> np.ndarray:
    """Compute one cosine score per row after checking the normalized shapes."""

    matrix = np.asarray(normalized_matrix, dtype=np.float32)
    query = np.asarray(normalized_query, dtype=np.float32)
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
        raise ValueError("matrix/query dimensions do not match")
    scores = matrix @ query
    if not np.isfinite(scores).all():
        raise ValueError("cosine scoring produced non-finite values")
    return scores


def deterministic_best(ids: Sequence[str], scores: Sequence[float]) -> tuple[str, float, int]:
    """Choose maximum score, breaking exact ties by keyframe identity."""

    names = [str(value) for value in ids]
    values = np.asarray(scores, dtype=np.float64)
    if not names or len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("candidate ids/scores must be non-empty, aligned, and unique")
    if not np.isfinite(values).all():
        raise ValueError("candidate scores contain non-finite values")
    index = min(range(len(names)), key=lambda item: (-float(values[item]), names[item]))
    return names[index], float(values[index]), index


def deterministic_injected_rank(
    corpus_ids: Sequence[str],
    corpus_scores: Sequence[float],
    *,
    candidate_id: str,
    candidate_score: float,
) -> int:
    """Rank a candidate by descending score then ascending keyframe identity."""

    names = np.asarray([str(value) for value in corpus_ids])
    scores = np.asarray(corpus_scores, dtype=np.float64)
    if names.ndim != 1 or scores.ndim != 1 or len(names) != len(scores):
        raise ValueError("corpus ids/scores are not aligned")
    if len(set(names.tolist())) != len(names):
        raise ValueError("corpus ids must be unique")
    if not np.isfinite(scores).all() or not np.isfinite(candidate_score):
        raise ValueError("rank inputs contain non-finite scores")
    other = names != candidate_id
    better = scores > candidate_score
    tied_before = (scores == candidate_score) & (names < candidate_id)
    return 1 + int(np.count_nonzero(other & (better | tied_before)))


def interval_candidate_indices(
    ids: Sequence[str],
    *,
    video_id: str,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    """Return corpus positions whose source identities lie in an inclusive interval."""

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("inclusive interval is invalid")
    positions = []
    for index, keyframe_id in enumerate(ids):
        candidate_video, frame_idx = parse_keyframe_id(str(keyframe_id))
        if candidate_video == video_id and start_frame <= frame_idx <= end_frame:
            positions.append(index)
    return np.asarray(positions, dtype=np.int64)


def classify_siglip_failure_stage(
    *,
    has_btc_in_interval: bool,
    source_signal_delta: float | None,
    best_source_rank: int,
    best_btc_rank: int | None,
    exposure_depth: int,
    improvement_epsilon: float,
) -> FailureStage:
    """Classify only stages observable from the locked SigLIP dense trace."""

    if exposure_depth < 1 or improvement_epsilon < 0:
        raise ValueError("failure-stage thresholds are invalid")
    if not has_btc_in_interval:
        return "temporal"
    if source_signal_delta is None or best_btc_rank is None:
        raise ValueError("represented intervals require source delta and BTC rank")
    if source_signal_delta > improvement_epsilon:
        return "retrieval" if best_source_rank > exposure_depth else "representation"
    return "retrieval" if best_btc_rank > exposure_depth else "none"


def validate_o1s_worker_manifest(
    manifest: Mapping[str, Any],
    *,
    runner: str,
    run_id: str,
    scan_plan_hash: str,
    expected_assigned: Sequence[str],
    expected_frame_count: int,
    expected_repo_commit: str,
    expected_notebook18_report_sha256: str,
    expected_targets_sha256: str,
    expected_siglip_canary_report_hash: str,
) -> bool:
    """Validate a complete worker handoff and its exact disjoint assignment."""

    payload = dict(manifest)
    claimed = payload.pop("manifest_payload_hash", None)
    if claimed != canonical_hash(payload):
        return False
    expected = {
        "status": "COMPLETE",
        "decision_authority": "diagnostic_only",
        "runner": runner,
        "runners": ["quang", "trong"],
        "run_id": run_id,
        "scan_plan_hash": scan_plan_hash,
        "sampling_policy": "inclusive_uniform_integer_lb64",
        "max_samples_per_event": 64,
        "full_o1s_status": "BLOCKED_SAMPLED_INTERVALS_ARE_LOWER_BOUND",
        "notebook18_report_sha256": expected_notebook18_report_sha256,
        "targets_sha256": expected_targets_sha256,
        "assigned_count": len(expected_assigned),
        "assigned_frame_count": expected_frame_count,
        "verified_complete_count": len(expected_assigned),
        "repo_commit": expected_repo_commit,
        "siglip_canary_report_hash": expected_siglip_canary_report_hash,
        "heldout_accessed": False,
        "btc_mutated": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    assigned = [str(value) for value in manifest.get("assigned", [])]
    verified = [str(value) for value in manifest.get("verified_complete", [])]
    return (
        assigned == list(expected_assigned)
        and verified == list(expected_assigned)
        and not manifest.get("failures_this_session")
    )


def o1s_math_call_site_ledger() -> list[dict[str, Any]]:
    """Machine-readable definitions for every decision-relevant O1-S calculation."""

    return [
        {
            "formula_id": "M-O1S-001",
            "symbol": "normalize_feature_rows",
            "formula": "z_i = x_i / ||x_i||_2",
            "rounding": "none_float32",
            "failure": "reject_nonfinite_or_zero_norm",
        },
        {
            "formula_id": "M-O1S-002",
            "symbol": "cosine_scores",
            "formula": "score(i,q) = z_i dot z_q",
            "rounding": "none_float32_matmul",
            "failure": "reject_shape_or_nonfinite",
        },
        {
            "formula_id": "M-O1S-003",
            "symbol": "deterministic_best",
            "formula": "argmax_i(score_i, tie=-lexicographic_keyframe_id)",
            "rounding": "none",
            "tie_break": "ascending_keyframe_id",
        },
        {
            "formula_id": "M-O1S-004",
            "symbol": "deterministic_injected_rank",
            "formula": "1 + count(score_j>score_c or (score_j=score_c and id_j<id_c))",
            "rounding": "none",
            "tie_break": "ascending_keyframe_id",
        },
        {
            "formula_id": "M-O1S-005",
            "symbol": "o1_s_delta",
            "formula": "best_sampled_source_siglip_score - best_interval_btc_siglip_score",
            "rounding": "none_float64_serialization",
            "missing_baseline": "null_and_failure_stage_temporal",
        },
        {
            "formula_id": "M-O1S-006",
            "symbol": "classify_siglip_failure_stage",
            "formula": (
                "temporal if no interval BTC; else representation/retrieval/none "
                "by epsilon and depth"
            ),
            "rounding": "strict_delta_gt_epsilon",
            "blocked": ["fusion", "reranker", "scorer"],
        },
        {
            "formula_id": "M-O1S-007",
            "symbol": "interval_candidate_indices",
            "formula": "video_i=video_gt and start_frame<=frame_i<=end_frame",
            "rounding": "none_integer_inclusive",
            "failure": "reject_invalid_interval_or_keyframe_identity",
        },
    ]


def adversarial_counterexample_summary(
    traces: Sequence[O1SEventTrace | Mapping[str, Any]],
    *,
    exposure_depth: int,
    improvement_epsilon: float,
) -> dict[str, int]:
    """Count observations that falsify common overclaims about O1-S-LB64."""

    if exposure_depth < 1 or improvement_epsilon < 0:
        raise ValueError("counterexample thresholds are invalid")
    rows = [
        value if isinstance(value, O1SEventTrace) else O1SEventTrace.model_validate(value)
        for value in traces
    ]
    return {
        "sampled_intervals_where_no_gain_is_not_full_o1s_evidence": sum(
            not row.full_interval_scan for row in rows
        ),
        "source_signal_gain_but_still_outside_exposure": sum(
            row.o1_s_delta is not None
            and row.o1_s_delta > improvement_epsilon
            and row.best_source_rank > exposure_depth
            for row in rows
        ),
        "source_signal_gain_but_nonpositive_o1r_score_delta": sum(
            row.o1_s_delta is not None
            and row.o1_s_delta > improvement_epsilon
            and (row.o1_r_delta is None or row.o1_r_delta <= 0)
            for row in rows
        ),
        "positive_o1r_score_delta_without_sampled_source_gain": sum(
            row.o1_r_delta is not None
            and row.o1_r_delta > 0
            and (row.o1_s_delta is None or row.o1_s_delta <= improvement_epsilon)
            for row in rows
        ),
        "temporal_missing_events": sum(row.failure_stage == "temporal" for row in rows),
    }


__all__ = [
    "FailureStage",
    "O1SEventTrace",
    "adversarial_counterexample_summary",
    "classify_siglip_failure_stage",
    "cosine_scores",
    "deterministic_best",
    "deterministic_injected_rank",
    "interval_candidate_indices",
    "normalize_feature_rows",
    "o1s_math_call_site_ledger",
    "validate_o1s_worker_manifest",
]
