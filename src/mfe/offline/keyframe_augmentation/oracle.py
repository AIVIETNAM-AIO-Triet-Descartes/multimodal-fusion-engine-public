"""Strict, fail-closed schema for the O1-T/O1-S/O1-R evidence contract.

The existing Gate 1 manifest is intentionally a coordinate diagnostic.  This
module defines the later evidence row that may only be emitted once source
frames, ranks, stage scores, and provenance are actually available.  It does
not infer retrieval evidence from temporal coverage.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

FailureStage = Literal[
    "temporal",
    "representation",
    "retrieval",
    "fusion",
    "reranker",
    "scorer",
    "none",
]
EvidenceStatus = Literal["MEASURED", "BLOCKED_MISSING_STAGE_ARTIFACT"]


class OracleEvidenceRow(BaseModel):
    """One query/event row shared by the three O1 ceilings.

    ``gt_interval_inclusive`` is inclusive.  Legacy v1 field names are accepted
    on input only so existing local artifacts remain readable; serialization
    emits the canonical v2 names.  A blocked row still has to carry the locked
    query/event identity and provenance; its missing fields are explicit
    ``None`` rather than guessed frame/rank values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal[1, 2] = 2
    query_id: str = Field(min_length=1)
    query_type: Literal["KIS", "QA", "TRAKE"]
    video_id: str = Field(min_length=1)
    gt_interval_inclusive: tuple[int, int] = Field(
        validation_alias=AliasChoices("gt_interval_inclusive", "interval")
    )
    best_btc_frame_idx: int | None = Field(default=None, ge=0)
    best_btc_rank: int | None = Field(default=None, ge=1)
    best_btc_score_by_stage: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "best_btc_score_by_stage", "best_btc_stage_scores"
        ),
    )
    best_source_frame_idx: int | None = Field(default=None, ge=0)
    best_source_rank: int | None = Field(default=None, ge=1)
    best_source_score_by_stage: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "best_source_score_by_stage", "best_source_stage_scores"
        ),
    )
    o1_t_delta: float | None = None
    o1_s_delta: float | None = None
    o1_r_delta: float | None = None
    failure_stage: FailureStage
    evidence_status: EvidenceStatus
    provenance_hashes: dict[str, str] = Field(min_length=1)

    @field_validator("gt_interval_inclusive")
    @classmethod
    def _inclusive_interval(cls, value: tuple[int, int]) -> tuple[int, int]:
        start, end = value
        if start < 0 or end < start:
            raise ValueError("interval must be an inclusive non-negative [start,end]")
        return value

    @field_validator("best_btc_score_by_stage", "best_source_score_by_stage")
    @classmethod
    def _finite_stage_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(float(score)) for score in value.values()):
            raise ValueError("stage scores must be finite")
        return value

    @field_validator("provenance_hashes")
    @classmethod
    def _non_empty_provenance(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("provenance hash names must be non-empty strings")
        if any(not isinstance(hash_value, str) or not hash_value for hash_value in value.values()):
            raise ValueError("provenance hashes must be non-empty strings")
        return value

    @model_validator(mode="after")
    def _measured_rows_are_complete(self) -> OracleEvidenceRow:
        if self.evidence_status == "MEASURED":
            if self.best_source_frame_idx is None or self.best_source_rank is None:
                raise ValueError("MEASURED O1 evidence requires the best source frame and rank")
            if not self.best_source_score_by_stage:
                raise ValueError("MEASURED O1 evidence requires source stage scores")
            if self.o1_s_delta is None or self.o1_r_delta is None:
                raise ValueError("MEASURED O1 evidence requires O1-S and O1-R deltas")
        return self

    @property
    def interval(self) -> tuple[int, int]:
        """Compatibility accessor for in-memory v1 callers."""

        return self.gt_interval_inclusive

    @property
    def best_btc_stage_scores(self) -> dict[str, float]:
        """Compatibility accessor for in-memory v1 callers."""

        return self.best_btc_score_by_stage

    @property
    def best_source_stage_scores(self) -> dict[str, float]:
        """Compatibility accessor for in-memory v1 callers."""

        return self.best_source_score_by_stage


def validate_oracle_rows(rows: list[dict[str, Any]]) -> list[OracleEvidenceRow]:
    """Validate an oracle JSON/JSONL payload without altering it."""

    validated = [OracleEvidenceRow.model_validate(row) for row in rows]
    seen: set[tuple[str, tuple[int, int]]] = set()
    for row in validated:
        identity = (row.query_id, row.gt_interval_inclusive)
        if identity in seen:
            raise ValueError(f"duplicate oracle query/event identity: {identity}")
        seen.add(identity)
    return validated


def blocked_oracle_row(
    event: dict[str, Any],
    *,
    provenance_hashes: dict[str, str],
    failure_stage: FailureStage = "representation",
) -> OracleEvidenceRow:
    """Build an honest placeholder when the canonical stage artifact is absent."""

    return OracleEvidenceRow(
        query_id=str(event["query_id"]),
        query_type=str(event["query_type"]),
        video_id=str(event["video_id"]),
        gt_interval_inclusive=(int(event["start_frame"]), int(event["end_frame"])),
        failure_stage=failure_stage,
        evidence_status="BLOCKED_MISSING_STAGE_ARTIFACT",
        provenance_hashes=provenance_hashes,
    )


__all__ = [
    "EvidenceStatus",
    "FailureStage",
    "OracleEvidenceRow",
    "blocked_oracle_row",
    "validate_oracle_rows",
]
