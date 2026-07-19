"""Validator — the last fence before submit. docs/05 §3.

The governing principle, from docs/05 §3: **submit sai còn tệ hơn không submit**
— a wrong submission costs the time to find the answer again and may be
penalised (docs/context/buoi-1.md §6). So: "Validator thà chặn nhầm còn hơn cho
lọt." When a rule is ambiguous, block.

The five rules:

    Tồn tại          keyframe_id is in the Relational Map        -> block
    Trong biên       0 <= frame_number < total_frames            -> block
    Timestamp hợp lệ 0 <= timestamp_ms <= duration_ms            -> block
    Đúng schema      record matches the required format          -> block
    Có bằng chứng    answer traces to a real evidence_id         -> warn/block
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mfe.contracts.errors import ErrorCode
from mfe.contracts.tools import EvidenceId


class ValidationFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    rule: str
    message: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    failures: list[ValidationFailure] = []

    @property
    def blocked(self) -> bool:
        return not self.valid


class Validator:
    """Owner: Nhật Bảo. TODO(Nhật Bảo): implement all five rules of docs/05 §3."""

    def __init__(self, relational_map: object) -> None:
        self.relational_map = relational_map

    def validate(
        self,
        keyframe_id: str,
        evidence_id: EvidenceId | None = None,
        schema_format: str = "by_timestamp",
    ) -> ValidationResult:
        """Run every rule. TODO(Nhật Bảo): implement.

        Report *all* failures, not just the first. An operator under time
        pressure who fixes one problem only to hit the next has lost twice the
        time — which is the metric that actually matters (docs/04 §5: optimize
        TTCA, not just Recall).
        """
        raise NotImplementedError("TODO(Nhật Bảo): run 5 rules -> ValidationResult")

    # --- individual rules, docs/05 §3 -------------------------------------

    def _check_exists(self, keyframe_id: str) -> ValidationFailure | None:
        """Rule "Tồn tại" -> KEYFRAME_NOT_FOUND."""
        raise NotImplementedError("TODO(Nhật Bảo)")

    def _check_frame_in_bounds(self, keyframe_id: str) -> ValidationFailure | None:
        """Rule "Trong biên" -> FRAME_OUT_OF_BOUNDS.

        Note the asymmetry with the timestamp rule below, and keep it: frames
        are 0-indexed so the bound is exclusive (`frame_number < total_frames`),
        while duration is inclusive (`timestamp_ms <= duration_ms`). Making both
        inclusive admits a frame that does not exist.
        """
        raise NotImplementedError("TODO(Nhật Bảo)")

    def _check_timestamp_in_bounds(self, keyframe_id: str) -> ValidationFailure | None:
        """Rule "Timestamp hợp lệ" -> TIMESTAMP_OUT_OF_BOUNDS."""
        raise NotImplementedError("TODO(Nhật Bảo)")

    def _check_schema(self, record: dict, schema_format: str) -> ValidationFailure | None:
        """Rule "Đúng schema" -> SCHEMA_MISMATCH.

        The BTC format is TO_CONFIRM (docs/00 §3), so this validates against
        whichever adapter config selected — not against a hardcoded shape.
        """
        raise NotImplementedError("TODO(Nhật Bảo)")

    def _check_evidence(self, evidence_id: EvidenceId | None) -> ValidationFailure | None:
        """Rule "Có bằng chứng" -> MISSING_EVIDENCE.

        docs/05 §3 marks this one "Cảnh báo/chặn" rather than a hard block.
        TODO(Nhật Bảo): settle which, with the Vỹ Triết. It matters more later than now:
        docs/context/buoi-3.md §7 requires an agent's answer to point at real
        evidence, and a warning is not a guardrail. Sprint 1 has a human in the
        loop, so warn may be enough — but decide it, do not default into it.
        """
        raise NotImplementedError("TODO(Nhật Bảo)")
