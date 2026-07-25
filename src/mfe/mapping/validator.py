"""Validator — the last fence before submit. docs/05 §3.

The governing principle, from docs/05 §3: **submit sai còn tệ hơn không submit**
— a wrong submission costs the time to find the answer again and may be
penalised (docs/context/buoi-1.md §6). So: "Validator thà chặn nhầm còn hơn cho
lọt." When a rule is ambiguous, block.

The five rules:

    Tồn tại          keyframe_id is in the Relational Map        -> block
    Trong biên       0 <= frame_idx < total_frames               -> block
    Timestamp hợp lệ 0 <= timestamp_ms <= duration_ms            -> block
    Đúng schema      record matches the required format          -> block
    Có bằng chứng    answer traces to a real evidence_id         -> warn/block
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mfe.contracts.errors import ErrorCode, ContractError
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
    """Owner: Nhật Bảo. Implement all five rules of docs/05 §3."""

    def __init__(self, relational_map: object) -> None:
        self.relational_map = relational_map

    def validate(
        self,
        keyframe_id: str,
        evidence_id: EvidenceId | None = None,
        schema_format: str = "by_timestamp",
    ) -> ValidationResult:
        """Run every rule.

        Report *all* failures, not just the first. An operator under time
        pressure who fixes one problem only to hit the next has lost twice the
        time — which is the metric that actually matters (docs/04 §5: optimize
        TTCA, not just Recall).
        """
        failures = []

        f1 = self._check_exists(keyframe_id)
        if f1:
            failures.append(f1)
            # If it doesn't exist, we can't check bounds
            return ValidationResult(valid=False, failures=failures)
            
        f2 = self._check_frame_in_bounds(keyframe_id)
        if f2: failures.append(f2)
            
        f3 = self._check_timestamp_in_bounds(keyframe_id)
        if f3: failures.append(f3)

        # Build the record to check its schema
        try:
            from mfe.mapping.submit import build_submission
            record = build_submission(keyframe_id, schema_format, self.relational_map)
            f4 = self._check_schema(record, str(schema_format))
            if f4: failures.append(f4)
        except Exception as e:
            failures.append(ValidationFailure(
                code=ErrorCode.SCHEMA_MISMATCH,
                rule="Đúng schema",
                message=f"Failed to build submission: {str(e)}"
            ))
            
        f5 = self._check_evidence(evidence_id)
        if f5: failures.append(f5)

        return ValidationResult(valid=len(failures) == 0, failures=failures)

    # --- individual rules, docs/05 §3 -------------------------------------

    def _check_exists(self, keyframe_id: str) -> ValidationFailure | None:
        """Rule "Tồn tại" -> KEYFRAME_NOT_FOUND."""
        try:
            self.relational_map.get_keyframe(keyframe_id)
            return None
        except ContractError as e:
            if e.code == ErrorCode.KEYFRAME_NOT_FOUND:
                return ValidationFailure(
                    code=ErrorCode.KEYFRAME_NOT_FOUND,
                    rule="Tồn tại",
                    message=f"Keyframe {keyframe_id} not in Relational Map"
                )
            raise

    def _check_frame_in_bounds(self, keyframe_id: str) -> ValidationFailure | None:
        """Rule "Trong biên" -> FRAME_OUT_OF_BOUNDS."""
        keyframe = self.relational_map.get_keyframe(keyframe_id)
        video = self.relational_map.get_video(keyframe.video_id)
        if not (0 <= keyframe.frame_idx < video.total_frames):
            return ValidationFailure(
                code=ErrorCode.FRAME_OUT_OF_BOUNDS,
                rule="Trong biên",
                message=f"frame_idx {keyframe.frame_idx} out of bounds for video {video.video_id} with {video.total_frames} frames"
            )
        return None

    def _check_timestamp_in_bounds(self, keyframe_id: str) -> ValidationFailure | None:
        """Rule "Timestamp hợp lệ" -> TIMESTAMP_OUT_OF_BOUNDS."""
        keyframe = self.relational_map.get_keyframe(keyframe_id)
        video = self.relational_map.get_video(keyframe.video_id)
        if not (0 <= keyframe.timestamp_ms <= video.duration_ms):
            return ValidationFailure(
                code=ErrorCode.TIMESTAMP_OUT_OF_BOUNDS,
                rule="Timestamp hợp lệ",
                message=f"timestamp_ms {keyframe.timestamp_ms} out of bounds for video {video.video_id} with duration {video.duration_ms} ms"
            )
        return None

    def _check_schema(self, record: dict, schema_format: str) -> ValidationFailure | None:
        """Rule "Đúng schema" -> SCHEMA_MISMATCH."""
        # Simple structural check
        if schema_format == "by_timestamp":
            if "video_id" not in record or "timestamp_ms" not in record:
                return ValidationFailure(code=ErrorCode.SCHEMA_MISMATCH, rule="Đúng schema", message="Missing video_id or timestamp_ms")
        elif schema_format == "by_frame":
            if "video_id" not in record or "frame_idx" not in record:
                return ValidationFailure(code=ErrorCode.SCHEMA_MISMATCH, rule="Đúng schema", message="Missing video_id or frame_idx")
        elif schema_format == "by_frame_range":
            if "video_id" not in record or "frame_start" not in record or "frame_end" not in record:
                return ValidationFailure(code=ErrorCode.SCHEMA_MISMATCH, rule="Đúng schema", message="Missing video_id, frame_start, or frame_end")
        elif schema_format == "raw_keyframe":
            if "keyframe_id" not in record:
                return ValidationFailure(code=ErrorCode.SCHEMA_MISMATCH, rule="Đúng schema", message="Missing keyframe_id")
        return None

    def _check_evidence(self, evidence_id: EvidenceId | None) -> ValidationFailure | None:
        """Rule "Có bằng chứng" -> MISSING_EVIDENCE."""
        if evidence_id is None or not str(evidence_id).strip():
            return ValidationFailure(
                code=ErrorCode.MISSING_EVIDENCE,
                rule="Có bằng chứng",
                message="Submission lacks an evidence_id tracing it to a retrieval result"
            )
        import re
        if not re.match(r"^[a-zA-Z0-9_]+:[a-zA-Z0-9_\-]+:\d+$", str(evidence_id)):
            return ValidationFailure(
                code=ErrorCode.MISSING_EVIDENCE,
                rule="Có bằng chứng",
                message=f"evidence_id '{evidence_id}' violates the format '{{tool_name}}:{{run_id}}:{{ordinal}}'"
            )
        return None
