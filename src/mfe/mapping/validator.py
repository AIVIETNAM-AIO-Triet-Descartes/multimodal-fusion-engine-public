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
    Có bằng chứng    answer traces to a real evidence_id         -> block (chốt)
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.contracts.tools import EVIDENCE_ID_PATTERN, EvidenceId


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
        failures: list[ValidationFailure] = []

        # Rule "Tồn tại" — one fetch; if it is missing, nothing else can be
        # checked, so return early. (This is also the only fetch: the keyframe
        # and its video are read once here and passed to the bounds rules,
        # instead of each rule hitting the DB again.)
        try:
            keyframe = self.relational_map.get_keyframe(keyframe_id)
        except ContractError as e:
            if e.code == ErrorCode.KEYFRAME_NOT_FOUND:
                return ValidationResult(
                    valid=False,
                    failures=[
                        ValidationFailure(
                            code=ErrorCode.KEYFRAME_NOT_FOUND,
                            rule="Tồn tại",
                            message=f"Keyframe {keyframe_id} not in Relational Map",
                        )
                    ],
                )
            raise
        video = self.relational_map.get_video(keyframe.video_id)

        for check in (
            self._check_frame_in_bounds(keyframe, video),
            self._check_timestamp_in_bounds(keyframe, video),
        ):
            if check is not None:
                failures.append(check)

        # Rule "Đúng schema" — build the record, then check its shape.
        try:
            from mfe.mapping.submit import build_submission

            record = build_submission(keyframe_id, schema_format, self.relational_map)
            schema_failure = self._check_schema(record, str(schema_format))
            if schema_failure is not None:
                failures.append(schema_failure)
        except Exception as e:  # noqa: BLE001 - a validator fails closed on any doubt
            failures.append(
                ValidationFailure(
                    code=ErrorCode.SCHEMA_MISMATCH,
                    rule="Đúng schema",
                    message=f"Failed to build submission: {e}",
                )
            )

        evidence_failure = self._check_evidence(evidence_id)
        if evidence_failure is not None:
            failures.append(evidence_failure)

        return ValidationResult(valid=len(failures) == 0, failures=failures)

    # --- individual rules, docs/05 §3 -------------------------------------

    def _check_frame_in_bounds(
        self, keyframe: KeyframeRow, video: VideoRow
    ) -> ValidationFailure | None:
        """Rule "Trong biên" -> FRAME_OUT_OF_BOUNDS. frame_idx is 0-indexed, so
        the upper bound is exclusive."""
        if not (0 <= keyframe.frame_idx < video.total_frames):
            return ValidationFailure(
                code=ErrorCode.FRAME_OUT_OF_BOUNDS,
                rule="Trong biên",
                message=(
                    f"frame_idx {keyframe.frame_idx} out of bounds for video "
                    f"{video.video_id} ({video.total_frames} frames)"
                ),
            )
        return None

    def _check_timestamp_in_bounds(
        self, keyframe: KeyframeRow, video: VideoRow
    ) -> ValidationFailure | None:
        """Rule "Timestamp hợp lệ" -> TIMESTAMP_OUT_OF_BOUNDS. duration is
        inclusive."""
        if not (0 <= keyframe.timestamp_ms <= video.duration_ms):
            return ValidationFailure(
                code=ErrorCode.TIMESTAMP_OUT_OF_BOUNDS,
                rule="Timestamp hợp lệ",
                message=(
                    f"timestamp_ms {keyframe.timestamp_ms} out of bounds for video "
                    f"{video.video_id} (duration {video.duration_ms} ms)"
                ),
            )
        return None

    def _check_schema(self, record: dict, schema_format: str) -> ValidationFailure | None:
        """Rule "Đúng schema" -> SCHEMA_MISMATCH. Structural check per format."""
        required: dict[str, tuple[str, ...]] = {
            "by_timestamp": ("video_id", "timestamp_ms"),
            "by_frame": ("video_id", "frame_idx"),
            "by_frame_range": ("video_id", "frame_start", "frame_end"),
            "raw_keyframe": ("keyframe_id",),
        }
        missing = [k for k in required.get(schema_format, ()) if k not in record]
        if missing:
            return ValidationFailure(
                code=ErrorCode.SCHEMA_MISMATCH,
                rule="Đúng schema",
                message=f"Record for {schema_format} missing: {', '.join(missing)}",
            )
        return None

    def _check_evidence(self, evidence_id: EvidenceId | None) -> ValidationFailure | None:
        """Rule "Có bằng chứng" -> MISSING_EVIDENCE."""
        if evidence_id is None or not str(evidence_id).strip():
            return ValidationFailure(
                code=ErrorCode.MISSING_EVIDENCE,
                rule="Có bằng chứng",
                message="Submission lacks an evidence_id tracing it to a retrieval result",
            )
        if not re.match(EVIDENCE_ID_PATTERN, str(evidence_id)):
            return ValidationFailure(
                code=ErrorCode.MISSING_EVIDENCE,
                rule="Có bằng chứng",
                message=f"evidence_id '{evidence_id}' violates format {{tool}}:{{run}}:{{ordinal}}",
            )
        return None
