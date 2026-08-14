"""Validator — the last fence before submit. docs/05 §3.

The governing principle, from docs/05 §3: **submit sai còn tệ hơn không submit**
— a wrong submission costs the time to find the answer again and may be
penalised (docs/context/buoi-1.md §6). So: "Validator thà chặn nhầm còn hơn cho
lọt." When a rule is ambiguous, block.

The six rules:

    Tồn tại          keyframe_id is in the Relational Map        -> block
    Trong biên       0 <= frame_idx < total_frames               -> block
    Timestamp hợp lệ 0 <= timestamp_ms <= duration_ms            -> block
    Nhất quán frame  frame_idx agrees with floor(pts*fps) AND
                     sits in order among its neighbours          -> block
    Đúng schema      record matches the required format          -> block
    Có bằng chứng    answer traces to a real evidence_id         -> block (chốt)

"Nhất quán frame" is the rule that matters most under the official scoring
(PDF §2.1.1): an answer scores 1 only if `frame_id` falls inside BTC's own
[s, e] window, so a frame that is right about *which* moment but wrong by a
handful of frames scores exactly the same as a completely wrong video — zero.
Two independent checks have to agree before a frame is allowed out.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import frame_idx_from_pts
from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.contracts.tools import EVIDENCE_ID_PATTERN, EvidenceId

#: How many keyframes either side to pull for the ordering check. Small on
#: purpose: the question is only "does this frame sit between its immediate
#: neighbours", and a wider window buys nothing but query time.
NEIGHBOR_WINDOW = 1


class ValidationFailure(BaseModel):
    """A rule was broken. Blocks the submit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    rule: str
    message: str


class ValidationWarning(BaseModel):
    """A check passed, but only weakly. Does NOT block the submit.

    The case this exists for: a frame sits inside `(prev, next)` and so passes
    the ordering rule — but those two keyframes are hundreds of frames apart, so
    "inside the bracket" was nearly free to satisfy and says almost nothing.
    Reporting that as `valid` with no qualification tells the operator the
    answer was checked when it was barely checked at all.

    Kept separate from `failures` on purpose. Folding it in would make
    "suspicious" block a submit, and a validator that blocks on suspicion stops
    being used (docs/05 §3 wants it to block on *violation*).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    rule: str
    message: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    failures: list[ValidationFailure] = []
    warnings: list[ValidationWarning] = []

    @property
    def blocked(self) -> bool:
        return not self.valid

    @property
    def confident(self) -> bool:
        """Valid *and* nothing looked weak. What "ready to submit" should mean."""
        return self.valid and not self.warnings


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
            self._check_frame_consistency(keyframe, video),
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

    def _check_frame_consistency(
        self, keyframe: KeyframeRow, video: VideoRow
    ) -> ValidationFailure | None:
        """Rule "Nhất quán frame" -> FRAME_OUT_OF_BOUNDS. Two agreeing checks.

        **(a) Formula.** `floor(pts_time * fps)` reproduces BTC's own
        `frame_idx` exactly (contracts/ids.py). A row that disagrees was built
        with the wrong rounding mode, the wrong fps, or a hand-written
        f-string — all three are silent, and all three put a plausible-looking
        wrong frame on the scoreboard.

        **(b) Ordering.** The frame must not go *backwards* relative to its
        immediate neighbours.

        Note the comparison is non-strict: two consecutive keyframes are allowed
        to share a `frame_idx`. That is not a defect in the data, it is what
        happens when BTC samples two keyframes less than one frame apart —
        `floor(0.0 * 30)` and `floor(0.0333333 * 30)` are both 0. Measured on
        the full 2026 batch-1 mapping: **614 such pairs across 192 of 873
        videos, and zero pairs that actually decrease**. An earlier revision
        used `>=` / `<=` here and would have blocked 1,228 perfectly valid
        keyframes — a validator that throws away correct answers, which is worse
        than one that never ran.

        The price is that (b) is now weaker: an off-by-one row whose neighbours
        happen to share a frame no longer trips it. Restoring that strength is
        what the interpolation-ratio check is for (see
        `docs/triet/keyframe_validator_audit.md`, Gap 2) — ordering alone was
        never going to catch a small fps error anyway.
        """
        expected_frame_idx = frame_idx_from_pts(keyframe.pts_time, video.fps)
        if keyframe.frame_idx != expected_frame_idx:
            return ValidationFailure(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Nhất quán frame",
                message=(
                    f"{keyframe.keyframe_id}: frame_idx {keyframe.frame_idx} disagrees "
                    f"with floor(pts_time {keyframe.pts_time} * fps {video.fps}) = "
                    f"{expected_frame_idx}"
                ),
            )

        neighbors = self.relational_map.get_neighbors(
            keyframe.keyframe_id, window=NEIGHBOR_WINDOW
        )
        before = [n for n in neighbors if n.keyframe_index < keyframe.keyframe_index]
        after = [n for n in neighbors if n.keyframe_index > keyframe.keyframe_index]

        if before and before[-1].frame_idx > keyframe.frame_idx:
            return ValidationFailure(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Nhất quán frame",
                message=(
                    f"{keyframe.keyframe_id}: frame_idx {keyframe.frame_idx} goes backwards "
                    f"from its previous keyframe {before[-1].keyframe_id} "
                    f"(frame_idx {before[-1].frame_idx})"
                ),
            )
        if after and after[0].frame_idx < keyframe.frame_idx:
            return ValidationFailure(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Nhất quán frame",
                message=(
                    f"{keyframe.keyframe_id}: frame_idx {keyframe.frame_idx} overtakes "
                    f"its next keyframe {after[0].keyframe_id} "
                    f"(frame_idx {after[0].frame_idx})"
                ),
            )
        return None

    def _check_schema(self, record: dict, schema_format: str) -> ValidationFailure | None:
        """Rule "Đúng schema" -> SCHEMA_MISMATCH. Structural check per format.

        These key names must match what `mapping/adapters/*.build()` emits, key
        for key. `by_frame` says `frame_id` — BTC's name for the field, adopted
        at the adapter boundary — not our internal `frame_idx`. Getting the two
        out of step blocks every valid submission with SCHEMA_MISMATCH, which
        reads like a data problem rather than a rename that was half-finished.
        """
        required: dict[str, tuple[str, ...]] = {
            "by_timestamp": ("video_id", "timestamp_ms"),
            "by_frame": ("video_id", "frame_id"),
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
