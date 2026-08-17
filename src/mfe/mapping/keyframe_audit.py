"""Audit frame indices derived for self-extracted keyframes.

Official BTC keyframes remain the source of truth.  This module checks a
custom keyframe (DAKE, uniform sampling, or another extractor) against the two
official keyframes surrounding its presentation timestamp.  It never inserts
the custom keyframe into the Relational Map.

Spec: ``docs/task/trong/trong-keyframe-validator.md``.
Owner: Pham Trong.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Protocol

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import frame_idx_from_pts
from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.validator import ValidationFailure, ValidationResult, ValidationWarning

#: Maximum absolute interpolation error, measured in frames.
#:
#: On the five-video sample, correct-fps candidates had at most 0.5 frame of
#: floor-rounding noise.  A 1.0-frame tolerance therefore produced no false
#: positives while detecting 98.1% of the simulated 30.0-vs-29.97 fps errors
#: (whose median error was 26 frames).
FRAME_ERROR_TOLERANCE: float = 1.0

#: Brackets no wider than this add nothing at the default tolerance.
#:
#: A zero-frame gap cannot distinguish integer candidates (614 adjacent pairs
#: in batch 1 had the same frame index).  With a one-frame gap, the inclusive
#: range check constrains every integer candidate and the interpolation error
#: cannot exceed the default 1.0-frame tolerance.  A caller choosing a stricter
#: tolerance still gets interpolation; see ``should_interpolate`` below.
MIN_INTERPOLATION_GAP_FRAMES: int = 1

#: A gap above 300 frames is considered weak evidence.
#:
#: The measured gap was 114 frames on average and 198 frames at p95.  Using 300
#: avoids warning on ordinary spacing while flagging unusually sparse regions
#: where merely falling inside the bracket says little about correctness.
LARGE_GAP_FRAMES: int = 300


class _RelationalMapReader(Protocol):
    """Read-only Relational Map surface required by the audit."""

    def get_video(self, video_id: str) -> VideoRow: ...

    def find_bracketing_keyframes(
        self, video_id: str, pts_time: float
    ) -> tuple[KeyframeRow | None, KeyframeRow | None]: ...


def audit_custom_keyframe(
    relational_map: _RelationalMapReader,
    video_id: str,
    pts_time: float,
    candidate_frame_idx: int | None = None,
    frame_error_tolerance: float = FRAME_ERROR_TOLERANCE,
) -> ValidationResult:
    """Assess whether a custom keyframe's frame index is safe to submit.

    ``candidate_frame_idx`` may be supplied to audit an extractor's result.  If
    omitted, it is derived with the shared :func:`frame_idx_from_pts` contract
    and the video's literal FPS stored in the Relational Map.

    A definite violation (out of video/bracket bounds or disagreement with an
    exact official keyframe) blocks the candidate.  Missing or sparse evidence,
    and an excessive interpolation error that is still inside the bracket,
    produce warnings so callers can distinguish ``valid`` from ``confident``.
    """
    if (
        isinstance(frame_error_tolerance, bool)
        or not isinstance(frame_error_tolerance, Real)
        or not math.isfinite(frame_error_tolerance)
        or frame_error_tolerance < 0
    ):
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            "frame_error_tolerance must be a finite number >= 0, "
            f"got {frame_error_tolerance!r}",
        )
    frame_error_tolerance = float(frame_error_tolerance)

    if (
        isinstance(pts_time, bool)
        or not isinstance(pts_time, Real)
    ):
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"pts_time must be a real number, got {pts_time!r}",
        )
    pts_time = float(pts_time)

    if candidate_frame_idx is not None:
        if isinstance(candidate_frame_idx, bool) or not isinstance(
            candidate_frame_idx, Integral
        ):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                "candidate_frame_idx must be an integer frame number, "
                f"got {candidate_frame_idx!r}",
            )
        candidate_frame_idx = int(candidate_frame_idx)

    try:
        video = relational_map.get_video(video_id)
    except ContractError as error:
        if error.code != ErrorCode.VIDEO_NOT_FOUND:
            raise
        return ValidationResult(
            valid=False,
            failures=[
                ValidationFailure(
                    code=ErrorCode.VIDEO_NOT_FOUND,
                    rule="Tồn tại video",
                    message=f"Video {video_id} not found in Relational Map",
                )
            ],
        )

    duration_seconds = video.duration_ms / 1000.0
    if not math.isfinite(pts_time) or not (0.0 <= pts_time <= duration_seconds):
        return ValidationResult(
            valid=False,
            failures=[
                ValidationFailure(
                    code=ErrorCode.TIMESTAMP_OUT_OF_BOUNDS,
                    rule="Timestamp hợp lệ",
                    message=(
                        f"pts_time {pts_time!r}s out of bounds for video {video_id} "
                        f"(duration {duration_seconds:.3f}s)"
                    ),
                )
            ],
        )

    candidate = (
        frame_idx_from_pts(pts_time, video.fps)
        if candidate_frame_idx is None
        else candidate_frame_idx
    )
    if not (0 <= candidate < video.total_frames):
        return ValidationResult(
            valid=False,
            failures=[
                ValidationFailure(
                    code=ErrorCode.FRAME_OUT_OF_BOUNDS,
                    rule="Trong biên",
                    message=(
                        f"candidate frame_idx {candidate} out of bounds for video {video_id} "
                        f"({video.total_frames} frames)"
                    ),
                )
            ],
        )

    previous, following = relational_map.find_bracketing_keyframes(video_id, pts_time)

    # find_bracketing_keyframes deliberately puts an exact timestamp in
    # ``previous``.  Equality with official data is stronger than interpolation.
    if previous is not None and previous.pts_time == pts_time:
        if candidate == previous.frame_idx:
            return ValidationResult(valid=True)
        return ValidationResult(
            valid=False,
            failures=[
                ValidationFailure(
                    code=ErrorCode.FRAME_INCONSISTENT,
                    rule="Trùng khớp keyframe chính thức",
                    message=(
                        f"{video_id}: candidate frame_idx {candidate} at pts_time "
                        f"{pts_time:.6f}s disagrees with official keyframe "
                        f"{previous.keyframe_id} (frame_idx {previous.frame_idx})"
                    ),
                )
            ],
        )

    failures: list[ValidationFailure] = []
    warnings: list[ValidationWarning] = []

    # An open-ended bracket cannot support interpolation.  It remains usable,
    # but never confident; the one known boundary can still expose a violation.
    if previous is None:
        warnings.append(
            ValidationWarning(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Độ tin cậy",
                message=(
                    f"{video_id}: missing official keyframe before pts_time "
                    f"{pts_time:.6f}s (before first keyframe)"
                ),
            )
        )
        if following is not None and candidate > following.frame_idx:
            failures.append(
                ValidationFailure(
                    code=ErrorCode.FRAME_INCONSISTENT,
                    rule="Biên keyframe",
                    message=(
                        f"{video_id}: candidate frame_idx {candidate} overtakes next official "
                        f"keyframe {following.keyframe_id} (frame_idx {following.frame_idx})"
                    ),
                )
            )

    if following is None:
        warnings.append(
            ValidationWarning(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Độ tin cậy",
                message=(
                    f"{video_id}: missing official keyframe after pts_time "
                    f"{pts_time:.6f}s (after last keyframe)"
                ),
            )
        )
        if previous is not None and candidate < previous.frame_idx:
            failures.append(
                ValidationFailure(
                    code=ErrorCode.FRAME_INCONSISTENT,
                    rule="Biên keyframe",
                    message=(
                        f"{video_id}: candidate frame_idx {candidate} precedes previous official "
                        f"keyframe {previous.keyframe_id} (frame_idx {previous.frame_idx})"
                    ),
                )
            )

    if previous is None or following is None:
        return ValidationResult(
            valid=not failures,
            failures=failures,
            warnings=warnings,
        )

    gap_frames = following.frame_idx - previous.frame_idx
    if candidate < previous.frame_idx:
        failures.append(
            ValidationFailure(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Biên keyframe",
                message=(
                    f"{video_id}: candidate frame_idx {candidate} precedes previous official "
                    f"keyframe {previous.keyframe_id} (frame_idx {previous.frame_idx})"
                ),
            )
        )
    elif candidate > following.frame_idx:
        failures.append(
            ValidationFailure(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Biên keyframe",
                message=(
                    f"{video_id}: candidate frame_idx {candidate} overtakes next official "
                    f"keyframe {following.keyframe_id} (frame_idx {following.frame_idx})"
                ),
            )
        )

    if gap_frames > LARGE_GAP_FRAMES:
        warnings.append(
            ValidationWarning(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Độ tin cậy thưa",
                message=(
                    f"{video_id}: large gap between bracketing keyframes "
                    f"({gap_frames} frames between {previous.keyframe_id} "
                    f"and {following.keyframe_id})"
                ),
            )
        )

    # Interpolation warnings only qualify a candidate that already passed the
    # hard inclusive range check.  A range violation is a failure, not merely a
    # suspicious interpolation result.
    if failures:
        return ValidationResult(valid=False, failures=failures, warnings=warnings)

    gap_seconds = following.pts_time - previous.pts_time
    if gap_seconds <= 0.0:
        warnings.append(
            ValidationWarning(
                code=ErrorCode.FRAME_INCONSISTENT,
                rule="Độ tin cậy",
                message=(
                    f"{video_id}: official bracket has a non-positive timestamp gap "
                    f"between {previous.keyframe_id} and {following.keyframe_id}"
                ),
            )
        )
    else:
        # At the default 1-frame tolerance, gaps of zero or one cannot produce
        # a useful warning beyond the inclusive bounds check.  If a caller asks
        # for a stricter tolerance, a one-frame gap can matter (for example an
        # expected position of 0.6 versus integer candidate 0), so do not skip
        # it.  A zero-frame gap remains safe: expected_frame == candidate after
        # the bounds check, and this formula never divides by ``gap_frames``.
        should_interpolate = (
            gap_frames > MIN_INTERPOLATION_GAP_FRAMES
            or frame_error_tolerance < gap_frames
        )
        if not should_interpolate:
            return ValidationResult(valid=True, warnings=warnings)

        expected_ratio = (pts_time - previous.pts_time) / gap_seconds
        expected_frame = previous.frame_idx + expected_ratio * gap_frames
        frame_error = abs(candidate - expected_frame)

        if frame_error > frame_error_tolerance:
            warnings.append(
                ValidationWarning(
                    code=ErrorCode.FRAME_INCONSISTENT,
                    rule="Tỉ lệ nội suy",
                    message=(
                        f"{video_id}: interpolation frame_error {frame_error:.3f} > "
                        f"tolerance {frame_error_tolerance:.3f} "
                        f"(expected {expected_frame:.3f}, candidate {candidate})"
                    ),
                )
            )

    return ValidationResult(valid=True, warnings=warnings)
