"""TRAKE (Tracking & Event Retrieval) Metric Evaluation Module.

Ref: docs/metrics-btc.html §4 (PDF §2.1.3).
Author: Phạm Trọng (TV3)

Evaluates sequential multi-event queries where N discrete keyframes/timestamps
are submitted for N ground-truth events within a single video.
"""

from __future__ import annotations

from dataclasses import dataclass

from mfe.contracts.errors import ContractError, ErrorCode


@dataclass(frozen=True)
class TRAKEEvent:
    """Ground truth frame window for a single event in TRAKE query."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.start_frame, int) or isinstance(self.start_frame, bool):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"start_frame must be an integer, got {type(self.start_frame)}",
            )
        if not isinstance(self.end_frame, int) or isinstance(self.end_frame, bool):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"end_frame must be an integer, got {type(self.end_frame)}",
            )
        if self.start_frame > self.end_frame:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"start_frame ({self.start_frame}) cannot be greater than "
                f"end_frame ({self.end_frame})",
            )

    def contains(self, frame_id: int) -> bool:
        """Check if frame_id falls inside [start_frame, end_frame] (inclusive)."""
        return self.start_frame <= frame_id <= self.end_frame


@dataclass(frozen=True)
class TRAKEGroundTruth:
    """Ground truth for a TRAKE query."""

    video_id: str
    events: list[TRAKEEvent]

    def __post_init__(self) -> None:
        if not self.video_id or not isinstance(self.video_id, str):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"video_id must be a non-empty string, got {self.video_id!r}",
            )
        if not isinstance(self.events, (list, tuple)) or not self.events:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                "TRAKE ground truth must contain a non-empty list of events",
            )
        for idx, event in enumerate(self.events):
            if not isinstance(event, TRAKEEvent):
                raise ContractError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"events[{idx}] must be a TRAKEEvent instance, got {type(event)}",
                )


@dataclass(frozen=True)
class TRAKESubmission:
    """Submission candidate for a TRAKE query."""

    video_id: str
    frame_ids: list[int]

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"video_id must be a string, got {type(self.video_id)}",
            )
        if not isinstance(self.frame_ids, (list, tuple)):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"frame_ids must be a list or tuple of integers, got {type(self.frame_ids)}",
            )
        for idx, fid in enumerate(self.frame_ids):
            if not isinstance(fid, int) or isinstance(fid, bool):
                raise ContractError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"frame_ids[{idx}] must be an integer, got {fid!r} ({type(fid)})",
                )


def score_trake(submission: TRAKESubmission, gt: TRAKEGroundTruth) -> float:
    """Calculate R-Score for a single TRAKE query submission candidate.

    Formula (PDF §2.1.3):
        If submission.video_id != gt.video_id:
            R-Score = 0.0
        If submission.video_id == gt.video_id:
            R-Score = (1 / N) * sum( I(frame_ids[j] in [start_j, end_j]) for j in 1..N )

    Args:
        submission: TRAKESubmission with video_id and list of N frame_ids.
        gt: TRAKEGroundTruth with video_id and list of N events [start, end].

    Returns:
        float: Score in range [0.0, 1.0].

    Raises:
        ContractError: If input types are invalid or length of frame_ids
            does not match the number of events N in ground truth.
    """
    if not isinstance(submission, TRAKESubmission):
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"submission must be a TRAKESubmission, got {type(submission)}",
        )
    if not isinstance(gt, TRAKEGroundTruth):
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"gt must be a TRAKEGroundTruth, got {type(gt)}",
        )

    # Rule 1: Video mismatch -> 0.0 immediately
    if submission.video_id != gt.video_id:
        return 0.0

    # Rule 2: Event count mismatch -> raise ContractError (don't silently pad or slice)
    num_events = len(gt.events)
    if len(submission.frame_ids) != num_events:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"Mismatched event count: submission has {len(submission.frame_ids)} frame_ids, "
            f"ground truth expects {num_events} events",
        )

    # Rule 3: Match each frame_id against corresponding event range [start_frame, end_frame]
    matched_count = sum(
        1
        for frame_id, event in zip(submission.frame_ids, gt.events, strict=True)
        if event.contains(frame_id)
    )

    score = matched_count / num_events
    return float(score)
