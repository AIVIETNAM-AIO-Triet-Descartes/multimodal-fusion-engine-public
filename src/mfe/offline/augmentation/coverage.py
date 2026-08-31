"""Coverage audit and detector-threshold calibration for Gate 1."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from mfe.contracts.ids import build_keyframe_id
from mfe.offline.augmentation.models import (
    AuditStatus,
    BtcFrame,
    CoverageAuditRow,
    CoverageClass,
    ShotProposal,
)


def read_btc_mapping(video_id: str, path: str | Path) -> list[BtcFrame]:
    """Read BTC ``frame_idx`` and ``pts_time`` literally from its CSV."""

    best: dict[int, BtcFrame] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                frame_idx = int(row["frame_idx"])
                pts_time = float(row["pts_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: invalid frame_idx/pts_time") from error
            candidate = BtcFrame(
                keyframe_id=build_keyframe_id(video_id, frame_idx),
                video_id=video_id,
                frame_idx=frame_idx,
                pts_time=pts_time,
            )
            # BTC contains adjacent rows that collapse to the same source frame.
            # Identity is frame_idx, so keep one deterministic row.
            best.setdefault(frame_idx, candidate)
    return sorted(best.values(), key=lambda item: (item.frame_idx, item.pts_time))


def btc_in_shot(shot: ShotProposal, btc_frames: list[BtcFrame]) -> list[BtcFrame]:
    return [
        item
        for item in btc_frames
        if item.video_id == shot.video_id
        and shot.start_frame <= item.frame_idx < shot.end_frame_exclusive
    ]


def temporal_gaps(shot: ShotProposal, btc_frames: list[BtcFrame]) -> list[float]:
    """Boundary-to-BTC and BTC-to-BTC gaps in direct PTS seconds."""

    points = [shot.start_pts_time]
    points.extend(item.pts_time for item in sorted(btc_frames, key=lambda item: item.pts_time))
    points.append(shot.end_pts_time)
    gaps = [right - left for left, right in zip(points, points[1:], strict=False)]
    if any(gap < 0 for gap in gaps):
        raise ValueError(f"{shot.shot_id}: BTC PTS is not monotonic inside the shot")
    return gaps


def k_target(duration_seconds: float, target_gap_seconds: float, k_max: int) -> int:
    if duration_seconds <= 0 or target_gap_seconds <= 0 or k_max < 1:
        raise ValueError("duration/target gap must be positive and k_max >= 1")
    return min(k_max, max(1, math.ceil(duration_seconds / target_gap_seconds)))


def classify_coverage(
    shot: ShotProposal,
    btc_frames: list[BtcFrame],
    *,
    target_gap_seconds: float,
    k_max: int,
    micro_shot_seconds: float,
    uncertain_below: float,
) -> CoverageAuditRow:
    """Classify one shot as C0/C1/CT/CE/OK under a frozen density policy."""

    inside = btc_in_shot(shot, btc_frames)
    gaps = temporal_gaps(shot, inside)
    maximum_gap = max(gaps, default=shot.duration_seconds)
    target = k_target(shot.duration_seconds, target_gap_seconds, k_max)
    deficit = max(0, target - len(inside))

    if shot.duration_seconds < micro_shot_seconds:
        coverage_class = CoverageClass.CT
        status = AuditStatus.NEEDS_REVIEW
        reason = f"micro-shot {shot.duration_seconds:.3f}s < {micro_shot_seconds:.3f}s"
    elif shot.boundary_confidence < uncertain_below:
        coverage_class = CoverageClass.CE
        status = AuditStatus.NEEDS_REVIEW
        reason = (
            f"boundary confidence {shot.boundary_confidence:.3f} "
            f"< {uncertain_below:.3f}"
        )
    elif not inside:
        coverage_class = CoverageClass.C0
        status = AuditStatus.AUTO_ACCEPTED
        reason = "shot contains no BTC keyframe"
    elif deficit > 0 or maximum_gap > target_gap_seconds:
        coverage_class = CoverageClass.C1
        status = AuditStatus.AUTO_ACCEPTED
        reason = (
            f"btc={len(inside)} target={target}; max_gap={maximum_gap:.3f}s "
            f"target_gap={target_gap_seconds:.3f}s"
        )
    else:
        coverage_class = CoverageClass.OK
        status = AuditStatus.AUTO_ACCEPTED
        reason = "BTC density and temporal gap satisfy policy"

    return CoverageAuditRow(
        video_id=shot.video_id,
        shot_id=shot.shot_id,
        start_frame=shot.start_frame,
        end_frame_exclusive=shot.end_frame_exclusive,
        start_pts_time=shot.start_pts_time,
        end_pts_time=shot.end_pts_time,
        detector_confidence=shot.boundary_confidence,
        btc_keyframe_ids=tuple(item.keyframe_id for item in inside),
        btc_count=len(inside),
        max_temporal_gap=maximum_gap,
        coverage_class=coverage_class,
        target_gap_seconds=target_gap_seconds,
        k_target=target,
        k_max=k_max,
        deficit=deficit,
        audit_status=status,
        audit_reason=reason,
    )


@dataclass(frozen=True)
class BoundaryLabel:
    frame_idx: int
    is_boundary: bool


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


def measure_threshold(scores, labels: list[BoundaryLabel], threshold: float) -> ThresholdMetrics:
    """Measure a threshold on manually reviewed frame positions."""

    tp = fp = fn = 0
    for label in labels:
        if not 0 <= label.frame_idx < len(scores):
            raise ValueError(f"manual label frame {label.frame_idx} is outside score array")
        predicted = float(scores[label.frame_idx]) >= threshold
        if predicted and label.is_boundary:
            tp += 1
        elif predicted:
            fp += 1
        elif label.is_boundary:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ThresholdMetrics(threshold, precision, recall, f1, tp, fp, fn)


def choose_threshold(scores, labels: list[BoundaryLabel], grid: tuple[float, ...]):
    """Choose max-F1, then max-precision, then the higher threshold."""

    if not labels:
        raise ValueError("at least one manual boundary label is required")
    rows = [measure_threshold(scores, labels, threshold) for threshold in grid]
    selected = max(rows, key=lambda row: (row.f1, row.precision, row.threshold))
    return selected, rows
