"""Deterministic B1--B4 selectors and the global 3% allocator."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping

from mfe.contracts.ids import build_keyframe_id
from mfe.offline.augmentation.models import (
    BtcFrame,
    CandidateRow,
    CoverageAuditRow,
    CoverageClass,
    Selector,
)
from mfe.offline.keyframe_augmentation.common import pts_time_for_frame_idx


def _frames_in_shot(row: CoverageAuditRow, pts_by_frame: Mapping[int, float]) -> list[int]:
    frames = [
        frame_idx
        for frame_idx in pts_by_frame
        if row.start_frame <= frame_idx < row.end_frame_exclusive
    ]
    if not frames:
        raise ValueError(f"{row.shot_id}: timeline contains no frame in shot interval")
    return sorted(frames)


def _nearest_frame(frames: list[int], pts_by_frame: Mapping[int, float], target_pts: float) -> int:
    return min(frames, key=lambda frame_idx: (abs(pts_by_frame[frame_idx] - target_pts), frame_idx))


def _distance(pts: float, anchors: Iterable[float]) -> float:
    values = list(anchors)
    return min((abs(pts - anchor) for anchor in values), default=0.0)


def _candidate(
    row: CoverageAuditRow,
    frame_idx: int,
    pts_by_frame: Mapping[int, float],
    btc_frames: list[BtcFrame],
    *,
    literal_fps_by_video: Mapping[str, float],
    selector: Selector,
    selection_round: int,
    decoder_fingerprint: str,
    config_hash: str,
) -> CandidateRow:
    decoder_pts_time = float(pts_by_frame[frame_idx])
    literal_fps = float(literal_fps_by_video[row.video_id])
    pts_time = pts_time_for_frame_idx(frame_idx, literal_fps)
    anchors = [item.pts_time for item in btc_frames]
    if not anchors:
        anchors = [row.start_pts_time, row.end_pts_time]
    return CandidateRow(
        keyframe_id=build_keyframe_id(row.video_id, frame_idx),
        video_id=row.video_id,
        frame_idx=frame_idx,
        literal_fps=literal_fps,
        pts_time=pts_time,
        decoder_pts_time=decoder_pts_time,
        shot_id=row.shot_id,
        selector=selector,
        coverage_class=row.coverage_class,
        selection_round=selection_round,
        distance_to_nearest_btc=_distance(decoder_pts_time, anchors),
        detector_confidence=row.detector_confidence,
        decoder_fingerprint=decoder_fingerprint,
        config_hash=config_hash,
        audit_status=row.audit_status,
    )


def _eligible(rows: Iterable[CoverageAuditRow]) -> list[CoverageAuditRow]:
    return sorted((row for row in rows if row.eligible), key=lambda row: row.shot_id)


def select_b1(
    rows: Iterable[CoverageAuditRow],
    pts_by_video: Mapping[str, Mapping[int, float]],
    btc_by_shot: Mapping[str, list[BtcFrame]],
    *,
    literal_fps_by_video: Mapping[str, float],
    decoder_fingerprints: Mapping[str, str],
    config_hash: str,
) -> list[CandidateRow]:
    """B1: one temporal center for every accepted C0 shot."""

    out: list[CandidateRow] = []
    for row in _eligible(rows):
        if row.coverage_class != CoverageClass.C0:
            continue
        timeline = pts_by_video[row.video_id]
        frames = _frames_in_shot(row, timeline)
        center_pts = (row.start_pts_time + row.end_pts_time) / 2.0
        frame_idx = _nearest_frame(frames, timeline, center_pts)
        out.append(
            _candidate(
                row,
                frame_idx,
                timeline,
                btc_by_shot.get(row.shot_id, []),
                literal_fps_by_video=literal_fps_by_video,
                selector=Selector.SHOT_CENTER,
                selection_round=1,
                decoder_fingerprint=decoder_fingerprints[row.video_id],
                config_hash=config_hash,
            )
        )
    return out


def _farthest_frame(
    frames: list[int],
    pts_by_frame: Mapping[int, float],
    anchor_pts: list[float],
    excluded: set[int],
) -> int | None:
    available = [frame_idx for frame_idx in frames if frame_idx not in excluded]
    if not available:
        return None
    return max(
        available,
        key=lambda frame_idx: (
            _distance(float(pts_by_frame[frame_idx]), anchor_pts),
            -frame_idx,
        ),
    )


def select_b2(
    rows: Iterable[CoverageAuditRow],
    pts_by_video: Mapping[str, Mapping[int, float]],
    btc_by_shot: Mapping[str, list[BtcFrame]],
    *,
    literal_fps_by_video: Mapping[str, float],
    decoder_fingerprints: Mapping[str, str],
    config_hash: str,
) -> list[CandidateRow]:
    """B2: one frame at the largest BTC blind gap in each eligible shot."""

    out: list[CandidateRow] = []
    for row in _eligible(rows):
        timeline = pts_by_video[row.video_id]
        frames = _frames_in_shot(row, timeline)
        btc = btc_by_shot.get(row.shot_id, [])
        protected = {item.frame_idx for item in btc}
        anchors = [item.pts_time for item in btc]
        if not anchors:
            # Farthest from both shot boundaries is the temporal center.
            anchors = [row.start_pts_time, row.end_pts_time]
        frame_idx = _farthest_frame(frames, timeline, anchors, protected)
        if frame_idx is None:
            continue
        out.append(
            _candidate(
                row,
                frame_idx,
                timeline,
                btc,
                literal_fps_by_video=literal_fps_by_video,
                selector=Selector.FARTHEST,
                selection_round=1,
                decoder_fingerprint=decoder_fingerprints[row.video_id],
                config_hash=config_hash,
            )
        )
    return out


def select_b3(
    rows: Iterable[CoverageAuditRow],
    pts_by_video: Mapping[str, Mapping[int, float]],
    btc_by_shot: Mapping[str, list[BtcFrame]],
    *,
    literal_fps_by_video: Mapping[str, float],
    decoder_fingerprints: Mapping[str, str],
    config_hash: str,
) -> list[CandidateRow]:
    """B3: dynamic quota with center first for C0, then greedy farthest."""

    out: list[CandidateRow] = []
    for row in _eligible(rows):
        timeline = pts_by_video[row.video_id]
        frames = _frames_in_shot(row, timeline)
        btc = btc_by_shot.get(row.shot_id, [])
        protected = {item.frame_idx for item in btc}
        chosen: list[int] = []
        quota = max(1, row.deficit)
        for selection_round in range(1, quota + 1):
            if row.coverage_class == CoverageClass.C0 and selection_round == 1:
                target_pts = (row.start_pts_time + row.end_pts_time) / 2.0
                available = [item for item in frames if item not in protected]
                if not available:
                    break
                frame_idx = _nearest_frame(available, timeline, target_pts)
                selector = Selector.SHOT_CENTER
            else:
                anchors = [item.pts_time for item in btc]
                anchors.extend(float(timeline[item]) for item in chosen)
                if not anchors:
                    anchors = [row.start_pts_time, row.end_pts_time]
                found = _farthest_frame(frames, timeline, anchors, protected | set(chosen))
                if found is None:
                    break
                frame_idx = found
                selector = Selector.FARTHEST
            chosen.append(frame_idx)
            out.append(
                _candidate(
                    row,
                    frame_idx,
                    timeline,
                    btc,
                    literal_fps_by_video=literal_fps_by_video,
                    selector=selector,
                    selection_round=selection_round,
                    decoder_fingerprint=decoder_fingerprints[row.video_id],
                    config_hash=config_hash,
                )
            )
    return out


def select_b4(
    b3: Iterable[CandidateRow],
    rows: Iterable[CoverageAuditRow],
    pts_by_video: Mapping[str, Mapping[int, float]],
    btc_by_shot: Mapping[str, list[BtcFrame]],
    *,
    literal_fps_by_video: Mapping[str, float],
    decoder_fingerprints: Mapping[str, str],
    config_hash: str,
    seed: int,
    quota_multiplier: int = 1,
) -> list[CandidateRow]:
    """B4 random control, optionally over-sampled for post-dedup matching."""

    if quota_multiplier < 1:
        raise ValueError("quota_multiplier must be >= 1")

    quotas = Counter(item.shot_id for item in b3)
    rows_by_shot = {row.shot_id: row for row in _eligible(rows)}
    out: list[CandidateRow] = []
    for shot_id in sorted(quotas):
        row = rows_by_shot[shot_id]
        timeline = pts_by_video[row.video_id]
        btc = btc_by_shot.get(shot_id, [])
        protected = {item.frame_idx for item in btc}
        frames = [item for item in _frames_in_shot(row, timeline) if item not in protected]
        rng = random.Random(f"{seed}:{shot_id}")
        rng.shuffle(frames)
        quota = quotas[shot_id] * quota_multiplier
        for selection_round, frame_idx in enumerate(frames[:quota], 1):
            out.append(
                _candidate(
                    row,
                    frame_idx,
                    timeline,
                    btc,
                    literal_fps_by_video=literal_fps_by_video,
                    selector=Selector.RANDOM,
                    selection_round=selection_round,
                    decoder_fingerprint=decoder_fingerprints[row.video_id],
                    config_hash=config_hash,
                )
            )
    return out


def apply_global_cap(
    candidates: Iterable[CandidateRow],
    audit_rows: Iterable[CoverageAuditRow],
    *,
    base_keyframes: int,
    cap_ratio: float,
) -> list[CandidateRow]:
    """Apply the round-first deterministic allocator from the final decision."""

    if base_keyframes < 0 or not 0 < cap_ratio <= 1:
        raise ValueError("base_keyframes must be non-negative and cap_ratio in (0, 1]")
    limit = int(base_keyframes * cap_ratio)
    if limit == 0:
        return []
    rows_by_shot = {row.shot_id: row for row in audit_rows}
    unique: dict[str, CandidateRow] = {}
    for candidate in candidates:
        unique.setdefault(candidate.keyframe_id, candidate)

    def priority(candidate: CandidateRow):
        row = rows_by_shot[candidate.shot_id]
        return (
            candidate.selection_round,
            -(row.end_pts_time - row.start_pts_time),
            -candidate.detector_confidence,
            -candidate.distance_to_nearest_btc,
            candidate.video_id,
            candidate.frame_idx,
        )

    return sorted(unique.values(), key=priority)[:limit]


def counts_by_shot(candidates: Iterable[CandidateRow]) -> dict[str, int]:
    return dict(sorted(Counter(item.shot_id for item in candidates).items()))


def match_random_control(
    treatment: Iterable[CandidateRow], random_pool: Iterable[CandidateRow]
) -> list[CandidateRow]:
    """Take a deterministic random-control subset with exact per-shot quotas."""

    quotas = Counter(item.shot_id for item in treatment)
    by_shot: dict[str, list[CandidateRow]] = {}
    for candidate in random_pool:
        by_shot.setdefault(candidate.shot_id, []).append(candidate)
    out: list[CandidateRow] = []
    for shot_id in sorted(quotas):
        available = sorted(
            by_shot.get(shot_id, []),
            key=lambda item: (item.selection_round, item.frame_idx),
        )
        if len(available) < quotas[shot_id]:
            raise ValueError(
                f"{shot_id}: random pool has {len(available)} post-dedup frames, "
                f"needs {quotas[shot_id]}"
            )
        out.extend(available[: quotas[shot_id]])
    return out
