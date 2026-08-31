"""DEV-only oracle target generation and force-rank helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from mfe.contracts.ids import build_keyframe_id
from mfe.contracts.tools import Candidate
from mfe.offline.augmentation.models import (
    AuditStatus,
    BtcFrame,
    CandidateRow,
    CoverageAuditRow,
    OracleTargetRow,
    Selector,
)
from mfe.offline.keyframe_augmentation.common import pts_time_for_frame_idx
from mfe.online.tools.evidence import create_evidence_id


def query_intervals(query: dict) -> list[tuple[int, int]]:
    """Return inclusive strict intervals, falling back to wide intervals."""

    strict_events = query.get("gt_events_strict")
    events = strict_events or query.get("gt_events")
    if events:
        return [(int(start), int(end)) for start, end in events]
    frame_range = query.get("gt_frame_range_strict") or query.get("gt_frame_range")
    if frame_range:
        return [(int(frame_range[0]), int(frame_range[1]))]
    return []


def _covering_row(frame_idx: int, rows: Iterable[CoverageAuditRow]) -> CoverageAuditRow | None:
    eligible = [
        row
        for row in rows
        if row.eligible and row.start_frame <= frame_idx < row.end_frame_exclusive
    ]
    return min(eligible, key=lambda row: row.shot_id) if eligible else None


def build_oracle_targets(
    queries: Iterable[dict],
    pts_by_video: Mapping[str, Mapping[int, float]],
    btc_by_video: Mapping[str, list[BtcFrame]],
    audit_by_video: Mapping[str, list[CoverageAuditRow]],
    *,
    literal_fps_by_video: Mapping[str, float],
    decoder_fingerprints: Mapping[str, str],
    config_hash: str,
) -> list[OracleTargetRow]:
    """Pick the valid GT frame farthest from existing BTC evidence."""

    out: list[OracleTargetRow] = []
    for query in queries:
        query_id = str(query.get("query_id") or "")
        video_id = str(query.get("gt_video_id") or "")
        if not query_id or video_id not in pts_by_video:
            continue
        timeline = pts_by_video[video_id]
        btc = btc_by_video.get(video_id, [])
        btc_pts = [item.pts_time for item in btc]
        for event_index, (start, end) in enumerate(query_intervals(query)):
            frames = [frame for frame in timeline if start <= frame <= end]
            candidates: list[tuple[float, int, CoverageAuditRow]] = []
            for frame_idx in frames:
                row = _covering_row(frame_idx, audit_by_video.get(video_id, []))
                if row is None:
                    continue
                pts_time = float(timeline[frame_idx])
                distance = min(
                    (abs(pts_time - item) for item in btc_pts),
                    default=row.max_temporal_gap,
                )
                candidates.append((distance, frame_idx, row))
            if not candidates:
                continue
            distance, frame_idx, row = max(candidates, key=lambda item: (item[0], -item[1]))
            target = CandidateRow(
                keyframe_id=build_keyframe_id(video_id, frame_idx),
                video_id=video_id,
                frame_idx=frame_idx,
                literal_fps=float(literal_fps_by_video[video_id]),
                pts_time=pts_time_for_frame_idx(
                    frame_idx, float(literal_fps_by_video[video_id])
                ),
                decoder_pts_time=float(timeline[frame_idx]),
                shot_id=row.shot_id,
                selector=Selector.ORACLE_TARGET,
                coverage_class=row.coverage_class,
                selection_round=1,
                distance_to_nearest_btc=distance,
                detector_confidence=row.detector_confidence,
                decoder_fingerprint=decoder_fingerprints[video_id],
                config_hash=config_hash,
                audit_status=AuditStatus.AUTO_ACCEPTED,
            )
            out.append(
                OracleTargetRow(
                    query_id=query_id,
                    event_index=event_index,
                    video_id=video_id,
                    frame_range_start=start,
                    frame_range_end=end,
                    target=target,
                )
            )
    return out


def force_rank(
    candidates: Iterable[Candidate],
    target: OracleTargetRow,
    *,
    run_id: str,
    top_k: int,
) -> list[Candidate]:
    """Prepend an oracle frame without modifying the production cascade."""

    if top_k < 1:
        return []
    forced = Candidate(
        keyframe_id=target.target.keyframe_id,
        video_id=target.video_id,
        score=1.0,
        evidence_id=create_evidence_id("oracle_force", run_id, target.event_index + 1),
    )
    out = [forced]
    out.extend(item for item in candidates if item.keyframe_id != forced.keyframe_id)
    return out[:top_k]
