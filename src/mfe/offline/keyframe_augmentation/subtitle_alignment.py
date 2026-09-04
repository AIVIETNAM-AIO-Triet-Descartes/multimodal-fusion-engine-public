"""Align supplemental frames to the immutable BTC subtitle windows.

The BTC subtitle stage assigns transcript text to half-open windows beginning at
one BTC keyframe and ending at the next.  Supplemental frames can reuse that
already-audited assignment without rerunning ASR and without deriving a
timestamp from FPS: locate the containing BTC window by the decoder-authored
``frame_idx`` and inherit its text.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mfe.contracts.ids import build_keyframe_id


class SubtitleAlignmentStatus(StrEnum):
    """Outcome for one supplemental keyframe."""

    ALIGNED = "aligned"
    BEFORE_FIRST_BTC = "before_first_btc"
    SOURCE_WINDOW_SILENT = "source_window_silent"


@dataclass(frozen=True)
class BtcSubtitleWindow:
    """One BTC keyframe window read literally from the canonical mapping CSV."""

    keyframe_id: str
    video_id: str
    frame_idx: int
    next_frame_idx: int | None


@dataclass(frozen=True)
class SubtitleAlignmentResult:
    """Text payload plus a complete per-candidate audit trail."""

    subtitles: dict[str, str]
    audit_rows: tuple[dict[str, Any], ...]


def build_btc_subtitle_windows(
    video_id: str,
    mapping_rows: Sequence[Mapping[str, Any]],
) -> tuple[BtcSubtitleWindow, ...]:
    """Build ordered BTC windows without calculating frame indices or PTS."""

    if not video_id.strip():
        raise ValueError("video_id must be non-empty")
    parsed: list[tuple[int, int]] = []
    for row_number, row in enumerate(mapping_rows, 1):
        try:
            ordinal = int(row["n"])
            frame_idx = int(row["frame_idx"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid BTC mapping row {row_number}") from error
        if frame_idx < 0:
            raise ValueError("BTC frame_idx must be non-negative")
        parsed.append((ordinal, frame_idx))

    ordinals = sorted(ordinal for ordinal, _ in parsed)
    if ordinals != list(range(1, len(parsed) + 1)):
        raise ValueError("BTC mapping n must be contiguous and one-based")

    # BTC contains adjacent rows that point at the same source frame, and a few
    # mapping files are not physically ordered by time.  Identity is based on
    # the literal CSV frame_idx, so duplicate frame rows represent one store
    # key and must be collapsed before half-open windows are built.  Sorting is
    # only ordering those copied coordinates; it does not derive a timestamp or
    # manufacture a frame number.
    frame_indices = sorted({frame_idx for _, frame_idx in parsed})

    return tuple(
        BtcSubtitleWindow(
            keyframe_id=build_keyframe_id(video_id, frame_idx),
            video_id=video_id,
            frame_idx=frame_idx,
            next_frame_idx=(
                frame_indices[index + 1] if index + 1 < len(frame_indices) else None
            ),
        )
        for index, frame_idx in enumerate(frame_indices)
    )


def align_supplemental_subtitles(
    video_id: str,
    candidates: Sequence[Mapping[str, Any]],
    btc_windows: Sequence[BtcSubtitleWindow],
    btc_subtitles: Mapping[str, Any],
) -> SubtitleAlignmentResult:
    """Inherit text from the BTC window containing each supplemental frame.

    The function deliberately ignores candidate ``pts_time`` fields.  Current
    B1 selection manifests label those values as FPS-derived estimates; using
    them as durable subtitle coordinates would violate the project's timestamp
    contract.  Decoder-authored ``frame_idx`` and canonical BTC mapping rows are
    sufficient to establish temporal containment.
    """

    if any(window.video_id != video_id for window in btc_windows):
        raise ValueError("BTC subtitle windows belong to another video")
    starts = [window.frame_idx for window in btc_windows]
    if starts != sorted(starts) or len(starts) != len(set(starts)):
        raise ValueError("BTC subtitle windows must be strictly ordered")

    btc_ids = {window.keyframe_id for window in btc_windows}
    seen: set[str] = set()
    subtitles: dict[str, str] = {}
    audit: list[dict[str, Any]] = []
    ordered_candidates = sorted(candidates, key=lambda row: int(row["frame_idx"]))
    for row in ordered_candidates:
        frame_idx = int(row["frame_idx"])
        keyframe_id = str(row["keyframe_id"])
        expected_id = build_keyframe_id(video_id, frame_idx)
        if keyframe_id != expected_id:
            raise ValueError(f"candidate keyframe_id must be {expected_id}")
        if keyframe_id in seen:
            raise ValueError(f"duplicate supplemental keyframe_id: {keyframe_id}")
        if keyframe_id in btc_ids:
            raise ValueError(f"supplemental keyframe overlaps BTC: {keyframe_id}")
        seen.add(keyframe_id)

        position = bisect_right(starts, frame_idx) - 1
        if position < 0:
            audit.append(
                {
                    "keyframe_id": keyframe_id,
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "status": SubtitleAlignmentStatus.BEFORE_FIRST_BTC.value,
                    "source_keyframe_id": None,
                    "source_window_start_frame_idx": None,
                    "source_window_end_frame_idx_exclusive": starts[0] if starts else None,
                }
            )
            continue

        window = btc_windows[position]
        raw_text = btc_subtitles.get(window.keyframe_id, "")
        text = str(raw_text).strip() if raw_text is not None else ""
        status = (
            SubtitleAlignmentStatus.ALIGNED
            if text
            else SubtitleAlignmentStatus.SOURCE_WINDOW_SILENT
        )
        if text:
            subtitles[keyframe_id] = text
        audit.append(
            {
                "keyframe_id": keyframe_id,
                "video_id": video_id,
                "frame_idx": frame_idx,
                "status": status.value,
                "source_keyframe_id": window.keyframe_id,
                "source_window_start_frame_idx": window.frame_idx,
                "source_window_end_frame_idx_exclusive": window.next_frame_idx,
            }
        )

    return SubtitleAlignmentResult(subtitles=subtitles, audit_rows=tuple(audit))
