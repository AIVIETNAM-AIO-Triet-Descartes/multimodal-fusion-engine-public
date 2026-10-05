"""Read technical metadata from a video file — step [1] of docs/02 §2.

This module is where `fps` enters the system. Everything downstream — every
timestamp, every submission — is derived from what it returns. docs/00 §7:
"Không hardcode FPS. Đọc từ metadata từng video." docs/context/buoi-1.md §7
records that BTC may ship videos with differing FPS and resolution.

Sprint 1 has two ways in, and uses the second:

    probe_video()          open the .mp4 and ask the container (needs PyAV)
    probe_from_provided()  read fps out of map-keyframes/{video_id}.csv

The provided path is preferred because the CSV's `fps` is the same number BTC
used to compute the `frame_idx` we submit. If the container disagreed with the
CSV, following the container would make our frames disagree with theirs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import validate_video_id
from mfe.contracts.records import VideoRow
from mfe.offline.provided import MapRow

logger = logging.getLogger(__name__)


def probe_from_provided(
    video_id: str,
    source_path: str,
    map_rows: list[MapRow],
    media_info: dict,
    frame_size: tuple[int, int],
) -> VideoRow:
    """Build a VideoRow from the shipped CSV + media-info. Owner: Quốc Đạt.

    Args:
        video_id: Validated video key.
        source_path: Path to the source .mp4, kept so the row can reopen it.
        map_rows: Parsed `map-keyframes/{video_id}.csv`.
        media_info: Parsed `media-info/{video_id}.json` (may be empty).
        frame_size: (width, height) read off the first keyframe image.

    `total_frames` and `duration_ms` are the two numbers the validator
    bound-checks a submission against (docs/05 §1.1), so their failure modes are
    asymmetric: too small blocks correct submissions, too large lets wrong ones
    through. Both are therefore taken as the max of an estimate and a fact:

        duration_s   = max(media-info "length", max(pts_time))
        total_frames = max(int(duration_s * fps) + 1, max(frame_idx) + 1)

    The second term of each is a fact — we hold a keyframe at that time and at
    that frame, so the video is at least that long. The estimate only ever
    widens the bound, never narrows it.

    Widening by `pts_time` is not hypothetical. BTC stores `media-info.length`
    as **whole seconds**, so any video whose last keyframe falls in the final
    fractional second is declared shorter than its own keyframes: on the 2026
    sample, L25_V003 has `length` 1162 against a last `pts_time` of 1162.12,
    and the resulting `timestamp_ms` 1162120 was rejected as out of bounds. The
    keyframe is real and BTC shipped it; the rounded duration is the wrong side
    of the comparison.
    """
    validate_video_id(video_id)
    if not map_rows:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, f"{video_id}: no keyframe rows to probe"
        )

    distinct_fps = {row.fps for row in map_rows}
    if len(distinct_fps) != 1:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{video_id}: map-keyframes declares more than one fps {sorted(distinct_fps)}; "
            "a variable-frame-rate video needs its frame mapping handled per row",
        )
    fps = distinct_fps.pop()
    if fps <= 0:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, f"{video_id}: fps must be > 0, got {fps}"
        )

    max_frame_idx = max(row.frame_idx for row in map_rows)
    last_pts_time = max(row.pts_time for row in map_rows)

    declared_length_s = media_info.get("length")
    if isinstance(declared_length_s, (int, float)) and declared_length_s > 0:
        duration_s = max(float(declared_length_s), last_pts_time)
        if duration_s != float(declared_length_s):
            logger.info(
                "%s: duration widened to %.3fs to cover the last keyframe's pts_time "
                "(media-info 'length' declared only %s s)",
                video_id,
                duration_s,
                declared_length_s,
            )
    else:
        # No media-info: the last keyframe is a floor on the real duration.
        duration_s = last_pts_time
        logger.warning(
            "%s: media-info has no usable 'length'; duration falls back to the last "
            "keyframe's pts_time (%.3fs), which understates the true duration",
            video_id,
            duration_s,
        )

    frames_from_duration = int(duration_s * fps) + 1
    total_frames = max(frames_from_duration, max_frame_idx + 1)
    if total_frames != frames_from_duration:
        logger.info(
            "%s: total_frames widened to %d to cover frame_idx %d "
            "(duration*fps implied only %d)",
            video_id,
            total_frames,
            max_frame_idx,
            frames_from_duration,
        )

    width, height = frame_size
    return VideoRow(
        video_id=video_id,
        source_path=source_path,
        fps=fps,
        duration_ms=round(duration_s * 1000),
        total_frames=total_frames,
        width=width,
        height=height,
    )


def probe_video(source_path: str, video_id: str) -> VideoRow:
    """Read fps / duration / resolution / frame count straight from the container.

    Owner: Quốc Đạt. Not the Sprint 1 path — see the module docstring — but kept
    working for a dataset that ships videos without a map-keyframes CSV.

    A warning if you use this fps to derive `frame_idx`: the container reports
    the exact NTSC rational (30000/1001 = 29.970029...), while the dataset
    states the truncated 29.97, and `floor(pts_time * fps)` only reproduces
    BTC's own `frame_idx` with the **truncated** value — 100% vs 41% on the
    sample (contracts/ids.py). Prefer the CSV's fps whenever one exists; the
    container's is for playback, not for agreeing with the scoreboard.
    """
    try:
        import av
    except ImportError as err:
        raise ContractError(
            ErrorCode.STORE_UNAVAILABLE,
            "probe_video needs PyAV: pip install -e '.[offline]'. "
            "For the shipped dataset use probe_from_provided instead.",
        ) from err

    validate_video_id(video_id)
    with av.open(source_path) as container:
        streams = container.streams.video
        if not streams:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT, f"{source_path} has no video stream"
            )
        stream = streams[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT, f"{source_path} declares no frame rate"
            )
        fps = float(rate)  # Fraction -> float, keeping 30000/1001 as 29.97002997
        duration_s = float(container.duration or 0) / 1_000_000.0

        total_frames = stream.frames
        if not total_frames:
            total_frames = int(duration_s * fps) + 1
            logger.warning(
                "%s: container reports no frame count; estimated %d from duration*fps. "
                "The validator bound-checks frames against this number (docs/05 §1.1).",
                source_path,
                total_frames,
            )

        return VideoRow(
            video_id=video_id,
            source_path=source_path,
            fps=fps,
            duration_ms=round(duration_s * 1000),
            total_frames=total_frames,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
        )


def derive_video_id(source_path: str) -> str:
    """Derive a `video_id` from a filename: data/sample/videos/L21_V001.mp4 -> L21_V001.

    Owner: Quốc Đạt.

    Validated before it is returned — a filename with a space or a '#' would
    silently break every keyframe_id built from it (contracts/ids.py).
    """
    return validate_video_id(Path(source_path).stem)
