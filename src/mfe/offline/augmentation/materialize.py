"""Extract delta images and build reproducible augmented ordinals."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mfe.offline.augmentation.io import sha256_file
from mfe.offline.augmentation.models import BtcFrame, CandidateRow


@dataclass(frozen=True)
class MaterializedImage:
    candidate: CandidateRow
    path: Path
    image_sha256: str
    width: int
    height: int
    byte_size: int


def extract_delta_images(
    video_path: str | Path,
    candidates: list[CandidateRow],
    output_root: str | Path,
    *,
    jpeg_quality: int = 95,
    pts_tolerance: float = 1e-6,
) -> list[MaterializedImage]:
    """Decode one video once and save only requested source frame indices."""

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if not candidates:
        return []
    video_ids = {item.video_id for item in candidates}
    if len(video_ids) != 1:
        raise ValueError("one extraction pass may contain candidates from one video only")
    try:
        import av
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("delta extraction requires PyAV") from error

    requested = {item.frame_idx: item for item in candidates}
    if len(requested) != len(candidates):
        raise ValueError("duplicate candidate frame_idx in one video")
    target_dir = Path(output_root) / candidates[0].video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    found: dict[int, MaterializedImage] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise RuntimeError(f"{video_path}: video stream has no time_base")
        for frame_idx, frame in enumerate(container.decode(stream)):
            candidate = requested.get(frame_idx)
            if candidate is None:
                continue
            if frame.pts is None:
                raise RuntimeError(f"{video_path}: selected frame {frame_idx} has no PTS")
            actual_pts = float(frame.pts * stream.time_base)
            if abs(actual_pts - candidate.decoder_pts_time) > pts_tolerance:
                raise RuntimeError(
                    f"{candidate.keyframe_id}: decoder PTS drift "
                    f"{actual_pts} != {candidate.decoder_pts_time}"
                )
            target = target_dir / f"frame_{frame_idx:06d}.jpg"
            temporary = target.with_suffix(".tmp.jpg")
            image = frame.to_image()
            image.save(temporary, format="JPEG", quality=jpeg_quality)
            temporary.replace(target)
            found[frame_idx] = MaterializedImage(
                candidate=candidate,
                path=target,
                image_sha256=sha256_file(target),
                width=image.width,
                height=image.height,
                byte_size=target.stat().st_size,
            )
            if len(found) == len(requested):
                break
    missing = sorted(set(requested) - set(found))
    if missing:
        raise RuntimeError(f"{video_path}: decoder ended before selected frames {missing[:10]}")
    return [found[index] for index in sorted(found)]


def assign_augmented_ordinals(
    btc_frames: Iterable[BtcFrame], candidates: Iterable[CandidateRow]
) -> dict[str, int]:
    """Create zero-based, time-ordered ordinals without changing identity."""

    rows: list[tuple[float, int, int, str]] = []
    for item in btc_frames:
        rows.append((item.pts_time, item.frame_idx, 0, item.keyframe_id))
    for item in candidates:
        rows.append((item.pts_time, item.frame_idx, 1, item.keyframe_id))
    ids = [item[3] for item in sorted(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError("BTC and delta contain a duplicate canonical keyframe_id")
    return {keyframe_id: ordinal for ordinal, keyframe_id in enumerate(ids)}


def build_stage_view(
    images: Iterable[MaterializedImage],
    target_dir: str | Path,
) -> dict[str, Path]:
    """Create ordinal filenames for existing offline stages without changing IDs.

    The durable delta assets keep ``frame_<frame_idx>.jpg`` names.  This view is
    disposable and exists only because legacy stage readers open images by
    ordinal ``n``.  It returns ``keyframe_id -> ordinal path``.
    """

    root = Path(target_dir)
    root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for ordinal, item in enumerate(sorted(images, key=lambda row: row.candidate.pts_time), 1):
        target = root / f"{ordinal:03d}.jpg"
        try:
            target.hardlink_to(item.path)
        except OSError:
            shutil.copy2(item.path, target)
        mapping[item.candidate.keyframe_id] = target
    return mapping
