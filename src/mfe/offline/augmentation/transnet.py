"""TransNetV2 adapter and deterministic conversion of scores to shots.

The adapter keeps model loading behind a dotted factory and imports heavy
dependencies lazily.  Unit tests exercise score-to-shot logic without a model;
real runs fail closed if the checkpoint, decoder PTS, or prediction shape is
not reproducible.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mfe.contracts.ids import build_shot_id
from mfe.offline.augmentation.io import canonical_hash, sha256_file, sha256_path
from mfe.offline.augmentation.models import ShotProposal


@dataclass(frozen=True)
class DecodedTimeline:
    """Low-resolution model frames plus source PTS for every decoded frame."""

    frames: Any
    pts_times: tuple[float, ...]
    duration_seconds: float
    decoder_fingerprint: str

    @property
    def total_frames(self) -> int:
        return len(self.pts_times)


def _load_attr(dotted: str) -> Any:
    if ":" not in dotted:
        raise ValueError("detector factory must use 'module:attribute' syntax")
    module_name, attribute = dotted.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def detector_fingerprint(factory: str, checkpoint: str | Path) -> str:
    path = Path(checkpoint)
    return canonical_hash(
        {
            "kind": "transnet_v2",
            "factory": factory,
            "checkpoint_sha256": sha256_path(path),
        }
    )


def decode_for_transnet(video_path: str | Path) -> DecodedTimeline:
    """Decode a video once, preserving source-frame order and direct PTS.

    TransNetV2 consumes 48x27 RGB frames.  The resize changes only model input;
    candidate identity remains the zero-based sequential decoder count.  PTS is
    read from ``frame.pts * stream.time_base`` and is never reconstructed from
    FPS.
    """

    try:
        import av
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("decode requires the offline extras: av, opencv, numpy") from error

    source = Path(video_path)
    frames: list[Any] = []
    pts_times: list[float] = []
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise RuntimeError(f"{source}: no video stream")
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise RuntimeError(f"{source}: stream has no duration/time_base; cannot lock PTS")
        duration_seconds = float(stream.duration * stream.time_base)
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise RuntimeError(f"{source}: decoded frame {frame_idx} has no PTS")
            pts_time = float(frame.pts * stream.time_base)
            if pts_times and pts_time <= pts_times[-1]:
                raise RuntimeError(
                    f"{source}: non-increasing PTS at frame {frame_idx}: "
                    f"{pts_time} <= {pts_times[-1]}"
                )
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(cv2.resize(rgb, (48, 27), interpolation=cv2.INTER_AREA))
            pts_times.append(pts_time)

    if not frames:
        raise RuntimeError(f"{source}: decoder returned no frames")
    if duration_seconds <= pts_times[-1]:
        raise RuntimeError(
            f"{source}: stream duration {duration_seconds} must exceed final PTS {pts_times[-1]}"
        )
    fingerprint = canonical_hash(
        {
            "decoder": "pyav",
            "av_version": getattr(av, "__version__", "unknown"),
            "opencv_version": getattr(cv2, "__version__", "unknown"),
            "video_sha256": sha256_file(source),
            "total_frames": len(frames),
            "duration_seconds": duration_seconds,
        }
    )
    return DecodedTimeline(
        frames=np.asarray(frames, dtype=np.uint8),
        pts_times=tuple(pts_times),
        duration_seconds=duration_seconds,
        decoder_fingerprint=fingerprint,
    )


class TransNetV2Adapter:
    """Load a pinned TransNetV2 checkpoint through a configurable factory."""

    def __init__(self, factory: str, checkpoint: str | Path, expected_sha256: str) -> None:
        self.factory = factory
        self.checkpoint = Path(checkpoint)
        actual = sha256_path(self.checkpoint)
        if actual != expected_sha256:
            raise RuntimeError(
                f"TransNet checkpoint SHA mismatch: expected {expected_sha256}, got {actual}"
            )
        self.fingerprint = detector_fingerprint(factory, self.checkpoint)
        constructor = _load_attr(factory)
        if self.checkpoint.is_dir():
            # Official TensorFlow TransNetV2 consumes a SavedModel directory.
            # Prefer the documented keyword and retain positional fallback for
            # small compatible wrappers used by other environments.
            try:
                self.model = constructor(model_dir=str(self.checkpoint))
            except TypeError:
                self.model = constructor(str(self.checkpoint))
        else:
            try:
                self.model = constructor()
            except TypeError:
                self.model = constructor(str(self.checkpoint))
            load_weights = getattr(self.model, "load_weights", None)
            if callable(load_weights):
                load_weights(str(self.checkpoint))

    def predict(self, timeline: DecodedTimeline):
        """Return one probability per decoded source frame."""

        import numpy as np

        model = self.model
        if callable(getattr(model, "predict_frames", None)):
            raw = model.predict_frames(timeline.frames)
        elif callable(getattr(model, "predict_raw", None)):
            # The reference TransNetV2 wrapper calls this method for an array
            # of pre-decoded RGB frames and returns the single-frame stream
            # together with an optional all-frame stream.
            raw = model.predict_raw(timeline.frames)
        elif callable(getattr(model, "predict_video", None)):
            raw = model.predict_video(timeline.frames)
        elif callable(model):
            raw = model(np.expand_dims(timeline.frames, axis=0))
        else:  # pragma: no cover - defensive integration error
            raise RuntimeError("TransNet factory returned an object with no prediction method")

        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
        if len(scores) != timeline.total_frames:
            raise RuntimeError(
                f"TransNet returned {len(scores)} scores for {timeline.total_frames} frames"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError("TransNet returned NaN/Inf scores")
        # Some wrappers expose logits while the official model exposes
        # probabilities.  Conversion is allowed only when the range proves they
        # are logits; silently applying sigmoid twice would change every cut.
        if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores))
        return scores


def transition_cuts(scores, threshold: float) -> list[tuple[int, float]]:
    """Collapse consecutive positive frames to one deterministic cut.

    Each return value is ``(end_frame_exclusive, confidence)``.  The midpoint
    of a predicted transition run is used; cuts at frame 0 or after the final
    frame are discarded because they would create an empty shot.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    positives = [index for index, score in enumerate(scores) if float(score) >= threshold]
    if not positives:
        return []

    groups: list[list[int]] = [[positives[0]]]
    for index in positives[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    total = len(scores)
    cuts: list[tuple[int, float]] = []
    for group in groups:
        cut = (group[0] + group[-1]) // 2 + 1
        if 0 < cut < total:
            cuts.append((cut, max(float(scores[index]) for index in group)))
    return cuts


def proposals_from_scores(
    video_id: str,
    scores,
    pts_times: tuple[float, ...],
    duration_seconds: float,
    threshold: float,
    fingerprint: str,
) -> list[ShotProposal]:
    """Convert frame probabilities into a complete non-overlapping shot list."""

    if len(scores) != len(pts_times):
        raise ValueError("scores and pts_times must have the same length")
    if not pts_times:
        return []
    if duration_seconds <= pts_times[-1] or not math.isfinite(duration_seconds):
        raise ValueError("duration must be finite and later than the final frame PTS")

    cuts = transition_cuts(scores, threshold)
    boundaries = [0, *(cut for cut, _ in cuts), len(pts_times)]
    confidence_by_end = {cut: confidence for cut, confidence in cuts}
    shots: list[ShotProposal] = []
    for number, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        start_pts = pts_times[start]
        end_pts = pts_times[end] if end < len(pts_times) else duration_seconds
        adjacent_confidences = [
            confidence_by_end[boundary]
            for boundary in (start, end)
            if boundary in confidence_by_end
        ]
        shots.append(
            ShotProposal(
                video_id=video_id,
                shot_id=build_shot_id(video_id, number),
                start_frame=start,
                end_frame_exclusive=end,
                start_pts_time=start_pts,
                end_pts_time=end_pts,
                boundary_confidence=min(adjacent_confidences, default=1.0),
                detector_fingerprint=fingerprint,
            )
        )
    return shots


def verify_shot_partition(shots: list[ShotProposal], total_frames: int) -> list[str]:
    """Return partition problems; an empty list means complete coverage."""

    problems: list[str] = []
    if total_frames < 1:
        return ["total_frames must be positive"]
    if not shots:
        return ["no shot proposals"]
    if shots[0].start_frame != 0:
        problems.append("first shot does not start at frame 0")
    if shots[-1].end_frame_exclusive != total_frames:
        problems.append("last shot does not end at total_frames")
    for left, right in zip(shots, shots[1:], strict=False):
        if left.video_id != right.video_id:
            problems.append("shot list mixes video ids")
        if left.end_frame_exclusive != right.start_frame:
            problems.append(
                f"gap/overlap between {left.shot_id} and {right.shot_id}: "
                f"{left.end_frame_exclusive} != {right.start_frame}"
            )
        if left.end_pts_time != right.start_pts_time:
            problems.append(f"PTS gap/overlap between {left.shot_id} and {right.shot_id}")
    return problems
