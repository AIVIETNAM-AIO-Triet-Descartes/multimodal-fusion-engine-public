"""Temporal window smoothing and event aggregation for video frame retrieval hits.

Inspired by DANTE and top-tier AIC/TRECVID video retrieval systems:
Single-frame retrieval often suffers from noise and transient visual false-positives.
Temporal window smoothing pools energy across consecutive keyframes within the same video,
boosting true continuous event segments and stabilizing ranking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from .contracts import StorageHit


class TemporalReranker:
    """Temporal window smoothing and cluster pooling for StorageHit lists."""

    @staticmethod
    def _kernel_weight(delta_frame: int, sigma: float, kernel: str) -> float:
        if delta_frame == 0:
            return 1.0
        if kernel == "gaussian":
            return math.exp(-0.5 * (delta_frame / sigma) ** 2)
        if kernel == "triangular":
            # Linear decay to 0 at 3*sigma
            max_dist = 3.0 * sigma
            if abs(delta_frame) >= max_dist:
                return 0.0
            return 1.0 - (abs(delta_frame) / max_dist)
        if kernel == "box":
            return 1.0 if abs(delta_frame) <= sigma else 0.0
        msg = f"unknown kernel: {kernel!r} (expected 'gaussian', 'triangular', or 'box')"
        raise ValueError(msg)

    @classmethod
    def smooth_hits(
        cls,
        hits: Sequence[StorageHit],
        *,
        window_size: int = 10,
        sigma: float = 3.0,
        kernel: str = "gaussian",
        decay_weight: float = 0.5,
        top_k: int | None = None,
    ) -> list[StorageHit]:
        """Apply temporal window smoothing to a list of StorageHits.

        For each hit (video_id, frame_id), energy is shared with other hits from the
        same video located within `window_size` frames.

        Parameters
        ----------
        hits:
            Initial retrieval hits from dense vector or sparse text search.
        window_size:
            Maximum frame distance |f_i - f_j| to consider for smoothing.
        sigma:
            Standard deviation for kernel decay weighting.
        kernel:
            'gaussian', 'triangular', or 'box'.
        decay_weight:
            Multiplier applied to the accumulated neighborhood energy (0.0 to 1.0).
        top_k:
            Optional limit for the returned list (defaults to len(hits)).
        """
        if kernel not in ("gaussian", "triangular", "box"):
            msg = f"unknown kernel: {kernel!r} (expected 'gaussian', 'triangular', or 'box')"
            raise ValueError(msg)

        if not hits:
            return []

        # Group hits by video_id
        video_groups: dict[str, list[StorageHit]] = defaultdict(list)
        for h in hits:
            video_groups[h.video_id].append(h)

        smoothed_hits: list[StorageHit] = []

        for _video_id, v_hits in video_groups.items():
            # Build quick frame_id -> score map for this video
            frame_map = {h.frame_id: h for h in v_hits}

            for h in v_hits:
                f_center = h.frame_id
                accumulated_boost = 0.0

                for other_f, other_h in frame_map.items():
                    if other_f == f_center:
                        continue
                    delta = abs(other_f - f_center)
                    if delta <= window_size:
                        w = cls._kernel_weight(delta, sigma=sigma, kernel=kernel)
                        accumulated_boost += other_h.score * w

                new_score = h.score + (decay_weight * accumulated_boost)
                smoothed_hits.append(
                    StorageHit(
                        video_id=h.video_id,
                        frame_id=h.frame_id,
                        score=float(new_score),
                        source=h.source,
                        text=h.text,
                    )
                )

        # Sort descending by new score
        smoothed_hits.sort(key=lambda x: x.score, reverse=True)
        if top_k is not None:
            return smoothed_hits[:top_k]
        return smoothed_hits

    @classmethod
    def pool_frame_clusters(
        cls,
        hits: Sequence[StorageHit],
        *,
        max_frame_gap: int = 15,
        top_k: int | None = None,
    ) -> list[StorageHit]:
        """Deduplicate tightly-clustered frames by retaining highest representative.

        If multiple hits occur within `max_frame_gap` of each other in the same video,
        only the peak frame is kept to ensure diversity of retrieved scenes.
        """
        if not hits:
            return []

        # Process in score order (highest first)
        accepted: list[StorageHit] = []
        video_accepted_frames: dict[str, list[int]] = defaultdict(list)

        for h in hits:
            existing_frames = video_accepted_frames[h.video_id]
            is_close = any(abs(h.frame_id - ef) <= max_frame_gap for ef in existing_frames)
            if not is_close:
                accepted.append(h)
                video_accepted_frames[h.video_id].append(h.frame_id)
                if top_k is not None and len(accepted) >= top_k:
                    break

        return accepted
