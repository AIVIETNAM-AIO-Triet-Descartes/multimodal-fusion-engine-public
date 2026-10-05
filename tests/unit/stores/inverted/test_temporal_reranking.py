"""Unit tests for TemporalReranker (DANTE-inspired window smoothing & frame clustering)."""

from __future__ import annotations

import pytest

from mfe.stores.production.contracts import StorageHit
from mfe.stores.production.temporal import TemporalReranker


def test_smooth_hits_empty() -> None:
    assert TemporalReranker.smooth_hits([]) == []


def test_smooth_hits_single_hit_unchanged_order() -> None:
    hits = [StorageHit(video_id="v1", frame_id=100, score=1.0)]
    smoothed = TemporalReranker.smooth_hits(hits)
    assert len(smoothed) == 1
    assert smoothed[0].video_id == "v1"
    assert smoothed[0].frame_id == 100
    assert smoothed[0].score == 1.0


def test_smooth_hits_boosts_contiguous_cluster() -> None:
    # v1 has contiguous frames 10, 11, 12 (event segment)
    # v2 has an isolated frame 50
    hits = [
        StorageHit(video_id="v2", frame_id=50, score=0.85),
        StorageHit(video_id="v1", frame_id=10, score=0.80),
        StorageHit(video_id="v1", frame_id=11, score=0.80),
        StorageHit(video_id="v1", frame_id=12, score=0.80),
    ]
    # Before smoothing, v2 frame 50 is rank 1 (score 0.85 > 0.80)
    smoothed = TemporalReranker.smooth_hits(hits, window_size=5, sigma=2.0, decay_weight=0.5)

    # After smoothing, v1 frames gain energy and surpass the isolated v2 hit
    top_hit = smoothed[0]
    assert top_hit.video_id == "v1"
    assert top_hit.score > 0.85
    # Center frame (11) has both 10 and 12 in close range, so it should receive the highest boost
    v1_11 = next(h for h in smoothed if h.video_id == "v1" and h.frame_id == 11)
    v1_10 = next(h for h in smoothed if h.video_id == "v1" and h.frame_id == 10)
    assert v1_11.score > v1_10.score


def test_smooth_hits_kernels() -> None:
    hits = [
        StorageHit(video_id="v1", frame_id=10, score=1.0),
        StorageHit(video_id="v1", frame_id=12, score=1.0),
    ]
    gaussian_res = TemporalReranker.smooth_hits(hits, kernel="gaussian")
    triangular_res = TemporalReranker.smooth_hits(hits, kernel="triangular")
    box_res = TemporalReranker.smooth_hits(hits, kernel="box")

    assert len(gaussian_res) == 2
    assert len(triangular_res) == 2
    assert len(box_res) == 2


def test_smooth_hits_invalid_kernel() -> None:
    hits = [StorageHit(video_id="v1", frame_id=10, score=1.0)]
    with pytest.raises(ValueError, match="unknown kernel"):
        TemporalReranker.smooth_hits(hits, kernel="invalid_kernel")


def test_smooth_hits_top_k_cutoff() -> None:
    hits = [
        StorageHit(video_id="v1", frame_id=i * 10, score=float(i))
        for i in range(10)
    ]
    smoothed = TemporalReranker.smooth_hits(hits, top_k=3)
    assert len(smoothed) == 3


def test_pool_frame_clusters_deduplication() -> None:
    hits = [
        StorageHit(video_id="v1", frame_id=100, score=0.95),
        StorageHit(video_id="v1", frame_id=102, score=0.90),  # Gap=2 -> dropped
        StorageHit(video_id="v1", frame_id=105, score=0.88),  # Gap=5 -> dropped
        StorageHit(video_id="v1", frame_id=200, score=0.80),  # Gap=100 -> kept
        StorageHit(video_id="v2", frame_id=101, score=0.75),  # Different video -> kept
    ]
    pooled = TemporalReranker.pool_frame_clusters(hits, max_frame_gap=10)
    assert len(pooled) == 3
    assert [h.frame_id for h in pooled if h.video_id == "v1"] == [100, 200]
    assert [h.video_id for h in pooled] == ["v1", "v1", "v2"]


def test_smooth_hits_cross_video_isolation() -> None:
    """Ensure energy from video A does not leak into video B even if frame_ids match exactly."""
    hits = [
        StorageHit(video_id="v1", frame_id=10, score=1.0),
        StorageHit(video_id="v2", frame_id=11, score=0.5),
    ]
    smoothed = TemporalReranker.smooth_hits(hits, window_size=5, decay_weight=0.5)
    v1_hit = next(h for h in smoothed if h.video_id == "v1")
    v2_hit = next(h for h in smoothed if h.video_id == "v2")
    # Neither should receive a boost since they belong to different videos
    assert v1_hit.score == 1.0
    assert v2_hit.score == 0.5


def test_smooth_hits_kernel_weights_differ() -> None:
    """Verify that different kernels produce distinct score boosts for delta > 0."""
    hits = [
        StorageHit(video_id="v1", frame_id=10, score=1.0),
        StorageHit(video_id="v1", frame_id=12, score=1.0),
    ]
    g_res = TemporalReranker.smooth_hits(hits, kernel="gaussian", sigma=2.0, decay_weight=0.5)
    t_res = TemporalReranker.smooth_hits(hits, kernel="triangular", sigma=2.0, decay_weight=0.5)
    b_res = TemporalReranker.smooth_hits(hits, kernel="box", sigma=2.0, decay_weight=0.5)

    # All three boost the score > 1.0, but by different amounts
    assert g_res[0].score > 1.0
    assert t_res[0].score > 1.0
    assert b_res[0].score > 1.0
    assert g_res[0].score != t_res[0].score or t_res[0].score != b_res[0].score

