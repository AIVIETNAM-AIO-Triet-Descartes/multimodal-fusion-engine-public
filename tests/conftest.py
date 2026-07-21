"""Shared test fixtures.

Test layout mirrors the package: tests/unit/<workstream>/ belongs to that
workstream's owner. Cross-workstream behaviour goes in tests/integration/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfe.contracts.records import KeyframeRow, VideoRow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_video() -> VideoRow:
    """A 29.97fps video. NTSC drop-frame is where `round(pts_time*fps)` breaks
    (docs/adr/0002-id-format.md), so the fixture uses it deliberately."""
    return VideoRow(
        video_id="L21_V001",
        source_path="data/raw/L21_V001.mp4",
        fps=29.97,
        duration_ms=600000,
        total_frames=17982,
        width=1280,
        height=720,
    )


@pytest.fixture
def sample_video_integer_fps() -> VideoRow:
    """A second video with different fps *and* resolution — the real dataset
    mixes 25.0, 29.97 and 30.0 (docs/00 §3), so nothing may assume one value."""
    return VideoRow(
        video_id="L21_V002",
        source_path="data/raw/L21_V002.mp4",
        fps=25.00,
        duration_ms=420000,
        total_frames=10500,
        width=1920,
        height=1080,
    )


@pytest.fixture
def sample_keyframe() -> KeyframeRow:
    """A keyframe in the real-data ID model (docs/00 §3): keyframe_index and
    frame_idx are different numbers, both from the map-keyframes CSV."""
    return KeyframeRow(
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        keyframe_index=1,
        frame_idx=90,
        pts_time=3.0,
        timestamp_ms=3000,
        shot_id="L21_V001#s02",
    )
