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
    """The video from docs/05 §1.1(c). Fractional fps on purpose — 29.97 is
    where rounding bugs live, and a 30.0 fixture would hide them."""
    return VideoRow(
        video_id="L01_V001",
        source_path="data/raw/L01_V001.mp4",
        fps=29.97,
        duration_ms=600000,
        total_frames=17982,
        width=1280,
        height=720,
    )


@pytest.fixture
def sample_video_integer_fps() -> VideoRow:
    """The second video from docs/05 §1.1(c). Different fps *and* resolution —
    docs/context/buoi-1.md §7 warns BTC's videos may vary in both, so nothing
    may assume one value."""
    return VideoRow(
        video_id="L01_V002",
        source_path="data/raw/L01_V002.mp4",
        fps=25.00,
        duration_ms=420000,
        total_frames=10500,
        width=1920,
        height=1080,
    )


@pytest.fixture
def sample_keyframe() -> KeyframeRow:
    """The keyframe from docs/05 §1.2(c)."""
    return KeyframeRow(
        keyframe_id="L01_V001#000250",
        video_id="L01_V001",
        frame_number=250,
        timestamp_ms=8342,
        shot_id="L01_V001#s02",
    )
