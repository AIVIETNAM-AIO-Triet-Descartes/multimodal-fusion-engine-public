"""probe_from_provided — the bounds the validator later checks against."""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.offline.probe import probe_from_provided
from mfe.offline.provided import MapRow


def _rows(*pairs: tuple[float, int], fps: float = 25.0) -> list[MapRow]:
    return [
        MapRow(n=index, pts_time=pts, fps=fps, frame_idx=frame)
        for index, (pts, frame) in enumerate(pairs, 1)
    ]


def _probe(rows: list[MapRow], media_info: dict) -> object:
    return probe_from_provided(
        video_id="L25_V003",
        source_path="data/sample_2026/videos/L25_V003.mp4",
        map_rows=rows,
        media_info=media_info,
        frame_size=(1280, 720),
    )


def test_duration_widens_to_cover_the_last_keyframe():
    """BTC stores `length` as whole seconds, and it can round *down*.

    L25_V003 of the 2026 sample declares length 1162 while its last keyframe
    sits at pts_time 1162.12. Trusting the declared length made the keyframe's
    own timestamp_ms out of bounds, and ingest refused the whole dataset over a
    keyframe BTC itself shipped.
    """
    row = _probe(_rows((0.0, 0), (1162.12, 29053)), {"length": 1162})

    assert row.duration_ms == 1162120
    assert row.total_frames >= 29054


def test_declared_length_is_kept_when_it_already_covers():
    """The common case: the video really is longer than its last keyframe."""
    row = _probe(_rows((0.0, 0), (174.8, 4370)), {"length": 176})

    assert row.duration_ms == 176000


def test_total_frames_widens_to_cover_the_last_frame_idx():
    row = _probe(_rows((0.0, 0), (100.0, 2600)), {"length": 100})

    assert row.total_frames == 2601


def test_missing_length_falls_back_to_the_last_keyframe():
    row = _probe(_rows((0.0, 0), (42.5, 1062)), {})

    assert row.duration_ms == 42500


def test_variable_frame_rate_is_refused():
    """One fps per video, or every frame_idx we submit is derived differently."""
    rows = [
        MapRow(n=1, pts_time=0.0, fps=25.0, frame_idx=0),
        MapRow(n=2, pts_time=10.0, fps=29.97, frame_idx=299),
    ]
    with pytest.raises(ContractError):
        _probe(rows, {"length": 10})


def test_no_rows_is_refused():
    with pytest.raises(ContractError):
        _probe([], {"length": 10})
