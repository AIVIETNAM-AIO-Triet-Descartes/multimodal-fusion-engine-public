"""Unit tests for dedup module (docs/04 §3.2).

The window is in *frames* now, not keyframe ordinals — see `mfe.online.dedup`.
These tests use a 100-frame spacing between candidates and a 500-frame window,
so "5 apart" reads the same as it used to while the unit underneath is real.
"""

from mfe.contracts.tools import Candidate
from mfe.online.dedup import dedup, group_by_shot, select_representatives

#: Frames between two adjacent test candidates. Close to the 128-frame median
#: measured on the shipped keyframes, and round enough to read.
SPACING = 100

#: Window covering 5 spacings, the old default expressed in frames.
WINDOW = 5 * SPACING


def _cand(step: int, score: float, ordinal: int) -> Candidate:
    """One candidate of L21_V001, `step` spacings into the video."""
    return Candidate(
        keyframe_id=f"L21_V001#{step * SPACING:06d}",
        video_id="L21_V001",
        score=score,
        evidence_id=f"search_text:t:{ordinal}",
    )


def test_group_by_shot_temporal():
    cands = [
        _cand(1, 0.9, 1),
        _cand(3, 0.8, 2),
        _cand(50, 0.7, 3),
    ]

    groups = group_by_shot(cands, temporal_window_frames=WINDOW)
    # steps 1 and 3 are within the window; step 50 is far away
    assert len(groups) == 2


def test_select_representatives():
    grp1 = [
        _cand(1, 0.9, 1),
        _cand(3, 0.8, 2),
    ]
    grp2 = [
        _cand(50, 0.7, 3),
    ]

    reps = select_representatives([grp1, grp2])
    assert len(reps) == 2
    assert reps[0].keyframe_id == "L21_V001#000100"
    assert reps[1].keyframe_id == "L21_V001#005000"


def test_dedup_pipeline():
    cands = [
        _cand(1, 0.9, 1),
        _cand(2, 0.85, 2),
    ]
    res = dedup(cands, temporal_window_frames=WINDOW)
    assert len(res) == 1
    assert res[0].keyframe_id == "L21_V001#000100"


def test_a_group_spans_at_most_the_window():
    """Groups must not chain: steps 1, 6, 11, ... are not all one shot.

    Each of these is within the window of its predecessor but the run covers 46
    spacings. Measuring the gap from the previous keyframe instead of the
    group's first one collapsed an entire video into one group on the 2026
    sample — 216 candidates spanning keyframes 1 to 262 — leaving a single
    representative and burying every correct answer behind it.
    """
    cands = [_cand(step, 0.9, ordinal) for ordinal, step in enumerate(range(1, 47, 5))]

    groups = group_by_shot(cands, temporal_window_frames=WINDOW)

    assert len(groups) > 1
    for group in groups:
        frames = [int(c.keyframe_id.split("#")[1]) for c in group]
        assert max(frames) - min(frames) <= WINDOW


def test_dense_run_is_split_into_windows():
    """A contiguous run of 12 keyframes becomes windows of at most 6."""
    cands = [_cand(step, 1.0 - step / 100, step) for step in range(1, 13)]

    groups = group_by_shot(cands, temporal_window_frames=WINDOW)

    assert [len(group) for group in groups] == [6, 6]


def test_the_window_unit_is_frames_not_keyframes():
    """The unit change itself, locked.

    A window of 5 used to mean "5 keyframes" and grouped neighbours. Read as
    frames it is 0.17s, so nothing groups — the failure mode this rename exists
    to prevent, and one that produces no error of its own.
    """
    cands = [_cand(1, 0.9, 1), _cand(2, 0.8, 2)]

    assert len(group_by_shot(cands, temporal_window_frames=5)) == 2
    assert len(group_by_shot(cands, temporal_window_frames=WINDOW)) == 1
