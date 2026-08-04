"""Unit tests for dedup module (docs/04 §3.2)."""

from mfe.contracts.tools import Candidate
from mfe.online.dedup import dedup, group_by_shot, select_representatives


def _cand(index: int, score: float, ordinal: int) -> Candidate:
    """One candidate of video L21_V001, with a contract-valid evidence_id."""
    return Candidate(
        keyframe_id=f"L21_V001#{index:03d}",
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

    groups = group_by_shot(cands, temporal_window=5)
    # L21_V001#001 and L21_V001#003 are in group 1; #050 is in group 2
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
    assert reps[0].keyframe_id == "L21_V001#001"
    assert reps[1].keyframe_id == "L21_V001#050"


def test_dedup_pipeline():
    cands = [
        _cand(1, 0.9, 1),
        _cand(2, 0.85, 2),
    ]
    res = dedup(cands, temporal_window=5)
    assert len(res) == 1
    assert res[0].keyframe_id == "L21_V001#001"
