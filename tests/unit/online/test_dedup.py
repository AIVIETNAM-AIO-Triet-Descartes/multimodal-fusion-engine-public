"""Unit tests for dedup module (docs/04 §3.2)."""

from mfe.contracts.tools import Candidate
from mfe.online.dedup import dedup, group_by_shot, select_representatives


def test_group_by_shot_temporal():
    cands = [
        Candidate(keyframe_id="L21_V001#001", video_id="L21_V001", score=0.9, evidence_id="ev1"),
        Candidate(keyframe_id="L21_V001#003", video_id="L21_V001", score=0.8, evidence_id="ev2"),
        Candidate(keyframe_id="L21_V001#050", video_id="L21_V001", score=0.7, evidence_id="ev3"),
    ]

    groups = group_by_shot(cands, temporal_window=5)
    # L21_V001#001 and L21_V001#003 are in group 1; #050 is in group 2
    assert len(groups) == 2


def test_select_representatives():
    grp1 = [
        Candidate(keyframe_id="L21_V001#001", video_id="L21_V001", score=0.9, evidence_id="ev1"),
        Candidate(keyframe_id="L21_V001#003", video_id="L21_V001", score=0.8, evidence_id="ev2"),
    ]
    grp2 = [
        Candidate(keyframe_id="L21_V001#050", video_id="L21_V001", score=0.7, evidence_id="ev3"),
    ]

    reps = select_representatives([grp1, grp2])
    assert len(reps) == 2
    assert reps[0].keyframe_id == "L21_V001#001"
    assert reps[1].keyframe_id == "L21_V001#050"


def test_dedup_pipeline():
    cands = [
        Candidate(keyframe_id="L21_V001#001", video_id="L21_V001", score=0.9, evidence_id="ev1"),
        Candidate(keyframe_id="L21_V001#002", video_id="L21_V001", score=0.85, evidence_id="ev2"),
    ]
    res = dedup(cands, temporal_window=5)
    assert len(res) == 1
    assert res[0].keyframe_id == "L21_V001#001"
