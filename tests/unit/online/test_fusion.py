"""Unit tests for fusion module (docs/04 §3.1)."""

from mfe.contracts.tools import Candidate
from mfe.online.fusion import normalize_scores, weighted_fusion


def test_normalize_scores_rrf():
    cands = [
        Candidate(
            keyframe_id="kf1",
            video_id="v1",
            score=10.0,
            evidence_id="search_text:t:1",
        ),
        Candidate(
            keyframe_id="kf2",
            video_id="v1",
            score=5.0,
            evidence_id="search_text:t:2",
        ),
    ]
    norm = normalize_scores(cands, method="rrf", rrf_k=60)
    assert len(norm) == 2
    assert norm[0].score > norm[1].score
    assert norm[0].score == 1.0 / (60 + 1)
    assert norm[1].score == 1.0 / (60 + 2)


def test_weighted_fusion():
    list_a = [
        Candidate(
            keyframe_id="kf1",
            video_id="v1",
            score=0.9,
            evidence_id="search_text:a:1",
            matched_terms=["cat"],
        ),
        Candidate(
            keyframe_id="kf2",
            video_id="v1",
            score=0.5,
            evidence_id="search_text:a:2",
            matched_terms=["dog"],
        ),
    ]
    list_b = [
        Candidate(
            keyframe_id="kf2",
            video_id="v1",
            score=0.8,
            evidence_id="vector_search:b:2",
            matched_terms=["dog"],
        ),
        Candidate(
            keyframe_id="kf3",
            video_id="v2",
            score=0.4,
            evidence_id="vector_search:b:3",
            matched_terms=["bird"],
        ),
    ]

    result_lists = {"tool_a": list_a, "tool_b": list_b}
    weights = {"tool_a": 1.0, "tool_b": 1.0}

    fused = weighted_fusion(result_lists, weights, run_id="test_fusion")
    assert len(fused) == 3
    # Check that candidate kf2 combined terms and merged evidence
    kf2_cand = next(c for c in fused if c.keyframe_id == "kf2")
    assert "dog" in kf2_cand.matched_terms
    assert kf2_cand.evidence_id.startswith("late_fusion:test_fusion:")
