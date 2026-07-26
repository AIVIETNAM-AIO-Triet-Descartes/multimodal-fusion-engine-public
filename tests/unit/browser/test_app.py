"""Unit tests for Browser MVP application helpers (docs/04 §4)."""

from mfe.browser.app import validate_submission_stub
from mfe.contracts.tools import Candidate


def test_validate_submission_stub_success():
    cand = Candidate(
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        score=0.95,
        evidence_id="search_text:run01:0001",
    )
    res = validate_submission_stub(cand)
    assert res.ok
    assert res.items[0]["status"] == "VALID"
    assert res.items[0]["keyframe_id"] == "L21_V001#001"
    assert res.items[0]["evidence_id"] == "search_text:run01:0001"


def test_validate_submission_stub_missing_evidence():
    cand = Candidate(
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        score=0.95,
        evidence_id="",
    )
    res = validate_submission_stub(cand)
    assert not res.ok
    assert "evidence_id" in res.error.message
