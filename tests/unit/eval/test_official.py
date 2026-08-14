"""Candidate -> official submission, and the TRAKE assembler."""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.contracts.tools import Candidate
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY
from mfe.eval.official import (
    KISQuery,
    QAQuery,
    TRAKEQuery,
    load_official_queryset,
    to_kis_submissions,
    to_qa_submissions,
    to_trake_submissions,
)


class _FakeKeyframe:
    def __init__(self, frame_idx: int) -> None:
        self.frame_idx = frame_idx


class _FakeMap:
    """Stands in for the Relational Map: keyframe_id -> frame_idx.

    A fake is right here. The point of these tests is that the translation reads
    `frame_idx` off the map rather than deriving it, so the map's contents must
    be something the test controls and the translation cannot recompute.
    """

    def __init__(self, frames: dict[str, int]) -> None:
        self._frames = frames

    def get_keyframe(self, keyframe_id: str) -> _FakeKeyframe:
        if keyframe_id not in self._frames:
            raise KeyError(keyframe_id)
        return _FakeKeyframe(self._frames[keyframe_id])


def _cand(keyframe_id: str, score: float, ordinal: int) -> Candidate:
    return Candidate(
        keyframe_id=keyframe_id,
        video_id=keyframe_id.split("#")[0],
        score=score,
        evidence_id=f"vector_search:t:{ordinal}",
    )


def test_frame_id_is_read_from_the_map_not_derived():
    """`frame_idx` 375 must survive even though keyframe_index is 8.

    The two numbers are unrelated — 8 is the ordinal, 375 the real source frame.
    Submitting the ordinal is a silent miss, so this pins which one is used.
    """
    relational = _FakeMap({"L23_V001#008": 375})

    submissions = to_kis_submissions([_cand("L23_V001#008", 0.9, 0)], relational)

    assert submissions[0].frame_id == 375
    assert submissions[0].video_id == "L23_V001"


def test_answer_is_attached_to_every_qa_submission():
    relational = _FakeMap({"L23_V001#001": 0, "L23_V001#002": 87})
    candidates = [_cand("L23_V001#001", 0.9, 0), _cand("L23_V001#002", 0.8, 1)]

    submissions = to_qa_submissions(candidates, relational, "linh vật")

    assert [s.answer for s in submissions] == ["linh vật", "linh vật"]
    assert [s.frame_id for s in submissions] == [0, 87]


def test_submission_list_is_capped():
    relational = _FakeMap({f"L23_V001#{i:03d}": i * 10 for i in range(1, 130)})
    candidates = [_cand(f"L23_V001#{i:03d}", 1.0 - i / 1000, i) for i in range(1, 130)]

    submissions = to_kis_submissions(candidates, relational, limit=MAX_ANSWERS_PER_QUERY)

    assert len(submissions) == MAX_ANSWERS_PER_QUERY


def test_unknown_keyframe_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        to_kis_submissions([_cand("L99_V999#001", 0.9, 0)], _FakeMap({}))


# --- TRAKE assembly --------------------------------------------------------


def test_trake_frames_are_strictly_increasing():
    relational = _FakeMap({"V#001": 10, "V#002": 20, "V#003": 30})
    per_event = [
        [_cand("V#001", 0.9, 0)],
        [_cand("V#002", 0.8, 1)],
        [_cand("V#003", 0.7, 2)],
    ]

    submissions = to_trake_submissions(per_event, relational)

    assert submissions[0].frame_ids == [10, 20, 30]


def test_a_high_scoring_late_frame_does_not_block_later_events():
    """The failure that made greedy assembly useless on the real sample.

    Event 1's best-scoring candidate sits at frame 900, after everything event 2
    can offer. Greedy takes it and the video is discarded; the DP takes the
    lower-scoring frame 10 and keeps a valid answer with two more events.
    """
    relational = _FakeMap({"V#009": 900, "V#001": 10, "V#002": 20})
    per_event = [
        [_cand("V#009", 0.99, 0), _cand("V#001", 0.10, 1)],
        [_cand("V#002", 0.80, 2)],
    ]

    submissions = to_trake_submissions(per_event, relational)

    assert len(submissions) == 1
    assert submissions[0].frame_ids == [10, 20]


def test_video_without_an_increasing_assignment_is_dropped():
    """Dropped, not padded — a padded frame is a wrong guess wearing a costume."""
    relational = _FakeMap({"V#009": 900, "V#002": 20})
    per_event = [[_cand("V#009", 0.9, 0)], [_cand("V#002", 0.8, 1)]]

    assert to_trake_submissions(per_event, relational) == []


def test_video_missing_an_event_entirely_is_dropped():
    relational = _FakeMap({"A#001": 10, "B#001": 10, "B#002": 20})
    per_event = [
        [_cand("A#001", 0.9, 0), _cand("B#001", 0.5, 1)],
        [_cand("B#002", 0.8, 2)],  # nothing for video A
    ]

    submissions = to_trake_submissions(per_event, relational)

    assert [s.video_id for s in submissions] == ["B"]


def test_videos_are_ranked_by_mean_score():
    relational = _FakeMap({"A#001": 10, "A#002": 20, "B#001": 10, "B#002": 20})
    per_event = [
        [_cand("B#001", 0.9, 0), _cand("A#001", 0.4, 1)],
        [_cand("B#002", 0.9, 2), _cand("A#002", 0.4, 3)],
    ]

    submissions = to_trake_submissions(per_event, relational)

    assert [s.video_id for s in submissions] == ["B", "A"]


# --- query set -------------------------------------------------------------


def test_baseline_queryset_parses_and_covers_all_three_types():
    queries = load_official_queryset("eval/queryset_btc_2026.jsonl")

    assert {type(query) for query in queries} == {KISQuery, QAQuery, TRAKEQuery}
    for query in queries:
        # Every ground truth must be constructible — a TRAKE query whose event
        # count disagrees with its interval count would only fail at run time.
        query.ground_truth()


def test_trake_query_with_mismatched_event_counts_is_refused():
    query = TRAKEQuery(
        query_id="bad",
        type="TRAKE",
        text="một chuỗi hai sự kiện",
        event_texts=["a", "b"],
        gt_video_id="L23_V001",
        gt_events=[(0, 5)],
    )
    with pytest.raises(ContractError):
        query.ground_truth()


def test_trake_without_a_decomposition_searches_the_whole_query() -> None:
    """`event_texts` is optional — splitting the sequence is the agent's job.

    Until that exists, every event searches the one sentence BTC would give us.
    That finds the right video and little else, which is the honest floor.
    """
    query = TRAKEQuery(
        query_id="q",
        type="TRAKE",
        text="vận động viên xuất phát rồi về đích",
        gt_video_id="L23_V001",
        gt_events=[(0, 5), (100, 110)],
    )

    assert query.searches() == [query.text, query.text]
    assert len(query.ground_truth().events) == 2


def test_strict_window_is_derived_from_the_anchor() -> None:
    """One annotation, two windows — they cannot drift apart by hand."""
    query = KISQuery(
        query_id="q",
        type="KIS",
        text="…",
        gt_video_id="L23_V001",
        gt_frame_range=(301, 625),
        gt_anchor_frame=375,
    )

    wide = query.ground_truth()
    strict = query.ground_truth_strict()

    assert (wide.start_frame_id, wide.end_frame_id) == (301, 625)
    assert (strict.start_frame_id, strict.end_frame_id) == (370, 380)


def test_no_anchor_means_no_strict_window() -> None:
    """Wide-only is legal: the query is still scored, just on one ruler."""
    query = KISQuery(
        query_id="q",
        type="KIS",
        text="…",
        gt_video_id="L23_V001",
        gt_frame_range=(301, 625),
    )

    assert query.ground_truth() is not None
    assert query.ground_truth_strict() is None


def test_strict_window_is_clamped_at_zero() -> None:
    """An anchor at frame 0 must not produce a negative start."""
    query = KISQuery(
        query_id="q",
        type="KIS",
        text="…",
        gt_video_id="L23_V001",
        gt_frame_range=(0, 86),
        gt_anchor_frame=0,
    )

    strict = query.ground_truth_strict()

    assert strict.start_frame_id == 0
    assert strict.end_frame_id == 5


def test_anchor_outside_its_own_scene_is_refused() -> None:
    """The two numbers would then describe different moments.

    Scoring either one would silently believe the wrong annotation, and the two
    columns of the report would disagree for a reason nobody could see.
    """
    query = KISQuery(
        query_id="q",
        type="KIS",
        text="…",
        gt_video_id="L23_V001",
        gt_frame_range=(301, 625),
        gt_anchor_frame=900,
    )
    with pytest.raises(ContractError):
        query.ground_truth_strict()


def test_trake_needs_one_anchor_per_event() -> None:
    query = TRAKEQuery(
        query_id="q",
        type="TRAKE",
        text="…",
        gt_video_id="L23_V001",
        gt_events=[(0, 86), (301, 625)],
        gt_anchor_frames=[0],
    )
    with pytest.raises(ContractError):
        query.ground_truth_strict()


def test_trake_strict_windows_track_their_events() -> None:
    query = TRAKEQuery(
        query_id="q",
        type="TRAKE",
        text="…",
        gt_video_id="L23_V001",
        gt_events=[(0, 86), (301, 625)],
        gt_anchor_frames=[40, 375],
    )

    strict = query.ground_truth_strict()

    assert [(e.start_frame, e.end_frame) for e in strict.events] == [(35, 45), (370, 380)]


def test_inverted_frame_range_is_refused() -> None:
    """A swapped range would otherwise score every answer as wrong."""
    query = KISQuery(
        query_id="q",
        type="KIS",
        text="…",
        gt_video_id="L23_V001",
        gt_frame_range=(510, 500),
    )
    with pytest.raises(ContractError):
        query.ground_truth()


def test_qa_falls_back_to_the_ground_truth_answer() -> None:
    """No VQA reader yet, so an unset `system_answer` scores retrieval alone.

    Pinned by a test because it makes the QA number an *upper bound*, and that
    caveat is easy to lose once the column appears in a report.
    """
    query = QAQuery(
        query_id="q",
        type="QA",
        text="màu gì",
        gt_video_id="L23_V001",
        gt_frame_range=(0, 5),
        gt_answer="màu vàng",
    )

    assert query.answer() == "màu vàng"
    assert query.model_copy(update={"system_answer": "màu đỏ"}).answer() == "màu đỏ"
