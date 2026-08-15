"""Coordinate ascent over fusion weights — `mfe.eval.tuning`.

Built on synthetic evidence rather than a live cascade: the point under test is
the search procedure, and a fake arm whose answer is known lets a test assert
*which* assignment wins instead of only that some number moved.
"""

from __future__ import annotations

import pytest

from mfe.contracts.tools import Candidate
from mfe.eval.official import KISQuery
from mfe.eval.tuning import (
    OFF,
    Assignment,
    Measurement,
    QueryEvidence,
    ablation,
    coordinate_ascent,
    fuse_and_group,
    measure,
)


class FakeRelational:
    """`keyframe_id` already carries the frame, so the map is pure formatting."""

    class _Row:
        def __init__(self, frame_idx: int) -> None:
            self.frame_idx = frame_idx

    def get_keyframe(self, keyframe_id: str):
        return self._Row(int(keyframe_id.split("#")[1]))


def _candidates(frames: list[int], video_id: str = "L21_V001") -> list[Candidate]:
    return [
        Candidate(
            keyframe_id=f"{video_id}#{frame:06d}",
            video_id=video_id,
            score=1.0 - index / 100,
            evidence_id=f"fake:tune:{index + 1}",
        )
        for index, frame in enumerate(frames)
    ]


def _query(query_id: str = "q1", answer_frame: int = 1000) -> KISQuery:
    return KISQuery(
        query_id=query_id,
        type="KIS",
        text="một truy vấn",
        gt_video_id="L21_V001",
        gt_frame_range=(answer_frame, answer_frame),
    )


def _evidence(result_lists: dict[str, list[Candidate]], **kwargs) -> QueryEvidence:
    return QueryEvidence(query=_query(**kwargs), per_event=[result_lists])


def _no_group(candidates):
    """Stage B as identity — grouping is not what these tests are about."""
    return candidates


def _measure(evidences, assignment, **kwargs) -> Measurement:
    return measure(evidences, assignment, FakeRelational(), _no_group, **kwargs)


# --- the OFF setting -----------------------------------------------------


def test_weight_zero_admits_candidates_that_off_removes() -> None:
    """What the two settings actually do, measured rather than assumed.

    A zero-weighted arm appends its candidates to the tail — it contributes no
    score, so it cannot outrank anything a weighted arm found — while OFF drops
    them from the list entirely. The submitted list differs; the rank of the
    answer does not. Both halves matter: the first is why OFF is in the grid, the
    second is why "weight 0 does not disable the arm" must not be read as "weight
    0 lets a noisy arm ruin the ranking".
    """
    lists = {"good": _candidates([11, 12, 13, 1000]), "noise": _candidates([21, 22, 23])}

    def frames(assignment: Assignment) -> list[int]:
        fused = fuse_and_group(lists, assignment, _no_group, "rrf", 60, top_k=100)
        return [int(c.keyframe_id.split("#")[1]) for c in fused]

    weighted = frames(Assignment({"good": 1.0, "noise": 1.0}))
    zeroed = frames(Assignment({"good": 1.0, "noise": 0.0}))
    switched_off = frames(Assignment({"good": 1.0, "noise": OFF}))

    assert zeroed == [11, 12, 13, 1000, 21, 22, 23]
    assert switched_off == [11, 12, 13, 1000]
    assert zeroed.index(1000) == switched_off.index(1000) == 3
    assert weighted.index(1000) == 6, "a *weighted* noise arm is what pushes it down"


def test_disabling_every_arm_scores_zero_rather_than_crashing() -> None:
    evidence = _evidence({"only": _candidates([1000])})
    result = _measure([evidence], Assignment({"only": OFF}))
    assert result.wide == 0.0
    assert result.hits == 0


# --- the ascent ----------------------------------------------------------


def test_ascent_beats_uniform_weights_when_one_arm_is_noisy() -> None:
    """The `ocr_sviptr` shape: right when it fires, noise the rest of the time.

    One query where the arm confirms the answer, two where it floods the top with
    junk and pushes the answer past rank 5 — the boundary where Final Score drops.

    Note what is *not* asserted: that the noisy arm gets silenced. Measured, the
    ascent instead raises the reliable arm to 4.0, which fixes both hurt queries
    and keeps the confirmation on the helpful one — 0.867 against the 0.800 that
    silencing would have scored. That is fusion behaving as §6.3 of the handoff
    says it does: not a choice between arms, so "this arm is noisy" does not
    imply "remove it".
    """
    helpful = _evidence({"good": _candidates([11, 1000]), "harmful": _candidates([1000])})
    hurt_a = _evidence(
        {"good": _candidates([11, 12, 13, 1000]), "harmful": _candidates([21, 22, 23])},
        query_id="q2",
    )
    hurt_b = _evidence(
        {"good": _candidates([11, 12, 13, 1000]), "harmful": _candidates([21, 22, 23])},
        query_id="q3",
    )

    result = coordinate_ascent(
        [helpful, hurt_a, hurt_b],
        Assignment({"good": 1.0, "harmful": 1.0}),
        ["good", "harmful"],
        FakeRelational(),
        _no_group,
        rounds=2,
    )

    assert result.best.wide > result.baseline.wide
    assert result.assignment.settings != {"good": 1.0, "harmful": 1.0}
    assert result.best.wide == pytest.approx(0.867, abs=0.005)


def test_a_tie_takes_the_least_intervention_and_never_off() -> None:
    """Which of the tied-best settings wins, and why it is not OFF.

    Below RRF's `1/(60+rank)` scale a small weight and a zero weight often score
    identically — here `OFF`, `0.0` and `0.5` all keep the answer at rank 4 while
    `1.0` pushes it to rank 7. Among tied winners the ascent takes the value
    closest to the uniform 1.0 baseline, because uniform weights are the strong
    default §3.4b warns against beating on noise, and it never takes OFF, because
    removing an arm from the system is a real change that a tie does not justify.
    """
    hurt = _evidence(
        {"good": _candidates([11, 12, 13, 1000]), "harmful": _candidates([21, 22, 23])}
    )

    result = coordinate_ascent(
        [hurt],
        Assignment({"good": 1.0, "harmful": 1.0}),
        ["harmful"],
        FakeRelational(),
        _no_group,
        rounds=2,
    )

    tried = dict(result.steps[0].tried)
    assert tried[OFF].wide == tried[0.0].wide == tried[0.5].wide > tried[1.0].wide
    assert result.assignment.settings["harmful"] == 0.5


def test_ascent_keeps_the_incumbent_when_nothing_beats_it() -> None:
    """No movement without a measured gain — 9 queries tie constantly."""
    evidence = _evidence({"good": _candidates([1000])})

    result = coordinate_ascent(
        [evidence],
        Assignment({"good": 1.0}),
        ["good"],
        FakeRelational(),
        _no_group,
        rounds=3,
    )

    assert result.assignment.settings["good"] == 1.0
    assert result.rounds_run == 1, "a round that changes nothing must stop the ascent"


def test_a_tie_never_switches_an_arm_off() -> None:
    """Removing an arm is a real change; a tie is not a reason for one."""
    # One arm, one query: every weight ranks the single answer first, so the
    # whole sweep ties — including OFF at 0.0? No: OFF scores worse, so build a
    # case where a second arm alone already answers.
    evidence = _evidence({"good": _candidates([1000]), "spare": _candidates([1000])})

    result = coordinate_ascent(
        [evidence],
        Assignment({"good": 1.0, "spare": 1.0}),
        ["good", "spare"],
        FakeRelational(),
        _no_group,
        rounds=2,
    )

    assert result.assignment.settings == {"good": 1.0, "spare": 1.0}


def test_ascent_records_every_value_it_tried() -> None:
    evidence = _evidence({"good": _candidates([1000])})
    result = coordinate_ascent(
        [evidence],
        Assignment({"good": 1.0}),
        ["good"],
        FakeRelational(),
        _no_group,
        rounds=1,
    )
    tried = [value for value, _ in result.steps[0].tried]
    assert tried == [OFF, 0.0, 0.5, 1.0, 2.0, 4.0]


def test_no_off_option_restricts_the_grid() -> None:
    evidence = _evidence({"good": _candidates([1000])})
    result = coordinate_ascent(
        [evidence],
        Assignment({"good": 1.0}),
        ["good"],
        FakeRelational(),
        _no_group,
        allow_off=False,
        rounds=1,
    )
    assert OFF not in [value for value, _ in result.steps[0].tried]


@pytest.mark.parametrize("objective", ["wide", "strict", "both"])
def test_objectives_all_run(objective: str) -> None:
    evidence = _evidence({"good": _candidates([1000])})
    result = coordinate_ascent(
        [evidence],
        Assignment({"good": 1.0}),
        ["good"],
        FakeRelational(),
        _no_group,
        rounds=1,
        objective=objective,
    )
    assert result.objective == objective


# --- ablation ------------------------------------------------------------


def test_ablation_reports_full_set_then_each_arm_removed() -> None:
    evidence = _evidence({"a": _candidates([1000]), "b": _candidates([2000])})

    rows = ablation(
        [evidence],
        Assignment({"a": 1.0, "b": 1.0}),
        ["a", "b"],
        FakeRelational(),
        _no_group,
    )

    assert [label for label, _ in rows] == ["<full>", "-a", "-b"]
    without_a = dict(rows)["-a"]
    assert without_a.hits == 0, "dropping the only arm that finds it must lose the hit"


def test_measurement_summary_is_readable_without_a_strict_window() -> None:
    evidence = _evidence({"a": _candidates([1000])})
    result = _measure([evidence], Assignment({"a": 1.0}))
    assert "HEP -" in result.summary()
    assert result.strict is None
