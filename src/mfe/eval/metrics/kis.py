"""Textual KIS scoring for one submitted answer — docs/metrics-btc.html §2.

Official formula (PDF §2.1.1)::

    R-Score(r_i) = I( v_i = GT_v  AND  id_i in [s, e] )

Binary: the right video *and* a frame inside the accepted interval, or nothing.
There is no partial credit at this query type, unlike TRAKE.

Scope, deliberately: this scores **one** answer. R@k and Final Score belong to
:mod:`mfe.eval.metrics.final_score`. Nothing here reads the Relational Map or
any store — it is a pure function over two frozen records, so it stays valid
however the retrieval side is rebuilt.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

__all__ = ("KISGroundTruth", "KISSubmission", "score_kis")

#: Same field policy as :mod:`mfe.eval.metrics.qa`, on purpose — the two scorers
#: are compared side by side in the harness and a silent difference in what
#: counts as a valid id would be read as a difference in retrieval quality.
_StrictString = Annotated[str, StringConstraints(min_length=1, strict=True)]

#: `strict=True` so `True`/`1.0`/`"505"` are rejected rather than coerced.
#: `bool` is an `int` in Python: without strict mode `frame_id=True` would score
#: as frame 1 and quietly produce a 0.0 that looks like an honest miss.
_FrameId = Annotated[int, Field(ge=0, strict=True)]


class KISSubmission(BaseModel):
    """One immutable KIS answer: which video, which source frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: _StrictString
    frame_id: _FrameId


class KISGroundTruth(BaseModel):
    """The accepted video and closed frame interval for one KIS query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: _StrictString
    start_frame_id: _FrameId
    end_frame_id: _FrameId

    @model_validator(mode="after")
    def _range_is_ordered(self) -> KISGroundTruth:
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("start_frame_id must be <= end_frame_id")
        return self


def score_kis(submission: KISSubmission, gt: KISGroundTruth) -> float:
    """Return the official binary R-Score for one KIS answer.

    The interval is treated as **closed at both ends**: `id_i == s` and
    `id_i == e` both count as a hit. The PDF writes `[s, e]` and never says
    otherwise, and square brackets are closed intervals in the notation it uses
    everywhere else. Flagged as an assumption in `docs/metrics-btc.html` §2, and
    locked by two boundary tests — if BTC later clarifies a half-open interval,
    those two tests are the only thing that changes.

    `video_id` is compared exactly, case-sensitively and without stripping. BTC
    ids are machine-generated (`L21_V001`), so a submission that differs by case
    or whitespace came from a bug in our own pipeline; normalising it here would
    hide that bug and score a wrong id as right.

    Raises `TypeError` for anything that is not the declared record type. A
    malformed submission scored as `0.0` is indistinguishable from an honest
    miss, which is the failure mode that matters at scoring time.
    """
    if not isinstance(submission, KISSubmission):
        raise TypeError("submission must be a KISSubmission")
    if not isinstance(gt, KISGroundTruth):
        raise TypeError("gt must be a KISGroundTruth")

    is_correct = (
        submission.video_id == gt.video_id
        and gt.start_frame_id <= submission.frame_id <= gt.end_frame_id
    )
    return 1.0 if is_correct else 0.0
