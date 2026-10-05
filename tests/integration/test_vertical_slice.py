"""The vertical slice — docs/00 §1.

    ingest -> search -> validate -> submit

Everything here is skipped until the workstreams land. That is deliberate: the
skips are the sprint's Definition of Done, written as code. When they all run,
Sprint 1 is done — and docs/01 §4 says the main branch should hold a working
slice at all times, which is only checkable if something checks it.

Owner: Vỹ Triết (harness), with each test unskipped by the workstream it covers.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="TODO: unskip as each workstream lands (docs/01 §4)")


def test_ingest_runs_on_sample_directory():
    """docs/00 §1: offline runs on any video directory with one command,
    producing all 4 stores. Owner: Quốc Đạt.

    Point it at tests/fixtures/, not data/raw/ — the test must not depend on
    whatever a developer happens to have downloaded.
    """


def test_all_four_stores_agree_on_keyframe_ids():
    """docs/02 §6: keyframe_id matches across all 4 stores. Owner: Quốc Đạt.

    A keyframe in the Vector Index but missing from the Relational Map is a
    candidate the validator will refuse mid-run. Catch it here.
    """


def test_mapping_is_100_percent_correct():
    """docs/00 §1: mapping correct 100% on the sample. Owner: Nhật Bảo.

    Via RelationalMap.verify_mapping(), not by sampling queries — a query set
    only touches the keyframes it happens to hit.
    """


def test_search_text_returns_candidates_with_evidence():
    """docs/04 §6: query -> top-k keyframes, each with an evidence_id. Owner: Phạm Trọng."""


def test_cascade_falls_back_when_rerank_fails():
    """docs/04 §1: rerank failure/timeout falls back to coarse. Owner: Phạm Trọng.

    Force the failure — do not wait to observe one. An untested fallback is a
    fallback that fires for the first time during a competition run.
    """


def test_validator_blocks_every_invalid_submit():
    """docs/00 §1: every malformed submit is blocked. Owner: Nhật Bảo.

    Needs one deliberately bad case per rule in docs/05 §3: unknown keyframe_id,
    frame past total_frames, timestamp past duration_ms, wrong schema, missing
    evidence.
    """


def test_submit_adapter_emits_both_timestamp_and_frame():
    """docs/05 §5: build_submission produces both formats. Owner: Nhật Bảo.

    Both, because BTC has not said which they want (docs/00 §3).
    """


def test_end_to_end_query_to_submission():
    """The whole slice: query -> cascade -> candidate -> validate -> submit record.

    The one test that proves docs/00 §1's "vertical slice" exists.
    """
