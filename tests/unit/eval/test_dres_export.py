import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mfe.contracts.tools import Candidate
from mfe.eval.dres_export import json_bytes, milliseconds, top1_payload
from mfe.eval.query_package import PackagedQuery
from mfe.eval.submission_bundle import replace_at_rank


def query(kind="KIS", events=()):
    return PackagedQuery("query-test-" + kind.lower(), kind, "test",
                         Path("test.txt"), events=events)


def candidate(n):
    return Candidate(keyframe_id=f"L21_V001#{n:06d}", video_id="L21_V001", score=1,
                     evidence_id=f"siglip:L21_V001:{n}")


class Map:
    def get_keyframe(self, key):
        # Deliberately not frame/FPS: mimics variable frame rate timing.
        return NS(video_id="L21_V001", pts_time={candidate(100).keyframe_id: 8.1235,
                                               candidate(200).keyframe_id: 29.701}[key])


def entry(payload):
    assert len(payload["answerSets"]) == 1
    assert len(payload["answerSets"][0]["answers"]) == 1
    return payload["answerSets"][0]["answers"][0]


def test_kis_manual_order_vfr():
    rows = [candidate(100), candidate(200)]
    value = top1_payload(query(), rows, Map(),
                         manual_order=[rows[1].keyframe_id, rows[0].keyframe_id])
    assert entry(value) == {"mediaItemName": "L21_V001", "start": 29701, "end": 29701}


def test_scrub_timestamp_overrides_colliding_catalogue_id():
    rows = replace_at_rank([candidate(100)], 1, 100, pts_time=99.123)
    assert entry(top1_payload(query(), rows, Map()))["start"] == 99123


def test_scrub_without_pts_blocked():
    with pytest.raises(ValueError):
        top1_payload(query(), replace_at_rank([candidate(100)], 1, 100), Map())


def test_qa_unicode_bytes():
    payload = top1_payload(query("QA"), [candidate(100)], Map(), answer=" hai người, áo đỏ ")
    raw = json_bytes(payload)
    assert "áo đỏ".encode() in raw
    assert entry(json.loads(raw))["text"] == "QA-hai người, áo đỏ-L21_V001-8124"


@pytest.mark.parametrize("frames", [[1], [1, 1], [2, 1], [-1, 2], [1, 2.5]])
def test_invalid_trake(frames):
    with pytest.raises(ValueError):
        top1_payload(query("TRAKE", ("a", "b")), [NS(video_id="N001", frame_ids=frames)], Map())


def test_complete_trake_sequence():
    payload = top1_payload(query("TRAKE", ("a", "b")),
                           [NS(video_id="N001", frame_ids=[100, 750]),
                            NS(video_id="N002", frame_ids=[1, 2])], Map())
    assert entry(payload) == {"text": "TR-N001-100,750"}


@pytest.mark.parametrize("seconds", [None, True, -1, float('nan'), float('inf')])
def test_invalid_time(seconds):
    with pytest.raises(ValueError):
        milliseconds(seconds)


def test_empty_results_answer_and_unknown_frame():
    with pytest.raises(ValueError):
        top1_payload(query(), [], Map())
    with pytest.raises(ValueError):
        top1_payload(query("QA"), [candidate(100)], Map(), answer=" ")
    with pytest.raises(KeyError):
        top1_payload(query(), [candidate(999)], Map())
