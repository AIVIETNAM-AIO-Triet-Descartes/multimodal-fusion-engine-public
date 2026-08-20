"""Khoa phan cua browser KHONG can Streamlit chay.

Ban truoc test `validate_submission_stub` — mot validator gia LUON tra VALID.
No khong con: dau ra that cua giao dien la mot file CSV dung dinh dang BTC, va
`mfe.eval.submission` moi la cho kiem dinh dang do.

Con lai o day la ba manh thuan tuy: doi ung vien sang toa do DANTE, xem truoc
mot dong nop, va anh thay the khi khong co keyframe tren dia.
"""

from __future__ import annotations

from mfe.browser.app import _row_preview, _to_event_hits, get_keyframe_image
from mfe.contracts.tools import Candidate
from mfe.eval.metrics.kis import KISSubmission
from mfe.eval.metrics.qa import QASubmission
from mfe.eval.metrics.trake import TRAKESubmission


class _Row:
    def __init__(self, frame_idx: int, keyframe_index: int) -> None:
        self.frame_idx = frame_idx
        self.keyframe_index = keyframe_index


class _Map:
    def __init__(self, rows: dict[str, _Row]) -> None:
        self.rows = rows

    def get_keyframe(self, keyframe_id: str) -> _Row:
        return self.rows[keyframe_id]      # KeyError khi khong co


def _cand(keyframe_id: str, score: float = 0.5) -> Candidate:
    return Candidate(
        keyframe_id=keyframe_id,
        video_id=keyframe_id.split("#")[0],
        score=score,
        evidence_id="ui:run:1",
    )


class TestEventHits:
    def test_it_reads_both_coordinates_off_the_map(self):
        """`ordinal` xep thu tu cho DANTE, `frame_id` la thu BTC cham. Hai truc
        khac nhau va ca hai deu doc tu map, khong suy ra tu `keyframe_id`."""
        relational = _Map({"V#000270": _Row(frame_idx=270, keyframe_index=3)})
        [hit] = _to_event_hits([_cand("V#000270")], relational)
        assert (hit.ordinal, hit.frame_id) == (3, 270)
        assert hit.video_id == "V"

    def test_a_keyframe_missing_from_the_map_is_dropped_not_guessed(self):
        """Doan toa do la nop sai khung hinh — loi trang (`buoi-1.md` §8). Bo mot
        ung vien lam ket qua ngheo di; doan lam no SAI."""
        relational = _Map({"V#000270": _Row(270, 3)})
        hits = _to_event_hits([_cand("V#000270"), _cand("V#999999")], relational)
        assert [h.keyframe_id for h in hits] == ["V#000270"]

    def test_candidate_order_and_scores_survive(self):
        relational = _Map({f"V#{n:06d}": _Row(n, n // 90) for n in (90, 180, 270)})
        hits = _to_event_hits(
            [_cand("V#000270", 0.9), _cand("V#000090", 0.4)], relational
        )
        assert [h.ordinal for h in hits] == [3, 1]
        assert [h.score for h in hits] == [0.9, 0.4]


class TestRowPreview:
    def test_kis_shows_video_and_frame(self):
        assert _row_preview(KISSubmission(video_id="V", frame_id=505)) == ["V", 505]

    def test_qa_shows_the_answer_too(self):
        row = _row_preview(QASubmission(video_id="V", frame_id=888, answer="màu xanh"))
        assert row == ["V", 888, "màu xanh"]

    def test_trake_shows_every_event_frame_in_order(self):
        row = _row_preview(TRAKESubmission(video_id="V", frame_ids=[101, 156, 203]))
        assert row == ["V", 101, 156, 203]


class TestPlaceholderImage:
    def test_a_missing_keyframe_gives_a_placeholder_that_says_so(self):
        """Mot cho trong trong nhu anh mo se khien nguoi van hanh tuong minh dang
        nhin khung hinh. O xem truoc phai noi ro no khong phai anh that."""
        image = get_keyframe_image("KHONG_CO#000001")
        assert image is not None
        # PIL Image, khong phai duong dan tren dia.
        assert not isinstance(image, str)
