"""Khoa duong tra loi Q&A — gom bang chung, va cho cam bo doc.

PDF §2.1.2 doi ba thu: video_id, frame_id, answer. Hai thu dau da co; module nay
la thu ba. Cai duoc kiem o day: bang chung gom DUNG va DU, thu tu xep hang duoc
giu, va khong co bo doc thi KHONG doan.
"""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.online.answering import (
    Answer,
    Evidence,
    gather_evidence,
    gather_with_context,
    propose,
)


class FakeHit:
    def __init__(self, video_id, frame_id, source, text):
        self.video_id = video_id
        self.frame_id = frame_id
        self.source = source
        self.text = text
        self.score = 0.0


class FakeStore:
    """Text store gia. Ghi lai khoa da hoi de kiem so lan goi."""

    def __init__(self, rows):
        self.rows = rows
        self.asked: list[list[tuple[str, int]]] = []

    def fetch(self, keys, *, source=None, limit=500):  # noqa: ARG002
        keys = list(keys)
        self.asked.append(keys)
        wanted = set(keys)
        return [r for r in self.rows if (r.video_id, r.frame_id) in wanted]


class FakeRow:
    def __init__(self, keyframe_id):
        self.keyframe_id = keyframe_id


class FakeMap:
    def __init__(self, neighbours):
        self.neighbours = neighbours

    def get_neighbors(self, keyframe_id, window):  # noqa: ARG002
        return [FakeRow(k) for k in self.neighbours]


ROWS = [
    FakeHit("L21_V001", 531, "ocr", "TIN CHINH 06:30:28"),
    FakeHit("L21_V001", 531, "ocr_vietocr", "TÌNH TRẠNG SỤT LÚN Ở ĐBSCL"),
    FakeHit("L21_V001", 531, "subtitle", "Sạt lở đang diễn ra rất nhanh"),
    FakeHit("L21_V001", 711, "ocr_vietocr", "TRÁI TIM ĐƯỢC VẬN CHUYỂN"),
]


class TestGatherEvidence:
    def test_it_groups_every_source_for_one_keyframe(self):
        [item] = gather_evidence(["L21_V001#000531"], FakeStore(ROWS))
        assert set(item.texts) == {"ocr", "ocr_vietocr", "subtitle"}
        assert item.frame_id == 531
        assert item.video_id == "L21_V001"

    def test_input_order_is_preserved(self):
        """Thu tu vao la XEP HANG cua tang truy hoi va nguoi van hanh doc tu tren
        xuong. Sap lai o day la nem di thong tin dat nhat vua duoc tao ra."""
        ids = ["L21_V001#000711", "L21_V001#000531"]
        assert [e.keyframe_id for e in gather_evidence(ids, FakeStore(ROWS))] == ids

    def test_a_keyframe_with_no_text_still_comes_back(self):
        """Bo mot keyframe khong co chu se lam danh sach bang chung lech khoi
        danh sach ung vien, va nguoi van hanh doc nham dong."""
        [item] = gather_evidence(["L21_V001#009999"], FakeStore(ROWS))
        assert item.texts == {}

    def test_an_empty_text_is_not_a_source(self):
        """'OCR im lang' va 'chua chay OCR' la hai chuyen; khoa rong lam chung
        trong giong nhau."""
        rows = [*ROWS, FakeHit("L21_V001", 531, "caption", "   ")]
        [item] = gather_evidence(["L21_V001#000531"], FakeStore(rows))
        assert "caption" not in item.texts

    def test_sources_can_be_narrowed(self):
        [item] = gather_evidence(
            ["L21_V001#000531"], FakeStore(ROWS), sources={"subtitle"}
        )
        assert set(item.texts) == {"subtitle"}

    def test_it_asks_the_store_once_for_the_whole_batch(self):
        """Mot lan goi cho N keyframe. Hoi tung cai la N vong mang trong 5 phut
        dong ho cua mot luot thi."""
        store = FakeStore(ROWS)
        gather_evidence(["L21_V001#000531", "L21_V001#000711"], store)
        assert len(store.asked) == 1
        assert len(store.asked[0]) == 2

    def test_no_store_is_empty_not_a_crash(self):
        assert gather_evidence(["L21_V001#000531"], None) == []

    def test_no_keyframes_is_empty(self):
        assert gather_evidence([], FakeStore(ROWS)) == []


class TestOrdering:
    def test_sources_come_back_in_answer_usefulness_order(self):
        """Chu tren man hinh tra loi thang phan lon cau hoi Q&A cua BTC; caption
        la cau sinh tu dong nen dung cuoi."""
        item = Evidence(
            keyframe_id="V#000001", video_id="V", frame_id=1,
            texts={"caption": "c", "subtitle": "s", "ocr_vietocr": "v", "ocr": "o"},
        )
        assert [s for s, _ in item.ordered()] == ["ocr_vietocr", "ocr", "subtitle", "caption"]

    def test_an_unknown_source_goes_last_not_lost(self):
        item = Evidence(
            keyframe_id="V#000001", video_id="V", frame_id=1,
            texts={"ocr": "o", "zzz_moi": "m"},
        )
        assert [s for s, _ in item.ordered()] == ["ocr", "zzz_moi"]

    def test_as_prompt_labels_every_line_with_its_source(self):
        item = Evidence(
            keyframe_id="V#000001", video_id="V", frame_id=1,
            texts={"ocr": "HTV9", "subtitle": "xin chào"},
        )
        assert item.as_prompt() == "[ocr] HTV9\n[subtitle] xin chào"


class TestContextWindow:
    def test_it_reads_neighbours_off_the_map_not_by_arithmetic(self):
        """`frame_idx ± k` KHONG phai keyframe lan can: khoang cach giua hai
        keyframe la hang chuc toi hang tram frame va khong deu."""
        store = FakeStore(ROWS)
        items = gather_with_context(
            "L21_V001#000531", store,
            FakeMap(["L21_V001#000531", "L21_V001#000711"]),
            window=1,
        )
        assert [e.frame_id for e in items] == [531, 711]

    def test_the_anchor_is_included_even_if_the_map_omits_it(self):
        items = gather_with_context(
            "L21_V001#000531", FakeStore(ROWS), FakeMap(["L21_V001#000711"]), window=1
        )
        assert "L21_V001#000531" in [e.keyframe_id for e in items]

    def test_a_negative_window_is_refused(self):
        with pytest.raises(ContractError, match="window"):
            gather_with_context("V#000001", FakeStore([]), FakeMap([]), window=-1)


class TestPropose:
    def test_without_a_reader_it_returns_none_rather_than_guessing(self):
        """Cho cam do nhat cua ca module. Mot heuristic kieu 'lay chuoi so dau
        tien trong OCR' tra loi dung vai cau va sai phan con lai, ma nguoi van
        hanh khong phan biet duoc hai truong hop."""
        evidence = gather_evidence(["L21_V001#000531"], FakeStore(ROWS))
        assert propose("mấy giờ?", evidence, reader=None) is None

    def test_a_reader_is_called_with_the_question_and_evidence(self):
        seen = {}

        def reader(question, evidence, image_paths=()):
            seen["question"] = question
            seen["n"] = len(evidence)
            seen["images"] = list(image_paths)
            return Answer(text="06:30:28", keyframe_id=evidence[0].keyframe_id,
                          sources=("ocr",), reader="fake")

        reader.name = "fake"
        evidence = gather_evidence(["L21_V001#000531"], FakeStore(ROWS))
        answer = propose("mấy giờ?", evidence, reader=reader)
        assert seen == {"question": "mấy giờ?", "n": 1, "images": []}
        assert answer.text == "06:30:28"
        assert answer.keyframe_id == "L21_V001#000531"

    def test_a_reader_that_raises_does_not_kill_the_query(self):
        """Mot luot thi dai 5 phut. Bo doc hong phai lam mat mot cau tra loi de
        xuat, khong phai mat ca luot."""
        def reader(question, evidence, image_paths=()):  # noqa: ARG001
            raise RuntimeError("model chet")

        reader.name = "no"
        assert propose("hỏi gì đó", [], reader=reader) is None

    def test_an_empty_question_is_refused(self):
        with pytest.raises(ContractError, match="cau hoi rong"):
            propose("   ", [], reader=lambda q, e, i=(): None)
