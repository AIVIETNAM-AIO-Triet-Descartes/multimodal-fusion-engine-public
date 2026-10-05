"""Khoa dinh dang file nop — PDF §2.1.

`docs/context/buoi-1.md` §8: "tim dung hinh nhung submit sai ID/frame la loi
trang". Nhung gi test o day deu la cho co the sai IM LANG: so cot, thu tu dong,
dau phay trong cau tra loi, va gioi han 100 dong.
"""

from __future__ import annotations

import csv
import io

import pytest

from mfe.contracts.errors import ContractError
from mfe.eval.metrics.kis import KISSubmission
from mfe.eval.metrics.qa import QASubmission
from mfe.eval.metrics.trake import TRAKESubmission
from mfe.eval.submission import to_csv_text, to_rows, write_csv


def parse(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline="")))


class TestShape:
    def test_kis_is_video_then_frame(self):
        rows = to_rows([KISSubmission(video_id="L21_V001", frame_id=505)])
        assert rows == [["L21_V001", "505"]]

    def test_qa_appends_the_answer(self):
        rows = to_rows(
            [QASubmission(video_id="L05_V005", frame_id=888, answer="màu xanh")]
        )
        assert rows == [["L05_V005", "888", "màu xanh"]]

    def test_trake_writes_every_event_frame_in_order(self):
        """PDF §2.1.3: mot dong TRAKE la mot video roi N frame, dung thu tu su
        kien. Doi thu tu la doi dap an."""
        rows = to_rows(
            [TRAKESubmission(video_id="L10_V010", frame_ids=[101, 156, 203, 251])]
        )
        assert rows == [["L10_V010", "101", "156", "203", "251"]]

    def test_no_header_row(self):
        """BTC doc dong dau tien la mot cau tra loi. Mot dong tieu de la mot cau
        tra loi sai o vi tri so 1 — vi tri dat nhat, vi R@1 vao Final Score."""
        text = to_csv_text([KISSubmission(video_id="V", frame_id=1)])
        assert parse(text)[0] == ["V", "1"]

    def test_mixing_query_types_is_refused(self):
        """Mot file nop la mot truy van, va mot truy van co dung mot dang. Tron
        hai dang cho ra file so cot khong deu — BTC doc duoc, va doc sai."""
        with pytest.raises(ContractError, match="tron nhieu dang"):
            to_rows([
                KISSubmission(video_id="V", frame_id=1),
                QASubmission(video_id="V", frame_id=2, answer="x"),
            ])


class TestOrderAndLimit:
    def test_row_order_is_preserved_exactly(self):
        """Thu tu dong CO TINH DIEM: Final Score la trung binh R@k tai
        k = 1, 5, 20, 50, 100, nen dap an dung o dong 1 an diem cao hon han cung
        dap an do o dong 30. Tang ghi file khong duoc xep lai gi."""
        items = [KISSubmission(video_id="V", frame_id=f) for f in (30, 10, 20)]
        assert [row[1] for row in to_rows(items)] == ["30", "10", "20"]

    def test_it_stops_at_a_hundred_answers(self):
        items = [KISSubmission(video_id="V", frame_id=f) for f in range(140)]
        assert len(to_rows(items)) == 100

    def test_going_over_the_limit_is_logged_not_silent(self, caplog):
        """'Nop 140 dong' va 'nop 100 dong' trong giong het nhau trong mot file
        da ghi."""
        items = [KISSubmission(video_id="V", frame_id=f) for f in range(140)]
        with caplog.at_level("WARNING"):
            to_rows(items)
        assert "cat 140 -> 100" in caplog.text

    def test_a_zero_limit_is_refused(self):
        with pytest.raises(ContractError, match="limit"):
            to_rows([KISSubmission(video_id="V", frame_id=1)], limit=0)


class TestQuoting:
    def test_a_comma_inside_a_qa_answer_does_not_become_a_new_column(self):
        """`",".join(...)` hong o day; `csv` chuan thi khong. Mot dap an nhu
        "5, hoac 6" tu tach thanh hai cot va day moi thu sau no sang sai vi tri."""
        text = to_csv_text(
            [QASubmission(video_id="V", frame_id=1, answer="5, hoặc 6")]
        )
        assert parse(text) == [["V", "1", "5, hoặc 6"]]

    def test_a_quote_inside_an_answer_round_trips(self):
        text = to_csv_text(
            [QASubmission(video_id="V", frame_id=1, answer='ông "Bảy" Nam')]
        )
        assert parse(text) == [["V", "1", 'ông "Bảy" Nam']]


class TestFile:
    def test_it_writes_utf8_without_blank_lines(self, tmp_path):
        """`newline=""` la bat buoc cua `csv`: thieu no, tren Windows moi dong ra
        hai ky tu xuong dong va file co mot dong trong xen ke."""
        path = write_csv(
            [
                QASubmission(video_id="V", frame_id=1, answer="màu xanh"),
                QASubmission(video_id="V", frame_id=2, answer="năm"),
            ],
            tmp_path / "sub" / "q1.csv",
        )
        raw = path.read_text(encoding="utf-8")
        assert "\r\r\n" not in raw
        assert [line for line in raw.splitlines() if line] == [
            "V,1,màu xanh",
            "V,2,năm",
        ]

    def test_it_creates_the_parent_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "q.csv"
        assert write_csv([KISSubmission(video_id="V", frame_id=1)], target).is_file()

    def test_an_empty_run_writes_an_empty_file_rather_than_failing(self, tmp_path):
        """Truy van khong tim duoc gi van la mot trang thai hop le; ném loi o day
        se lam nguoi van hanh mat ca phien vi mot truy van."""
        path = write_csv([], tmp_path / "empty.csv")
        assert path.read_text(encoding="utf-8") == ""
