"""Khoá bộ nạp gói đề BTC — mfe.eval.query_package.

Đề thật là file text người gõ tay, nên mọi test ở đây đều về chuyện **đọc sai một
file trông không giống đọc sai**: một hậu tố lạ thành KIS, một nhãn sự kiện gõ
nhầm thành thiếu sự kiện. Cả hai đều nộp được và đều bị chấm 0 mà không kêu.
"""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.eval.query_package import (
    load_package,
    load_query,
    parse_query_type,
    split_events,
)

BO_DE = "docs/context/bo-de-thu-nghiem"


def write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestDangTruyVan:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("query-p1-1-kis.txt", "KIS"),
            ("query-p1-15-qa.txt", "QA"),
            ("query-p1-16-trake.txt", "TRAKE"),
            ("query-2-KIS.txt", "KIS"),
        ],
    )
    def test_doc_tu_hau_to_ten_file(self, name, expected):
        assert parse_query_type(name) == expected

    def test_hau_to_la_thi_ném_loi_chu_khong_mac_dinh_ve_kis(self):
        """Mặc định về KIS ghi ra một file QA thiếu hẳn cột đáp án — nộp được,
        chấm 0, và không có gì trong quá trình đó phát ra tiếng."""
        with pytest.raises(ContractError):
            parse_query_type("query-p1-9-vqa.txt")


class TestTachSuKien:
    def test_cau_mo_dau_duoc_ghep_vao_tung_su_kien(self, tmp_path):
        """Một truy vấn TRAKE hỏi hai chuyện chồng lên nhau — *video nào* và
        *khoảnh khắc nào*. Câu mở đầu trả lời chuyện thứ nhất, và DANTE chỉ xếp
        được sự kiện TRONG một video nên nó không cứu được việc chọn nhầm video.
        """
        path = write(
            tmp_path,
            "query-p1-9-trake.txt",
            "Đoạn video múa lân, tìm các sự kiện sau:\nE1: Lân xoay vòng.\nE2: Lân tiếp đất.\n",
        )
        query = load_query(path)
        assert query.n_events == 2
        assert query.searches() == [
            "Đoạn video múa lân, tìm các sự kiện sau: Lân xoay vòng.",
            "Đoạn video múa lân, tìm các sự kiện sau: Lân tiếp đất.",
        ]

    def test_khong_co_cau_mo_dau_thi_su_kien_di_mot_minh(self, tmp_path):
        """Mất mát thật, ghi lại ở `preamble is None` — không bù bằng cách đoán
        ra một câu mô tả video."""
        path = write(tmp_path, "q-trake.txt", "E1: Bột vào tô.\nE2: Thả vào chảo.\n")
        query = load_query(path)
        assert query.preamble is None
        assert query.searches() == ["Bột vào tô.", "Thả vào chảo."]

    def test_nhan_go_nham_van_doc_duoc_theo_thu_tu_dong(self, caplog):
        """`query-p1-18-trake.txt` của BTC ghi E1, E2, **E2**, E4 cho bốn dòng.

        Bốn dòng vẫn là bốn sự kiện đúng thứ tự thời gian; chỉ cái nhãn sai. Từ
        chối cả đề vì một chữ số gõ nhầm là tự bỏ một câu — nhưng đi qua trong im
        lặng thì người vận hành không bao giờ biết để đối chiếu lại.
        """
        text = "Video nấu ăn:\nE1: cắt nấm.\nE2: cắt củ năng.\nE2: cắt đậu hũ.\nE4: bắc chảo.\n"
        with caplog.at_level("WARNING"):
            preamble, events = split_events(text)
        assert len(events) == 4
        assert events[2] == "cắt đậu hũ."
        assert preamble == "Video nấu ăn:"
        assert "THU TU DONG" in caplog.text

    def test_thieu_han_mot_dong_thi_dung(self):
        """Nhãn chạy đến E4 mà chỉ đọc được 3 dòng: thiếu hẳn một sự kiện, không
        phải lỗi gõ nhãn. Mọi dòng nộp sẽ sai số cột — thể lệ đòi số Frame ID
        khớp chính xác số events."""
        with pytest.raises(ContractError, match="E4"):
            split_events("E1: a.\nE2: b.\nE4: d.\n")

    def test_de_trake_khong_co_dong_su_kien_nao(self):
        with pytest.raises(ContractError):
            split_events("Tìm cảnh con lân nhảy qua cột.")

    def test_chu_E_giua_cau_khong_bi_hieu_thanh_moc(self):
        """Neo `^` chứ không phải tìm ở bất kỳ đâu: "vitamin E1" giữa câu không
        được mở thêm một sự kiện."""
        _, events = split_events("E1: nói về vitamin E2 trong rau.\nE2: cận cảnh rau.\n")
        assert len(events) == 2


class TestCaGoi:
    def test_sap_theo_so_khong_theo_chuoi(self, tmp_path):
        """Sắp theo chuỗi đặt `query-p1-10` trước `query-p1-2`, và đó là thứ tự
        người vận hành đối chiếu với đề trên màn hình."""
        for n in (1, 2, 10, 21):
            write(tmp_path, f"query-p1-{n}-kis.txt", f"truy van {n}")
        assert [q.query_id for q in load_package(tmp_path)] == [
            "query-p1-1-kis",
            "query-p1-2-kis",
            "query-p1-10-kis",
            "query-p1-21-kis",
        ]

    def test_ten_file_nop_giu_nguyen_ten_de(self, tmp_path):
        """Ghép lại tên từ các mảnh là chỗ một `p1` bị rơi và cả gói lệch tên."""
        write(tmp_path, "query-p1-15-qa.txt", "Xã này tên gì?")
        assert load_package(tmp_path)[0].csv_name == "query-p1-15-qa.csv"

    def test_thu_muc_rong(self, tmp_path):
        with pytest.raises(ContractError):
            load_package(tmp_path)

    def test_file_rong(self, tmp_path):
        path = write(tmp_path, "query-p1-1-kis.txt", "   \n")
        with pytest.raises(ContractError):
            load_query(path)


class TestBoDeThat:
    """Chạy trên chính bộ đề BTC gửi. Một test đọc dữ liệu thật, cố ý."""

    def test_ca_bo_de_thu_nghiem_doc_duoc(self):
        queries = load_package(BO_DE)
        assert len(queries) == 24
        counts = {t: sum(1 for q in queries if q.type == t) for t in ("KIS", "QA", "TRAKE")}
        assert counts == {"KIS": 18, "QA": 3, "TRAKE": 3}

    def test_moi_de_trake_deu_co_4_su_kien(self):
        for query in load_package(BO_DE):
            if query.type == "TRAKE":
                assert query.n_events == 4, query.query_id
                assert len(query.searches()) == 4
