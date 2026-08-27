"""Khoá tầng dựng cả gói file nộp — mfe.eval.submission_bundle.

Đây là phần `app/submit_lab.py` gọi để biến kết quả truy hồi thành thứ BTC chấm.
Sai ở đây là **lỗi trắng** (`docs/context/buoi-1.md` §8): tìm đúng hình nhưng nộp
sai id/frame thì không ai kêu, file vẫn nộp được, điểm vẫn 0. Nên nó nằm ở thư
viện chứ không nằm trong script Streamlit — script chạy mã lúc import nên không
test nào với tới được.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from mfe.contracts.errors import ContractError
from mfe.contracts.tools import Candidate
from mfe.eval.query_package import parse_query
from mfe.eval.submission_bundle import apply_manual_order, build_zip, submissions_for


def cand(video: str, ordinal: int, score: float = 1.0) -> Candidate:
    return Candidate(
        keyframe_id=f"{video}#{ordinal:06d}",
        video_id=video,
        score=score,
        evidence_id=f"siglip:test:{ordinal}",
    )


class FakeMap:
    """`RelationalMap` giả — chỉ trả `frame_idx`, đủ cho tầng này.

    `frame_idx` KHÁC phần sau `#` một cách CỐ Ý (nhân 7): nếu chỗ nào trong tầng
    này suy `frame_id` ra từ `keyframe_id` thay vì tra bản đồ, test sẽ thấy.
    """

    def get_keyframe(self, keyframe_id: str):
        ordinal = int(keyframe_id.split("#")[1])

        class Row:
            frame_idx = ordinal * 7
            keyframe_index = ordinal

        return Row()


@pytest.fixture
def relational():
    return FakeMap()


# --------------------------------------------------------------------------
# Thứ tự người xếp
# --------------------------------------------------------------------------


class TestThuTuTay:
    def test_khong_co_thu_tu_tay_thi_giu_nguyen(self):
        items = [cand("L01_V001", 1), cand("L01_V001", 2)]
        assert apply_manual_order(items, None) == items
        assert apply_manual_order(items, []) == items

    def test_dua_id_len_dau_theo_dung_thu_tu_da_keo(self):
        items = [cand("L01_V001", 1), cand("L01_V001", 2), cand("L01_V001", 3)]
        out = apply_manual_order(items, ["L01_V001#000003", "L01_V001#000001"])
        assert [c.keyframe_id for c in out][:2] == [
            "L01_V001#000003", "L01_V001#000001",
        ]

    def test_ung_vien_chua_xep_thi_NOI_VAO_CUOI_chu_khong_bi_bo(self):
        """Bỏ chúng đi sẽ cắt ngắn bài nộp, mà R@100 chỉ tăng khi nộp thêm dòng."""
        items = [cand("L01_V001", i) for i in (1, 2, 3)]
        out = apply_manual_order(items, ["L01_V001#000002"])
        assert len(out) == 3
        assert out[0].keyframe_id == "L01_V001#000002"
        assert {c.keyframe_id for c in out[1:]} == {
            "L01_V001#000001", "L01_V001#000003",
        }

    def test_id_khong_con_trong_ket_qua_thi_ROI_RA(self):
        """Tàn dư của một lượt chạy trước không được kéo theo vào bài nộp."""
        items = [cand("L01_V001", 1)]
        out = apply_manual_order(items, ["L09_V009#000099", "L01_V001#000001"])
        assert [c.keyframe_id for c in out] == ["L01_V001#000001"]


# --------------------------------------------------------------------------
# Ba dạng đề, ba hình dạng dòng
# --------------------------------------------------------------------------


class TestHinhDangNop:
    def test_kis_hai_cot_va_frame_id_TRA_TU_BAN_DO(self, relational):
        query = parse_query("query-p1-1-kis.txt", "một cảnh nào đó")
        rows = submissions_for(query, [cand("L01_V001", 3)], relational)
        assert len(rows) == 1
        # 3 * 7 = 21, khong phai 3 — chung minh no TRA chu khong SUY.
        assert rows[0].frame_id == 21
        assert rows[0].video_id == "L01_V001"

    def test_qa_co_cot_dap_an(self, relational):
        query = parse_query("query-p1-15-qa.txt", "màu gì?")
        rows = submissions_for(
            query, [cand("L01_V001", 3)], relational, answer="màu đỏ"
        )
        assert rows[0].answer == "màu đỏ"

    def test_qa_khong_nhap_dap_an_thi_NEM_kem_ten_cau(self, relational):
        """Không ghi cột rỗng: một file QA đủ 100 dòng với cột đáp án trống
        trông y hệt bài nộp hoàn chỉnh và ghi 0 điểm."""
        query = parse_query("query-p1-15-qa.txt", "màu gì?")
        with pytest.raises(ContractError, match="query-p1-15-qa"):
            submissions_for(query, [cand("L01_V001", 3)], relational)

    def test_thu_tu_tay_di_VAO_bai_nop(self, relational):
        query = parse_query("query-p1-1-kis.txt", "một cảnh nào đó")
        items = [cand("L01_V001", 1), cand("L01_V001", 5)]
        rows = submissions_for(
            query, items, relational, manual_order=["L01_V001#000005"]
        )
        assert [r.frame_id for r in rows] == [35, 7]

    def test_limit_cat_dung_so_dong(self, relational):
        query = parse_query("query-p1-1-kis.txt", "x")
        rows = submissions_for(
            query, [cand("L01_V001", i) for i in range(1, 30)], relational, limit=5
        )
        assert len(rows) == 5


class TestTrake:
    """TRAKE nhận `Alignment`, không nhận `Candidate`, và không xếp tay."""

    class FakeAlignment:
        def __init__(self, video_id, frame_ids):
            self.video_id = video_id
            self.frame_ids = frame_ids

    def test_moi_dong_la_mot_video_kem_N_khung(self, relational):
        query = parse_query(
            "query-p1-16-trake.txt", "Nhảy sào.\nE1 chạy đà\nE2 giậm nhảy"
        )
        result = [self.FakeAlignment("L01_V001", [100, 250])]
        rows = submissions_for(query, result, relational)
        assert rows[0].video_id == "L01_V001"
        assert rows[0].frame_ids == [100, 250]

    def test_thu_tu_tay_bi_BO_QUA_voi_trake(self, relational):
        """Kéo một dòng lên đầu không sửa được chuỗi bên trong dòng đó."""
        query = parse_query("query-p1-16-trake.txt", "E1 a\nE2 b")
        result = [
            self.FakeAlignment("L01_V001", [1, 2]),
            self.FakeAlignment("L02_V002", [3, 4]),
        ]
        rows = submissions_for(
            query, result, relational, manual_order=["bat ky thu gi"]
        )
        assert [r.video_id for r in rows] == ["L01_V001", "L02_V002"]


# --------------------------------------------------------------------------
# ZIP cả gói
# --------------------------------------------------------------------------


class TestZipCaGoi:
    def test_mot_file_moi_cau_dung_ten_de_cong_csv(self, relational):
        queries = [
            parse_query("query-p1-1-kis.txt", "a"),
            parse_query("query-p1-15-qa.txt", "b"),
        ]
        results = {
            "query-p1-1-kis": [cand("L01_V001", 1)],
            "query-p1-15-qa": [cand("L02_V002", 2)],
        }
        bundle = build_zip(
            queries, results, relational, answers={"query-p1-15-qa": "màu đỏ"}
        )
        with zipfile.ZipFile(io.BytesIO(bundle.payload)) as z:
            assert sorted(z.namelist()) == [
                "query-p1-1-kis.csv", "query-p1-15-qa.csv",
            ]
        assert bundle.skipped == {}

    def test_mot_cau_QA_thieu_dap_an_KHONG_lam_hong_ca_goi(self, relational):
        """48 câu đúng vẫn phải tải về được, dù câu thứ 49 quên gõ đáp án."""
        queries = [
            parse_query("query-p1-1-kis.txt", "a"),
            parse_query("query-p1-15-qa.txt", "b"),
        ]
        results = {
            "query-p1-1-kis": [cand("L01_V001", 1)],
            "query-p1-15-qa": [cand("L02_V002", 2)],
        }
        bundle = build_zip(queries, results, relational)
        with zipfile.ZipFile(io.BytesIO(bundle.payload)) as z:
            assert z.namelist() == ["query-p1-1-kis.csv"]
        assert "query-p1-15-qa" in bundle.skipped
        assert "dap an" in bundle.skipped["query-p1-15-qa"]

    def test_cau_CHUA_CHAY_thi_KHONG_co_file_chu_khong_phai_file_rong(self, relational):
        """Một CSV rỗng nộp được và ghi 0 điểm y như không nộp, nhưng nó làm
        người vận hành tin là đã nộp đủ. File thiếu thì đếm ra ngay."""
        queries = [
            parse_query("query-p1-1-kis.txt", "a"),
            parse_query("query-p1-2-kis.txt", "b"),
        ]
        results = {"query-p1-1-kis": [cand("L01_V001", 1)]}
        bundle = build_zip(queries, results, relational)
        with zipfile.ZipFile(io.BytesIO(bundle.payload)) as z:
            assert z.namelist() == ["query-p1-1-kis.csv"]
        assert "chua chay" in bundle.skipped["query-p1-2-kis"]

    def test_khong_dong_tieu_de(self, relational):
        queries = [parse_query("query-p1-1-kis.txt", "a")]
        results = {"query-p1-1-kis": [cand("L01_V001", 1)]}
        bundle = build_zip(queries, results, relational)
        with zipfile.ZipFile(io.BytesIO(bundle.payload)) as z:
            body = z.read("query-p1-1-kis.csv").decode("utf-8")
        assert body.splitlines()[0] == "L01_V001,7"

    def test_dap_an_QA_co_dau_phay_duoc_TRICH_DAN(self, relational):
        """`",".join(...)` sẽ tách "5, hoặc 6" thành hai cột và đẩy lệch mọi thứ
        sau nó. `csv` chuẩn thì không."""
        queries = [parse_query("query-p1-15-qa.txt", "bao nhiêu?")]
        results = {"query-p1-15-qa": [cand("L01_V001", 1)]}
        payload = build_zip(
            queries, results, relational, answers={"query-p1-15-qa": "5, hoặc 6"}
        )
        with zipfile.ZipFile(io.BytesIO(payload.payload)) as z:
            body = z.read("query-p1-15-qa.csv").decode("utf-8")
        assert body.splitlines()[0] == 'L01_V001,7,"5, hoặc 6"'

    def test_thu_tu_tay_theo_TUNG_CAU_khong_lan_sang_cau_khac(self, relational):
        queries = [
            parse_query("query-p1-1-kis.txt", "a"),
            parse_query("query-p1-2-kis.txt", "b"),
        ]
        results = {
            "query-p1-1-kis": [cand("L01_V001", 1), cand("L01_V001", 5)],
            "query-p1-2-kis": [cand("L02_V002", 1), cand("L02_V002", 5)],
        }
        payload = build_zip(
            queries, results, relational,
            manual_orders={"query-p1-1-kis": ["L01_V001#000005"]},
        )
        with zipfile.ZipFile(io.BytesIO(payload.payload)) as z:
            first = z.read("query-p1-1-kis.csv").decode("utf-8").splitlines()
            second = z.read("query-p1-2-kis.csv").decode("utf-8").splitlines()
        assert first[0] == "L01_V001,35"   # da keo len dau
        assert second[0] == "L02_V002,7"   # giu nguyen thu tu may
