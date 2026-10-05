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


class TestMenhDeTachTay:
    """`run_package(clauses=…)` — cần gạt độ phủ của tầng thứ tự cảnh.

    Bộ luật chỉ đạt ngưỡng ≥3 mệnh đề ở **3/38** đề KIS thật, nên đường người
    tách tay là chỗ tầng này lấy lại độ phủ. Test khoá việc nó **tới được**
    `Cascade.search`, vì một tham số rơi giữa đường sẽ im lặng: giao diện vẫn
    hiện mệnh đề đã gõ, còn truy hồi thì chạy bằng bộ luật.
    """

    class FakeCascade:
        def __init__(self):
            self.calls = []
            self.kwargs = []

        def search(
            self, *, query_text, top_k, run_id, clauses=None,
            query_text_en=None, ocr_keywords=None,
        ):
            self.calls.append((query_text, clauses))
            self.kwargs.append({
                "query_text": query_text, "run_id": run_id,
                "query_text_en": query_text_en, "ocr_keywords": ocr_keywords,
            })
            return [cand("L01_V001", 1)]

    def test_menh_de_tay_di_TOI_cascade(self, relational):
        from mfe.eval.submission_bundle import run_package

        cascade = self.FakeCascade()
        query = parse_query("query-p1-1-kis.txt", "cảnh A rồi cảnh B rồi cảnh C")
        run_package(
            cascade, relational, [query], pool_k=10,
            clauses={"query-p1-1-kis": ["cảnh A", "cảnh B", "cảnh C"]},
        )
        assert cascade.calls == [
            ("cảnh A rồi cảnh B rồi cảnh C", ["cảnh A", "cảnh B", "cảnh C"])
        ]

    def test_khong_tach_tay_thi_truyen_None(self, relational):
        from mfe.eval.submission_bundle import run_package

        cascade = self.FakeCascade()
        query = parse_query("query-p1-1-kis.txt", "một cảnh")
        run_package(cascade, relational, [query], pool_k=10)
        assert cascade.calls == [("một cảnh", None)]

    def test_menh_de_cua_cau_khac_KHONG_lan_sang(self, relational):
        from mfe.eval.submission_bundle import run_package

        cascade = self.FakeCascade()
        queries = [
            parse_query("query-p1-1-kis.txt", "câu một"),
            parse_query("query-p1-2-kis.txt", "câu hai"),
        ]
        run_package(
            cascade, relational, queries, pool_k=10,
            clauses={"query-p1-2-kis": ["a", "b", "c"]},
        )
        assert cascade.calls == [("câu một", None), ("câu hai", ["a", "b", "c"])]


class TestChuoiTimGhiDe:
    """`run_package(searches=…)` — ô ghi đè chuỗi tìm ở bàn nộp.

    Đo được vì sao cần: pool 500 phủ **40/47** câu QA dev khi tìm bằng
    `retrieval_text` viết tay, so với **32/47** khi tìm bằng nguyên văn câu đề —
    và `PackagedQuery` không có `retrieval_text`. Test khoá hai thứ: chuỗi ghi đè
    **tới được** cascade, và `run_id` **không** đổi theo nó (đổi là đổi tên file
    nộp).
    """

    def test_chuoi_ghi_de_thay_query_text_nhung_giu_run_id(self, relational):
        from mfe.eval.submission_bundle import SearchText, run_package

        cascade = TestMenhDeTachTay.FakeCascade()
        query = parse_query("query-p1-9-qa.txt", "Xe lội nước. Biển ghi số mấy?")
        run_package(
            cascade, relational, [query], pool_k=10,
            searches={"query-p1-9-qa": SearchText(vi="Xe lội nước")},
        )
        call = cascade.kwargs[0]
        assert call["query_text"] == "Xe lội nước"
        # run_id di thang vao evidence_id VA quyet dinh ten file nop.
        assert call["run_id"] == "query-p1-9-qa"

    def test_tieng_anh_dua_tay_di_TOI_cascade(self, relational):
        """Nửa quan trọng hơn: envit5 chính là một trong hai nguyên nhân."""
        from mfe.eval.submission_bundle import SearchText, run_package

        cascade = TestMenhDeTachTay.FakeCascade()
        query = parse_query("query-p1-9-qa.txt", "Xe ô tô lội nước")
        run_package(
            cascade, relational, [query], pool_k=10,
            searches={
                "query-p1-9-qa": SearchText(
                    vi="Xe ô tô lội nước", en="cars wading through water",
                    ocr_keywords="biển báo",
                )
            },
        )
        call = cascade.kwargs[0]
        assert call["query_text_en"] == "cars wading through water"
        assert call["ocr_keywords"] == "biển báo"

    def test_khong_ghi_de_thi_dung_nguyen_van_va_KHONG_dat_tieng_anh(self, relational):
        """`None` phải tới cascade, không phải chuỗi rỗng.

        Chuỗi rỗng sẽ được `Cascade` hiểu là "đã có bản dịch" và tầng dịch ngồi
        ngoài — arm `siglip` mất luôn, mà nó là arm trọng số 2,0.
        """
        from mfe.eval.submission_bundle import run_package

        cascade = TestMenhDeTachTay.FakeCascade()
        query = parse_query("query-p1-9-qa.txt", "Xe lội nước")
        run_package(cascade, relational, [query], pool_k=10)
        call = cascade.kwargs[0]
        assert call["query_text"] == "Xe lội nước"
        assert call["query_text_en"] is None
        assert call["ocr_keywords"] is None

    def test_ghi_de_cua_cau_khac_KHONG_lan_sang(self, relational):
        from mfe.eval.submission_bundle import SearchText, run_package

        cascade = TestMenhDeTachTay.FakeCascade()
        queries = [
            parse_query("query-p1-1-kis.txt", "câu một"),
            parse_query("query-p1-2-kis.txt", "câu hai"),
        ]
        run_package(
            cascade, relational, queries, pool_k=10,
            searches={"query-p1-2-kis": SearchText(vi="ghi đè hai")},
        )
        assert [c["query_text"] for c in cascade.kwargs] == ["câu một", "ghi đè hai"]


class TestCauTuGoKhongVaoZip:
    """Câu `tim-tay-*` là để dò, không phải đề BTC — nó không được vào gói nộp.

    Tên file nộp phải là tên đề BTC ra. Một file `tim-tay-1-qa.csv` trong ZIP là
    thứ BTC không hỏi, và nó lọt vào im lặng vì câu tự gõ đi qua đúng
    `parse_query` như đề thật.
    """

    def test_tim_tay_bi_loai_khoi_zip_va_co_ly_do(self, relational):
        real = parse_query("query-p1-1-kis.txt", "cảnh thật")
        adhoc = parse_query("tim-tay-1-kis.txt", "dò thử")
        results = {
            "query-p1-1-kis": [cand("L01_V001", 1)],
            "tim-tay-1-kis": [cand("L01_V002", 2)],
        }
        bundle = build_zip([real, adhoc], results, relational)
        with zipfile.ZipFile(io.BytesIO(bundle.payload)) as archive:
            assert archive.namelist() == ["query-p1-1-kis.csv"]
        # Bien mat KEM ly do: mot cau vang mat khong loi giai thich trong y het
        # mot cau chay hong.
        assert "tim-tay-1-kis" in bundle.skipped
        assert "khong vao goi nop" in bundle.skipped["tim-tay-1-kis"]

    def test_tim_tay_bi_loai_ke_ca_khi_chay_ra_ket_qua_tot(self, relational):
        """Loại theo TÊN, không theo việc nó có kết quả hay không."""
        adhoc = parse_query("tim-tay-3-qa.txt", "dò thử")
        bundle = build_zip(
            [adhoc], {"tim-tay-3-qa": [cand("L01_V001", i) for i in range(1, 51)]},
            relational, answers={"tim-tay-3-qa": "một đáp án hợp lệ"},
        )
        with zipfile.ZipFile(io.BytesIO(bundle.payload)) as archive:
            assert archive.namelist() == []
