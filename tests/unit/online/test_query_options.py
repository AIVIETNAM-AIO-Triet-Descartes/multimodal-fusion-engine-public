"""Bộ tuỳ chọn có phạm vi — mfe.online.query_options.

Nó quyết định arm nào và tầng rerank nào chạy cho câu nào, tức nó quyết định
**nội dung file nộp**. Đó là lý do nó ở thư viện chứ không nằm trong
`app/submit_lab.py`: script Streamlit chạy mã lúc import nên không test nào với
tới được.
"""

from __future__ import annotations

from mfe.online.query_options import QUERY_TYPES, Options, resolve, scope_keys


class TestGop:
    def test_lop_sau_de_len_lop_truoc(self):
        goi = Options(tiers=("vlm",), scene_order=0.5)
        cau = Options(tiers=(), scene_order=0.0)
        assert goi.merge(cau) == Options(arms={}, tiers=(), scene_order=0.0)

    def test_None_KE_THUA_chu_khong_XOA(self):
        """Ô bỏ trống ở cấp hẹp phải kế thừa cấp rộng.

        Nếu `None` ghi đè thì: đặt `vlm` cho cả gói, mở bảng dạng KIS ra xem rồi
        đóng lại, và `vlm` biến mất khỏi KIS mà không có gì báo.
        """
        goi = Options(tiers=("vlm",), scene_order=0.5)
        assert goi.merge(Options()) == goi

    def test_arms_gop_THEO_KHOA_chu_khong_thay_ca_dict(self):
        """Phạm vi hẹp thường chỉ muốn nói về một arm; thay cả dict sẽ lặng lẽ
        tắt mọi arm nó không nhắc tới."""
        goi = Options(arms={"siglip": True, "ocr": True, "clap": False})
        cau = Options(arms={"ocr": False})
        assert goi.merge(cau).arms == {"siglip": True, "ocr": False, "clap": False}

    def test_tat_TUONG_MINH_van_de_duoc_len_bat(self):
        """`False` khác `None`: tắt tường minh phải thắng cấp trên."""
        assert Options(arms={"clap": True}).merge(
            Options(arms={"clap": False})
        ).arms == {"clap": False}


class TestPhamVi:
    def test_ba_khoa_tu_RONG_toi_HEP(self):
        assert scope_keys("query-p1-1-kis", "KIS") == (
            "goi", "dang_KIS", "cau_query-p1-1-kis",
        )

    def test_cau_thang_dang_thang_goi(self):
        store = {
            "goi": Options(tiers=("vlm",), scene_order=0.5),
            "dang_KIS": Options(scene_order=0.9),
            "cau_q1": Options(tiers=()),
        }
        got = resolve(store, "q1", "KIS")
        assert got.tiers == ()      # cau thang
        assert got.scene_order == 0.9   # dang thang goi

    def test_khoa_thieu_thi_coi_nhu_chua_chon(self):
        store = {"goi": Options(tiers=("vlm",))}
        assert resolve(store, "chua-tung-thay", "TRAKE").tiers == ("vlm",)

    def test_store_rong_khong_no(self):
        assert resolve({}, "q", "KIS") == Options()

    def test_dang_khong_lan_sang_nhau(self):
        store = {"dang_KIS": Options(tiers=("vlm",)), "dang_TRAKE": Options(tiers=())}
        assert resolve(store, "a", "KIS").tiers == ("vlm",)
        assert resolve(store, "b", "TRAKE").tiers == ()
        assert resolve(store, "c", "QA").tiers is None


def test_ba_dang_khop_hau_to_ten_file():
    """`QUERY_TYPES` phải khớp thứ `parse_query` sinh ra, nếu không một dạng sẽ
    không có tab nào và câu đó biến mất khỏi giao diện."""
    from mfe.eval.query_package import parse_query

    for suffix, expected in (("kis", "KIS"), ("qa", "QA"), ("trake", "TRAKE")):
        got = parse_query(f"query-p1-1-{suffix}.txt", "E1 mot canh nao do")
        assert got.type == expected
        assert expected in QUERY_TYPES


class TestQuest:
    """`quest` là cần gạt có phạm vi, và `None` phải KẾ THỪA chứ không TẮT.

    Cùng bẫy như `tiers`: nếu `None` ghi đè thì bật quest cho cả gói, mở bảng
    của một câu ra xem rồi đóng lại, và quest tắt mất ở câu đó mà không có gì
    báo. Ở đây hậu quả cụ thể hơn — câu đó đi tìm bằng một chuỗi khác hẳn.
    """

    def test_None_ke_thua_cap_tren(self):
        assert Options(quest=True).merge(Options()).quest is True

    def test_False_TUONG_MINH_de_duoc_len_True(self):
        """Tắt tường minh khác với chưa chọn — nếu không thì không tắt lại được."""
        assert Options(quest=True).merge(Options(quest=False)).quest is False

    def test_mac_dinh_la_chua_chon(self):
        assert Options().quest is None

    def test_resolve_theo_day_goi_dang_cau(self):
        store = {
            "goi": Options(quest=True),
            "dang_QA": Options(),
            "cau_q1": Options(quest=False),
        }
        assert resolve(store, "q1", "QA").quest is False
        assert resolve(store, "q2", "QA").quest is True
