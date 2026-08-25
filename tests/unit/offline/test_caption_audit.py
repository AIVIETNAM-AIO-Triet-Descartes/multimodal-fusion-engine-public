"""Khoá phép đếm năng lực của caption — `mfe.offline.caption_audit`.

Phép đếm này quyết định câu "prompt mới có hoạt động không", và nó được đọc trên
307 ảnh probe **trước khi** ai đó bỏ ~95 giờ GPU. Một phép đếm thổi phồng ở đây
sẽ cho phép một prompt hỏng đi qua cổng.

Đó không phải giả thuyết: bản đầu của module này dùng `word in text` và đo được
**72,2%** caption **tiếng Anh** của Bảo "có màu" — vì `"be"` nằm trong "before"
và `"cam"` nằm trong "camera". Con số sai trông hợp lý hơn con số đúng (1,2%),
nên nó sẽ không tự lộ ra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfe.offline.caption_audit import (
    audit,
    format_audit,
    has_banned_opener,
    has_camera,
    has_colour,
    has_count,
    has_position,
    has_relation,
    looks_vietnamese,
)


class TestQuanHeGiuaCacVat:
    """Điều khoản số 1 của biến thể `aic`: caption phải nói các vật LÀM GÌ VỚI NHAU."""

    @pytest.mark.parametrize(
        "caption",
        [
            "a man in a blue shirt waters a row of potted plants",
            "an adult points at an open notebook held by a child",
            "người đàn ông đang tưới mấy chậu cây",
            "cô giáo chỉ bài cho học trò",
            "a woman handed a bowl of noodles to a customer",
        ],
    )
    def test_bat_duoc_tuong_tac(self, caption):
        assert has_relation(caption)

    @pytest.mark.parametrize(
        "caption",
        [
            "a man, a green hose, and several potted plants",
            "there are two people and a notebook in the frame",
            "hai người đứng cạnh một chiếc bàn trắng",
            "three men wear dark suits and stand in a row",
        ],
    )
    def test_LIET_KE_khong_phai_quan_he(self, caption):
        """`stand`, `wear`, `đứng`, `mặc` CỐ Ý không tính.

        Thêm chúng vào là biến phép đo thành hằng số 100% — mọi caption đều có
        người đứng hoặc mặc gì đó.
        """
        assert not has_relation(caption)

    @pytest.mark.parametrize(
        "caption",
        [
            "brown water covers the road",            # water = DANH tu
            "a tall building behind the market",      # building = DANH tu
            "the presenter wears a dark tie",         # tie = DANH tu
            "his hands rest on the desk",             # hands = DANH tu
            "rows of plants in the field",            # plants = DANH tu
        ],
    )
    def test_DANH_TU_trung_dong_tu_khong_duoc_tinh(self, caption):
        """Đo được trên 855 caption của Bảo: khớp cả dạng gốc thì `water` 16,1%,
        `tie` 10,6%, `build` 10,4%, `hand` 8,9% — toàn bộ là danh từ, và chúng
        thổi "quan hệ" từ 50,5% lên 71,3%."""
        assert not has_relation(caption)


class TestKhopTheoRanhGioiTu:
    """Chuỗi con là cái bẫy, và nó chỉ lộ ra khi đo trên văn bản tiếng Anh.

    Danh sách giờ SONG NGỮ, nên "tiếng Anh thì không có màu" không còn đúng — và
    đó là chủ ý. Thứ các ca dưới đây khoá là chuyện khác: một từ khoá tiếng Việt
    không được khớp vì nó tình cờ nằm TRONG một từ tiếng Anh.
    """

    @pytest.mark.parametrize(
        "english",
        [
            "The camera is placed before the desk",   # "cam" (mau) trong "camera"
            "Two people can be seen between the cars",  # "be" (mau be) trong "between"
            "A man in a dark suit",                   # "dark" la do sang, khong phai mau
        ],
    )
    def test_tu_tieng_Viet_nam_trong_tu_tieng_Anh_khong_duoc_tinh(self, english):
        assert not has_colour(english)

    @pytest.mark.parametrize(
        "english",
        [
            "a rider in a yellow jersey and black shorts",
            "the blue plastic drums behind the stall",
        ],
    )
    def test_mau_TIENG_ANH_that_thi_van_bat_duoc(self, english):
        """Không có nhánh này thì phép đo mù trước chính dữ liệu sắp sinh ra.

        Bản chỉ-tiếng-Việt đọc 855 caption tiếng Anh của Bảo ra `màu 1,2%`, và
        `0%` trông y hệt một kết luận chứ không giống một lỗi.
        """
        assert has_colour(english)

    def test_tu_mau_that_thi_van_bat_duoc(self):
        assert has_colour("tay đua áo vàng quần đen")
        assert has_colour("chiếc xe màu xanh dương")
        assert has_colour("nồi đất trên bếp, đĩa thức ăn màu vàng cam")

    def test_cum_nhieu_tu_cung_khop_theo_ranh_gioi(self):
        assert has_position("người phụ nữ bên trái cầm ly")
        assert has_position("ở giữa là một người mặc áo xanh lá")
        assert not has_position("beneath the bridge")

    def test_dem_bat_ca_chu_lan_so(self):
        assert has_count("ba người đội nón đỏ")
        assert has_count("nhóm hơn 5 người xếp hàng")   # bắt bằng chữ SỐ
        assert has_count("một số người đứng xem")       # "một" vẫn là từ đếm
        assert not has_count("người đứng xem")

    def test_goc_may(self):
        assert has_camera("cận cảnh bàn tay cầm dao")
        assert has_camera("quay từ trên cao xuống")
        assert not has_camera("người đàn ông đứng cạnh xe")


class TestMoDauBiCam:
    @pytest.mark.parametrize(
        "caption",
        [
            "Hình ảnh cho thấy hai người đứng",
            "hình ảnh cho thấy hai người đứng",
            "(The image shows two anchors)",
            "Trong hình có ba chiếc xe",
        ],
    )
    def test_bat_duoc(self, caption):
        assert has_banned_opener(caption)

    def test_vao_thang_noi_dung_thi_khong_bi_tinh(self):
        assert not has_banned_opener("Hai người dẫn chương trình đứng sau bàn dẫn")

    def test_khong_bat_khi_cum_do_nam_GIUA_cau(self):
        """Cấm là cấm MỞ ĐẦU. Một câu nhắc tới 'bức ảnh' ở giữa thì không sao —
        thứ làm hỏng idf là cụm lặp lại ở ĐẦU mọi caption."""
        assert not has_banned_opener("Một người cầm bức ảnh gia đình")


class TestNhanDangTiengViet:
    def test_co_dau_thi_la_tieng_viet(self):
        assert looks_vietnamese("Hai người đàn ông ngồi cạnh nhau")

    def test_tieng_anh_thuan_thi_khong(self):
        assert not looks_vietnamese("Two news anchors standing behind a desk")


class TestBangTongHop:
    def test_bo_rong_khong_nem(self):
        assert audit([]) == {"n": 0}
        assert "khong co du lieu" in format_audit({"n": 0}, "x")

    def test_dem_dung_ti_le(self):
        pairs = [
            ("áo vàng quần đen", ""),
            ("hai người đứng", ""),
            ("người đàn ông", ""),
            ("bên trái là cái ghế", ""),
        ]
        report = audit(pairs)
        assert report["n"] == 4
        assert report["mau"] == pytest.approx(0.25)
        assert report["dem"] == pytest.approx(0.25)
        assert report["vi_tri"] == pytest.approx(0.25)

    def test_dem_so_dong_cua_nua_TEXT(self):
        """Mỗi vùng chữ rời nhau phải là MỘT DÒNG.

        Gộp logo, đồng hồ và dòng chạy chân màn hình vào một dòng là mất ranh
        giới giữa chúng, và BM25 sẽ khớp một cụm vắt qua hai vùng không liên quan.
        """
        report = audit([("x", "HTV9 HD\n06:31:14\nTIN CHÍNH"), ("y", "MỘT DÒNG"), ("z", "")])
        assert report["co_chu"] == pytest.approx(2 / 3)
        assert report["ocr_nhieu_dong"] == pytest.approx(1 / 3)
        assert report["ocr_dong_trung_vi"] == 2   # trung vi cua [3, 1]

    def test_chay_tren_855_cau_that_cua_Bao_va_ra_con_so_TRUNG_THUC(self):
        """Mốc nền, và nó là chỗ một kết luận sai đã bị bắt.

        Bản đầu của phép đo chỉ có từ khoá tiếng Việt. Đem nó đo 855 caption
        **tiếng Anh** này thì ra `màu 1,2% · vị trí 0,1%`, và tôi đã đọc con số
        đó như "caption của Bảo không nêu màu". Sai: nó chỉ có nghĩa bộ dò không
        biết đọc tiếng Anh. Đo lại bằng danh sách song ngữ ra `màu 78,6% ·
        vị trí 59,2% · quan hệ 50,5%`.

        Các ngưỡng dưới đây khoá cả hai phía. Sàn bắt phép đo mù ngôn ngữ quay
        lại; trần bắt phép đo thổi phồng (đúng lỗi `word in text` và lỗi khớp
        dạng gốc động từ đã mắc — `water`/`build`/`hand` là danh từ).
        """
        source = Path("notebooks/reference/dense_captions_qwen.json")
        if not source.is_file():  # pragma: no cover - file cua may Lead
            pytest.skip("khong co ban chay cua Bao")
        from mfe.offline.stages import parse_qwen25_caption_ocr

        payload = json.loads(source.read_text(encoding="utf-8"))
        report = audit(parse_qwen25_caption_ocr(v)[:2] for v in payload.values())
        assert report["n"] == 855
        assert 0.70 < report["mau"] < 0.90
        assert 0.50 < report["vi_tri"] < 0.70
        assert 0.40 < report["quan_he"] < 0.65
        # Mo dau sao mon: con so nay la LY DO doi prompt, khong phai chi tiet phu.
        assert report["mo_dau_cam"] > 0.50
        assert report["tieng_viet"] < 0.20
        # Nua `[TEXT]` thi NGUOC LAI: no chep chu tren man hinh, phan lon tieng Viet.
        assert report["co_chu"] > 0.90
        assert report["ocr_nhieu_dong"] > 0.90
