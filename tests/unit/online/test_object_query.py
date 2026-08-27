"""Khoá bộ viết lại truy vấn cho arm `object` — `mfe.online.object_query`.

Đây là **nửa còn lại** của thay đổi ở `mfe.offline.object_text`, và nếu hai nửa
lệch nhau thì chỉ mục và truy vấn **không còn một token chung nào** — đúng bài
học `labels_vi`, lần này tự ta gây ra. Không có gì báo: arm vẫn chạy, vẫn trả
về hạng, chỉ là hạng của một phép khớp không xảy ra.

Nên test đầu tiên ở đây soi chiếu HAI nửa với nhau, chứ không kiểm từng nửa.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfe.offline.object_text import Detection, render
from mfe.online.object_query import BLOCKED_COMPOUNDS, QUERY_REWRITERS, rewrite


def _extra(query: str) -> list[str]:
    """Chỉ phần token ghép được thêm vào."""
    return rewrite(query)[len(query) :].split()


class TestHaiNuaGapNhau:
    """Test quan trọng nhất file này: token truy vấn phải NẰM TRONG chỉ mục."""

    @pytest.mark.parametrize(
        ("query", "detections"),
        [
            ("ba người", [Detection("Person", "giữa", 24)] * 3),
            ("hai xe máy", [Detection("Motorcycle", "giữa", 24)] * 2),
            # Đồng nghĩa: truy vấn gõ "mô tô", chỉ mục ghi dạng chính "xe máy".
            ("hai mô tô", [Detection("Motorcycle", "giữa", 24)] * 2),
            ("người bên trái", [Detection("Person", "trái", 21)]),
            ("bên phải là hai chiếc xe máy",
             [Detection("Motorcycle", "phải", 27)] * 2),
        ],
    )
    def test_moi_token_truy_van_deu_co_trong_chi_muc(self, query, detections):
        indexed = set(render(detections).split())
        produced = _extra(query)
        assert produced, f"khong sinh token nao cho {query!r}"
        missing = [t for t in produced if t not in indexed]
        assert not missing, f"truy van sinh {missing} ma chi muc khong co"

    def test_dong_nghia_quy_ve_CUNG_dang_chinh(self):
        assert _extra("hai mô tô") == _extra("hai xe máy") == ["hai_xe_máy"]


class TestTrattTuTiengViet:
    def test_so_dem_dung_TRUOC_nhan(self):
        assert "ba_người" in _extra("ba người đang đứng")

    def test_mau_dung_SAU_nhan(self):
        assert "đỏ_mũ" in _extra("người đội mũ đỏ")

    def test_vi_tri_bat_ca_HAI_chieu(self):
        assert "trái_người" in _extra("người bên trái")
        assert "trái_người" in _extra("bên trái là một người")

    def test_tu_chen_giua_van_ghep_duoc(self):
        """`"ba chiếc xe máy"` — `chiếc` chen vào giữa số đếm và nhãn."""
        assert "ba_xe_máy" in _extra("ba chiếc xe máy")

    def test_qua_xa_thi_KHONG_ghep(self):
        """`"ba người đứng cạnh một chiếc xe tải lớn"` không được sinh
        `ba_xe_tải` — đó là một ràng buộc đề không hề nêu."""
        produced = _extra("ba người đang đứng ở cạnh một chiếc xe tải lớn")
        assert not any(t.startswith("ba_xe") for t in produced)


class TestKhongBIA_RANG_BUOC:
    """Ghép nhầm không làm arm trượt — nó làm arm khớp SAI, tệ hơn nhiều."""

    def test_tu_ghep_tieng_Viet_khong_bi_doc_thanh_nhan(self):
        # `tay đua` là vận động viên, không phải bàn tay.
        assert not any(t.endswith("_tay") for t in _extra("một tay đua áo vàng"))
        # `mũi tên` là arrow, không phải cái mũi (đề dot-1/23).
        assert not any(t.endswith("_mũi") for t in _extra("mũi tên vàng"))

    def test_mau_KHONG_nhin_toi_truoc(self):
        """Tiếng Việt đặt màu SAU danh từ. Cho phép nhìn tới trước thì
        `"con lân trắng, mũi đỏ"` sinh `trắng_mũi` — trắng thuộc về con lân."""
        assert "trắng_mũi" not in _extra("con lân trắng, mũi đỏ")

    def test_xanh_la_cay_la_MOT_MAU_khong_phai_mau_cong_cay(self):
        assert "lá_cây" not in _extra("phông nền xanh lá cây")

    def test_KHONG_la_phu_dinh_khong_phai_so_khong(self):
        """Đề dot-1/15: *"Không tính bảng chú giải…"*."""
        assert not any(t.startswith("không_") for t in _extra("không tính bảng trắng"))


class TestKhongPhaHongDuongCu:
    def test_truy_van_khong_co_nhan_thi_GIU_NGUYEN(self):
        query = "cảnh sạt lở đèo lúc trời mưa"
        assert rewrite(query) == query

    def test_truy_van_rong(self):
        assert rewrite("") == ""
        assert rewrite("   ") == "   "

    def test_token_GOC_van_con(self):
        """Chỉ mục của khung không có màu vẫn phải khớp được phần trần."""
        out = rewrite("ba người")
        assert "ba người" in out and "ba_người" in out

    def test_chi_arm_object_duoc_viet_lai(self):
        """Ghép token vào `ocr` hay `subtitle` sẽ sinh token không nguồn nào có."""
        assert set(QUERY_REWRITERS) == {"object"}


class TestTrenDeThat:
    def test_chay_duoc_tren_49_de_chinh_thuc(self):
        """Không câu nào được ném, và phải có câu sinh ra token.

        Con số đo ngày 26-08-2026: **40/49** đề sinh ít nhất một token ghép.
        Ngưỡng đặt lỏng hơn — nó canh bộ viết lại CHẾT, không canh vốn từ.
        """
        files = sorted(Path("docs/context").rglob("query-*.txt"))
        if not files:  # pragma: no cover - bo de chi co tren may Lead
            pytest.skip("khong co bo de")
        produced = sum(1 for f in files if _extra(f.read_text(encoding="utf-8")))
        assert produced > len(files) // 2

    def test_moi_cum_bi_chan_deu_dung_hai_tieng(self):
        """`_find_labels` tra `tokens[i:i+size+1]`, tức luôn là nhãn một tiếng
        cộng một tiếng nữa. Một cụm ba tiếng ở đây sẽ không bao giờ khớp."""
        assert all(len(c) == 2 for c in BLOCKED_COMPOUNDS)
