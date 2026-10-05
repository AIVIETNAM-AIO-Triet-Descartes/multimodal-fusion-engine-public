"""Luật cắt câu hỏi — nó quyết định chuỗi hệ đi tìm, nên nó quyết định bài nộp."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mfe.online.query_text import (
    Rewrite,
    has_description,
    quest_apply,
    quest_rewrite,
    strip_questions,
)


def test_giu_mo_ta_bo_cau_hoi():
    """Dạng phổ biến nhất của đề QA: `<mô tả>. <câu hỏi>?`"""
    text = (
        "Đoạn phim ghi lại cảnh những chiếc xe ô tô lội nước, chiếc xe màu vàng, "
        "màu đỏ và màu đen lần lượt chuẩn bị đi qua cầu. Con số được ghi trên "
        "biển báo bên trái của cây cầu là bao nhiêu?"
    )
    out = strip_questions(text)
    assert out.endswith("chuẩn bị đi qua cầu.")
    assert "bao nhiêu" not in out


def test_cau_de_chi_co_mot_cau_hoi_thi_GIU_NGUYEN():
    """Cắt hết thì trả nguyên văn — tìm bằng chuỗi rỗng cho ra 0 ứng viên.

    Không phải trường hợp hiếm: phần lớn câu QA dev thuộc loại này.
    """
    text = "Theo slide bài giảng Vật lí, hạt nhân nguyên tử gồm hai loại hạt nào?"
    assert strip_questions(text) == text
    assert not has_description(text)


def test_bo_nhieu_cau_hoi_lien_tiep():
    text = "Một người đàn ông đứng trước cổng. Anh ta mặc áo gì? Màu gì?"
    assert strip_questions(text) == "Một người đàn ông đứng trước cổng."


def test_cau_hoi_o_giua_cung_bi_bo():
    """Bỏ theo dấu chấm hỏi, không theo vị trí — đề BTC không cố định thứ tự."""
    text = "Cảnh quay bắt đầu ở bờ biển. Có bao nhiêu người? Sau đó máy lia lên trời."
    out = strip_questions(text)
    assert out == "Cảnh quay bắt đầu ở bờ biển. Sau đó máy lia lên trời."


def test_has_description_phan_biet_duoc_hai_loai():
    """Giao diện cần nó để vô hiệu nút — bấm xong không đổi gì trông như nút hỏng."""
    assert has_description("Cảnh bờ biển. Có mấy người?")
    assert not has_description("Có mấy người?")


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_chuoi_rong_khong_lam_no(text):
    assert strip_questions(text) == ""


class TestQuestRewrite:
    """`quest_rewrite` — lớp bọc BẮT được lúc quest im lặng rơi về câu hỏi gốc.

    `quest_decompose` trả về chính câu hỏi khi API hỏng, không log gì. Một nút
    bấm xong không đổi gì trông y hệt một nút chạy rồi mà vô ích, nên toàn bộ
    giá trị của lớp này nằm ở chỗ phân biệt hai thứ đó.
    """

    QUESTION = "Cảnh xe lội nước qua cầu. Biển báo ghi số mấy?"

    def test_phan_vai_hai_chuoi(self):
        """`search_hint` là TIẾNG ANH -> ô `en`; `ocr_keywords` tiếng Việt -> ô `vi`."""
        out = quest_rewrite(self.QUESTION, decompose=lambda _: {
            "search_hint": "cars crossing a bridge over water with a height sign",
            "ocr_keywords": "biển báo chiều cao cầu",
            "logic_question": "Biển báo ghi số mấy?",
        })
        assert out.ok
        assert out.en == "cars crossing a bridge over water with a height sign"
        assert out.vi == "biển báo chiều cao cầu"
        assert out.reason is None

    def test_quest_tra_ve_chinh_cau_hoi_la_HONG(self):
        """Đường dự phòng của `quest_decompose`, phải bị bắt chứ không đi tiếp."""
        out = quest_rewrite(self.QUESTION, decompose=lambda q: {
            "search_hint": q, "ocr_keywords": q, "logic_question": q,
        })
        assert not out.ok
        assert out.reason is not None
        assert "API hong" in out.reason
        # Vẫn phải có chuỗi dùng được: rơi về đường luật tự cắt.
        assert out.vi == "Cảnh xe lội nước qua cầu."
        assert out.en is None

    def test_search_hint_rong_cung_la_HONG(self):
        out = quest_rewrite(self.QUESTION, decompose=lambda _: {"search_hint": ""})
        assert not out.ok
        assert out.vi == "Cảnh xe lội nước qua cầu."

    def test_nem_thi_tra_ly_do_chu_khong_lan_ra(self):
        def boom(_):
            raise RuntimeError("429 het quota")

        out = quest_rewrite(self.QUESTION, decompose=boom)
        assert not out.ok
        assert out.reason is not None
        assert "RuntimeError" in out.reason and "429" in out.reason
        assert out.vi == "Cảnh xe lội nước qua cầu."

    def test_thieu_ocr_keywords_thi_dung_ban_tu_cat_cho_o_tieng_viet(self):
        """Quest được phép trả `ocr_keywords` rỗng — đó là câu không có chữ trên màn."""
        out = quest_rewrite(self.QUESTION, decompose=lambda _: {
            "search_hint": "cars crossing a bridge", "ocr_keywords": "",
        })
        assert out.ok
        assert out.vi == "Cảnh xe lội nước qua cầu."
        assert out.en == "cars crossing a bridge"


class TestQuestApply:
    """`quest_apply` — chọn PHÉP đúng theo dạng đề.

    Ba việc của `mfe.online.quest` khác nhau, và dùng nhầm việc thì hại chứ
    không phải vô ích: đem `quest_decompose` (vốn để tách một câu HỎI) vào một
    câu KIS là viết lại một mô tả thị giác vốn đã đúng thể loại SigLIP ăn.
    """

    KIS = "Cảnh trong bộ phim của đạo diễn Steven Spielberg năm 1975."
    QA = "Cảnh xe lội nước qua cầu. Biển báo ghi số mấy?"

    def test_QA_di_duong_viet_lai(self):
        goi = []

        def rewrite(text):
            goi.append(text)
            return Rewrite(vi="biển báo", en="a height sign", ok=True)

        out = quest_apply(self.QA, "QA", rewrite=rewrite, expand=_no_expand)
        assert goi == [self.QA]
        assert out.ok and out.en == "a height sign"

    def test_KIS_di_duong_MO_RONG_va_NOI_VAO_CUOI(self):
        """Nối vào cuối, KHÔNG thay thế — câu gốc vẫn mang phần mô tả thị giác."""
        out = quest_apply(
            self.KIS, "KIS", rewrite=_no_rewrite,
            expand=lambda _: [_Term(vi="cá mập", en="shark")],
        )
        assert out.ok
        assert out.vi.startswith(self.KIS)
        assert "cá mập" in out.vi
        assert "cá mập" in (out.reason or "")

    def test_TRAKE_dung_chung_duong_voi_KIS(self):
        out = quest_apply(
            "Chuỗi động tác nhảy cao.", "TRAKE", rewrite=_no_rewrite,
            expand=lambda _: [_Term(vi="xà ngang", en="crossbar")],
        )
        assert out.ok and "xà ngang" in out.vi

    def test_KHONG_goi_ten_them_gi_la_ok_False_co_LY_DO(self):
        """42/49 đề thật không cần thêm gì — rỗng là câu trả lời ĐÚNG.

        Nhưng nó phải NÓI RA: một nút bấm xong không đổi gì trông y hệt một nút
        hỏng, và người vận hành sẽ đi sửa khoá API thay vì đi tìm tiếp.
        """
        out = quest_apply(self.KIS, "KIS", rewrite=_no_rewrite, expand=lambda _: [])
        assert not out.ok
        assert out.vi == self.KIS
        assert out.reason and "khong goi ten" in out.reason

    def test_expand_NEM_thi_van_tra_ve_cau_goc(self):
        def boom(_):
            raise RuntimeError("het quota")

        out = quest_apply(self.KIS, "KIS", rewrite=_no_rewrite, expand=boom)
        assert not out.ok
        assert out.vi == self.KIS
        assert "het quota" in (out.reason or "")

    @pytest.mark.parametrize("text", ["", "   "])
    def test_cau_rong_khong_goi_LLM(self, text):
        out = quest_apply(text, "KIS", rewrite=_no_rewrite, expand=_no_expand)
        assert not out.ok and out.reason == "cau rong"


@dataclass(frozen=True)
class _Term:
    vi: str
    en: str
    why: str = ""


def _no_rewrite(_text):
    raise AssertionError("khong duoc di duong viet lai o day")


def _no_expand(_text):
    raise AssertionError("khong duoc di duong mo rong o day")
