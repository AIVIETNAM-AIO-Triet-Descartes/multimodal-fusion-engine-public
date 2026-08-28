"""Luật cắt câu hỏi — nó quyết định chuỗi hệ đi tìm, nên nó quyết định bài nộp."""

from __future__ import annotations

import pytest

from mfe.online.query_text import has_description, strip_questions


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
