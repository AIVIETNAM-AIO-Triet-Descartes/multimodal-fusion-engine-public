"""Khoá hành vi của `dung_dau_vao` — phần phụ đề (24-09).

Ba thứ được khoá ở đây đều là quyết định đã đo, không phải sở thích:

  - **Không có thoại thì KHÔNG in nhãn `[speech]`.** In một nhãn rỗng là dạy
    model rằng im lặng có nghĩa. Toàn bộ 296 video nhóm N (camera giao thông)
    là video câm — đếm 24-09 trên volume `keyframes-batch-2-v2`.
  - **Khử trùng trong CÙNG một cụm.** `phu_de_cua` trả một câu whisper ở mọi
    cửa sổ nó chạm, nên một câu dài lặp nguyên văn 3 lần trong một prompt (đo
    trên `L26_V213`: đoạn 700 ký tự nằm ở 3 cửa sổ liền nhau).
  - **Giữ nguyên THỨ TỰ.** Thứ tự nói là thông tin; `set()` sẽ trộn nó.
"""

from __future__ import annotations

from mfe.offline.shot_record_prompt import PROMPT_BAN_GHI_SHOT, dung_dau_vao


def _doan(sid: str, caps: list[str], subs: list[str] | None = None) -> dict:
    d = {"segment_id": sid, "duration_seconds": 15.0, "captions": caps}
    if subs is not None:
        d["subtitles"] = subs
    return d


class TestKhoiSpeech:
    def test_co_thoai_thi_in_khoi_speech(self):
        ra = dung_dau_vao([_doan("V#w0001", ["a man walks"], ["Xin chào quý vị"])])
        assert "[speech] Xin chào quý vị" in ra

    def test_khong_co_thoai_thi_KHONG_in_nhan(self):
        """Video câm: nhãn phải biến mất hẳn, không phải `[speech] ` rỗng."""
        ra = dung_dau_vao([_doan("V#w0001", ["a car stops"], [])])
        assert "[speech]" not in ra

    def test_thieu_han_khoa_subtitles_van_chay(self):
        """Đường gọi cũ (chưa có phụ đề) không được vỡ."""
        ra = dung_dau_vao([_doan("V#w0001", ["a car stops"])])
        assert "[speech]" not in ra
        assert "a car stops" in ra

    def test_chuoi_trang_khong_tao_ra_khoi_rong(self):
        ra = dung_dau_vao([_doan("V#w0001", ["x"], ["   ", ""])])
        assert "[speech]" not in ra


class TestKhuTrungTrongCum:
    def test_cau_vat_qua_hai_cua_so_chi_in_MOT_lan(self):
        cau = "Chúng ta sẽ pha nước xốt như sau"
        ra = dung_dau_vao([_doan("V#w0001", ["a"], [cau]),
                           _doan("V#w0002", ["b"], [cau])])
        assert ra.count(cau) == 1

    def test_khu_trung_khong_nuot_cau_khac(self):
        ra = dung_dau_vao([_doan("V#w0001", ["a"], ["câu một", "câu hai"]),
                           _doan("V#w0002", ["b"], ["câu hai", "câu ba"])])
        assert "câu ba" in ra
        assert ra.count("câu hai") == 1

    def test_giu_nguyen_thu_tu_noi(self):
        ra = dung_dau_vao([_doan("V#w0001", ["a"], ["một", "hai", "ba"])])
        dong = next(x for x in ra.splitlines() if x.startswith("[speech]"))
        assert dong == "[speech] một hai ba"

    def test_khu_trung_khong_vuot_qua_loi_goi(self):
        """Cụm sau là một lượt gọi khác: câu vắt qua ranh giới phải in lại."""
        cau = "nội dung lặp"
        a = dung_dau_vao([_doan("V#w0001", ["a"], [cau])])
        b = dung_dau_vao([_doan("V#w0002", ["b"], [cau])])
        assert cau in a
        assert cau in b


class TestLuatChongBia:
    """Luật 8 là hàng rào duy nhất giữa ASR sai và bản ghi — đừng gỡ lặng lẽ."""

    def test_prompt_cam_lay_ten_rieng_tu_thoai(self):
        assert "NEVER take a proper noun from [speech]" in PROMPT_BAN_GHI_SHOT

    def test_prompt_tach_nghe_khoi_thay(self):
        assert "DO NOT MERGE WHAT IS HEARD INTO WHAT IS SEEN" in PROMPT_BAN_GHI_SHOT

    def test_prompt_co_truong_speech_topic(self):
        assert '"speech_topic"' in PROMPT_BAN_GHI_SHOT

    def test_prompt_chan_cau_phu_dinh(self):
        """Luật 5c: câu phủ định nằm ở ~90% tài liệu là stopword."""
        assert "NEVER WRITE A SENTENCE ABOUT SOMETHING THAT IS ABSENT" in PROMPT_BAN_GHI_SHOT

    def test_prompt_gioi_han_onscreen_text(self):
        """Luật 6b: đo được 33 mục trong MỘT cửa sổ batch 2 trước khi siết."""
        assert '"onscreen_text" IS AT MOST 5 ITEMS' in PROMPT_BAN_GHI_SHOT
