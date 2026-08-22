"""Khoá bộ lọc keyframe lân cận — mfe.offline.dedup_keyframes.

Bộ lọc này **xoá dữ liệu**, nên mọi test ở đây đều về chuyện xoá nhầm: một chữ ký
đổi giá trị giữa hai lần chạy, một chuỗi trôi chậm bị cắt mất khúc giữa, một
ngưỡng nuốt nội dung thật. Sai theo hướng giữ lại thì tốn đĩa; sai theo hướng bỏ
đi thì mất keyframe và không có gì báo.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfe.offline.dedup_keyframes import (
    PHASH_KEEP,
    embedding_filter,
    hamming,
    phash,
    signature_filter,
    sweep_embedding,
    sweep_signature,
)


def gradient(width: int = 64, height: int = 48, shift: int = 0) -> np.ndarray:
    """Ảnh BGR có cấu trúc — vài tần số chồng lên nhau, không phải dốc tuyến tính.

    **Đừng thay bằng một gradient tuyến tính.** DCT của dốc tuyến tính có rất
    nhiều hệ số bằng đúng nhau nằm sát ngưỡng median, nên vài sai số float cuối
    cùng lật bit và `test_doi_do_sang_khong_doi_chu_ky` trượt — trên ảnh tổng hợp
    đó khoảng cách là 8 bit, trong khi trên keyframe thật (đo trên `L21_V001`)
    cùng phép tăng sáng cho 0–4 bit. Fixture sai, không phải hàm sai.

    Cũng đừng thay bằng nhiễu ngẫu nhiên: pHash trên nhiễu cho bit gần như ngẫu
    nhiên, nên test sẽ "đạt" ngay cả khi phép biến đổi hỏng.
    """
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    x = np.roll(x, shift, axis=1)
    plane = (
        90
        + 40 * np.sin(2 * np.pi * x / width)
        + 25 * np.cos(2 * np.pi * y / height)
        + 15 * np.sin(6 * np.pi * (x + y) / width)
    )
    plane = np.clip(plane, 0, 200)
    return np.stack([plane, plane, plane], axis=2).astype(np.uint8)


class TestChuKy:
    def test_dai_dung_64_bit(self):
        value = phash(gradient())
        assert 0 <= value < 2 ** (PHASH_KEEP * PHASH_KEEP - 1)

    def test_cung_anh_cung_chu_ky(self):
        """Tất định. Một chữ ký đổi giữa hai lần chạy làm mọi so sánh đã lưu
        thành vô nghĩa, và nó hỏng trong im lặng."""
        image = gradient()
        assert phash(image) == phash(image.copy())

    def test_anh_xam_va_anh_mau_deu_nhan(self):
        colour = gradient()
        grey = colour[:, :, 0]
        assert isinstance(phash(grey), int)
        assert isinstance(phash(colour), int)

    def test_doi_do_sang_khong_doi_chu_ky(self):
        """Hệ số DC bị bỏ có chủ ý: cùng một cảnh sáng hơn vẫn là cùng cảnh.

        Đây là lý do pHash được chọn thay vì so pixel thẳng — hai keyframe cách
        nhau 0,2 giây thường chỉ khác nhau ở phơi sáng tự động.
        """
        base = gradient()
        brighter = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        assert hamming(phash(base), phash(brighter)) <= 2

    def test_anh_khac_han_thi_chu_ky_khac_han(self):
        left = gradient(shift=0)
        right = gradient(shift=32)
        assert hamming(phash(left), phash(right)) > 5


class TestLocTheoChuKy:
    def test_bo_khung_trung(self):
        report = signature_filter([0b1010, 0b1010, 0b1111_0000], max_distance=1)
        assert report.kept == [0, 2]
        assert report.dropped == [(1, 0, 0.0)]

    def test_so_voi_khung_DA_GIU_chu_khong_phai_khung_lien_truoc(self):
        """Chuỗi trôi chậm A→B→C: mỗi bước cách nhau 2 bit, A và C cách 4.

        So theo cặp liền kề sẽ bỏ cả B lẫn C và giữ mỗi A — mất hẳn khúc cuối
        của một cú lia máy chậm. So với khung đã giữ thì C được giữ lại, đúng.
        """
        signatures = [0b0000, 0b0011, 0b1111]
        report = signature_filter(signatures, max_distance=2)
        assert report.kept == [0, 2]
        assert [d[0] for d in report.dropped] == [1]

    def test_khung_dau_luon_duoc_giu(self):
        report = signature_filter([0b1111, 0b1111, 0b1111], max_distance=64)
        assert report.kept == [0]
        assert report.n_dropped == 2

    def test_danh_sach_rong(self):
        report = signature_filter([], max_distance=5)
        assert report.kept == [] and report.n_dropped == 0
        assert report.ratio_dropped == 0.0

    def test_nguong_0_van_bat_trung_tuyet_doi(self):
        report = signature_filter([7, 7, 8], max_distance=0)
        assert report.kept == [0, 2]

    def test_bao_cao_giu_ca_khung_bi_bo_LAN_khung_no_trung(self):
        """Chỉ nói 'đã bỏ N khung' thì không kiểm lại được — người vận hành cần
        mở đúng hai ảnh đó ra nhìn mới biết ngưỡng đúng hay sai."""
        report = signature_filter([0, 1, 255], max_distance=2)
        dropped_index, kept_index, distance = report.dropped[0]
        assert (dropped_index, kept_index) == (1, 0)
        assert distance == 1.0

    def test_nguong_cang_long_cang_bo_nhieu(self):
        signatures = [0, 1, 3, 7, 15, 255]
        counts = [r.n_dropped for _, r in sweep_signature(signatures, (0, 1, 2, 4, 8))]
        assert counts == sorted(counts)


class TestLocTheoEmbedding:
    def test_bo_khung_gan_trung(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.001], [0.0, 1.0]])
        assert embedding_filter(vectors, max_cosine=0.99).kept == [0, 2]

    def test_chuan_hoa_lai_vector_chua_chuan_hoa(self):
        """`n4_siglip` ghi ra vector có norm lệch 1e-4 (float16 -> float32).

        Cosine tính trên vector chưa chuẩn hoá không phải cosine — nó vẫn ra một
        con số, vẫn xếp hạng được, và vẫn sai.
        """
        unit = np.array([[3.0, 4.0], [30.0, 40.0], [0.0, 1.0]])  # hai cai dau cung huong
        report = embedding_filter(unit, max_cosine=0.99)
        assert report.kept == [0, 2]

    def test_so_voi_khung_DA_GIU(self):
        drift = np.array([[1.0, 0.0], [0.95, 0.312], [0.8, 0.6]])
        report = embedding_filter(drift, max_cosine=0.99)
        assert 0 in report.kept and 2 in report.kept

    def test_vector_khong_thi_khong_chia_cho_0(self):
        vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
        report = embedding_filter(vectors, max_cosine=0.99)
        assert report.n_kept + report.n_dropped == 2

    def test_ma_tran_sai_chieu_thi_nem_loi(self):
        with pytest.raises(ValueError, match="ma tran"):
            embedding_filter(np.array([1.0, 2.0, 3.0]))

    def test_nguong_cang_chat_cang_bo_it(self):
        rng = np.random.default_rng(0)
        vectors = np.cumsum(rng.normal(scale=0.05, size=(40, 8)), axis=0) + 1.0
        counts = [r.n_dropped for _, r in sweep_embedding(vectors, (0.999, 0.99, 0.9))]
        assert counts == sorted(counts)


class TestBaoCao:
    def test_ti_le_tinh_tren_TONG_dau_vao(self):
        report = signature_filter([0, 0, 0, 255], max_distance=1)
        assert report.n_kept == 2 and report.n_dropped == 2
        assert report.ratio_dropped == pytest.approx(0.5)

    def test_summary_doc_duoc(self):
        text = signature_filter([0, 0, 255], max_distance=1).summary()
        assert "giu 2" in text and "bo 1" in text
