"""`BM25Thua` phải ra ĐÚNG điểm của `rank_bm25.BM25Okapi`.

Đây là điều kiện sống còn của bản vá, không phải phép lịch sự: bảng trọng số
trong `config/btc_b1.yaml` được leo dốc trên điểm của `BM25Okapi`. Một cách
tính "gần đúng" sẽ nhanh hơn, vẫn trả về thứ hạng trông hợp lý, và lặng lẽ làm
mọi con số đã đo mất nghĩa — đúng lớp lỗi §5 CLAUDE.md.

Nên test ở đây **không** so với hằng số chép tay. Nó dựng `BM25Okapi` thật trên
cùng corpus rồi so từng điểm một.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfe.online.bm25_thua import EPSILON, K1, B, BM25Thua

rank_bm25 = pytest.importorskip("rank_bm25")


CORPUS = [
    "người đàn ông mặc áo xanh đứng cạnh xe máy".split(),
    "người phụ nữ nấu ăn trong bếp với dầu hành phi".split(),
    "đoàn xe đạp đua qua cầu dưới trời mưa".split(),
    "xe máy xe máy xe máy chạy trên đường".split(),   # từ lặp nhiều lần
    "sạt lở đất chắn ngang đèo".split(),
    "người đàn ông".split(),                           # tài liệu rất ngắn
    ("mưa " * 60).split(),                             # tài liệu rất dài
    "bảng hiệu công ty màu đỏ trước toà nhà".split(),
]

#: `người` có mặt ở 3/8 tài liệu, `xe` ở 3/8 — đủ phổ biến để idf ÂM trên một
#: corpus 8 dòng, tức chạm đúng nhánh `EPSILON` mà bản kẹp-về-0 sẽ làm sai.
TRUY_VAN = [
    ["người"],
    ["người", "đàn", "ông"],
    ["xe", "máy"],
    ["mưa"],
    ["mưa", "mưa"],                     # từ lặp trong truy vấn
    ["không", "hề", "tồn", "tại"],
    ["người", "khongtontai", "xe"],     # nửa trúng nửa trượt
    [],
]


@pytest.fixture(scope="module")
def cap():
    return rank_bm25.BM25Okapi(CORPUS), BM25Thua(CORPUS)


class TestKhopVoiRankBm25:
    @pytest.mark.parametrize("tokens", TRUY_VAN)
    def test_diem_khop_tung_tai_lieu(self, cap, tokens):
        cu, moi = cap
        a = np.asarray(cu.get_scores(tokens), dtype=np.float64)
        b = moi.get_scores(tokens)
        assert b.shape == a.shape
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("tokens", TRUY_VAN)
    def test_thu_hang_khop(self, cap, tokens):
        """Điểm khớp chưa đủ — thứ hạng mới là thứ đi vào hợp nhất."""
        cu, moi = cap
        a = np.argsort(np.asarray(cu.get_scores(tokens)))[::-1]
        b = np.argsort(moi.get_scores(tokens))[::-1]
        assert list(a) == list(b)

    def test_tham_so_dung_mac_dinh_rank_bm25(self):
        """Ghim ba hằng số. Đổi chúng là đổi mọi điểm đã leo dốc."""
        assert (K1, B, EPSILON) == (1.5, 0.75, 0.25)

    def test_idf_am_dung_epsilon_khong_kep_ve_0(self, cap):
        """`người` ở 3/8 tài liệu ⇒ idf âm. Kẹp về 0 thì điểm ra 0 hết.

        `BM25Okapi` thay idf âm bằng `eps × idf dương trung bình`, nên điểm vẫn
        DƯƠNG và vẫn xếp hạng được. Đây là chỗ một bản cài lại dễ sai nhất, và
        sai thì chỉ lộ ở đuôi bảng xếp hạng.
        """
        _cu, moi = cap
        diem = moi.get_scores(["người"])
        assert (diem > 0).any(), "idf âm bị kẹp về 0 — mọi từ phổ biến biến mất"


class TestTruongHopBien:
    def test_corpus_rong(self):
        moi = BM25Thua([])
        assert moi.get_scores(["gì", "đó"]).shape == (0,)

    def test_tai_lieu_rong(self):
        """Một dòng không có token nào không được làm đổ cả chỉ mục."""
        cu = rank_bm25.BM25Okapi([[], ["mưa"], []])
        moi = BM25Thua([[], ["mưa"], []])
        np.testing.assert_allclose(
            moi.get_scores(["mưa"]),
            np.asarray(cu.get_scores(["mưa"]), dtype=np.float64),
            rtol=1e-9, atol=1e-12,
        )

    def test_truy_van_toan_tu_la(self, cap):
        _cu, moi = cap
        diem = moi.get_scores(["khongheton", "taibaogio"])
        assert diem.shape == (len(CORPUS),)
        assert not diem.any()

    def test_tu_lap_trong_truy_van_cong_don(self, cap):
        """`BM25Okapi` lặp `for q in query`, KHÔNG khử trùng — phải giữ y vậy."""
        _cu, moi = cap
        mot = moi.get_scores(["mưa"])
        hai = moi.get_scores(["mưa", "mưa"])
        np.testing.assert_allclose(hai, mot * 2, rtol=1e-9)

    def test_float64(self, cap):
        _cu, moi = cap
        assert moi.get_scores(["người"]).dtype == np.float64
