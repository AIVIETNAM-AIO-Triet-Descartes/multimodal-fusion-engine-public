"""Khoá cách dựng chỉ mục arm `object` — `mfe.offline.object_text`.

Arm này từng đo được **~0** trên bộ dev (handoff §A.1: 2/42), và nguyên nhân đã
được chẩn đoán bằng idf chứ không bằng điểm: token thuộc tính là **stopword tự
do**. Các test ở đây khoá đúng ba thứ làm nó hết là stopword, và cả ba đều thuộc
loại hỏng mà **không có gì báo** — chỉ mục vẫn dựng, truy vấn vẫn chạy, thứ hạng
vẫn ra, chỉ vô dụng.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfe.offline.object_text import (
    GRID_SIZE,
    Detection,
    canonical_vi,
    compose,
    count_word,
    dominant_colour,
    grid_cell,
    read_detections,
    render,
    zone_of,
)


class TestTrucToaDo:
    """Thứ tự trục `[ymin, xmin, ymax, xmax]` — đã xác minh trên dữ liệu thật.

    Đảo nhầm x/y thì `zone_of` và `grid_cell` sai **âm thầm**: không lỗi, không
    cảnh báo, chỉ là mọi vật đều nằm sai chỗ. §6 của
    `docs/triet/huong-di-grid-vitri-mausac.md` gọi đúng tên cái bẫy này.
    """

    def test_dai_ngang_doc_theo_x_la_phan_tu_1_va_3(self):
        # Hộp bám mép TRÁI: x nhỏ (phần tử 1, 3), y bất kỳ.
        assert zone_of([0.0, 0.02, 1.0, 0.10]) == "trái"
        assert zone_of([0.0, 0.90, 1.0, 0.98]) == "phải"
        assert zone_of([0.0, 0.45, 1.0, 0.55]) == "giữa"

    def test_neu_doc_nham_truc_thi_ket_qua_KHAC_HAN(self):
        """Ca này tồn tại để phép kiểm ở trên không vô nghĩa.

        Một hộp cao-và-hẹp bên trái, dưới cách đọc đảo trục, sẽ thành hộp
        thấp-và-rộng ở giữa. Nếu hai cách đọc cho cùng kết quả thì test kia
        không chứng minh được gì.
        """
        box = [0.10, 0.02, 0.90, 0.10]
        swapped = [box[1], box[0], box[3], box[2]]
        assert zone_of(box) != zone_of(swapped)

    def test_hop_hong_khong_nem(self):
        assert zone_of(None) == "giữa"
        assert zone_of([]) == "giữa"
        assert zone_of(["a", "b", "c", "d"]) == "giữa"


class TestLuoiONhung:
    def test_goc_tren_trai_la_o_0(self):
        assert grid_cell([0.0, 0.0, 0.05, 0.05]) == 0

    def test_goc_duoi_phai_la_o_cuoi(self):
        assert grid_cell([0.95, 0.95, 1.0, 1.0]) == GRID_SIZE**2 - 1

    def test_tam_o_bien_1_0_khong_tran_ra_ngoai_luoi(self):
        """`int(1.0 * 7)` là 7, ngoài dải 0..6. Không chặn thì ô cuối vượt lưới."""
        assert grid_cell([1.0, 1.0, 1.0, 1.0]) == GRID_SIZE**2 - 1

    def test_hang_va_cot_KHONG_bi_hoan_doi(self):
        # Trên-phải: y nhỏ, x lớn -> hàng 0, cột 6 -> ô 6.
        assert grid_cell([0.0, 0.95, 0.05, 1.0]) == 6
        # Dưới-trái: y lớn, x nhỏ -> hàng 6, cột 0 -> ô 42.
        assert grid_cell([0.95, 0.0, 1.0, 0.05]) == 42


class TestTokenGhep:
    """`_` phải sống sót qua bộ tách từ, nếu không cả hướng đi này vô nghĩa."""

    def test_dau_gach_duoi_song_sot_qua_tokenize(self):
        from mfe.stores.inverted.index import _tokenize

        assert _tokenize(compose("ba", "người")) == ["ba_người"]

    def test_nhan_NHIEU_CHU_thanh_MOT_token(self):
        """`hai_xe máy` tách thành `hai_xe` + `máy`: vừa mất nhãn, vừa đẻ rác."""
        from mfe.stores.inverted.index import _tokenize

        assert compose("hai", "xe máy") == "hai_xe_máy"
        assert _tokenize(compose("hai", "xe máy")) == ["hai_xe_máy"]

    def test_nhan_rong_thi_khong_sinh_token_cut(self):
        assert compose("ba", "") == ""
        assert compose("ba", "   ") == ""

    def test_dang_chinh_la_dang_TIENG_VIET_dau_tien(self):
        assert canonical_vi(["Person", "người", "con người"], "Person") == "người"

    def test_nhan_chua_dich_thi_roi_ve_chinh_no(self):
        assert canonical_vi(["Zebra"], "Zebra") == "Zebra"


class TestChuoiChiMuc:
    def test_dem_di_vao_TOKEN_chu_khong_vao_so_lan_lap(self):
        """Bản cũ lặp nhãn `min(n,3)` lần để gắn số đếm vào ĐIỂM SỐ.

        Nó không phân biệt được "ba người" với "ba cái ghế", và nó làm tài liệu
        dài ra — mà BM25 phạt tài liệu dài. Một token `ba_người` làm đúng việc
        đó, có idf, và không tốn thêm độ dài.
        """
        from mfe.offline.labels_vi import translate_labels

        text = render([Detection("Person", "giữa", 24) for _ in range(3)])
        assert "ba_người" in text
        # Phần token TRẦN phải đúng bằng bộ đồng nghĩa, xuất hiện MỘT lần.
        # (Đếm chữ `người` không dùng được: đồng nghĩa `con người` chứa nó.)
        bare = " ".join(translate_labels(["Person"]))
        assert text.startswith(bare)
        assert text.count(bare) == 1, "khong duoc lap nhan nua"
        assert [t for t in text.split() if "_" in t] == ["ba_người", "giữa_người"]

    def test_qua_nguong_thi_ghi_nhieu(self):
        text = render([Detection("Person", "giữa", 24) for _ in range(20)])
        assert "nhiều_người" in text

    def test_moi_dai_deu_co_token_rieng(self):
        text = render([
            Detection("Person", "trái", 21),
            Detection("Person", "phải", 27),
        ])
        assert "trái_người" in text and "phải_người" in text

    def test_mau_chi_xuat_hien_khi_CO_du_lieu_mau(self):
        assert "đỏ_người" in render([Detection("Person", "giữa", 24, "đỏ")])
        assert "đỏ_người" not in render([Detection("Person", "giữa", 24, None)])

    def test_token_ghep_chi_cho_MOT_dang_dong_nghia(self):
        """Ghép cho cả ba đồng nghĩa của `Motorcycle` là 12 token cho một vật —
        đúng cái tội `min(n,3)` mắc phải, chỉ đổi cơ chế."""
        text = render([Detection("Motorcycle", "phải", 32)])
        composites = [t for t in text.split() if "_" in t]
        assert composites == ["một_xe_máy", "phải_xe_máy"]
        # Đồng nghĩa vẫn còn ở token TRẦN, nên phần thu hồi không mất gì.
        assert "mô" in text and "tô" in text

    def test_o_luoi_KHONG_vao_chuoi_bm25(self):
        """0/49 đề thật hỏi theo góc. Mỗi ô thêm vào là một token không bao giờ
        được truy vấn tới, và nó chỉ làm loãng BM25."""
        text = render([Detection("Person", "giữa", 24)])
        assert "o24" not in text

    def test_bo_rong_ra_chuoi_rong(self):
        assert render([]) == ""


class TestDocHopThat:
    def test_bo_hop_duoi_nguong(self):
        payload = {
            "detection_class_entities": ["Person", "Car"],
            "detection_scores": ["0.9", "0.1"],
            "detection_boxes": [[0.1, 0.1, 0.9, 0.2], [0.1, 0.8, 0.9, 0.9]],
        }
        hits = read_detections(payload, 0.30)
        assert [h.label for h in hits] == ["Person"]

    def test_KHONG_khu_trung_lap(self):
        """Khử trùng lặp là chỗ bản đầu mất trắng số đếm: năm hộp `Person`
        thành đúng một chữ "người"."""
        payload = {
            "detection_class_entities": ["Person"] * 5,
            "detection_scores": ["0.9"] * 5,
            "detection_boxes": [[0.1, 0.1, 0.9, 0.2]] * 5,
        }
        assert len(read_detections(payload, 0.30)) == 5

    def test_diem_hong_bi_bo_chu_khong_nem(self):
        payload = {
            "detection_class_entities": ["Person", "Car"],
            "detection_scores": ["khong phai so", "0.9"],
            "detection_boxes": [[0.1, 0.1, 0.9, 0.2], [0.1, 0.1, 0.9, 0.2]],
        }
        assert [h.label for h in read_detections(payload, 0.30)] == ["Car"]

    def test_chay_tren_FILE_THAT_cua_BTC(self):
        source = Path("data/sample/objects/L21_V001/001.json")
        if not source.is_file():  # pragma: no cover - may khong co mau
            pytest.skip("khong co file object mau")
        payload = json.loads(source.read_text(encoding="utf-8"))
        hits = read_detections(payload, 0.30)
        assert hits, "file that phai co it nhat mot hop tren nguong"
        assert all(0 <= h.cell < GRID_SIZE**2 for h in hits)
        assert all(h.zone in ("trái", "giữa", "phải") for h in hits)


class TestMau:
    def test_vung_rong_tra_None_chu_khong_doan_bua(self):
        """Một màu sai gắn vào nhãn tạo ra token ghép sai — nó khớp NHẦM chứ
        không chỉ trượt."""
        assert dominant_colour([]) is None

    @pytest.mark.parametrize(
        ("rgb", "want"),
        [((250, 250, 250), "trắng"), ((5, 5, 5), "đen"),
         ((200, 10, 10), "đỏ"), ((10, 10, 200), "dương")],
    )
    def test_map_ve_mau_pho_quat_gan_nhat(self, rgb, want):
        assert dominant_colour([rgb] * 4) == want

    def test_dung_TRUNG_VI_chu_khong_trung_binh(self):
        """Trung bình của nửa đỏ nửa xanh là xám — một màu không hề có trong
        ảnh. Trung vị luôn rơi vào một màu thật."""
        pixels = [(255, 0, 0)] * 7 + [(0, 0, 255)] * 3
        assert dominant_colour(pixels) == "đỏ"


def test_count_word():
    assert count_word(0) == "không"
    assert count_word(3) == "ba"
    assert count_word(9) == "chín"
    assert count_word(10) == "nhiều"
    assert count_word(500) == "nhiều"
