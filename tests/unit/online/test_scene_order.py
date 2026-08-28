"""Thứ tự cảnh cho KIS — tách mệnh đề, đảo trình tự, thiên kiến mức video.

Phần lớn test ở đây bảo vệ **chỗ KHÔNG cắt**. Tách sai một câu một-khoảnh-khắc
thành hai cảnh là bịa ra một ràng buộc video không có, và câu đang đúng sẽ tụt
hạng — đắt hơn nhiều so với việc bỏ lỡ một câu có trình tự.
"""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.contracts.tools import Candidate
from mfe.online.dante import EventHit
from mfe.online.scene_order import (
    MAX_CLAUSES,
    MIN_CLAUSE_WORDS,
    ScenePrior,
    apply_prior,
    clauses_from_texts,
    split_clauses,
    video_prior,
)

# --------------------------------------------------------------------------
# Tách mệnh đề — chỗ KHÔNG cắt
# --------------------------------------------------------------------------


def test_cau_khong_co_tu_noi_tra_ve_nguyen_cau():
    text = "Một người đàn ông mặc áo xanh ngồi trên xuồng giữa sông"
    clauses = split_clauses(text)
    assert len(clauses) == 1
    assert clauses[0].text == text
    assert clauses[0].order == 0


def test_tu_noi_yeu_mot_minh_khong_du_de_cat():
    """`triet_kis_01` — "sau khi" ở đây là MỘT khoảnh khắc, không phải hai cảnh.

    Không có từ nối mạnh nào trong câu, nên `sau khi` không được quyền cắt. Đây
    là câu đã có thật trong bộ dev và nó phải đi nguyên khối xuống các arm.
    """
    text = (
        "tay đua mặc áo vàng giơ hai tay lên ăn mừng sau khi về đích, "
        "hai bên đường có hàng rào quảng cáo"
    )
    assert len(split_clauses(text)) == 1


def test_roi_mot_minh_khong_cat():
    text = (
        "Một người đàn ông mặc áo xanh ngồi trên xuồng, dùng dao chặt tàu lá "
        "dừa nước dài rồi kéo phần lá vừa cắt lên thuyền"
    )
    assert len(split_clauses(text)) == 1


def test_menh_de_qua_ngan_thi_khong_cat():
    """Hai bên chỗ cắt phải đủ `MIN_CLAUSE_WORDS` từ."""
    text = "Người đàn ông đứng trên bờ sông rộng, sau đó đi"
    clauses = split_clauses(text)
    assert len(clauses) == 1
    assert _words(clauses[0].text) >= MIN_CLAUSE_WORDS


def test_lan_luot_khong_phai_tu_noi_trinh_tu():
    """Đo trên đề thật: cả ba lần "lần lượt" đều liệt kê vật trong CÙNG một cảnh.

    Có một từ nối mạnh trong câu, nên nếu "lần lượt" nằm trong `WEAK_CUES` thì
    nó sẽ cắt — và xé đôi đúng cái danh sách mà mệnh đề đang liệt kê.
    """
    text = (
        "Slide bài giảng lịch sử nền trắng có ba ô bấm bo tròn màu xanh dương "
        "xếp dọc, lần lượt ghi HOA KÌ, NHẬT BẢN và TÂY ÂU, mỗi ô kèm một khung "
        "trắng rỗng bên phải. Ngay sau đó slide thu nhỏ về nửa phải màn hình "
        "trên nền xanh đậm lấp lánh"
    )
    clauses = split_clauses(text)
    assert len(clauses) == 2
    assert "lần lượt ghi HOA KÌ" in clauses[0].text


def test_qua_nhieu_manh_thi_bo_ca_phep_tach():
    text = " ".join(
        f"cảnh thứ {i} có một người mặc áo màu khác nhau đứng yên, sau đó"
        for i in range(MAX_CLAUSES + 3)
    )
    assert len(split_clauses(text)) == 1


# --------------------------------------------------------------------------
# Tách mệnh đề — chỗ PHẢI cắt
# --------------------------------------------------------------------------


def test_ba_canh_that_tach_thanh_ba_menh_de():
    """`triet_order_01` — câu nhiều cảnh thật, đúng ba mốc."""
    text = (
        "Đoạn clip bắt đầu bằng cảnh đổ giá đỗ từ một tô nhựa màu đỏ vào chảo "
        "bột bánh xèo vàng đang chín, sau đó một chiếc muỗng màu xanh lá xúc "
        "nhân từ hộp nhựa trong rải đều lên mặt bánh. Cảnh cuối cùng là cận "
        "cảnh một chiếc bánh xèo vàng gập đôi nằm một mình trên đĩa sứ tròn "
        "màu trắng."
    )
    clauses = split_clauses(text)
    assert len(clauses) == 3
    assert [c.order for c in clauses] == [0, 1, 2]
    assert "đổ giá đỗ" in clauses[0].text
    assert "muỗng màu xanh lá" in clauses[1].text
    assert "bánh xèo vàng gập đôi" in clauses[2].text


def test_tu_noi_ghi_vao_menh_de_dung_sau_no():
    text = (
        "Đoạn clip bắt đầu bằng cảnh những con tôm đã được lột vỏ nằm trên dĩa "
        "sứ trắng, sau đó là cảnh quay các đầu bếp đang trang trí món ăn khác"
    )
    clauses = split_clauses(text)
    assert len(clauses) == 2
    assert clauses[0].cue is None
    assert clauses[1].cue == "sau do"


def test_bo_he_tu_la_va_dau_cau_o_hai_dau_cho_cat():
    text = (
        "Đoạn clip bắt đầu bằng cảnh cà rốt cắt hình ngôi sao đang luộc trong "
        "nồi nước sôi, kết thúc bằng hình ảnh đĩa rau củ luộc trình bày đẹp mắt"
    )
    clauses = split_clauses(text)
    assert len(clauses) == 2
    assert not clauses[1].text.startswith((",", " ", "là "))


# --------------------------------------------------------------------------
# Đảo trình tự — thứ tự chữ ngược thứ tự thời gian
# --------------------------------------------------------------------------


def test_truoc_do_dao_nguoc_thu_tu_chu():
    """`query-p1-12-kis` — "Trước đó là cảnh X" đẩy X lên TRƯỚC mệnh đề chính.

    Không đảo thì DANTE đi tìm một chuỗi tăng dần theo đúng chiều sai, và câu
    tụt hạng thay vì lên.
    """
    text = (
        "Có thể thấy trong cảnh quay có 4 tài xế xe ôm công nghệ trong trạm "
        "xăng, trong đó 3 người đứng đợi còn 1 người lái xe. Trước đó là cảnh "
        "một người đậy nắp bình xăng xe máy của họ."
    )
    clauses = split_clauses(text)
    assert len(clauses) == 2
    assert "đậy nắp bình xăng" in clauses[0].text
    assert "4 tài xế" in clauses[1].text


def test_truoc_khi_cung_dao_khi_da_co_tu_noi_manh():
    text = (
        "Đoạn clip bắt đầu bằng cảnh con lân trắng pha cam hạ thấp đầu trên "
        "dãy cọc gỗ cao, trước khi nó nhảy sang các cột tiếp theo phía xa"
    )
    clauses = split_clauses(text)
    assert len(clauses) == 2
    assert "nhảy sang các cột" in clauses[0].text


# --------------------------------------------------------------------------
# Mệnh đề tách sẵn — người gõ tay hoặc LLM
# --------------------------------------------------------------------------


def test_menh_de_tach_san_giu_nguyen_thu_tu_truyen_vao():
    """Phần gọi chịu trách nhiệm về trình tự; hàm này KHÔNG đảo lại lần nữa.

    Đảo ở cả hai chỗ là đảo hai lần. `split_clauses` đã xử "trước đó" xong xuôi
    trước khi trả về, còn người gõ tay thì gõ thẳng theo thứ tự thời gian.
    """
    clauses = clauses_from_texts(["cảnh đầu tiên", "cảnh thứ hai", "cảnh cuối"])
    assert [c.order for c in clauses] == [0, 1, 2]
    assert [c.text for c in clauses] == ["cảnh đầu tiên", "cảnh thứ hai", "cảnh cuối"]


def test_menh_de_tach_san_bo_manh_rong():
    assert len(clauses_from_texts(["cảnh một", "   ", "", "cảnh hai"])) == 2


def test_menh_de_tach_san_khong_bi_gioi_han_do_dai_cua_bo_luat():
    """`MIN_CLAUSE_WORDS` là cái chốt chống bộ luật xé bừa, không phải luật chung.

    Người gõ "bánh xèo vàng" làm một cảnh thì đó là chủ ý, không phải tai nạn
    của một biểu thức chính quy.
    """
    clauses = clauses_from_texts(["bánh xèo vàng", "đĩa sứ trắng"])
    assert len(clauses) == 2


# --------------------------------------------------------------------------
# Thiên kiến mức video
# --------------------------------------------------------------------------


def _hit(video: str, ordinal: int, score: float) -> EventHit:
    return EventHit(
        keyframe_id=f"{video}#{ordinal * 100:06d}",
        video_id=video,
        ordinal=ordinal,
        frame_id=ordinal * 100,
        score=score,
    )


def test_prior_thuong_video_dung_thu_tu():
    per_clause = [
        [_hit("L01_V001", 5, 1.0), _hit("L02_V002", 40, 1.0)],
        [_hit("L01_V001", 20, 1.0), _hit("L02_V002", 25, 1.0)],
        [_hit("L01_V001", 30, 1.0), _hit("L02_V002", 10, 1.0)],
    ]
    prior = video_prior(per_clause)
    # V001: 5 -> 20 -> 30 tang dan, ca ba menh de tu tim duoc cho cua no.
    assert "L01_V001" in prior
    # V002: 40 -> 25 -> 10 nguoc chieu hoan toan; khong phep gan tang dan nao
    # cho ca ba menh de bang chung that -> truot `MIN_SUPPORT`.
    assert "L02_V002" not in prior


def test_prior_mang_theo_vi_tri_da_ghep():
    """`ordinals` là thứ giới hạn vùng thưởng — thiếu nó là quay về thưởng cả
    video, tức là bản đã đo được là tệ hơn nền."""
    per_clause = [
        [_hit("L01_V001", 5, 1.0)],
        [_hit("L01_V001", 20, 1.0)],
        [_hit("L01_V001", 30, 1.0)],
    ]
    assert video_prior(per_clause)["L01_V001"].ordinals == (5, 20, 30)


def test_prior_rong_khi_chi_co_mot_menh_de():
    assert video_prior([[_hit("L01_V001", 5, 1.0)]]) == {}


def test_prior_rong_voi_HAI_menh_de():
    """Hai mệnh đề thì ràng buộc tăng dần lọc được ~½ số video — gần như không
    lọc gì. Đo trên bộ dev cũ: câu 2 mệnh đề tệ đi ở **cả ba** cấu hình, câu 3
    mệnh đề không bao giờ tệ đi. `MIN_CLAUSES_FOR_PRIOR` là chỗ ghi lại điều đó.
    """
    per_clause = [
        [_hit("L01_V001", 5, 1.0)],
        [_hit("L01_V001", 20, 1.0)],
    ]
    assert video_prior(per_clause) == {}
    # Ha nguong xuong thi no chay lai — de do duoc hai ban canh nhau.
    assert video_prior(per_clause, min_clauses=2) != {}


def test_min_support_None_doi_MOI_menh_de_co_bang_chung():
    """Một hằng số cố định có nghĩa khác nhau tuỳ số mệnh đề: `MIN_SUPPORT=2`
    trên câu 3 mệnh đề cho **một** mệnh đề đi nhờ miễn phí."""
    # Ba vi tri duoc truy hoi (5, 20, 30) nhung menh de 2 KHONG tim ra cai nao —
    # no phai "di nho" vi tri 20 do menh de 3 tim ra, voi S = 0.
    per_clause = [
        [_hit("L01_V001", 5, 1.0)],
        [],
        [_hit("L01_V001", 20, 1.0), _hit("L01_V001", 30, 1.0)],
    ]
    assert video_prior(per_clause) == {}
    assert video_prior(per_clause, min_support=2) != {}


def test_prior_nam_trong_khoang_don_vi():
    per_clause = [
        [_hit("L01_V001", 5, 1.0), _hit("L02_V002", 6, 0.4)],
        [_hit("L01_V001", 20, 1.0), _hit("L02_V002", 21, 0.4)],
        [_hit("L01_V001", 30, 1.0), _hit("L02_V002", 31, 0.4)],
    ]
    prior = video_prior(per_clause)
    assert max(p.score for p in prior.values()) == pytest.approx(1.0)
    assert all(0.0 < p.score <= 1.0 for p in prior.values())


# --------------------------------------------------------------------------
# Áp thiên kiến
# --------------------------------------------------------------------------


def _cand(video: str, ordinal: int, score: float) -> Candidate:
    return Candidate(
        keyframe_id=f"{video}#{ordinal:06d}",
        video_id=video,
        score=score,
        evidence_id=f"siglip:{video}:{ordinal}",
    )


def _ordinal_of(keyframe_id: str) -> int:
    return int(keyframe_id.split("#")[1])


def _prior(score: float, *ordinals: int) -> ScenePrior:
    return ScenePrior(score=score, ordinals=ordinals)


def test_trong_so_khong_thi_giu_nguyen_thu_tu():
    """Đường mặc định của mọi câu không kể trình tự phải là KHÔNG-LÀM-GÌ thật."""
    items = [_cand("L01_V001", 1, 0.9), _cand("L02_V002", 2, 0.8)]
    prior = {"L01_V001": _prior(1.0, 1)}
    assert apply_prior(items, prior, ordinal_of=_ordinal_of, weight=0.0) == items
    assert apply_prior(items, {}, ordinal_of=_ordinal_of, weight=0.5) == items


def test_window_can_ordinal_of_va_noi_ra_khi_thieu():
    """Đòi `window` mà quên `ordinal_of` thì NÉM, không lặng lẽ bỏ qua `window`.

    Bỏ qua trong im lặng sẽ cho ra một phép đo của cấu hình **khác** với cấu
    hình người đo tưởng mình đang đo.
    """
    items = [_cand("L01_V001", 1, 0.9)]
    with pytest.raises(ContractError):
        apply_prior(items, {"L01_V001": _prior(1.0, 1)}, weight=0.5, window=3)


def test_mac_dinh_la_thuong_CA_VIDEO():
    """`WINDOW = None` — đo được là thắng ở cả hai bộ, xem docstring của nó.

    Nên đường mặc định KHÔNG cần `ordinal_of`, và đó là chuyện có ích: phần gọi
    không phải cầm `RelationalMap` chỉ để bật tầng này.
    """
    items = [_cand("L01_V001", 99, 0.79)]
    out = apply_prior(items, {"L01_V001": _prior(1.0, 10)}, weight=0.5)
    assert out[0].score == pytest.approx(0.79 * 1.5)


def test_chi_thuong_khung_gan_vi_tri_da_ghep():
    """`trong_kis_58`: prior 0,994 ĐÚNG video mà hạng vẫn tụt 17 -> 32, vì một
    video sai đạt 1,0 nâng cả ~50 khung của nó lên trên khung đúng."""
    items = [
        _cand("L01_V001", 10, 0.80),   # sat vi tri da ghep -> duoc thuong
        _cand("L01_V001", 99, 0.79),   # cung video, xa -> KHONG duoc thuong
        _cand("L02_V002", 1, 0.85),
    ]
    out = apply_prior(
        items, {"L01_V001": _prior(1.0, 10)},
        ordinal_of=_ordinal_of, weight=0.5, window=3,
    )
    assert [c.keyframe_id for c in out][0] == "L01_V001#000010"
    xa = next(c for c in out if c.keyframe_id == "L01_V001#000099")
    assert xa.score == pytest.approx(0.79)


def test_window_none_khoi_phuc_thuong_ca_video():
    """Giữ để đo hai bản cạnh nhau, không phải để dùng."""
    items = [_cand("L01_V001", 99, 0.79)]
    out = apply_prior(items, {"L01_V001": _prior(1.0, 10)}, weight=0.5, window=None)
    assert out[0].score == pytest.approx(0.79 * 1.5)


def test_video_khong_co_prior_giu_nguyen_diem():
    items = [_cand("L01_V001", 1, 0.9), _cand("L02_V002", 1, 0.8)]
    out = apply_prior(
        items, {"L01_V001": _prior(1.0, 1)}, ordinal_of=_ordinal_of, weight=0.5
    )
    kept = next(c for c in out if c.video_id == "L02_V002")
    assert kept.score == pytest.approx(0.8)


def _words(text: str) -> int:
    return len([w for w in text.split() if any(ch.isalpha() for ch in w)])


# --------------------------------------------------------------------------
# Xung đột với rerank_tiers
# --------------------------------------------------------------------------


class TestBoQuaRerankKhiCoThuTuCanh:
    """`vlm` xoá đúng phần thứ tự cảnh vừa được — nên nó phải ngồi ngoài.

    Đo qua `mfe.eval.official` trên 14 câu KIS ba cảnh: thứ tự cảnh một mình
    0,771 -> 0,814; thêm `vlm` lên trên thì tụt thẳng về 0,771. `vlm` dời 16/20
    ứng viên mỗi câu, nên phần thứ tự cảnh đẩy lên nó xáo lại hết.
    """

    def _cascade(self, **retrieval):
        from types import SimpleNamespace

        from mfe.online.cascade import Cascade

        config = SimpleNamespace(retrieval=SimpleNamespace(**retrieval))
        return Cascade(config=config)

    def test_cau_CO_thu_tu_canh_thi_bo_qua_tang(self):
        cascade = self._cascade(rerank_tiers=["vlm"], rerank_skip_when_scene_order=True)
        items = [_cand("L01_V001", 1, 0.9)]
        out = cascade._rerank_tiers(items, "x", None, scene_ordered=True)
        assert out == items

    def test_cau_KHONG_co_thu_tu_canh_thi_tang_van_chay(self):
        """Nếu cửa thoát nuốt cả câu thường thì `vlm` thành vô hiệu hoàn toàn,
        và bảng đo +0,015 trên 163 câu không còn đúng với hệ đang chạy."""
        cascade = self._cascade(rerank_tiers=["khong_co_tang_nao"])
        items = [_cand("L01_V001", 1, 0.9)]
        # Ten tang la -> `build_tier` nem. Nem tuc la ĐA di qua cua thoat.
        with pytest.raises(ValueError, match="khong co tang rerank"):
            cascade._rerank_tiers(items, "x", None, scene_ordered=False)

    def test_tat_cua_thoat_thi_tang_chay_ca_tren_cau_thu_tu_canh(self):
        cascade = self._cascade(
            rerank_tiers=["khong_co_tang_nao"], rerank_skip_when_scene_order=False
        )
        items = [_cand("L01_V001", 1, 0.9)]
        with pytest.raises(ValueError):
            cascade._rerank_tiers(items, "x", None, scene_ordered=True)
