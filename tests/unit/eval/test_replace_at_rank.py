"""Thay khung tự tua vào đúng một hạng, và nó phải đi được tới file nộp.

Phép kiểm quan trọng nhất ở đây KHÔNG phải "hàm đổi đúng phần tử" — mà là khung
tự tua **ra được `frame_id` đúng ở bước xuất CSV**. `candidate_frame_id` cố ý ném
khi tra Relational Map không thấy, và khung tự tua thì chắc chắn không có trong
bản đồ. Quên nối hai đầu đó thì cả câu bị bỏ khỏi ZIP — im lặng, vì `build_zip`
bắt `ContractError` rồi ghi vào `skipped`.
"""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.contracts.tools import Candidate
from mfe.eval.official import candidate_frame_id
from mfe.eval.submission_bundle import replace_at_rank


def _ung_vien(frame: int, score: float = 1.0) -> Candidate:
    return Candidate(
        keyframe_id=f"L21_V001#{frame:06d}",
        video_id="L21_V001",
        score=score,
        evidence_id=f"siglip:L21_V001:{frame}",
    )


class _BanDoGia:
    """Chỉ biết những khung của BTC. Khung tự tua tra là ném — đúng như thật."""

    def get_keyframe(self, keyframe_id: str):
        if keyframe_id not in {"L21_V001#000100", "L21_V001#000200"}:
            raise ContractError("keyframe_not_found", keyframe_id)
        raise AssertionError("test này chỉ hỏi tới khung tự tua")


class TestThayDungCho:
    def test_thay_tai_cho_giu_nguyen_do_dai_va_hang(self) -> None:
        rows = [_ung_vien(100), _ung_vien(200), _ung_vien(300)]
        moi = replace_at_rank(rows, 2, 12345)
        assert len(moi) == 3
        assert moi[0] is rows[0] and moi[2] is rows[2]
        assert moi[1].keyframe_id == "L21_V001#012345"

    def test_giu_nguyen_diem_cua_ung_vien_cu(self) -> None:
        rows = [_ung_vien(100, score=0.87)]
        assert replace_at_rank(rows, 1, 999)[0].score == pytest.approx(0.87)

    def test_khung_moi_mang_theo_frame_idx(self) -> None:
        moi = replace_at_rank([_ung_vien(100)], 1, 777)
        assert moi[0].frame_idx == 777

    def test_evidence_id_hop_khuon(self) -> None:
        """`ghim-tay:` từng bị từ chối vì khuôn là `^[a-zA-Z0-9_]+:`."""
        moi = replace_at_rank([_ung_vien(100)], 1, 777)
        assert moi[0].evidence_id.startswith("tua_tay:")

    def test_khong_doi_danh_sach_goc(self) -> None:
        rows = [_ung_vien(100)]
        replace_at_rank(rows, 1, 777)
        assert rows[0].keyframe_id == "L21_V001#000100"


class TestTuChoi:
    @pytest.mark.parametrize("rank", [0, 3, -1])
    def test_hang_ngoai_danh_sach(self, rank: int) -> None:
        with pytest.raises(ContractError):
            replace_at_rank([_ung_vien(100), _ung_vien(200)], rank, 5)

    def test_frame_am(self) -> None:
        with pytest.raises(ContractError):
            replace_at_rank([_ung_vien(100)], 1, -1)


class TestDiDuocToiFileNop:
    def test_khung_tu_tua_ra_dung_frame_id_du_khong_co_trong_ban_do(self) -> None:
        moi = replace_at_rank([_ung_vien(100)], 1, 45678)
        assert candidate_frame_id(moi[0], _BanDoGia()) == 45678

    def test_khung_thuong_VAN_phai_tra_ban_do(self) -> None:
        """Chỗ đè chỉ được mở cho khung tự khai, không phải mở cho tất cả.

        Nếu `candidate_frame_id` lùi về việc bóc số từ `keyframe_id` cho mọi
        trường hợp thì một khung thiếu thật sẽ lặng lẽ thành hợp lệ.
        """
        with pytest.raises(ContractError):
            candidate_frame_id(_ung_vien(999), _BanDoGia())
