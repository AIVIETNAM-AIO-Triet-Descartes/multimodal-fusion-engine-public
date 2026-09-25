"""`ArmRegistry.run` chạy song song, và `arms_disabled` TẮT HẲN arm.

Hai thứ khoá ở đây, cả hai đều từ số đo 25-09 trên 163 câu dev:

1. **Song song không được đổi kết quả.** Các arm độc lập và `weighted_fusion`
   không phụ thuộc thứ tự, nên đổi từ tuần tự sang luồng chỉ được đổi ĐỒNG HỒ.
   Nếu một ngày nào đó kết quả lệch, test này phải đỏ trước khi ai kịp tin vào
   một bảng điểm sai.

2. **Trọng số 0 KHÁC tắt arm.** `fusion.py` ghi thẳng: *"A weight of 0 does not
   disable an arm — the arm still contributes its candidates to the pool"*. Đó
   là lý do ba arm trọng số 0,0 vẫn ăn **1.829 ms trên 7.402 ms** = 25% độ trễ
   một truy vấn, và vì sao phải có `arms_disabled` chứ không phải đặt 0.
"""

from __future__ import annotations

import threading
import time

import pytest

from mfe.contracts.tools import Candidate
from mfe.online.arms import Arm, ArmRegistry


def _cand(arm: str, i: int) -> Candidate:
    return Candidate(
        keyframe_id=f"L21_V001#{i:06d}", video_id="L21_V001",
        score=1.0 / (i + 1), evidence_id=f"{arm}:run:{i}",
    )


def _arm(ten: str, *, tre: float = 0.0, so: int = 3, **kw) -> Arm:
    def search(q, k, run, _ten=ten, _tre=tre, _so=so):
        if _tre:
            time.sleep(_tre)
        return [_cand(_ten, i) for i in range(_so)]

    return Arm(name=ten, search=search, **kw)


class TestSongSongKhongDoiKetQua:
    def test_cung_ket_qua_voi_tuan_tu(self):
        ten = [f"a{i}" for i in range(6)]
        song = ArmRegistry([_arm(t) for t in ten], max_workers=6).run("q", 3, "run")
        tuan = ArmRegistry([_arm(t) for t in ten], max_workers=1).run("q", 3, "run")
        assert song == tuan

    def test_giu_thu_tu_arm_khong_theo_luong_nao_xong_truoc(self):
        """Arm chậm nhất đứng ĐẦU: nếu thứ tự theo luồng thì nó sẽ tụt xuống cuối.

        Thứ tự khoá `results` là thứ tự arm, vì `weighted_fusion` phá hoà bằng
        thứ tự chèn — một bảng đổi thứ tự giữa hai lượt là bảng không so được.
        """
        arms = [_arm("cham", tre=0.05), _arm("nhanh1"), _arm("nhanh2")]
        ra = ArmRegistry(arms, max_workers=3).run("q", 3, "run")
        assert list(ra) == ["cham", "nhanh1", "nhanh2"]

    def test_that_su_chay_song_song(self):
        """Ba arm mỗi cái ngủ 0,1s: tuần tự ≥0,3s, song song ~0,1s."""
        arms = [_arm(f"a{i}", tre=0.1) for i in range(3)]
        t = time.perf_counter()
        ArmRegistry(arms, max_workers=3).run("q", 3, "run")
        assert time.perf_counter() - t < 0.25

    def test_moi_arm_mot_luong_khac_nhau(self):
        thay: set[int] = set()
        khoa = threading.Lock()

        def search(q, k, run):
            time.sleep(0.05)
            with khoa:
                thay.add(threading.get_ident())
            return [_cand("x", 0)]

        arms = [Arm(name=f"a{i}", search=search) for i in range(4)]
        ArmRegistry(arms, max_workers=4).run("q", 1, "run")
        assert len(thay) > 1, "không có gì chạy song song"


class TestMotArmHongKhongGietCaTruyVan:
    """`run` bắt mọi lỗi — một arm hỏng không được kéo cả truy vấn xuống."""

    def test_arm_nem_bi_bo_qua(self, caplog):
        def no(q, k, run):
            raise RuntimeError("vỡ")

        arms = [_arm("tot"), Arm(name="hong", search=no), _arm("tot2")]
        ra = ArmRegistry(arms, max_workers=3).run("q", 3, "run")
        assert set(ra) == {"tot", "tot2"}

    def test_value_error_la_TRUOT_khong_phai_HONG(self, caplog):
        """Truy vấn không có token đánh chỉ mục được: miss, không phải sập."""
        def rong(q, k, run):
            raise ValueError("không có token nào")

        arms = [_arm("tot"), Arm(name="rong", search=rong)]
        with caplog.at_level("INFO"):
            ra = ArmRegistry(arms, max_workers=2).run("q", 3, "run")
        assert set(ra) == {"tot"}
        assert any("returned nothing" in r.message for r in caplog.records)

    def test_arm_tra_rong_khong_vao_ket_qua(self):
        arms = [_arm("co"), _arm("khong", so=0)]
        ra = ArmRegistry(arms, max_workers=2).run("q", 3, "run")
        assert set(ra) == {"co"}


class TestTatArm:
    def test_arm_tat_KHONG_chay(self):
        """Khác trọng số 0: arm tắt không tốn một mili giây nào."""
        da_chay: list[str] = []

        def search(q, k, run):
            da_chay.append("x")
            return [_cand("x", 0)]

        arms = [Arm(name="bat", search=search),
                Arm(name="tat", search=search, enabled=False)]
        ra = ArmRegistry(arms, max_workers=2).run("q", 1, "run")
        assert set(ra) == {"bat"}
        assert len(da_chay) == 1, "arm đã tắt vẫn chạy"

    def test_trong_so_0_VAN_chay(self):
        """Đây là hành vi ĐÚNG, và là lý do `arms_disabled` phải tồn tại.

        `fusion.py`: "A weight of 0 does not disable an arm". Đo được: ba arm
        trọng số 0,0 tốn 1.829 ms / 7.402 ms = 25% độ trễ.
        """
        da_chay: list[str] = []

        def search(q, k, run):
            da_chay.append("x")
            return [_cand("x", 0)]

        arms = [Arm(name="khong_trong_so", search=search, weight=0.0)]
        ra = ArmRegistry(arms, max_workers=1).run("q", 1, "run")
        assert set(ra) == {"khong_trong_so"}
        assert len(da_chay) == 1


class TestArmsDisabledTrongConfig:
    def test_ten_arm_la_thi_NEM(self):
        """Gõ sai tên thì không gì bị tắt và không gì báo — phải ném.

        Đúng lớp lỗi `ArmRegistry.run` đã dính với `["bm25","ocr","subtitles"]`:
        ba cái tên không tồn tại, và nhánh đó lặng lẽ trỏ vào hư không.
        """
        from mfe.core.config import load_config
        from mfe.online.cascade import Cascade

        # Config là frozen (pydantic) — `model_copy` chứ không gán.
        config = load_config("config/btc_b1.yaml")
        hong = config.model_copy(update={
            "retrieval": config.retrieval.model_copy(
                update={"arms_disabled": ["khong_he_ton_tai"]})
        })
        with pytest.raises(ValueError, match="không tồn tại"):
            Cascade.from_config(hong, text_store=None, dense_store_factory=None)

    def test_config_that_khai_dung_nam_arm_da_tat(self):
        """Khoá hai quyết định khác nhau, cố ý gộp vào một danh sách.

        `ocr` · `ocr_paddle` · `clap` tắt vì **hiệu năng**: ablation 25-09 ra
        +0,000 cho cả ba, mà chúng ăn 1.829 ms trên 7.402 ms.

        `object` · `ocr_vietocr` tắt vì **dữ liệu** (Lead chốt 25-09): batch 2
        không kịp sinh vietocr và `ocr_qwen25` đã phủ phần OCR. Đo trước khi
        đổi: dev 0,5192 → 0,5197 · heldout 0,5419 → 0,5486, tức bỏ còn tốt hơn
        một chút.

        Test này đỏ khi ai đó thêm/bớt mà không cập nhật lý do — và lý do mới
        là thứ đáng đọc, không phải danh sách.
        """
        from mfe.core.config import load_config

        config = load_config("config/btc_b1.yaml")
        assert set(config.retrieval.arms_disabled) == {
            "ocr", "ocr_paddle", "clap", "object", "ocr_vietocr",
            "window_bm25", "window_bge_m3",
        }
