"""Khoá ba tầng rerank dựng lại từ SOTA_AIC — mfe.online.rerank_tiers.

Ba tầng này đều gọi ra ngoài: một model 6 GB, một HTTP API, một VLM tính tiền.
Nên test ở đây **không** kiểm chúng xếp hạng giỏi hay dở — cái đó phải đo trên
bộ dev. Nó kiểm đúng một thứ: **khi hỏng thì hỏng thế nào**.

Lý do là chính sách: bật một tầng hỏng phải *không bao giờ tệ hơn* tắt nó. Một
tầng làm mất ứng viên, hay ném lỗi ra ngoài, hay im lặng trả về danh sách rỗng,
sẽ biến một lượt nộp bình thường thành một lượt nộp trắng — và trên giao diện thì
người vận hành chỉ thấy "đã bật rerank".
"""

from __future__ import annotations

import json

import pytest

from mfe.contracts.tools import Candidate
from mfe.online.rerank_tiers import (
    TRAMMAY_ENV_KEY,
    ColPaliTier,
    HfCrossEncoderTier,
    TierOutcome,
    VlmTier,
    _parse_id_array,
    _reorder,
    _score_of,
    apply_tiers,
)

#: Mọi khoá nhà cung cấp mà một tầng có thể hỏi tới.
KHOA_NHA_CUNG_CAP = (TRAMMAY_ENV_KEY, "GEMINI_API_KEY", "NVIDIA_API_KEY", "HF_TOKEN")


@pytest.fixture(autouse=True)
def khong_dung_khoa_that(monkeypatch):
    """Xoá mọi khoá khỏi môi trường TRƯỚC mỗi test trong module này.

    Không phải phép lịch sự — đây là lỗi ĐÃ XẢY RA hai lần. Cả file này kiểm
    "khi hỏng thì hỏng thế nào", mà phần lớn nhánh hỏng được kích hoạt bằng
    **thiếu khoá**. Nên kết quả từng phụ thuộc vào `.env` của máy đang chạy:
    `test_tang_khong_chay_khong_lam_doi_thu_tu` xanh trên máy không có khoá và
    đỏ trên máy có, với một `TypeError: 'NoneType' object is not callable` chẳng
    liên quan gì tới thứ nó định kiểm.

    Nó nổ lần gần nhất khi `VlmTier` học thêm đường Trammay (25-09): hai test
    xoá tay `GEMINI_API_KEY` + `NVIDIA_API_KEY` bỗng thiếu một tên. Xoá theo
    DANH SÁCH ở một chỗ thì thêm nhà cung cấp mới không đẻ ra lỗi kiểu đó nữa.
    """
    for ten in KHOA_NHA_CUNG_CAP:
        monkeypatch.delenv(ten, raising=False)


def candidates(n: int = 5) -> list[Candidate]:
    return [
        Candidate(
            keyframe_id=f"L21_V001#{i:06d}",
            video_id="L21_V001",
            score=1.0 / (i + 1),
            evidence_id=f"arm:run:{i + 1}",
        )
        for i in range(n)
    ]


class TestXepLai:
    def test_theo_dung_thu_tu_duoc_giao(self):
        pool = candidates(3)
        order = [pool[2].keyframe_id, pool[0].keyframe_id, pool[1].keyframe_id]
        assert [c.keyframe_id for c in _reorder(pool, order)] == order

    def test_id_thieu_thi_bam_duoi_chu_khong_bi_bo(self):
        """Model trả thiếu id là chuyện thường — nó tóm tắt, nó gõ sai.

        Bỏ ứng viên đi thì tầng rerank có thể làm kết quả TỆ HƠN lúc chưa rerank,
        và đó là điều duy nhất nó không được phép làm.
        """
        pool = candidates(4)
        after = _reorder(pool, [pool[3].keyframe_id])
        assert len(after) == 4
        assert after[0] is pool[3]
        assert [c.keyframe_id for c in after[1:]] == [c.keyframe_id for c in pool[:3]]

    def test_id_bia_ra_thi_bo_qua(self):
        pool = candidates(2)
        after = _reorder(pool, ["KHONG_TON_TAI#000000", pool[1].keyframe_id])
        assert [c.keyframe_id for c in after] == [
            pool[1].keyframe_id, pool[0].keyframe_id
        ]

    def test_id_TRUNG_khong_duoc_nhan_doi_ung_vien(self):
        """Model trả trùng id cũng thường như trả thiếu — và trùng thì đắt hơn.

        Bản trước lọc theo `k in by_id`, nên một id lặp hai lần trong `order`
        sinh ra HAI dòng cùng một keyframe. Trên đường nộp đó là mất trắng một
        suất trong 100 dòng: BTC chấm R@k theo thứ hạng, nên dòng thứ hai không
        thêm cơ hội nào mà vẫn chiếm chỗ của một ứng viên khác.
        """
        pool = candidates(4)
        after = _reorder(
            pool,
            [pool[2].keyframe_id, pool[2].keyframe_id, pool[0].keyframe_id],
        )
        ids = [c.keyframe_id for c in after]
        assert len(ids) == len(set(ids)) == 4
        assert ids[:2] == [pool[2].keyframe_id, pool[0].keyframe_id]

    def test_danh_sach_rong(self):
        assert _reorder([], ["gi do"]) == []


class TestDocMangId:
    def test_json_tran(self):
        assert _parse_id_array('["a", "b"]') == ["a", "b"]

    def test_co_markdown_va_loi_dan(self):
        text = 'Here is the ranking:\n```json\n["b", "a"]\n```\nHope this helps!'
        assert _parse_id_array(text) == ["b", "a"]

    def test_khong_co_mang_thi_tra_rong(self):
        assert _parse_id_array("I cannot rank these images.") == []

    def test_json_hong_thi_tra_rong_chu_khong_nem(self):
        assert _parse_id_array('["a", "b"') == []
        assert _parse_id_array("[oops]") == []

    def test_json_khong_phai_mang(self):
        assert _parse_id_array('{"a": 1}') == []


class TestDocDiemHf:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ([{"label": "LABEL_0", "score": 0.9}], 0.9),
            ({"score": 0.7}, 0.7),
            (0.5, 0.5),
            ("khong phai so", 0.0),
            ([], 0.0),
        ],
    )
    def test_ba_hinh_dang_HF_tra_ve(self, payload, expected):
        """HF đổi hình dạng theo model. Đọc cả ba, và cái lạ thì về 0 chứ không nổ."""
        assert _score_of(payload) == pytest.approx(expected)


class TestKhongChayDuoc:
    """Mọi lý do không chạy đều phải là `ran=False` + `reason`, không phải exception."""

    def test_colpali_thieu_ban_dich(self):
        pool = candidates()
        after, outcome = ColPaliTier().rerank(
            pool, query_vi="mua lan", query_en=None, images=lambda _: None
        )
        assert after == pool
        assert not outcome.ran and "tieng Anh" in outcome.reason

    def test_colpali_thieu_anh(self, monkeypatch):
        monkeypatch.setattr(ColPaliTier, "available", lambda self: (True, None))
        pool = candidates()
        after, outcome = ColPaliTier().rerank(
            pool, query_vi="x", query_en="x", images=lambda _: None
        )
        assert after == pool and not outcome.ran
        assert "anh" in outcome.reason

    def test_hf_thieu_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        pool = candidates()
        after, outcome = HfCrossEncoderTier().rerank(
            pool, query_vi="x", query_en=None, texts=lambda _: "co chu"
        )
        assert after == pool and not outcome.ran
        assert "HF_TOKEN" in outcome.reason

    def test_hf_khong_ung_vien_nao_co_van_ban(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        pool = candidates()
        after, outcome = HfCrossEncoderTier().rerank(
            pool, query_vi="x", query_en=None, texts=lambda _: "   "
        )
        assert after == pool and not outcome.ran

    def test_vlm_thieu_key(self):
        # Khoá đã được `khong_dung_khoa_that` xoá sạch — xoá tay ở đây là chỗ
        # sẽ quên một tên khi có nhà cung cấp mới.
        pool = candidates()
        after, outcome = VlmTier().rerank(
            pool, query_vi="x", query_en="x", images=lambda _: None
        )
        assert after == pool and not outcome.ran
        assert "API_KEY" in outcome.reason

    def test_hf_loi_mang_khong_thoat_ra_ngoai(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        import mfe.online.rerank_tiers as module

        class FakeRequests:
            @staticmethod
            def post(*_args, **_kwargs):
                raise TimeoutError("het gio")

        monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)
        pool = candidates()
        after, outcome = module.HfCrossEncoderTier().rerank(
            pool, query_vi="x", query_en=None, texts=lambda _: "co chu"
        )
        assert after == pool and not outcome.ran
        assert "TimeoutError" in outcome.reason


class TestChuoi:
    def test_tang_khong_chay_khong_lam_doi_thu_tu(self):
        pool = candidates(4)
        after, outcomes = apply_tiers(
            pool, [ColPaliTier(), VlmTier()], query_vi="x", query_en=None
        )
        assert [c.keyframe_id for c in after] == [c.keyframe_id for c in pool]
        assert all(not o.ran for o in outcomes)

    def test_tang_sau_an_dau_ra_tang_truoc(self):
        pool = candidates(3)

        class Reverse:
            name = "dao"

            def rerank(self, cands, **_):
                out = list(reversed(cands))
                return out, TierOutcome(self.name, True, None, len(out))

        class First:
            name = "dau"

            def rerank(self, cands, **_):
                out = [cands[-1], *cands[:-1]]
                return out, TierOutcome(self.name, True, None, len(out))

        after, outcomes = apply_tiers(pool, [Reverse(), First()], query_vi="x")
        # dao: [2,1,0] -> dau lay cuoi len dau: [0,2,1]
        assert [c.keyframe_id for c in after] == [
            pool[0].keyframe_id, pool[2].keyframe_id, pool[1].keyframe_id
        ]
        assert [o.tier for o in outcomes] == ["dao", "dau"]

    def test_tang_lam_mat_ung_vien_thi_no_ngay(self):
        """Một tầng chỉ được đổi THỨ TỰ. Cắt bớt là biến rerank thành bộ lọc, và
        một bộ lọc ở đây làm bài nộp thiếu dòng mà không ai thấy."""
        pool = candidates(4)

        class Truncating:
            name = "cat"

            def rerank(self, cands, **_):
                return list(cands[:2]), TierOutcome(self.name, True, None, 2)

        with pytest.raises(AssertionError, match="mat ung vien"):
            apply_tiers(pool, [Truncating()], query_vi="x")


class TestGhiNhan:
    def test_moved_chi_ghi_cai_doi_cho(self):
        pool = candidates(3)

        class Swap:
            name = "hoan"

            def rerank(self, cands, **_):
                out = [cands[1], cands[0], cands[2]]
                from mfe.online.rerank_tiers import _moved

                return out, TierOutcome(
                    self.name, True, None, 3, 0.0, _moved(cands, out)
                )

        _, outcomes = apply_tiers(pool, [Swap()], query_vi="x")
        moved = outcomes[0].moved
        assert set(moved) == {pool[0].keyframe_id, pool[1].keyframe_id}
        assert moved[pool[1].keyframe_id] == (2, 1)

    def test_summary_doc_duoc_ca_hai_truong_hop(self):
        assert "khong chay" in TierOutcome("x", False, "thieu key").summary()
        assert "cham 5" in TierOutcome("x", True, None, 5, 1.2).summary()


class TestPromptVlm:
    def test_prompt_liet_ke_dung_thu_tu_id(self):
        ids = ["a#000001", "b#000002"]
        prompt = VlmTier._prompt("mua lan", ids)
        assert json.dumps(ids) in prompt
        assert "JSON array" in prompt


class TestGoBocLoHfTraVe:
    """HF router trả `[[s1, s2, …]]` — lồng sâu hơn một lớp so với ai cũng đoán.

    Đây là lỗi ĐÃ XẢY RA và nó im lặng: đọc thẳng `payload[i]` lấy đúng phần tử
    đầu, 11 ứng viên còn lại giữ điểm cũ, và tầng báo "đã chấm 12". Bản gốc trong
    `SOTA_AIC` vướng đúng chỗ này.
    """

    def test_hinh_dang_that_cua_hf_router(self):
        from mfe.online.rerank_tiers import _unwrap_batch

        payload = [[{"score": 0.5}, {"score": 0.1}, {"score": 0.9}]]
        flat = _unwrap_batch(payload, 3)
        assert flat is not None and len(flat) == 3
        assert [_score_of(x) for x in flat] == [0.5, 0.1, 0.9]

    def test_hinh_dang_phang_cung_nhan(self):
        from mfe.online.rerank_tiers import _unwrap_batch

        assert _unwrap_batch([{"score": 0.5}, {"score": 0.1}], 2) is not None

    def test_lech_so_luong_thi_TU_CHOI_chu_khong_cham_mot_phan(self):
        from mfe.online.rerank_tiers import _unwrap_batch

        assert _unwrap_batch([[{"score": 0.5}]], 3) is None
        assert _unwrap_batch([{"score": 0.5}], 3) is None
        assert _unwrap_batch("khong phai list", 3) is None

    def test_tang_bao_khong_chay_khi_hinh_dang_la(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return [{"score": 0.9}]          # 1 diem cho 3 cap

        class FakeRequests:
            @staticmethod
            def post(*_a, **_k):
                return FakeResponse()

        monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)
        pool = candidates(3)
        after, outcome = HfCrossEncoderTier().rerank(
            pool, query_vi="x", query_en=None, texts=lambda _: "co chu", depth=3
        )
        assert [c.keyframe_id for c in after] == [c.keyframe_id for c in pool]
        assert not outcome.ran and "hinh dang la" in outcome.reason


class TestTangCuaTa:
    """`OurRerankTier` — `Reranker` của ta khoác giao diện tầng.

    Nó tồn tại để bàn thử Streamlit và bản đo chạy **cùng một** tầng. Nên thứ
    phải khoá là hợp đồng chung của mọi tầng, không phải chất lượng xếp hạng:
    không mất ứng viên, không ném ra ngoài, `depth` được tôn trọng.
    """

    class FakeReranker:
        def __init__(self, order=None, error=None):
            self.order = order
            self.error = error
            self.seen = None

        def rerank(self, query, cands, run_id="r"):
            self.seen = list(cands)
            if self.error is not None:
                raise self.error
            if self.order is None:
                return list(reversed(cands))
            by_id = {c.keyframe_id: c for c in cands}
            return [by_id[k] for k in self.order if k in by_id]

    def test_xep_lai_va_giu_du_ung_vien(self):
        from mfe.online.rerank_tiers import OurRerankTier

        pool = candidates(4)
        after, outcome = OurRerankTier(self.FakeReranker()).rerank(pool, query_vi="x")
        assert len(after) == 4
        assert [c.keyframe_id for c in after] == [c.keyframe_id for c in reversed(pool)]
        assert outcome.ran and outcome.scored == 4

    def test_depth_chi_cham_phan_dau_con_duoi_giu_nguyen(self):
        from mfe.online.rerank_tiers import OurRerankTier

        pool = candidates(5)
        fake = self.FakeReranker()
        after, outcome = OurRerankTier(fake).rerank(pool, query_vi="x", depth=2)
        assert len(fake.seen) == 2, "chi hai ung vien dau duoc cham"
        assert len(after) == 5
        assert [c.keyframe_id for c in after[2:]] == [c.keyframe_id for c in pool[2:]]

    def test_loi_khong_thoat_ra_ngoai_va_giu_nguyen_thu_tu(self):
        from mfe.online.rerank_tiers import OurRerankTier

        pool = candidates(3)
        tier = OurRerankTier(self.FakeReranker(error=RuntimeError("model chet")))
        after, outcome = tier.rerank(pool, query_vi="x")
        assert [c.keyframe_id for c in after] == [c.keyframe_id for c in pool]
        assert not outcome.ran and "RuntimeError" in outcome.reason

    def test_khong_co_reranker_thi_bao_ly_do(self):
        from mfe.online.rerank_tiers import OurRerankTier

        pool = candidates(2)
        after, outcome = OurRerankTier(None).rerank(pool, query_vi="x")
        assert after == pool and not outcome.ran

    def test_cam_duoc_vao_chuoi_nhu_moi_tang_khac(self):
        from mfe.online.rerank_tiers import OurRerankTier

        pool = candidates(3)
        after, outcomes = apply_tiers(
            pool, [OurRerankTier(self.FakeReranker()), ColPaliTier()],
            query_vi="x", query_en=None,
        )
        assert len(after) == 3
        assert [o.tier for o in outcomes] == ["ta", "colpali"]
