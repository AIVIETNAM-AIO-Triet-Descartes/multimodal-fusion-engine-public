"""Khoá `mfe.online.quest` — phần MỞ RỘNG truy vấn, không phải phần gọi LLM.

Test ở đây không gọi API nào. Cái nó khoá là ba tính chất mà một lượt gọi thật
không kiểm được, vì lượt gọi thật lúc nào cũng trả về *gì đó*:

1. **Rỗng là câu trả lời hợp lệ.** 42/49 đề đã nêu đủ tên. Một từ bịa thêm vào
   truy vấn là recall mất đi, nên đường "không mở rộng" phải là đường mặc định
   khi có bất cứ nghi ngờ nào — kể cả khi LLM chết.
2. **Mở rộng là CỘNG THÊM, không phải THAY.** Đổi *"phim Spielberg 1975"* thành
   *"cá mập"* là mất luôn "thị trấn ven biển", "du khách" — mọi thứ đã đúng.
3. **Không đường nào được ném ra ngoài.** Không có bản mở rộng thì vẫn tìm được
   bằng nguyên câu; một exception ở tầng 0 làm hỏng cả truy vấn.
"""

from __future__ import annotations

import pytest

from mfe.online.quest import (
    ExpansionTerm,
    QuestClient,
    expanded_queries,
    quest_expand,
)


class FakeClient:
    """`QuestClient` giả — trả đúng chuỗi được giao, hoặc ném."""

    def __init__(self, reply: str | None = None, error: Exception | None = None,
                 ready: tuple[bool, str | None] = (True, None)) -> None:
        self.reply = reply
        self.error = error
        self.ready = ready
        self.calls = 0

    def available(self):
        return self.ready

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.reply or ""


class TestMoRong:
    def test_doc_duoc_json_tran(self):
        client = FakeClient(
            '{"terms": [{"vi": "cá mập", "en": "shark", "why": "phim Jaws 1975"}]}'
        )
        terms = quest_expand("phim Spielberg 1975", client)
        assert terms == [ExpansionTerm(vi="cá mập", en="shark", why="phim Jaws 1975")]

    def test_co_markdown_va_loi_dan(self):
        client = FakeClient(
            'Sure!\n```json\n{"terms": [{"vi": "EPFL", "en": "EPFL"}]}\n```\n'
        )
        assert [t.vi for t in quest_expand("x", client)] == ["EPFL"]

    def test_danh_sach_rong_la_cau_tra_loi_HOP_LE(self):
        """Đề đã nêu đủ tên thì không được thêm gì. Đây là trường hợp THƯỜNG."""
        client = FakeClient('{"terms": []}')
        assert quest_expand("tay dua ao vang gio hai tay", client) == []

    def test_provider_chet_thi_tra_rong_chu_khong_nem(self):
        client = FakeClient(ready=(False, "khong co GEMINI_API_KEY"))
        assert quest_expand("x", client) == []
        assert client.calls == 0, "khong duoc goi API khi da biet no khong chay"

    def test_loi_khi_goi_thi_tra_rong_chu_khong_nem(self):
        client = FakeClient(error=TimeoutError("het gio"))
        assert quest_expand("x", client) == []

    @pytest.mark.parametrize(
        "reply",
        ["khong co json o day", '{"terms": [', '{"khac": 1}', ""],
    )
    def test_tra_loi_khong_doc_duoc_thi_tra_rong(self, reply):
        assert quest_expand("x", FakeClient(reply)) == []

    def test_muc_thieu_ca_hai_ngon_ngu_bi_bo(self):
        client = FakeClient('{"terms": [{"vi": "", "en": "  ", "why": "z"}, {"vi": "hổ"}]}')
        assert [t.vi for t in quest_expand("x", client)] == ["hổ"]

    def test_muc_khong_phai_dict_bi_bo(self):
        client = FakeClient('{"terms": ["ca map", {"en": "shark"}]}')
        assert [t.en for t in quest_expand("x", client)] == ["shark"]


class TestGhepVaoTruyVan:
    def test_noi_vao_CUOI_chu_khong_thay_the(self):
        terms = [ExpansionTerm(vi="cá mập", en="shark")]
        vi, en = expanded_queries("thị trấn ven biển", "a coastal town", terms)
        assert vi == "thị trấn ven biển cá mập"
        assert en == "a coastal town shark"

    def test_khong_co_tu_nao_thi_giu_nguyen(self):
        vi, en = expanded_queries("nguyen cau", "the query", [])
        assert (vi, en) == ("nguyen cau", "the query")

    def test_khong_co_ban_tieng_anh_thi_van_mo_rong_ban_tieng_viet(self):
        """`siglip` ngồi ngoài khi thiếu tiếng Anh, nhưng arm văn bản thì không."""
        vi, en = expanded_queries("cau hoi", None, [ExpansionTerm(vi="hổ", en="tiger")])
        assert vi == "cau hoi hổ"
        assert en is None

    def test_tu_thieu_mot_ngon_ngu_chi_gop_vao_ngon_ngu_no_co(self):
        terms = [ExpansionTerm(vi="hổ Bengal", en=""), ExpansionTerm(vi="", en="Jaws")]
        vi, en = expanded_queries("q", "q_en", terms)
        assert vi == "q hổ Bengal"
        assert en == "q_en Jaws"


class TestProviderSanSang:
    def test_khong_khoa_nao_thi_noi_ro_thieu_khoa(self, monkeypatch):
        for key in ("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        ok, reason = QuestClient().available()
        assert not ok
        assert "API_KEY" in reason
