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
    quest_clauses,
    quest_expand,
    split_trake_events,
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
        for key in ("TRAMMAY_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        ok, reason = QuestClient().available()
        assert not ok
        assert "API_KEY" in reason


class TestTachMenhDe:
    """`quest_clauses` — nguồn THỨ BA cho mệnh đề, cạnh bộ luật và ô gõ tay.

    Bộ luật cứng (`scene_order.split_clauses`) dò từ nối, nên nó chỉ đạt ngưỡng
    ≥3 mệnh đề ở **3/38** đề KIS thật. Hàm này đọc được câu không có từ nối nào.
    Nhưng nó phải im lặng đúng chỗ: câu tả MỘT khoảnh khắc mà bị tách ra là truy
    vấn hỏng, và đó là trường hợp thường gặp nhất.
    """

    THREE = (
        '{"clauses": ['
        '{"text": "người đàn ông mở cửa tiệm", "order": 1},'
        '{"text": "người đàn ông bày hàng lên kệ", "order": 2},'
        '{"text": "khách bước vào mua đồ", "order": 3}]}'
    )

    def test_doc_duoc_va_giu_thu_tu(self):
        client = FakeClient(self.THREE)
        assert quest_clauses("x", client) == [
            "người đàn ông mở cửa tiệm",
            "người đàn ông bày hàng lên kệ",
            "khách bước vào mua đồ",
        ]

    def test_XEP_LAI_theo_order_chu_khong_theo_thu_tu_tra_ve(self):
        """Đề kể ngược thì `order` mới là trình tự thật, không phải vị trí trong JSON.

        DANTE thưởng cho chuỗi TĂNG DẦN, nên một danh sách đảo chiều không chỉ
        vô ích — nó chấm điểm cho đúng những video kể sai thứ tự.
        """
        client = FakeClient(
            '{"clauses": ['
            '{"text": "giơ tay ăn mừng", "order": 3},'
            '{"text": "xuất phát khỏi vạch", "order": 1},'
            '{"text": "về đích đầu tiên", "order": 2}]}'
        )
        assert quest_clauses("x", client) == [
            "xuất phát khỏi vạch", "về đích đầu tiên", "giơ tay ăn mừng",
        ]

    def test_co_markdown_va_loi_dan(self):
        client = FakeClient(f"Được thôi!\n```json\n{self.THREE}\n```")
        assert len(quest_clauses("x", client)) == 3

    def test_RONG_la_cau_tra_loi_dung(self):
        """Câu tả một khoảnh khắc -> `[]`. Không phải lỗi, và là ca thường gặp nhất."""
        assert quest_clauses("x", FakeClient('{"clauses": []}')) == []

    def test_client_khong_san_sang_thi_KHONG_goi(self):
        client = FakeClient(ready=(False, "chua co khoa"))
        assert quest_clauses("x", client) == []
        assert client.calls == 0

    def test_LLM_nem_thi_tra_rong(self):
        client = FakeClient(error=RuntimeError("het quota"))
        assert quest_clauses("x", client) == []

    @pytest.mark.parametrize("reply", ["khong co json", "", '{"clauses": khong-json}'])
    def test_tra_ve_khong_doc_duoc_thi_rong(self, reply):
        assert quest_clauses("x", FakeClient(reply)) == []

    def test_bo_menh_de_RONG_va_khong_phai_dict(self):
        client = FakeClient(
            '{"clauses": [{"text": "  ", "order": 1}, "rac", '
            '{"text": "cảnh thật", "order": 2}]}'
        )
        assert quest_clauses("x", client) == ["cảnh thật"]

    def test_thieu_order_thi_dung_thu_tu_tra_ve_chu_KHONG_bo(self):
        """Bỏ một mệnh đề vì thiếu `order` là mất hẳn một cảnh — đắt hơn nhiều
        so với dùng thứ tự LLM đã trả về, vốn đã là một phỏng đoán hợp lý."""
        client = FakeClient(
            '{"clauses": [{"text": "một"}, {"text": "hai"}, {"text": "ba"}]}'
        )
        assert quest_clauses("x", client) == ["một", "hai", "ba"]

    def test_order_hong_kieu_van_giu_menh_de(self):
        client = FakeClient(
            '{"clauses": [{"text": "một", "order": "x"}, {"text": "hai", "order": 2}]}'
        )
        assert set(quest_clauses("x", client)) == {"một", "hai"}


def test_split_trake_paste_keeps_preamble_and_event_order():
    preamble, events = split_trake_events(
        "Một món ăn làm từ tôm và các loại gia vị.\n"
        "E1: Khoảnh khắc người đầu bếp bắt đầu trộn hỗn hợp nước xốt.\n"
        "E2: Khoảnh khắc người đầu bếp bắt đầu cắt bỏ đầu tôm.\n"
        "E3: Người đầu bếp cho tôm ra dĩa.\n"
        "E4: Khoảnh khắc tép cam thứ 4 được xếp lên dĩa."
    )
    assert preamble == "Một món ăn làm từ tôm và các loại gia vị."
    assert len(events) == 4
    assert events[0].startswith("Khoảnh khắc người đầu bếp bắt đầu trộn")
    assert events[3].startswith("Khoảnh khắc tép cam thứ 4")


def test_split_trake_paste_uses_line_order_when_labels_are_wrong():
    _, events = split_trake_events("E1: một\nE3: hai\nE2: ba")
    assert events == ["một", "hai", "ba"]


def test_prompt_tach_menh_de_noi_ro_ba_luat_dat_gia():
    """Prompt là DỮ LIỆU, và ba luật này là ba chỗ đã trả giá ở nơi khác.

    Kiểm chúng còn trong prompt vì một lần sửa prompt cho gọn sẽ lặng lẽ bỏ đi
    ràng buộc, và hậu quả chỉ lộ ra sau một lượt đo.
    """
    from mfe.online.quest import QUEST_CLAUSE_PROMPT

    assert "DANH SÁCH RỖNG" in QUEST_CLAUSE_PROMPT      # mot khoanh khac -> rong
    assert "dưới 3 mệnh đề" in QUEST_CLAUSE_PROMPT      # nguong MIN_CLAUSES_FOR_PRIOR
    assert "THỜI GIAN" in QUEST_CLAUSE_PROMPT           # order theo thoi gian
    assert "KHÔNG thêm chi tiết" in QUEST_CLAUSE_PROMPT  # cam bia them
