"""Khoa bo doc Gemini — voi client GIA, khong goi mang lan nao.

Cai duoc kiem la phan CUA TA quanh API: prompt mang du rang buoc, duong thoat
duoc ton trong, thieu mot anh khong lam mat ca cau, va khoa API khong bao gio
nam trong code.
"""

from __future__ import annotations

import sys
import types

import pytest

from mfe.online.answering import Evidence
from mfe.online.readers import BAIL, GeminiAnswerReader


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return FakeResponse(self.reply)


class FakeClient:
    def __init__(self, reply):
        self.models = FakeModels(reply)


@pytest.fixture
def genai(monkeypatch):
    """Cai `google.genai` gia, va tra ve mot ham dat cau tra loi cua model."""
    box: dict = {}

    class Part:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"bytes": len(data), "mime": mime_type}

    module = types.ModuleType("google.genai")
    module.types = types.SimpleNamespace(Part=Part)
    module.Client = lambda api_key: box["client"]  # noqa: ARG005
    google = types.ModuleType("google")
    google.genai = module
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", module)
    monkeypatch.setenv("GEMINI_API_KEY", "khoa-gia")

    def install(reply):
        box["client"] = FakeClient(reply)
        return box["client"]

    return install


def ev(keyframe_id="L21_V001#000531", **texts):
    return Evidence(
        keyframe_id=keyframe_id,
        video_id=keyframe_id.split("#")[0],
        frame_id=int(keyframe_id.split("#")[1]),
        texts=texts or {"ocr": "TIN CHINH 06:30:28"},
    )


class TestAnswer:
    def test_it_returns_the_models_short_answer(self, genai):
        genai("06:30:28")
        answer = GeminiAnswerReader()("mấy giờ?", [ev()])
        assert answer.text == "06:30:28"
        assert answer.keyframe_id == "L21_V001#000531"
        assert answer.reader.startswith("gemini:")

    def test_only_the_first_line_is_kept(self, genai):
        """Model doi khi them mot cau giai thich duoi dap an. BTC so `answer`
        theo ngu nghia — mot cau dai chua dap an dung van co the truot."""
        genai("màu xanh\nVì trong ảnh có một chiếc ly màu xanh.")
        assert GeminiAnswerReader()("màu gì?", [ev()]).text == "màu xanh"

    def test_backticks_are_stripped(self, genai):
        genai("`5`")
        assert GeminiAnswerReader()("mấy người?", [ev()]).text == "5"


class TestBail:
    def test_the_bail_token_becomes_none_not_a_string_answer(self, genai):
        """Nop chuoi 'KHONG_DOC_DUOC' lam dap an la nop mot dap an chac chan sai
        ma trong nhu he thong da tra loi."""
        genai(BAIL)
        reader = GeminiAnswerReader()
        assert reader("hỏi gì đó", [ev()]) is None
        assert reader.bails == 1

    def test_an_empty_reply_is_a_bail(self, genai):
        genai("   ")
        assert GeminiAnswerReader()("hỏi gì đó", [ev()]) is None

    def test_bails_are_counted_against_calls(self, genai):
        """Duong thoat la co y (nguoc voi bai hoc o `_QWEN_PROMPT`), nhung phai
        DEM duoc de thay neu model lam dung no."""
        genai(BAIL)
        reader = GeminiAnswerReader()
        for _ in range(3):
            reader("hỏi", [ev()])
        assert (reader.bails, reader.calls) == (3, 3)


class TestPrompt:
    def _prompt(self, client) -> str:
        contents = client.models.calls[0]["contents"]
        return next(part for part in contents if isinstance(part, str))

    def test_the_prompt_carries_the_question_and_the_evidence(self, genai):
        client = genai("x")
        GeminiAnswerReader()("cầm ly màu gì?", [ev(ocr="LY MAU XANH")])
        prompt = self._prompt(client)
        assert "cầm ly màu gì?" in prompt
        assert "LY MAU XANH" in prompt

    def test_it_forbids_outside_knowledge_and_demands_a_short_answer(self, genai):
        """Khong co hai rang buoc nay thi model tra loi tu kien thuc chung —
        nghe hop ly, sai video, va nguoi van hanh khong phan biet duoc."""
        client = genai("x")
        GeminiAnswerReader()("hỏi", [ev()])
        prompt = self._prompt(client)
        assert "Không dùng kiến thức bên ngoài" in prompt
        assert "NGẮN NHẤT" in prompt
        assert BAIL in prompt

    def test_it_names_the_model_it_was_configured_with(self, genai):
        client = genai("x")
        GeminiAnswerReader(model="gemini-2.5-pro")("hỏi", [ev()])
        assert client.models.calls[0]["model"] == "gemini-2.5-pro"

    def test_evidence_is_capped_at_the_image_budget(self, genai):
        """Qua nhieu khung hinh thi cau hoi bi pha loang va model bat dau tra loi
        ve khung hinh khac."""
        client = genai("x")
        many = [ev(f"L21_V001#{n:06d}", ocr=f"khung {n}") for n in (1, 2, 3, 4, 5)]
        GeminiAnswerReader(images=2)("hỏi", many)
        prompt = self._prompt(client)
        assert "khung 1" in prompt and "khung 2" in prompt
        assert "khung 5" not in prompt


class TestImages:
    def test_images_are_sent_as_bytes(self, genai, tmp_path):
        client = genai("x")
        path = tmp_path / "kf.jpg"
        path.write_bytes(b"\xff\xd8fake-jpeg")
        GeminiAnswerReader()("hỏi", [ev()], [path])
        blobs = [c for c in client.models.calls[0]["contents"] if isinstance(c, dict)]
        assert blobs == [{"bytes": path.stat().st_size, "mime": "image/jpeg"}]

    def test_a_missing_image_does_not_lose_the_whole_question(self, genai, tmp_path):
        """Thieu mot anh thi doc bang phan con lai. Mot luot thi dai 5 phut."""
        client = genai("06:30")
        answer = GeminiAnswerReader()("hỏi", [ev()], [tmp_path / "khong-co.jpg"])
        assert answer.text == "06:30"
        assert not [c for c in client.models.calls[0]["contents"] if isinstance(c, dict)]

    def test_more_images_than_the_budget_are_dropped(self, genai, tmp_path):
        client = genai("x")
        paths = []
        for n in range(5):
            path = tmp_path / f"{n}.jpg"
            path.write_bytes(b"\xff\xd8x")
            paths.append(path)
        GeminiAnswerReader(images=2)("hỏi", [ev()], paths)
        blobs = [c for c in client.models.calls[0]["contents"] if isinstance(c, dict)]
        assert len(blobs) == 2


class TestApiKey:
    def test_a_missing_key_says_where_to_put_it(self, genai, monkeypatch):
        """§9.3: khoa nam trong bien moi truong, khong trong code hay config."""
        genai("x")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GeminiAnswerReader()("hỏi", [ev()])

    def test_the_client_is_built_once_and_reused(self, genai):
        genai("x")
        reader = GeminiAnswerReader()
        reader("hỏi", [ev()])
        first = reader._client
        reader("hỏi lần hai", [ev()])
        assert reader._client is first
