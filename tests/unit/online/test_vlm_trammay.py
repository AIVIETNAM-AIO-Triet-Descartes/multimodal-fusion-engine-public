"""`VlmTier` đi qua cổng Trammay (Lead chốt 25-09-2026).

Vì sao cần test: đổi nhà cung cấp cho tầng này **không phải đổi khoá**.
`rerank_tiers` cũ gọi `genai.Client(api_key=os.environ["GEMINI_API_KEY"])` —
giao thức Gemini-native. Trammay là cổng **OpenAI-compatible**, nên nhét khoá
Trammay vào `genai.Client` cho ra **401**, không phải kết quả kém hơn.

Và `vlm` là tầng rerank **duy nhất** còn bằng chứng dương (+0,015), nó **bật
cho lượt nộp**, còn `Cascade._rerank_tiers` chỉ ghi `WARNING` khi một tầng ngồi
ngoài. Tức hỏng ở đây là hỏng im lặng trên đúng đường nộp bài.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfe.online import rerank_tiers
from mfe.online.rerank_tiers import TRAMMAY_ENV_KEY, VlmTier


@pytest.fixture
def anh(tmp_path: Path) -> list[Path]:
    from PIL import Image

    ra = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (1280, 720), (i * 40, 80, 120)).save(p, format="JPEG")
        ra.append(p)
    return ra


@pytest.fixture(autouse=True)
def sach_khoa(monkeypatch):
    """Không để khoá thật trong `.env` của máy Lead lọt vào test.

    Đây là lỗi ĐÃ XẢY RA trong bộ này: một test rò `GEMINI_API_KEY` sang test
    khác và làm nó đỏ tuỳ thứ tự chạy.
    """
    for k in (TRAMMAY_ENV_KEY, "GEMINI_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(k, raising=False)


class TestTenKhoaKhongTroi:
    def test_trung_voi_trammay_ENV_KEY(self):
        """Hai chỗ khai cùng một tên biến môi trường — lệch là hỏng im lặng."""
        from mfe.online.trammay import ENV_KEY

        assert TRAMMAY_ENV_KEY == ENV_KEY


class TestAvailable:
    def test_co_khoa_trammay_la_kha_dung(self, monkeypatch):
        monkeypatch.setenv(TRAMMAY_ENV_KEY, "x")
        assert VlmTier().available() == (True, None)

    def test_khong_khoa_nao_thi_ly_do_NEU_DU_BA_TEN(self):
        ok, ly_do = VlmTier().available()
        assert ok is False
        # Lý do phải nêu đủ ba đường, nếu không người vận hành đặt nhầm khoá
        # rồi không hiểu vì sao tầng vẫn ngồi ngoài.
        for ten in (TRAMMAY_ENV_KEY, "GEMINI_API_KEY", "NVIDIA_API_KEY"):
            assert ten in ly_do


class TestChonNhaCungCap:
    def test_trammay_di_TRUOC_gemini(self, monkeypatch, anh):
        """Khi có cả hai khoá, Trammay thắng. Đây chính là quyết định của Lead."""
        monkeypatch.setenv(TRAMMAY_ENV_KEY, "x")
        monkeypatch.setenv("GEMINI_API_KEY", "y")
        goi: list[str] = []
        monkeypatch.setattr(VlmTier, "_ask_trammay", lambda s, p, a: goi.append("trammay") or "[]")
        monkeypatch.setattr(VlmTier, "_ask_gemini", lambda s, p, a: goi.append("gemini") or "[]")
        VlmTier()._ask("prompt", anh)
        assert goi == ["trammay"]

    def test_trammay_di_truoc_ca_nvidia_mot_anh(self, monkeypatch, anh):
        monkeypatch.setenv(TRAMMAY_ENV_KEY, "x")
        monkeypatch.setenv("NVIDIA_API_KEY", "y")
        goi: list[str] = []
        monkeypatch.setattr(VlmTier, "_ask_trammay", lambda s, p, a: goi.append("trammay") or "[]")
        monkeypatch.setattr(VlmTier, "_ask_nvidia", lambda s, p, a: goi.append("nvidia") or "[]")
        VlmTier()._ask("prompt", anh[:1])
        assert goi == ["trammay"]

    def test_khong_co_khoa_nao_thi_NEM(self, anh):
        with pytest.raises(RuntimeError, match="TRAMMAY_API_KEY"):
            VlmTier()._ask("prompt", anh)


class TestGuiDungThuGi:
    def test_gui_anh_DA_THU_NHO_khong_phai_file_goc(self, monkeypatch, anh):
        """`_encode` hạ cạnh dài về 512 px. Gửi file gốc là gửi thừa vài MB.

        Kiểm bằng cách so kích thước base64 với base64 của chính file gốc —
        nếu ai đó đổi sang `client.chat(images=paths)` thì hai số này bằng nhau
        và test đỏ.
        """
        import base64

        monkeypatch.setenv(TRAMMAY_ENV_KEY, "x")
        ghi: dict = {}

        class GiaClient:
            def chat(self, prompt, **kw):
                ghi.update(kw)
                ghi["prompt"] = prompt
                return '["a"]'

        monkeypatch.setattr(
            "mfe.online.trammay.TrammayClient.from_env", classmethod(lambda cls, **kw: GiaClient())
        )
        VlmTier()._ask("xep hang di", anh)

        assert ghi["prompt"] == "xep hang di"
        assert "images" not in ghi, "đang gửi Path — mất bước thu nhỏ 512 px"
        assert len(ghi["images_b64"]) == 3
        for p, b64 in zip(anh, ghi["images_b64"], strict=True):
            goc = base64.b64encode(p.read_bytes()).decode()
            assert len(b64) < len(goc), "ảnh gửi đi không nhỏ hơn file gốc"

    def test_from_env_tra_None_thi_NEM_ro_rang(self, monkeypatch, anh):
        """Có khoá mà client vẫn None là mâu thuẫn — phải nổ, không im lặng."""
        monkeypatch.setenv(TRAMMAY_ENV_KEY, "x")
        monkeypatch.setattr(
            "mfe.online.trammay.TrammayClient.from_env", classmethod(lambda cls, **kw: None)
        )
        with pytest.raises(RuntimeError, match="from_env"):
            VlmTier()._ask("prompt", anh)


class TestClientNhanAnhMaHoaSan:
    def test_images_b64_vao_dung_hinh_dang_openai(self, monkeypatch):
        from mfe.online.trammay import TrammayClient

        client = TrammayClient(api_key="x", base_url="http://khong-dung", model="m")
        gui: dict = {}
        monkeypatch.setattr(
            TrammayClient, "_post",
            lambda self, payload: gui.update(payload)
            or {"choices": [{"message": {"content": "ok"}}]},
        )
        client.chat("hoi", images_b64=["QUJD", "REVG"])

        noi_dung = gui["messages"][0]["content"]
        assert noi_dung[0] == {"type": "text", "text": "hoi"}
        assert [c["image_url"]["url"] for c in noi_dung[1:]] == [
            "data:image/jpeg;base64,QUJD",
            "data:image/jpeg;base64,REVG",
        ]

    def test_khong_truyen_anh_thi_van_la_luot_hoi_chu(self, monkeypatch):
        from mfe.online.trammay import TrammayClient

        client = TrammayClient(api_key="x", base_url="http://khong-dung", model="m")
        gui: dict = {}
        monkeypatch.setattr(
            TrammayClient, "_post",
            lambda self, payload: gui.update(payload)
            or {"choices": [{"message": {"content": "ok"}}]},
        )
        client.chat("hoi")
        assert gui["messages"][0]["content"] == [{"type": "text", "text": "hoi"}]


def test_module_khong_con_bat_buoc_genai():
    """`rerank_tiers` không được import `google.genai` ở cấp module.

    Nó vẫn dùng cho nhánh dự phòng, nhưng phải nằm trong hàm — nếu không, một
    máy chỉ có khoá Trammay mà thiếu SDK Google sẽ không import nổi module, và
    `vlm` biến mất trước cả khi ai kịp hỏi `available()`.
    """
    nguon = Path(rerank_tiers.__file__).read_text(encoding="utf-8")
    for dong in nguon.splitlines():
        if dong.startswith(("import ", "from ")):
            assert "genai" not in dong, f"import genai ở cấp module: {dong}"
