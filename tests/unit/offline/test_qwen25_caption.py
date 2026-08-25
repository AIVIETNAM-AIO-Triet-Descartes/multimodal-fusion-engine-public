"""Khoá hai tầng N6 Qwen2.5-VL — `mfe.offline.stages`.

Không test nào ở đây nạp vLLM. Cái được khoá là ba thứ mà một lượt chạy GPU thật
**không** phát hiện được, vì lượt đó lúc nào cũng trả về *gì đó*:

1. **Phép tách hai trường.** Một lần sinh, hai artifact. Mất nhãn `[TEXT]` thì cả
   câu trả lời chảy vào caption và bảng kết quả trông vẫn bình thường — nên phép
   tách phải nói ra được là nó có thấy nhãn hay không.
2. **Cache câu trả lời.** Tầng thứ hai chạy lại `llm.generate()` là gấp đôi tiền
   GPU của cả lượt, không lỗi nào báo. Thứ duy nhất lộ ra là số lần gọi.
3. **Địa chỉ.** `n` (tên file ảnh) và `frame_idx` (khoá của kho) là hai con số
   khác nhau. Nhầm chúng thì mọi thứ vẫn chạy, chỉ trỏ vào khung hình khác.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfe.offline import stages
from mfe.offline.stages import Keyframe, StageContext, parse_qwen25_caption_ocr


class TestTachHaiTruong:
    def test_dung_dinh_dang_thi_tach_dung(self):
        caption, ocr, marked = parse_qwen25_caption_ocr(
            "[CAPTION] Two anchors at a desk.\n\n[TEXT] HTV9 HD\n06:31:14"
        )
        assert caption == "Two anchors at a desk."
        assert ocr == "HTV9 HD\n06:31:14"
        assert marked is True

    def test_go_cap_ngoac_don_prompt_da_goi_y(self):
        """Prompt viết `[CAPTION] (your description here)` nên model trả cả ngoặc.

        Giữ nguyên thì mọi caption bắt đầu bằng `(` — vô hại với BM25 nhưng làm
        phép đối chiếu với bản của Bảo so hai chuỗi khác nhau.
        """
        caption, ocr, _ = parse_qwen25_caption_ocr(
            "[CAPTION] (The image shows a road.)\n\n[TEXT] (BIEN BAO)"
        )
        assert caption == "The image shows a road."
        assert ocr == "BIEN BAO"

    def test_ngoac_long_nhau_thi_GIU_NGUYEN(self):
        """Chỉ gỡ khi chắc chắn. Một caption có ngoặc bên trong thì cặp ngoài
        không còn là bao bì, và cắt bừa hai đầu là ăn mất chữ."""
        caption, _, _ = parse_qwen25_caption_ocr(
            "[CAPTION] (A man (in blue) walks.)\n\n[TEXT] None"
        )
        assert caption.startswith("(A man")

    def test_mat_nhan_TEXT_thi_bao_ra_va_KHONG_doan(self):
        """Không có `[TEXT]` thì OCR để TRỐNG, không cắt bừa nửa sau.

        Đoán sai ở đây là đổ một đoạn mô tả cảnh vật vào chỉ mục OCR — im lặng,
        và chỉ lộ ra khi ai đó đọc một dòng ngẫu nhiên của `qwen_ocr.json`.
        """
        caption, ocr, marked = parse_qwen25_caption_ocr("[CAPTION] A road at night.")
        assert caption == "A road at night."
        assert ocr == ""
        assert marked is False

    def test_khong_nhan_nao_thi_ca_cau_la_caption(self):
        caption, ocr, marked = parse_qwen25_caption_ocr("A road at night.")
        assert caption == "A road at night."
        assert ocr == "" and marked is False

    @pytest.mark.parametrize("bail", ["None", "none", "no text", "N/A", "khong", "-"])
    def test_tu_bo_cuoc_bi_loc_theo_TUNG_DONG(self, bail):
        """Prompt CÓ cho sẵn đường thoát `'None'`, nên model sẽ dùng nó.

        Lọc theo dòng chứ không theo cả chuỗi: chỗ mà phép lọc cũ thất bại là khi
        model trả CẢ chữ thật LẪN từ bỏ cuộc trong cùng một câu trả lời.
        """
        _, ocr, _ = parse_qwen25_caption_ocr(f"[CAPTION] x\n\n[TEXT] BARBER SHOP\n{bail}")
        assert ocr == "BARBER SHOP"

    def test_chi_co_tu_bo_cuoc_thi_ocr_rong(self):
        _, ocr, marked = parse_qwen25_caption_ocr("[CAPTION] x\n\n[TEXT] None")
        assert ocr == ""
        assert marked is True, "van co du hai nhan — 'khong co chu' khac 'prompt hong'"

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_rong_khong_nem(self, text):
        assert parse_qwen25_caption_ocr(text) == ("", "", False)

    def test_chay_duoc_tren_toan_bo_855_cau_that_cua_Bao(self):
        """Bản đối chiếu duy nhất ta có. Nếu phép tách vỡ ở đây thì nó vỡ thật.

        Đo ngày 25-08-2026 trên `notebooks/reference/dense_captions_qwen.json`:
        **0/855** câu mất nhãn, **3/855** không có chữ nào đọc được. Ngưỡng dưới
        đặt lỏng hơn số đo — nó canh một phép tách HỎNG, không canh model.
        """
        source = Path("notebooks/reference/dense_captions_qwen.json")
        if not source.is_file():  # pragma: no cover - file cua may Lead
            pytest.skip("khong co ban chay cua Bao")
        payload = json.loads(source.read_text(encoding="utf-8"))
        parsed = [parse_qwen25_caption_ocr(v) for v in payload.values()]
        marked = sum(1 for _c, _o, m in parsed if m)
        with_text = sum(1 for _c, o, _m in parsed if o)
        assert len(parsed) == 855
        assert marked / len(parsed) > 0.95
        assert with_text / len(parsed) > 0.90
        assert all(not c.startswith("[CAPTION]") for c, _o, _m in parsed)


class _FakeEngine:
    """Đứng chỗ vLLM. Đếm số lần `generate` được gọi và bao nhiêu ảnh mỗi lần."""

    def __init__(self) -> None:
        self.calls = 0
        self.sizes: list[int] = []

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.calls += 1
        self.sizes.append(len(prompts))

        class _Out:
            def __init__(self, text):
                self.outputs = [type("O", (), {"text": text})()]

        return [_Out(f"[CAPTION] anh {i}\n\n[TEXT] CHU {i}") for i in range(len(prompts))]


@pytest.fixture
def fake_scan(monkeypatch):
    """`qwen25_scan` với vLLM và `qwen_vl_utils` được thay bằng bản giả."""
    import sys
    import types

    engine = _FakeEngine()

    class _Processor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

    monkeypatch.setattr(stages, "_qwen25_engine", lambda ctx: (_Processor(), engine))
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    utils = types.ModuleType("qwen_vl_utils")
    utils.process_vision_info = lambda messages: (["IMG"], None)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", utils)

    stages._QWEN_VL_ANSWERS.clear()
    stages._QWEN_VL_SCOPE = None
    yield engine
    stages._QWEN_VL_ANSWERS.clear()
    stages._QWEN_VL_SCOPE = None


def _ctx(tmp_path, n_frames=5, **params) -> StageContext:
    keyframes = [
        # `n` va `frame_idx` CO Y khac nhau (n=1 -> frame 100): mot test dung
        # n == frame_idx se khong the bat duoc phep tra cuu bi bo qua.
        Keyframe(
            n=i, pts_time=i * 4.0, fps=25.0, frame_idx=i * 100,
            image=tmp_path / f"{i:03d}.jpg",
        )
        for i in range(1, n_frames + 1)
    ]
    ctx = StageContext("L21_V001", Path("/dev/null"), keyframes, None, None, tmp_path)
    ctx.params = {"batch_size": 2, "progress_every": 0, **params}
    return ctx


class TestCacheCauTraLoi:
    def test_tang_thu_hai_KHONG_sinh_lai(self, fake_scan, tmp_path):
        """Đây là một nửa tiền GPU của cả lượt, và không lỗi nào báo nếu mất."""
        ctx = _ctx(tmp_path)
        first = stages.qwen25_scan(ctx, ctx.keyframes, label="a")
        calls_after_first = fake_scan.calls
        second = stages.qwen25_scan(ctx, ctx.keyframes, label="b")
        assert second == first
        assert fake_scan.calls == calls_after_first, "tang thu hai da goi lai generate()"

    def test_lo_dung_batch_size(self, fake_scan, tmp_path):
        ctx = _ctx(tmp_path, n_frames=5)
        stages.qwen25_scan(ctx, ctx.keyframes, label="a")
        assert fake_scan.sizes == [2, 2, 1]

    def test_doi_video_thi_cache_bi_don(self, fake_scan, tmp_path):
        ctx = _ctx(tmp_path)
        stages.qwen25_scan(ctx, ctx.keyframes, label="a")
        ctx.video_id = "L21_V002"
        stages.qwen25_scan(ctx, ctx.keyframes, label="b")
        assert fake_scan.calls > 3, "doi video ma van dung cache cua video truoc"

    def test_doi_THAM_SO_SINH_thi_cache_TRUOT(self, fake_scan, tmp_path):
        """Đổi `max_tokens` mà vẫn trả về câu cũ là trả về kết quả của tham số khác."""
        ctx = _ctx(tmp_path)
        stages.qwen25_scan(ctx, ctx.keyframes, label="a")
        before = fake_scan.calls
        ctx.params = {**ctx.params, "max_tokens": 512}
        stages.qwen25_scan(ctx, ctx.keyframes, label="b")
        assert fake_scan.calls > before

    def test_unload_models_don_ca_cache_cau_tra_loi(self, fake_scan, tmp_path):
        ctx = _ctx(tmp_path)
        stages.qwen25_scan(ctx, ctx.keyframes, label="a")
        assert stages._QWEN_VL_ANSWERS
        stages.unload_models()
        assert not stages._QWEN_VL_ANSWERS
        assert stages._QWEN_VL_SCOPE is None


class TestHaiTangGhiGi:
    """Hai tầng phải ghi HAI artifact khác nhau, khoá theo `frame_idx`."""

    class _Writer:
        def __init__(self):
            self.written = {}

        def write_text_part(self, artifact, video_id, mapping, **kwargs):
            self.written[artifact] = (dict(mapping), kwargs)
            return {"artifact": artifact, "stage": "x", "rows": len(mapping)}

    def test_caption_va_ocr_di_vao_hai_artifact_khac_nhau(self, fake_scan, tmp_path):
        # MOT lo (batch_size 8 > 3 anh): bo sinh gia danh so theo TUNG LO, nen mot
        # lo la cach duy nhat doc duoc thu tu toan cuc tu chuoi tra ve.
        ctx = _ctx(tmp_path, n_frames=3, batch_size=8)
        writer = self._Writer()
        ctx.writer = writer
        stages.stage_n6_caption_qwen(ctx)
        stages.stage_n6_ocr_qwen25(ctx)
        assert set(writer.written) == {"qwen_captions", "qwen_ocr"}
        captions, _ = writer.written["qwen_captions"]
        ocr, _ = writer.written["qwen_ocr"]
        assert list(captions.values()) == ["anh 0", "anh 1", "anh 2"]
        assert list(ocr.values()) == ["CHU 0", "CHU 1", "CHU 2"]

    def test_khoa_dung_frame_idx_KHONG_dung_n(self, fake_scan, tmp_path):
        """`n` là tên file ảnh, `frame_idx` là khoá của kho. Đây là bất biến §2."""
        ctx = _ctx(tmp_path, n_frames=3, batch_size=8)
        writer = self._Writer()
        ctx.writer = writer
        stages.stage_n6_caption_qwen(ctx)
        captions, _ = writer.written["qwen_captions"]
        assert sorted(captions) == ["L21_V001#000100", "L21_V001#000200", "L21_V001#000300"]

    def test_tham_so_MAY_nam_o_facts_khong_o_params(self, fake_scan, tmp_path):
        """`params` bị `check_fingerprints` so giữa các shard.

        `tensor_parallel_size` là 2 trên Kaggle T4×2 và 1 trên P100 — cả hai đều
        hợp lệ. Để nó trong `params` là từ chối cả bundle vì hai người dùng hai
        loại GPU, một phép từ chối đúng hình thức và rỗng về nội dung.
        """
        ctx = _ctx(tmp_path, n_frames=2, tensor_parallel_size=1)
        writer = self._Writer()
        ctx.writer = writer
        stages.stage_n6_caption_qwen(ctx)
        _, kwargs = writer.written["qwen_captions"]
        assert "tensor_parallel_size" not in kwargs["params"]
        assert kwargs["facts"]["tensor_parallel_size"] == 1
        assert kwargs["params"]["engine"] == "vllm"

    def test_dem_cau_mat_nhan_vao_facts(self, fake_scan, tmp_path, monkeypatch):
        """Không có bộ đếm này thì prompt hỏng là một sự kiện không quan sát được."""
        ctx = _ctx(tmp_path, n_frames=2)
        writer = self._Writer()
        ctx.writer = writer
        monkeypatch.setattr(
            stages, "qwen25_scan", lambda ctx, kfs, label: ["khong co nhan"] * len(kfs)
        )
        stages.stage_n6_caption_qwen(ctx)
        _, kwargs = writer.written["qwen_captions"]
        assert kwargs["facts"]["unmarked"] == 2

    def test_dong_rong_bi_bo_khoi_mapping(self, fake_scan, tmp_path, monkeypatch):
        ctx = _ctx(tmp_path, n_frames=2)
        writer = self._Writer()
        ctx.writer = writer
        monkeypatch.setattr(
            stages, "qwen25_scan", lambda ctx, kfs, label: ["[CAPTION] x\n\n[TEXT] None"] * len(kfs)
        )
        stages.stage_n6_ocr_qwen25(ctx)
        ocr, _ = writer.written["qwen_ocr"]
        assert ocr == {}, "khung khong co chu khong duoc chiem mot dong trong chi muc"


class TestBienThePrompt:
    """Hai biến thể cùng tồn tại, và nhầm biến thể là sinh sai cả lượt."""

    def test_hai_bien_the_khac_ngon_ngu_va_khac_max_tokens(self, tmp_path):
        from mfe.offline.caption_prompt import PROMPT_VARIANTS

        bao, bao_tokens = PROMPT_VARIANTS["bao"]
        aic, aic_tokens = PROMPT_VARIANTS["aic"]
        assert "in English" in bao
        assert "TIẾNG VIỆT" in aic
        assert aic_tokens > bao_tokens, "caption tieng Viet ton nhieu token hon"

    def test_max_tokens_mac_dinh_THEO_bien_the(self, tmp_path):
        """Không có một `max_tokens` chung: 256 của Bảo cắt cụt caption tiếng Việt."""
        assert stages.qwen25_max_tokens(_ctx(tmp_path, prompt_variant="bao")) == 256
        assert stages.qwen25_max_tokens(_ctx(tmp_path, prompt_variant="aic")) == 384

    def test_khai_tay_max_tokens_thi_thang_mac_dinh(self, tmp_path):
        ctx = _ctx(tmp_path, prompt_variant="aic", max_tokens=512)
        assert stages.qwen25_max_tokens(ctx) == 512

    def test_bien_the_la_thi_NEM_chu_khong_am_tham_dung_mac_dinh(self, tmp_path):
        with pytest.raises(ValueError, match="prompt_variant"):
            stages.qwen25_prompt(_ctx(tmp_path, prompt_variant="khong_co"))

    def test_doi_bien_the_thi_cache_TRUOT(self, fake_scan, tmp_path):
        ctx = _ctx(tmp_path, prompt_variant="aic")
        stages.qwen25_scan(ctx, ctx.keyframes, label="a")
        before = fake_scan.calls
        ctx.params = {**ctx.params, "prompt_variant": "bao"}
        stages.qwen25_scan(ctx, ctx.keyframes, label="b")
        assert fake_scan.calls > before

    def test_few_shot_day_du_ba_vi_du_va_mot_vi_du_KHONG_co_chu(self):
        """Ví dụ `[TEXT]` trống là ví dụ quan trọng nhất.

        Prompt cấm viết "None". Không có một ví dụ cho thấy "không có chữ" được
        thể hiện bằng khoảng TRỐNG thì model sẽ tự nghĩ ra một từ — và
        `stage_n5_ocr_qwen` đã trả giá cho đúng chuyện đó (305/307 ảnh trả "None").
        """
        from mfe.offline.caption_prompt import PROMPT_VARIANTS

        aic = PROMPT_VARIANTS["aic"][0]
        assert aic.count("### Ví dụ") == 3
        assert aic.count("[CAPTION]") >= 4   # 1 trong luat + 3 vi du
        assert "\n[TEXT]\n" in aic, "phai co mot vi du voi [TEXT] de TRONG"

    def test_few_shot_KHONG_kem_anh_mau(self):
        """Text-only. Một ảnh mẫu tốn hàng trăm token thị giác MỖI request."""
        from mfe.offline.caption_prompt import PROMPT_VARIANTS

        aic = PROMPT_VARIANTS["aic"][0]
        assert "<|image" not in aic and "<image>" not in aic


class TestThuTuTextAnh:
    def _content(self, fake_scan, tmp_path, **params):
        seen = {}

        def capture(messages):
            seen["content"] = messages[0]["content"]
            return (["IMG"], None)

        import sys
        sys.modules["qwen_vl_utils"].process_vision_info = capture
        ctx = _ctx(tmp_path, n_frames=1, **params)
        stages.qwen25_scan(ctx, ctx.keyframes, label="x")
        return seen["content"]

    def test_mac_dinh_TEXT_TRUOC_de_prefix_caching_co_tac_dung(self, fake_scan, tmp_path):
        """vLLM tái dùng KV-cache theo TIỀN TỐ tính từ token đầu tiên.

        Đặt ảnh trước thì hàng trăm token thị giác — khác nhau ở mọi ảnh — nằm
        ngay đầu, nên khối chỉ dẫn + few-shot phía sau không bao giờ trùng tiền tố.
        """
        content = self._content(fake_scan, tmp_path)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"

    def test_text_first_False_giu_dung_thu_tu_cua_Bao(self, fake_scan, tmp_path):
        content = self._content(fake_scan, tmp_path, text_first=False)
        assert content[0]["type"] == "image"

    def test_doi_thu_tu_thi_cache_TRUOT(self, fake_scan, tmp_path):
        ctx = _ctx(tmp_path)
        stages.qwen25_scan(ctx, ctx.keyframes, label="a")
        before = fake_scan.calls
        ctx.params = {**ctx.params, "text_first": False}
        stages.qwen25_scan(ctx, ctx.keyframes, label="b")
        assert fake_scan.calls > before


class TestFactsMoi:
    class _Writer:
        def __init__(self):
            self.written = {}

        def write_text_part(self, artifact, video_id, mapping, **kwargs):
            self.written[artifact] = (dict(mapping), kwargs)
            return {"artifact": artifact, "stage": "x", "rows": len(mapping)}

    def test_ghi_bien_the_van_tay_prompt_va_thu_tu_vao_params(self, fake_scan, tmp_path):
        ctx = _ctx(tmp_path, n_frames=2, prompt_variant="bao")
        ctx.writer = self._Writer()
        stages.stage_n6_caption_qwen(ctx)
        _, kwargs = ctx.writer.written["qwen_captions"]
        assert kwargs["params"]["prompt_variant"] == "bao"
        assert kwargs["params"]["max_tokens"] == 256
        assert kwargs["params"]["text_first"] is True
        assert len(kwargs["params"]["prompt_sha"]) == 16

    def test_van_tay_prompt_doi_theo_bien_the(self, fake_scan, tmp_path):
        """Vân tay của CHUỖI prompt, không phải của cái tên.

        Sửa một chữ trong prompt mà giữ tên biến thể thì hai người sinh ra hai bộ
        caption khác nhau dưới cùng một nhãn, và `check_fingerprints` không thấy gì.
        """
        shas = {}
        for variant in ("bao", "aic"):
            ctx = _ctx(tmp_path, n_frames=1, prompt_variant=variant)
            ctx.writer = self._Writer()
            stages.stage_n6_caption_qwen(ctx)
            shas[variant] = ctx.writer.written["qwen_captions"][1]["params"]["prompt_sha"]
        assert shas["bao"] != shas["aic"]

    def test_throughput_va_prefix_caching_nam_o_facts(self, fake_scan, tmp_path):
        ctx = _ctx(tmp_path, n_frames=4)
        ctx.writer = self._Writer()
        stages.stage_n6_caption_qwen(ctx)
        _, kwargs = ctx.writer.written["qwen_captions"]
        assert "images_per_s" in kwargs["facts"]
        assert kwargs["facts"]["images_per_s"] > 0
        assert "prefix_caching" in kwargs["facts"]
        # Toc do la su that cua MAY, khong phai cua bundle.
        assert "images_per_s" not in kwargs["params"]
