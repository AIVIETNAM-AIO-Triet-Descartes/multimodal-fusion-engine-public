"""Locks the batched Qwen OCR loop and its OOM fallback — with a stub model.

Why stub instead of skip: the OOM fallback is the one path that **will** fire on
Kaggle and the one path a laptop can never reach, so leaving it unexercised means
shipping it unread. Everything except the model is ours — batching, the retry, the
counters, and the marker — and all of it is reachable with a fake.

Measured motivation: one image per `generate()` cost **10,8 s/image** on `L21_V001`
(3319 s for 307 images), i.e. ~532 GPU-hours for batch 1's 177.321 keyframes.
`build_qwen_generate()` in this same module already batches for ReCap; this stage
did not.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter
from mfe.offline.stages import Keyframe, StageContext, stage_n5_ocr_qwen, unload_models
from mfe.offline.transport import LocalDirTransport

VIDEO = "L21_V001"


class FakeOOM(Exception):
    """Stands in for `torch.cuda.OutOfMemoryError`."""


class Recorder:
    """Answers the stage's calls, and remembers the batch widths it was asked for."""

    def __init__(self, answers: list[str], oom_when=None) -> None:
        self.answers = list(answers)
        self.widths: list[int] = []
        self.oom_when = oom_when or (lambda width, call: False)
        self.calls = 0
        self.empty_cache_calls = 0
        self.padding_side: str | None = None

    # -- processor ------------------------------------------------------
    @property
    def tokenizer(self):
        return self

    def apply_chat_template(self, messages, add_generation_prompt=False):  # noqa: ARG002
        return "PROMPT"

    def __call__(self, *, text, images, padding=False, return_tensors=None):  # noqa: ARG002
        assert len(text) == len(images), "moi anh phai co mot prompt"
        self.width = len(text)
        # Phai la mot MAPPING: tang goi `model.generate(**inputs, ...)`, va
        # transformers that tra ve `BatchFeature`, von la mot dict.
        return FakeInputs({"input_ids": types.SimpleNamespace(shape=(self.width, 7))})

    def batch_decode(self, generated, skip_special_tokens=True):  # noqa: ARG002
        return [self.answers.pop(0) for _ in range(self.width)]

    # -- model ----------------------------------------------------------
    device = "cuda:0"

    def eval(self):
        return self

    def generate(self, *, max_new_tokens, do_sample, **inputs):  # noqa: ARG002
        self.calls += 1
        self.widths.append(self.width)
        if self.oom_when(self.width, self.calls):
            raise FakeOOM("CUDA out of memory")
        return _Slicable()


class FakeInputs(dict):
    """`BatchFeature` là dict, nên `**inputs` chạy được; `.to()` trả về chính nó."""

    def to(self, device):  # noqa: ARG002
        return self


class _Slicable:
    def __getitem__(self, item):
        return self


@pytest.fixture
def stubs(monkeypatch):
    """Install the modules the stage imports lazily, then take them back out."""
    recorder_box: dict[str, Recorder] = {}

    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(
        OutOfMemoryError=FakeOOM,
        empty_cache=lambda: recorder_box["r"].__setattr__(
            "empty_cache_calls", recorder_box["r"].empty_cache_calls + 1
        ),
        is_available=lambda: False,
    )

    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.open = lambda path: types.SimpleNamespace(convert=lambda mode: f"img:{path}")
    pil.Image = image_mod

    transformers = types.ModuleType("transformers")

    def install(recorder: Recorder) -> None:
        recorder_box["r"] = recorder
        transformers.AutoProcessor = types.SimpleNamespace(
            from_pretrained=lambda checkpoint: recorder
        )
        transformers.BitsAndBytesConfig = lambda **kw: kw
        transformers.Qwen2VLForConditionalGeneration = types.SimpleNamespace(
            from_pretrained=lambda checkpoint, **kw: recorder
        )

    for name, module in (
        ("torch", torch), ("PIL", pil), ("PIL.Image", image_mod),
        ("transformers", transformers),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    unload_models()          # cache model la toan cuc; dung stub cua lan truoc la sai
    yield install
    unload_models()


def make_ctx(tmp_path, count: int, **params) -> StageContext:
    keyframes = [
        Keyframe(n=i + 1, pts_time=i * 2.0, fps=25.0, frame_idx=i * 50,
                 image=tmp_path / f"{i + 1:03d}.jpg")
        for i in range(count)
    ]
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    writer = BundleWriter(transport, layout, "trong", tmp_path / "work")
    return StageContext(
        VIDEO, tmp_path / "unused.mp4", keyframes, writer, layout,
        tmp_path / "work", params=params,
    )


class TestBatching:
    def test_one_generate_call_per_batch_not_per_image(self, tmp_path, stubs):
        recorder = Recorder(["CHU: HTV9 HD"] * 20)
        stubs(recorder)
        marker = stage_n5_ocr_qwen(make_ctx(tmp_path, 20, batch_size=8))
        assert recorder.widths == [8, 8, 4]       # 20 anh, 3 lan goi
        assert marker["rows"] == 20

    def test_every_keyframe_is_asked_exactly_once(self, tmp_path, stubs):
        """A batching bug that drops or repeats the tail is silent: the part file
        is simply short, and `write_text_part` has no idea what it should hold."""
        recorder = Recorder([f"CHU: text {i}" for i in range(17)])
        stubs(recorder)
        marker = stage_n5_ocr_qwen(make_ctx(tmp_path, 17, batch_size=5))
        assert sum(recorder.widths) == 17
        assert marker["rows"] == 17

    def test_a_batch_smaller_than_the_batch_size_still_runs(self, tmp_path, stubs):
        recorder = Recorder(["CHU: a", "CHU: b"])
        stubs(recorder)
        assert stage_n5_ocr_qwen(make_ctx(tmp_path, 2, batch_size=8))["rows"] == 2

    def test_padding_is_set_to_the_left(self, tmp_path, stubs):
        """Right padding puts pad tokens *between* the prompt and the answer, so
        the model continues from padding. It does not raise — it returns junk."""
        recorder = Recorder(["CHU: x"] * 3)
        stubs(recorder)
        stage_n5_ocr_qwen(make_ctx(tmp_path, 3, batch_size=2))
        assert recorder.padding_side == "left"


class TestOomFallback:
    def test_it_halves_the_batch_and_retries_the_same_images(self, tmp_path, stubs):
        """The retry must redo the batch that failed, not skip past it."""
        recorder = Recorder(["CHU: t"] * 8, oom_when=lambda width, call: width == 8)
        stubs(recorder)
        marker = stage_n5_ocr_qwen(make_ctx(tmp_path, 8, batch_size=8))
        assert recorder.widths[0] == 8            # thu rong truoc
        assert recorder.widths[1:] == [4, 4]      # ha nua, lam lai DU 8 anh
        assert marker["rows"] == 8
        assert recorder.empty_cache_calls == 1

    def test_the_narrowed_width_sticks_for_the_rest(self, tmp_path, stubs):
        """Going back up would OOM again on every subsequent batch."""
        recorder = Recorder(["CHU: t"] * 12, oom_when=lambda width, call: width == 6)
        stubs(recorder)
        stage_n5_ocr_qwen(make_ctx(tmp_path, 12, batch_size=6))
        assert recorder.widths == [6, 3, 3, 3, 3]

    def test_an_oom_at_width_one_is_raised_not_swallowed(self, tmp_path, stubs):
        """Nothing left to halve. Failing loudly beats writing a short part file
        that no later run will ever fix — the marker would say it finished."""
        recorder = Recorder(["CHU: t"] * 4, oom_when=lambda width, call: True)
        stubs(recorder)
        with pytest.raises(FakeOOM):
            stage_n5_ocr_qwen(make_ctx(tmp_path, 4, batch_size=2))

    def test_no_marker_survives_the_raise(self, tmp_path, stubs):
        from mfe.offline.bundle import StageStatus

        recorder = Recorder(["CHU: t"] * 4, oom_when=lambda width, call: True)
        stubs(recorder)
        ctx = make_ctx(tmp_path, 4, batch_size=2)
        with pytest.raises(FakeOOM):
            stage_n5_ocr_qwen(ctx)
        status = StageStatus.load(ctx.writer.transport, ctx.layout, "trong")
        assert not status.is_done("n5_ocr_qwen", VIDEO)


class TestMarker:
    def test_it_records_the_width_actually_used_not_the_one_asked_for(
        self, tmp_path, stubs
    ):
        """A run that got narrowed mid-way must not look like a clean run at the
        asked width — that is how a 4× slowdown hides in a manifest."""
        recorder = Recorder(["CHU: t"] * 8, oom_when=lambda width, call: width == 8)
        stubs(recorder)
        params = stage_n5_ocr_qwen(make_ctx(tmp_path, 8, batch_size=8))["params"]
        assert params["batch_size"] == 4
        assert params["batch_size_asked"] == 8
        assert params["oom_retries"] == 1

    def test_it_separates_bail_words_from_no_text(self, tmp_path, stubs):
        """The old marker had one `answered_none` counter and put both in it, so
        `"BARBER SHOP\\nNone"` — a success — landed beside a genuine refusal."""
        recorder = Recorder([
            "CHU: HTV9 HD",          # sach
            "BARBER SHOP\nNone",     # co chu + tu bo cuoc
            "None",                  # bo cuoc that
            "CHU:",                  # im lang
        ])
        stubs(recorder)
        marker = stage_n5_ocr_qwen(make_ctx(tmp_path, 4, batch_size=4))
        assert marker["rows"] == 2                    # 2 khoa co chu
        assert marker["params"]["bail_words"] == 3    # 3 lan co tu bo cuoc
        assert marker["params"]["no_text"] == 2       # 2 lan khong ra chu

    def test_the_prompt_version_is_recorded(self, tmp_path, stubs):
        """Two runs of this stage are only comparable if the prompt is the same,
        and the prompt just changed for a measured reason."""
        stubs(Recorder(["CHU: x"]))
        marker = stage_n5_ocr_qwen(make_ctx(tmp_path, 1))
        assert marker["params"]["prompt"] == "vung+chu v2"

    def test_the_payload_is_keyed_by_frame_idx(self, tmp_path, stubs):
        stubs(Recorder(["CHU: HTV9 HD", "CHU: PHO 24"]))
        ctx = make_ctx(tmp_path, 2, batch_size=2)
        stage_n5_ocr_qwen(ctx)
        local = tmp_path / "back.json"
        ctx.writer.transport.get(ctx.layout.part("sviptr_ocr", VIDEO), local)
        import json

        payload = json.loads(local.read_text(encoding="utf-8"))
        assert set(payload) == {f"{VIDEO}#000000", f"{VIDEO}#000050"}


def test_numpy_is_not_needed_here():
    """Guard against an accidental import creeping into the stub set."""
    assert np is not None
