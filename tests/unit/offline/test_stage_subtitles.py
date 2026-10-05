"""Regression tests for Whisper-to-keyframe subtitle coverage."""

from __future__ import annotations

import json
import logging
import sys
import types
from types import SimpleNamespace

import pytest

from mfe.offline import stages
from mfe.offline.bundle import BundleLayout, BundleWriter, StageStatus
from mfe.offline.stages import Keyframe, StageContext, stage_n2_subtitles
from mfe.offline.transport import LocalDirTransport

VIDEO = "L24_V045"


def _context(tmp_path, *, keyframes: int = 5) -> StageContext:
    frames = [
        Keyframe(
            n=index + 1,
            pts_time=index * 10.0,
            fps=25.0,
            frame_idx=index * 250,
            image=tmp_path / f"{index + 1:03d}.jpg",
        )
        for index in range(keyframes)
    ]
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    writer = BundleWriter(transport, layout, "trong", tmp_path / "work")
    return StageContext(
        VIDEO,
        tmp_path / f"{VIDEO}.mp4",
        frames,
        writer,
        layout,
        tmp_path / "work",
        params={"beam_size": 5, "window": "next_keyframe"},
    )


def _install_fake_whisper(monkeypatch, spans):
    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, *_args, **_kwargs):
            segments = [SimpleNamespace(**span) for span in spans]
            return iter(segments), SimpleNamespace(language="vi")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(stages, "cached_model", lambda _key, build: build())
    monkeypatch.setattr(StageContext, "audio_path", lambda self: self.workdir / "audio.wav")


def _read_json(ctx: StageContext, remote: str) -> dict:
    local = ctx.workdir / "readback.json"
    ctx.writer.transport.get(remote, local)
    return json.loads(local.read_text(encoding="utf-8"))


def test_high_empty_ratio_stops_the_stage_by_default(tmp_path, monkeypatch):
    """§4.5, và không còn là giả thuyết.

    Một video phiên âm ra hàng chục đoạn mà rơi vào gần như không keyframe nào
    thì hoặc sai cửa sổ, hoặc sai kênh audio — chạy tiếp sẽ chôn chuyện đó trong
    một manifest. Đúng chuyện đã xảy ra: 469 video của Quang đi lọt với 69–87%
    rỗng vì bản cài đặt của bạn ấy không có chốt này.
    """
    _install_fake_whisper(
        monkeypatch,
        [{"start": 10.5, "end": 11.5, "text": "mot doan phu de"}],
    )
    ctx = _context(tmp_path, keyframes=5)

    with pytest.raises(RuntimeError, match="80% keyframe khong co subtitle"):
        stage_n2_subtitles(ctx)

    # Chốt TRƯỚC khi cắm mốc: không được để lại marker nói rằng đã xong.
    assert not StageStatus.load(ctx.writer.transport, ctx.layout, "trong").is_done(
        "n2_subtitles", VIDEO
    )


def test_raising_the_threshold_lets_a_quiet_video_through(tmp_path, monkeypatch):
    """Bộ video nhiều khoảng lặng vẫn chạy được — nhưng phải nâng ngưỡng CÓ CHỦ Ý.

    Và quyết định đó đi vào `params`, nên `check_fingerprints` nhìn thấy nó: "nửa
    bundle này được phép thưa" trở thành một sự thật so được giữa các mảnh, thay
    vì một dòng log không ai đọc. Đó là cả lý do ngưỡng nằm ở `params` chứ không
    phải một hằng số.
    """
    _install_fake_whisper(
        monkeypatch,
        [{"start": 10.5, "end": 11.5, "text": "mot doan phu de"}],
    )
    ctx = _context(tmp_path, keyframes=5)
    ctx.params = {"max_empty_ratio": 0.9}

    marker = stage_n2_subtitles(ctx)

    assert marker["rows"] == 1
    assert marker["segments"] == 1
    assert marker["keyframes_total"] == 5
    assert marker["keyframes_with_subtitle"] == 1
    assert marker["keyframes_without_subtitle"] == 4
    assert marker["empty_ratio"] == pytest.approx(0.8)
    assert marker["params"]["max_empty_ratio"] == 0.9
    assert StageStatus.load(ctx.writer.transport, ctx.layout, "trong").is_done(
        "n2_subtitles", VIDEO
    )
    assert _read_json(ctx, ctx.layout.part("subtitles", VIDEO)) == {
        f"{VIDEO}#000250": "mot doan phu de"
    }


def test_no_whisper_segments_is_a_valid_empty_artifact(tmp_path, monkeypatch, caplog):
    _install_fake_whisper(monkeypatch, [])
    ctx = _context(tmp_path, keyframes=3)

    with caplog.at_level(logging.WARNING, logger="mfe.offline.stages"):
        marker = stage_n2_subtitles(ctx)

    assert marker["rows"] == 0
    assert marker["segments"] == 0
    assert marker["empty_ratio"] == pytest.approx(1.0)
    assert "keyframe khong co subtitle" not in caplog.text
    assert _read_json(ctx, ctx.layout.part("subtitles", VIDEO)) == {}
