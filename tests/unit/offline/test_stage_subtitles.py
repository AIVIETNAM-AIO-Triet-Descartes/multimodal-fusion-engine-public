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


def test_high_empty_ratio_warns_but_writes_artifact_and_marker(
    tmp_path, monkeypatch, caplog
):
    """Silent keyframe windows must not block N2 or the following N3 stage."""
    _install_fake_whisper(
        monkeypatch,
        [{"start": 10.5, "end": 11.5, "text": "mot doan phu de"}],
    )
    ctx = _context(tmp_path, keyframes=5)

    with caplog.at_level(logging.WARNING, logger="mfe.offline.stages"):
        marker = stage_n2_subtitles(ctx)

    assert marker["rows"] == 1
    assert marker["segments"] == 1
    assert marker["keyframes_total"] == 5
    assert marker["keyframes_with_subtitle"] == 1
    assert marker["keyframes_without_subtitle"] == 4
    assert marker["empty_ratio"] == pytest.approx(0.8)
    assert "80% keyframe khong co subtitle" in caplog.text
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
