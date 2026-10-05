from __future__ import annotations

import json

import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter
from mfe.offline.fingerprint_gate import (
    FingerprintContract,
    FingerprintLockedWriter,
    FingerprintMismatch,
    discover_fingerprint_contract,
)
from mfe.offline.transport import LocalDirTransport


def _manifest() -> dict:
    return {
        "video_id": "L26_V010",
        "author": "quang",
        "bundle": "aic_btc_b1_v1",
        "env": {
            "torch": "2.8.0",
            "faster-whisper": "1.1.0",
            "pillow": "11.3.0",
        },
        "artifacts": {
            "subtitles": {
                "model": "faster-whisper large-v3",
                "params": {"window": "next_keyframe"},
            },
            "paddle_ocr": {
                "model": "paddleocr 3.2.0 lang=vi",
                "params": {"lang": "vi", "api": "predict"},
            },
            "viet_ocr": {
                "model": "vietocr 0.3.13/vgg_transformer + paddleocr 3.2.0 (phat hien)",
                "params": {"beamsearch": False},
            },
        },
    }


def _contract() -> FingerprintContract:
    return FingerprintContract.from_manifest(
        _manifest(),
        reference_video="L26_V010",
        reference_author="quang",
        bundle="aic_btc_b1_v1",
        required_artifacts=("paddle_ocr", "viet_ocr"),
    )


def test_same_hard_environment_merges_and_pillow_only_warns() -> None:
    warnings = _contract().check_environment(
        {"torch": "2.8.0", "faster-whisper": "1.1.0", "pillow": "12.0.0"}
    )

    assert len(warnings) == 1
    assert "pillow" in warnings[0]


def test_faster_whisper_difference_is_env_lech() -> None:
    with pytest.raises(FingerprintMismatch, match="env_lech") as caught:
        _contract().check_environment(
            {"torch": "2.8.0", "faster-whisper": "1.0.3", "pillow": "11.3.0"}
        )
    assert "faster-whisper" in str(caught.value)


def test_model_params_difference_is_model_lech() -> None:
    with pytest.raises(FingerprintMismatch, match="model_lech"):
        _contract().check_artifact(
            "paddle_ocr",
            model="paddleocr 3.2.0 lang=vi",
            params={"lang": "vi", "api": "ocr"},
        )


def test_retained_artifact_params_difference_is_model_lech() -> None:
    with pytest.raises(FingerprintMismatch, match="model_lech") as caught:
        _contract().check_existing_manifest(
            {
                "artifacts": {
                    "subtitles": {
                        "model": "faster-whisper large-v3",
                        "params": {"window": "fixed3"},
                    }
                }
            }
        )
    assert "subtitles" in str(caught.value)


def test_retained_mismatch_is_rejected_before_new_part_upload(tmp_path) -> None:
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    video_id = "L21_V001"
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "artifacts": {
                    "subtitles": {
                        "model": "faster-whisper large-v3",
                        "params": {"window": "fixed3"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    transport.put(previous, layout.part_manifest(video_id))
    writer = FingerprintLockedWriter(
        transport,
        layout,
        "trong",
        tmp_path / "work",
        fingerprint=_contract(),
    )

    with pytest.raises(FingerprintMismatch, match="model_lech"):
        writer.write_text_part(
            "paddle_ocr",
            video_id,
            {"L21_V001_001": "chu"},
            model="paddleocr 3.2.0 lang=vi",
            params={"lang": "vi", "api": "predict"},
        )

    assert not transport.exists(layout.part("paddle_ocr", video_id))


def test_reference_must_have_both_artifacts() -> None:
    manifest = _manifest()
    del manifest["artifacts"]["viet_ocr"]

    with pytest.raises(FingerprintMismatch, match="CA HAI"):
        FingerprintContract.from_manifest(
            manifest,
            reference_video="L26_V010",
            reference_author="quang",
            bundle="aic_btc_b1_v1",
            required_artifacts=("paddle_ocr", "viet_ocr"),
        )


def test_discovers_a_completed_reference_without_stopping_quang(tmp_path) -> None:
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    quang_writer = BundleWriter(transport, layout, "quang", tmp_path / "quang-work")
    for stage in ("n5_ocr_paddle", "n5_ocr_vietocr"):
        quang_writer.mark_done(stage, "L26_V010")
    manifest = tmp_path / "reference.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    transport.put(manifest, layout.part_manifest("L26_V010"))

    contract = discover_fingerprint_contract(
        transport,
        layout,
        tmp_path / "gate-work",
        reference_author="quang",
        candidate_videos=["L22_V001", "L26_V010"],
        required_artifacts=("paddle_ocr", "viet_ocr"),
    )

    assert contract.reference_video == "L26_V010"


def test_discovery_waits_until_quang_has_one_complete_video(tmp_path) -> None:
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    quang_writer = BundleWriter(transport, layout, "quang", tmp_path / "quang-work")
    quang_writer.mark_done("n5_ocr_paddle", "L22_V001")

    with pytest.raises(FingerprintMismatch, match="chua co video nao"):
        discover_fingerprint_contract(
            transport,
            layout,
            tmp_path / "gate-work",
            reference_author="quang",
            candidate_videos=["L22_V001"],
            required_artifacts=("paddle_ocr", "viet_ocr"),
        )
