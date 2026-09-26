from types import SimpleNamespace

from mfe.online.answering import Evidence, gather_evidence
from mfe.online.flat_evidence import FlatTextStore


def test_flat_bundle_texts_are_exposed_as_keyframe_evidence():
    bundle = SimpleNamespace(
        texts={
            "caption_qwen": {"V#000007": "một cảnh dài"},
            "ocr_qwen25": {"V#000007": "BẢNG TIN"},
        }
    )
    evidence = gather_evidence(["V#000007"], FlatTextStore(bundle))
    assert evidence == [
        Evidence(
            keyframe_id="V#000007",
            video_id="V",
            frame_id=7,
            texts={"caption_qwen": "một cảnh dài", "ocr_qwen25": "BẢNG TIN"},
        )
    ]


def test_caption_prompt_is_limited_without_cutting_ocr():
    caption = "x" * 250
    item = Evidence(
        keyframe_id="V#000001",
        video_id="V",
        frame_id=1,
        texts={"caption_qwen": caption, "ocr_qwen25": "BẢNG TIN"},
    )
    prompt = item.as_prompt()
    assert "BẢNG TIN" in prompt
    assert "x" * 201 not in prompt
    assert "x…\n[ocr_qwen25]" in prompt
