"""Locks `parse_qwen_ocr` — the post-processing that let a real bug through.

Measured on `L21_V001` with the first prompt ("transcribe, answer None if no
text"): Qwen2-VL answered `None` on **305 of 307** keyframes, while easyocr found
text on **all 307** of the same images. Same pictures, one model saw text
everywhere and the other saw none — so the fault was the prompt, not the frames.

The old filter was `answer.lower().strip(" .") == "none"`, comparing the **whole**
answer. So `"BARBER SHOP\nNone"` — an answer that did find text — was stored with
the junk word glued on, and went into the BM25 index that way. Bảo's shipped file
has the same contamination (`"East 42nd St\nNone"`), which is how the shape was
recognised.
"""

from __future__ import annotations

import pytest

from mfe.offline.stages import _QWEN_PROMPT, parse_qwen_ocr


class TestFormattedAnswers:
    def test_reads_the_text_after_the_marker(self):
        assert parse_qwen_ocr("VUNG: logo kenh\nCHU: HTV9 HD") == ("HTV9 HD", False)

    def test_drops_the_region_line_it_does_not_want(self):
        """Step 1 exists to make the model look at the image, not to be stored.
        Leaking `VUNG: logo kênh` into `sviptr_ocr` would put words describing the
        frame into a field the index treats as text read *from* the frame."""
        text, _ = parse_qwen_ocr("VUNG: logo kenh, dong ho\nCHU: HTV9 HD 06:31:48")
        assert "logo" not in text
        assert text == "HTV9 HD 06:31:48"

    def test_an_empty_text_line_is_silence_not_a_word(self):
        assert parse_qwen_ocr("VUNG: khong\nCHU:") == ("", True)

    @pytest.mark.parametrize("marker", ["CHU:", "CHỮ:", "TEXT:"])
    def test_accepts_the_marker_spellings_the_model_actually_emits(self, marker):
        assert parse_qwen_ocr(f"VUNG: bien hieu\n{marker} PHO 24")[0] == "PHO 24"


class TestTheBugThatShipped:
    def test_real_text_plus_a_bail_word_keeps_the_text_and_drops_the_word(self):
        """The exact string found on the Volume: `L21_V001#037011`."""
        text, bailed = parse_qwen_ocr("BARBER SHOP\nNone")
        assert text == "BARBER SHOP"
        assert bailed is True     # dem lai, vi no la dau hieu prompt con lung lay

    def test_the_same_answer_with_a_marker(self):
        assert parse_qwen_ocr("CHU: BARBER SHOP\nNone")[0] == "BARBER SHOP"

    def test_no_bail_word_survives_into_the_text(self):
        """Whatever the model appends, none of these may reach the BM25 index."""
        for word in ("None", "none", "khong", "không", "no text", "N/A", "-", "empty"):
            text, _ = parse_qwen_ocr(f"CHU: PHO 24\n{word}")
            assert text == "PHO 24", word


class TestBailingOut:
    @pytest.mark.parametrize("answer", ["None", "none.", "khong", "không", "", "   "])
    def test_a_pure_bail_yields_no_text(self, answer):
        text, bailed = parse_qwen_ocr(answer)
        assert (text, bailed) == ("", True)

    def test_an_unformatted_answer_still_yields_its_text(self):
        """The model does not always keep the format, and an answer that ignored
        the format can still be a correct reading. Throwing it away would repeat
        the failure this parser exists to fix, in the opposite direction."""
        assert parse_qwen_ocr("HTV9 HD\n06:31:48") == ("HTV9 HD\n06:31:48", False)

    def test_markdown_bullets_are_stripped(self):
        assert parse_qwen_ocr("CHU:\n* HTV9 HD\n* 60 giây")[0] == "HTV9 HD\n60 giây"


class TestPromptItself:
    def test_it_no_longer_offers_an_escape_hatch_for_the_transcription(self):
        """Ban dau: "If there is no text, answer exactly: None". Cho san mot token
        de bo cuoc thi model 7B 4-bit dung no — 305/307 lan."""
        assert "None" not in _QWEN_PROMPT

    def test_it_asks_where_the_text_is_before_asking_what_it_says(self):
        assert _QWEN_PROMPT.index("BƯỚC 1") < _QWEN_PROMPT.index("BƯỚC 2")
        assert "VUNG:" in _QWEN_PROMPT and "CHU:" in _QWEN_PROMPT

    def test_it_does_not_ask_for_a_description(self):
        """§2.2 — `caption` and `ocr_sviptr` must not be two fields cut from one
        generation. Step 1 asks *where* text is, never what the scene shows."""
        assert "mô tả hình ảnh" in _QWEN_PROMPT  # ...trong cau CAM
        assert "Không dịch, không giải thích, không mô tả hình ảnh." in _QWEN_PROMPT
