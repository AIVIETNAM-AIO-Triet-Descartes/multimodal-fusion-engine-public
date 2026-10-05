"""VLM đọc đáp án QA — mọi đường hỏng phải nói ra, không được im lặng.

Đáp án sai không mất một dòng mà mất **cả 100 dòng**, vì BTC chấm cột thứ ba
trước. Nên phần lớn test ở đây kiểm đúng một tính chất: khi không chắc thì
`ok=False` kèm lý do, chứ không phải một chuỗi trông như đáp án.
"""

from __future__ import annotations

import json

import pytest

from mfe.contracts.tools import Candidate
from mfe.online.vqa_answer import Answer, parse_answer, read_answer, read_answer_hybrid


def _candidates(n: int = 5) -> list[Candidate]:
    return [
        Candidate(
            keyframe_id=f"L21_V003#{26000 + i:06d}",
            video_id="L21_V003",
            score=1.0 / (i + 1),
            evidence_id=f"siglip:local:{i + 1}",
        )
        for i in range(n)
    ]


@pytest.fixture
def images(tmp_path):
    """`keyframe_id -> Path` cho một ảnh thật sự tồn tại trên đĩa.

    `read_answer` lọc theo `p.is_file()`, nên một fixture trả đường dẫn ma sẽ
    khiến mọi test đi vào nhánh "khong lay duoc anh nao" mà vẫn xanh.
    """
    def provider(keyframe_id: str):
        path = tmp_path / f"{keyframe_id.replace('#', '_')}.jpg"
        path.write_bytes(b"khong-phai-jpeg-that-nhung-co-tren-dia")
        return path

    return provider


def test_doc_duoc_dap_an_va_chi_ra_khung(images):
    """Đường thành công: có đáp án, có nguồn, `ok` True."""
    reply = json.dumps({
        "answer": "2,15", "frame": "L21_V003#026002",
        "confidence": "cao", "why": "bien bao ben trai cau",
    })
    out = read_answer(
        "Con số trên biển báo là bao nhiêu?", _candidates(), images,
        ask=lambda *_: reply,
    )
    assert out.ok
    assert out.text == "2,15"
    assert out.frame == "L21_V003#026002"
    assert out.confidence == "cao"
    assert out.looked_at == 5


def test_dap_an_rong_la_ket_qua_hop_le_khong_phai_loi(images):
    """Model nói "không đọc được" phải ra `ok=False` KÈM lý do, không ném.

    Đây là hành vi mong muốn chứ không phải suy biến: thà để người vận hành gõ
    tay còn hơn điền một chuỗi bịa vào cả 100 dòng.
    """
    reply = json.dumps({
        "answer": "", "frame": "", "confidence": "thap",
        "why": "khong khung nao thay bien bao",
    })
    out = read_answer("...?", _candidates(), images, ask=lambda *_: reply)
    assert not out.ok
    assert out.text == ""
    assert out.reason
    assert "khong doc duoc" in out.reason


def test_khung_model_bia_ra_thi_bo_nguon_chu_khong_bo_dap_an(images):
    """Id không có trong danh sách gửi đi = model không thực sự đọc từ khung nào.

    Đáp án vẫn có thể đúng nên giữ lại, nhưng nguồn phải mất đi — người vận hành
    cần thấy là không có gì để đối chiếu.
    """
    reply = json.dumps({"answer": "2,15", "frame": "L99_V999#000001"})
    out = read_answer("...?", _candidates(), images, ask=lambda *_: reply)
    assert out.ok
    assert out.text == "2,15"
    assert out.frame == ""


def test_api_nem_thi_tra_ly_do_chu_khong_lan_ra_ngoai(images):
    """Hàm này chạy giữa vòng vẽ giao diện; ném là làm trắng cả trang."""
    def boom(*_):
        raise RuntimeError("429 quota het")

    out = read_answer("...?", _candidates(), images, ask=boom)
    assert not out.ok
    assert out.reason is not None
    assert "RuntimeError" in out.reason
    assert "429" in out.reason


def test_tra_loi_khong_phai_json_thi_bao_ra(images):
    out = read_answer("...?", _candidates(), images, ask=lambda *_: "chac la 2,15 do")
    assert not out.ok
    assert out.reason is not None
    assert "khong doc duoc JSON" in out.reason


def test_khong_co_anh_nao_thi_khong_goi_api():
    """Thiếu ảnh phải dừng TRƯỚC khi trả tiền cho một lượt gọi."""
    called = []

    def ask(*args):
        called.append(args)
        return "{}"

    out = read_answer("...?", _candidates(), lambda _: None, ask=ask)
    assert not out.ok
    assert out.reason == "khong lay duoc anh nao cho top ung vien"
    assert not called


def test_depth_cat_dung_so_khung_gui_di(images):
    seen: dict[str, int] = {}

    def ask(prompt, paths):
        seen["paths"] = len(paths)
        return json.dumps({"answer": "x"})

    read_answer("...?", _candidates(10), images, depth=3, ask=ask)
    assert seen["paths"] == 3


def test_prompt_mang_du_cau_hoi_va_danh_sach_id(images):
    """Prompt phải chứa câu hỏi VÀ id — model được yêu cầu khai nguồn từ đó."""
    captured: dict[str, str] = {}

    def ask(prompt, _paths):
        captured["prompt"] = prompt
        return json.dumps({"answer": "x"})

    read_answer("Biển báo ghi gì?", _candidates(2), images, ask=ask)
    assert "Biển báo ghi gì?" in captured["prompt"]
    assert "L21_V003#026000" in captured["prompt"]
    assert "L21_V003#026001" in captured["prompt"]


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"answer": "2,15"}\n```',
        'Day la ket qua: {"answer": "2,15"} — hy vong dung.',
        '{"answer": "2,15"}',
    ],
)
def test_parse_answer_chiu_duoc_markdown_va_loi_dan(raw):
    """Model được bảo "chỉ JSON" và phần lớn lượt nó nghe lời — phần lớn thôi."""
    assert parse_answer(raw) == {"answer": "2,15"}


@pytest.mark.parametrize("raw", ["khong co json", "[1, 2, 3]", "{hong}"])
def test_parse_answer_tra_none_khi_khong_doc_duoc(raw):
    assert parse_answer(raw) is None


def test_answer_mac_dinh_la_chua_co_dap_an():
    """`Answer()` trần phải là "chưa có", không phải "có đáp án rỗng"."""
    assert not Answer().ok
    assert Answer().text == ""


def test_hybrid_reader_uses_continuous_and_scattered_evidence_with_budget(tmp_path):
    candidates = _candidates(20)

    class Store:
        def fetch(self, keys):
            return [
                type(
                    "Hit",
                    (),
                    {
                        "video_id": video,
                        "frame_id": frame,
                        "source": "subtitle",
                        "text": f"text {frame}",
                    },
                )()
                for video, frame in keys
            ]

    class Relational:
        def get_neighbors(self, _keyframe_id, window):
            return [
                type("Row", (), {"keyframe_id": f"L21_V003#{26000 + i:06d}"})()
                for i in range(-2, 3)
            ]

    def image(keyframe_id):
        path = tmp_path / f"{keyframe_id.replace('#', '_')}.jpg"
        path.write_bytes(b"image")
        return path

    seen = {}

    def ask(prompt, paths):
        seen["prompt"] = prompt
        seen["paths"] = paths
        return '{"answer":"x", "frame":"L21_V003#026000"}'

    out = read_answer_hybrid(
        "cau hoi",
        candidates,
        image,
        Store(),
        Relational(),
        ask=ask,
        max_images=15,
    )
    assert out.ok
    assert len(seen["paths"]) == 15
    assert "continuous frames" in seen["prompt"]
