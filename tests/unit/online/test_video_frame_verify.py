from pathlib import Path

from PIL import Image

from mfe.contracts.records import KeyframeRow
from mfe.online.video_frame_verify import shortlist_video_keyframes, verify_video_keyframe


class _Bundle:
    def __init__(self) -> None:
        self.texts = {
            "caption_gemini": {
                "L21_V001#000100": "a person walks into a room",
                "L21_V001#000200": "the first shrimp touches the serving plate",
                "L21_V001#000300": "the chef adds an orange segment",
            }
        }
        self.vectors = {}


class _Client:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


def _rows() -> list[KeyframeRow]:
    return [
        KeyframeRow(
            keyframe_id=f"L21_V001#{frame:06d}",
            video_id="L21_V001",
            keyframe_index=index,
            frame_idx=frame,
            pts_time=frame / 25,
            timestamp_ms=frame * 40,
        )
        for index, frame in enumerate((100, 200, 300), 1)
    ]


def test_shortlist_scans_text_and_keeps_exact_caption_hit():
    selected = shortlist_video_keyframes(
        "con tôm đầu tiên chạm vào dĩa",
        _rows(),
        _Bundle(),
        anchor_keyframe_id="L21_V001#000100",
        max_images=2,
    )
    assert "L21_V001#000200" in {row.keyframe_id for row in selected}
    assert len(selected) == 2


def test_high_confidence_result_is_marked_for_auto_selection(tmp_path: Path):
    paths = {}
    for row in _rows():
        path = tmp_path / f"{row.frame_idx}.jpg"
        Image.new("RGB", (80, 45), "navy").save(path)
        paths[row.keyframe_id] = path
    client = _Client(
        '{"best_keyframe_id":"L21_V001#000200","alternatives":'
        '["L21_V001#000300"],"confidence":0.91,"is_match":true,'
        '"reason":"Đúng khoảnh khắc tôm chạm đĩa."}'
    )
    result = verify_video_keyframe(
        "the first shrimp touches the plate",
        "L21_V001",
        "L21_V001#000100",
        _rows(),
        _Bundle(),
        lambda keyframe_id: paths.get(keyframe_id),
        model="ag/gemini-3.8-flash-high",
        client=client,
        max_images=3,
        auto_threshold=0.8,
    )
    assert result.best_keyframe_id == "L21_V001#000200"
    assert result.auto_selected is True
    assert result.confidence_label == "cao"
    assert result.total_keyframes_scanned == 3
    assert len(client.calls) == 1
    assert result.image_parts_sent == 1
    assert len(client.calls[0][1]["images_b64"]) == 1


def test_low_confidence_result_requires_operator_confirmation(tmp_path: Path):
    path = tmp_path / "100.jpg"
    Image.new("RGB", (80, 45), "black").save(path)
    client = _Client(
        '{"best_keyframe_id":"L21_V001#000100","alternatives":[],"confidence":0.54,'
        '"is_match":true,"reason":"Khoảnh khắc chưa rõ."}'
    )
    result = verify_video_keyframe(
        "ambiguous moment",
        "L21_V001",
        "L21_V001#000100",
        _rows()[:1],
        _Bundle(),
        lambda _keyframe_id: path,
        model="ag/gemini-3.8-flash-high",
        client=client,
    )
    assert result.auto_selected is False
    assert result.confidence_label == "thấp"


def test_trake_shortlist_respects_adjacent_event_times():
    selected = shortlist_video_keyframes(
        "the shrimp touches the plate",
        _rows(),
        _Bundle(),
        anchor_keyframe_id="L21_V001#000200",
        min_timestamp_ms=4000,
        max_timestamp_ms=12000,
        max_images=12,
    )
    assert [row.keyframe_id for row in selected] == ["L21_V001#000200"]


def test_qa_prompt_verifies_answer_evidence(tmp_path: Path):
    path = tmp_path / "200.jpg"
    Image.new("RGB", (80, 45), "green").save(path)
    client = _Client(
        '{"best_keyframe_id":"L21_V001#000200","alternatives":[],"confidence":0.9,'
        '"is_match":true,"reason":"Phụ đề chứng minh đáp án."}'
    )
    result = verify_video_keyframe(
        "What touches the plate? Shrimp",
        "L21_V001",
        "L21_V001#000200",
        _rows()[1:2],
        _Bundle(),
        lambda _keyframe_id: path,
        model="ag/gemini-3.8-flash-high",
        client=client,
        task_type="QA",
        task_context={"question": "What touches the plate?", "proposed_answer": "Shrimp"},
        auto_threshold=0.85,
    )
    prompt = client.calls[0][0]
    assert "This is QA verification" in prompt
    assert '"proposed_answer": "Shrimp"' in prompt
    assert result.task_type == "QA"
    assert result.auto_selected is True


def test_trake_verification_reports_temporal_eligibility(tmp_path: Path):
    paths = {}
    for row in _rows():
        path = tmp_path / f"{row.frame_idx}.jpg"
        Image.new("RGB", (80, 45), "orange").save(path)
        paths[row.keyframe_id] = path
    client = _Client(
        '{"best_keyframe_id":"L21_V001#000200","alternatives":[],"confidence":0.88,'
        '"is_match":true,"reason":"Đúng E2 và nằm giữa E1, E3."}'
    )
    result = verify_video_keyframe(
        "E2: the shrimp touches the plate",
        "L21_V001",
        "L21_V001#000200",
        _rows(),
        _Bundle(),
        lambda keyframe_id: paths.get(keyframe_id),
        model="ag/gemini-3.8-flash-high",
        client=client,
        task_type="TRAKE",
        task_context={
            "current_event_index": 2,
            "previous_timestamp_ms": 4000,
            "next_timestamp_ms": 12000,
        },
        auto_threshold=0.85,
    )
    assert result.task_type == "TRAKE"
    assert result.eligible_keyframes == 1
    assert result.best_keyframe_id == "L21_V001#000200"
