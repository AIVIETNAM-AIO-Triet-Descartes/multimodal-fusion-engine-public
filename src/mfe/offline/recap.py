"""ReCap — recurrent shot captioning, batched across videos.

    docs/triet/ke-hoach-offline-ingest.md §4.6
    U-CESE, arXiv:2605.23274 §4.1

        (C_t, M_t) = LVLM(S_t, M_{t-1})

`M_t` is a memory string the model rewrites each shot: it carries setting,
characters and action forward, and the model decides what to keep. The paper
likens it to an RNN's hidden state.

**Why this module exists instead of another entry in `STAGES`.** Every other
stage is `(ctx) -> marker`, one video at a time, and that shape is correct for
them. It is fatal here. The recurrence makes a single video strictly sequential,
so a 7B model on one T4 would spend ~20-40s per shot with the GPU mostly idle
between short decodes — measured against 116.028 shots that is 645-1289 hours,
which is not a plan. Nothing forbids running *different videos* concurrently
though: each carries its own `M_t`. Batching sixteen of them brings the same work
to ~64 hours. So the unit of scheduling has to be "a batch of videos, one shot
each", and that does not fit `(ctx) -> marker`.

**Everything here is testable without a GPU.** The model is a callable —
`generate(prompts) -> list[str]` — so the state machine, the prompt, the parsing
and the failure counting are exercised by `tests/unit/offline/test_recap.py` with
a stub. Only the callable itself needs Kaggle.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: How many context keyframes go into one shot's prompt. Images dominate the
#: token bill and the prefill time, so this is the main cost knob.
CONTEXT_IMAGES = 4

#: Videos in flight. The recurrence serialises one video; this is where the
#: throughput comes from. VRAM-bound in practice — a 7B 4-bit model plus KV cache
#: for N sequences of ~5k tokens — so the working value has to be measured, not
#: assumed.
BATCH_VIDEOS = 8


@dataclass
class Shot:
    """One group of adjacent keyframes, and the speech that overlaps it."""

    index: int
    keyframe_ids: list[str]
    image_paths: list[Path]
    speech: str = ""
    start: float = 0.0
    end: float = 0.0


@dataclass
class VideoState:
    """One video's position in the recurrence, and its accumulated memory.

    Checkpointed after every shot rather than at the end, for a reason that does
    not apply to the other stages: this one is **not reproducible**. Re-running a
    video from the start produces different captions for the same shot, because
    `M_t` took a different path. Losing progress does not just cost time, it
    changes the data.
    """

    video_id: str
    shots: list[Shot]
    memory: str = ""
    shot_index: int = -1
    captions: dict[str, str] = field(default_factory=dict)
    parse_failures: int = 0
    empty_captions: int = 0

    @property
    def done(self) -> bool:
        return self.shot_index >= len(self.shots) - 1

    @property
    def next_shot(self) -> Shot:
        return self.shots[self.shot_index + 1]

    def to_checkpoint(self) -> dict:
        return {
            "video_id": self.video_id,
            "shot_index": self.shot_index,
            "memory": self.memory,
            "captions": self.captions,
            "parse_failures": self.parse_failures,
            "empty_captions": self.empty_captions,
        }

    @classmethod
    def resume(cls, video_id: str, shots: list[Shot], checkpoint: dict | None) -> VideoState:
        state = cls(video_id=video_id, shots=shots)
        if checkpoint:
            state.memory = checkpoint.get("memory", "")
            state.shot_index = int(checkpoint.get("shot_index", -1))
            state.captions = dict(checkpoint.get("captions", {}))
            state.parse_failures = int(checkpoint.get("parse_failures", 0))
            state.empty_captions = int(checkpoint.get("empty_captions", 0))
        return state


@dataclass
class Prompt:
    """What one call needs. Images stay as paths so the backend loads them."""

    video_id: str
    shot_index: int
    text: str
    images: list[Path]


_VI = """Bạn đang mô tả các cảnh liên tiếp của một video.

TRÍ NHỚ (bối cảnh đã tích luỹ từ các cảnh trước): {memory}
LỜI THOẠI trong đoạn này: {speech}

Viết mô tả bằng tiếng Việt cho cảnh này: thực thể, thuộc tính, tương tác.
Dùng trí nhớ để HIỂU cảnh, nhưng KHÔNG nhắc tới nó trong mô tả.
Sau đó cập nhật trí nhớ: giữ cái còn liên quan, bỏ cái đã cũ, tối đa 80 từ.

Chỉ trả về JSON, không thêm gì khác:
{{"caption": "...", "memory": "..."}}"""

_EN = """You are describing consecutive scenes from one video.

MEMORY (context accumulated from earlier scenes): {memory}
SPEECH in this segment: {speech}

Write a description of this scene in English: entities, attributes, interactions.
Use the memory to UNDERSTAND the scene, but do NOT mention it in the description.
Then update the memory: keep what is still relevant, drop what is stale, max 80 words.

Return only JSON, nothing else:
{{"caption": "...", "memory": "..."}}"""


def build_prompt(state: VideoState, language: str = "vi") -> Prompt:
    """The prompt for this video's next shot.

    Memory and speech go in as plain text; the images are the shot's own
    keyframes, capped at `CONTEXT_IMAGES`.
    """
    shot = state.next_shot
    template = _VI if language == "vi" else _EN
    return Prompt(
        video_id=state.video_id,
        shot_index=shot.index,
        text=template.format(
            memory=state.memory or "(chưa có)",
            speech=shot.speech or "(không có)",
        ),
        images=shot.image_paths[:CONTEXT_IMAGES],
    )


def parse_answer(text: str) -> tuple[str, str | None]:
    """`(caption, memory)`. `memory is None` means the answer did not parse.

    Returning `None` rather than an empty string is the whole point. A failed
    parse must **not** overwrite `M_{t-1}` with nothing: that silently breaks the
    recurrence for the rest of the video, and every later caption is then produced
    without the context the method exists to provide. Bảo's notebook had the same
    shape of bug — a missing `[CAPTION]` marker dumped the whole response into the
    caption field with no counter — and here it would cost far more than one row.
    """
    cleaned = (text or "").strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    cleaned = cleaned.removesuffix("```").strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return "", None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return "", None
    if not isinstance(payload, dict):
        return "", None
    caption = str(payload.get("caption", "")).strip()
    memory = payload.get("memory")
    return caption, str(memory).strip() if isinstance(memory, str) else None


def advance(state: VideoState, answer: str) -> None:
    """Apply one model answer to one video's state.

    On a parse failure the shot is counted and skipped, and **memory is left
    untouched** — a broken answer is missing information, not a reason to forget.
    """
    shot = state.next_shot
    caption, memory = parse_answer(answer)

    if memory is None:
        state.parse_failures += 1
    else:
        state.memory = memory

    if caption:
        for keyframe_id in shot.keyframe_ids:
            state.captions[keyframe_id] = caption
    else:
        state.empty_captions += 1

    state.shot_index = shot.index


Generate = Callable[[Sequence[Prompt]], list[str]]
OnShot = Callable[[VideoState], None]


def run_recap(
    states: list[VideoState],
    generate: Generate,
    *,
    language: str = "vi",
    batch_videos: int = BATCH_VIDEOS,
    on_shot: OnShot | None = None,
) -> list[VideoState]:
    """Drive every video forward one shot at a time, `batch_videos` at once.

    Each round takes the next shot from up to `batch_videos` unfinished videos,
    asks for all of them in one call, and applies the answers. Videos have
    different shot counts, so the batch refills from whatever is still running
    rather than waiting for the slowest.

    `on_shot` fires after each video advances — that is where the checkpoint gets
    written. Per shot, not per video: this stage is not reproducible, so a lost
    round is not re-doable work, it is different data.
    """
    if batch_videos < 1:
        raise ValueError(f"batch_videos must be >= 1, got {batch_videos}")

    pending = [s for s in states if not s.done]
    while pending:
        batch = pending[:batch_videos]
        prompts = [build_prompt(s, language) for s in batch]
        answers = generate(prompts)
        if len(answers) != len(prompts):
            raise RuntimeError(
                f"backend tra ve {len(answers)} cau tra loi cho {len(prompts)} prompt"
            )
        for state, answer in zip(batch, answers, strict=True):
            advance(state, answer)
            if on_shot is not None:
                on_shot(state)
        pending = [s for s in pending if not s.done]
    return states


def summarise(states: list[VideoState]) -> dict:
    """Counts worth putting in a marker — including the ones that mean trouble."""
    shots = sum(len(s.shots) for s in states)
    failures = sum(s.parse_failures for s in states)
    return {
        "videos": len(states),
        "shots": shots,
        "captions": sum(len(s.captions) for s in states),
        "parse_failures": failures,
        "empty_captions": sum(s.empty_captions for s in states),
        # The number to watch. A 7B model holding a memory string coherent across
        # ~130 shots is the open question of this whole approach (§4.6b), and a
        # rising parse-failure rate is the first place it shows.
        "parse_failure_rate": round(failures / shots, 4) if shots else 0.0,
    }
