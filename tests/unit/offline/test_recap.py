"""Locks ReCap's state machine — mfe.offline.recap.

The model is a callable, so everything except the model runs here: batching
across videos, the recurrence, checkpoint/resume, and what happens when the
answer does not parse. That last one carries the most weight — a 7B model
emitting reliable JSON on every one of ~116.000 calls is exactly the assumption
this approach cannot make.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mfe.offline.recap import (
    Shot,
    VideoState,
    advance,
    build_prompt,
    parse_answer,
    run_recap,
    summarise,
)


def make_shots(video_id: str, count: int) -> list[Shot]:
    return [
        Shot(
            index=i,
            keyframe_ids=[f"{video_id}#{i * 100:06d}", f"{video_id}#{i * 100 + 30:06d}"],
            image_paths=[Path(f"{video_id}/{i}_{k}.jpg") for k in range(6)],
            speech=f"loi thoai {i}",
        )
        for i in range(count)
    ]


def answer(caption: str, memory: str) -> str:
    import json

    return json.dumps({"caption": caption, "memory": memory}, ensure_ascii=False)


class TestParseAnswer:
    def test_reads_plain_json(self):
        assert parse_answer(answer("mot canh", "nho gi do")) == ("mot canh", "nho gi do")

    def test_tolerates_code_fences(self):
        text = "```json\n" + answer("a", "b") + "\n```"
        assert parse_answer(text) == ("a", "b")

    def test_tolerates_chatter_around_the_object(self):
        """7B models prepend explanations. The object is still in there."""
        text = "Here is the JSON you asked for:\n" + answer("a", "b") + "\nHope that helps!"
        assert parse_answer(text) == ("a", "b")

    @pytest.mark.parametrize("text", ["", "khong phai json", "{broken", '{"caption": "x"}'])
    def test_missing_memory_is_None_not_empty(self, text):
        """The distinction the whole recurrence rests on.

        An unparseable answer must not overwrite `M_{t-1}` with nothing — that
        breaks the memory chain for the rest of the video, and every later caption
        is then produced without the context the method exists to supply. `None`
        says "no answer"; `""` would say "the memory is now empty".
        """
        assert parse_answer(text)[1] is None


class TestAdvance:
    def test_applies_the_caption_to_every_keyframe_of_the_shot(self):
        state = VideoState("L21_V001", make_shots("L21_V001", 3))
        advance(state, answer("canh dau", "nho 1"))
        assert state.shot_index == 0
        assert state.memory == "nho 1"
        assert set(state.captions) == set(state.shots[0].keyframe_ids)

    def test_a_parse_failure_keeps_the_previous_memory(self):
        state = VideoState("L21_V001", make_shots("L21_V001", 3))
        advance(state, answer("a", "nho tot"))
        advance(state, "model noi lung tung")
        assert state.memory == "nho tot"      # KHONG bi xoa
        assert state.parse_failures == 1
        assert state.shot_index == 1          # van di tiep

    def test_an_empty_caption_is_counted_not_stored(self):
        state = VideoState("L21_V001", make_shots("L21_V001", 2))
        advance(state, answer("", "nho"))
        assert state.captions == {}
        assert state.empty_captions == 1


class TestBatching:
    def test_runs_every_video_to_completion(self):
        """Videos have different shot counts, so the batch must refill from
        whatever is still running rather than wait for the slowest."""
        states = [
            VideoState("A", make_shots("A", 3)),
            VideoState("B", make_shots("B", 7)),
            VideoState("C", make_shots("C", 1)),
        ]
        rounds = []

        def generate(prompts):
            rounds.append([p.video_id for p in prompts])
            return [answer(f"{p.video_id}-{p.shot_index}", f"m{p.shot_index}") for p in prompts]

        run_recap(states, generate, batch_videos=2)
        assert all(s.done for s in states)
        assert [len(s.captions) for s in states] == [6, 14, 2]
        assert max(len(r) for r in rounds) == 2

    def test_each_video_keeps_its_own_memory(self):
        """Two videos in one batch must not share `M_t` — that is the whole
        reason batching across videos is safe while batching within one is not.
        """
        states = [VideoState("A", make_shots("A", 3)), VideoState("B", make_shots("B", 3))]
        seen: list[tuple[str, str]] = []

        def generate(prompts):
            seen.extend((p.video_id, p.text) for p in prompts)
            return [answer("c", f"nho-cua-{p.video_id}") for p in prompts]

        run_recap(states, generate, batch_videos=2)
        assert states[0].memory == "nho-cua-A"
        assert states[1].memory == "nho-cua-B"

        # Sau vong dau, moi prompt chi duoc mang tri nho CUA CHINH video do.
        later = [(vid, text) for vid, text in seen if "nho-cua-" in text]
        assert later, "khong vong nao mang tri nho — test khong kiem duoc gi"
        for video_id, text in later:
            assert f"nho-cua-{video_id}" in text
            other = "B" if video_id == "A" else "A"
            assert f"nho-cua-{other}" not in text

    def test_memory_from_the_previous_shot_reaches_the_next_prompt(self):
        state = VideoState("A", make_shots("A", 2))
        advance(state, answer("c", "bối cảnh: chợ đêm"))
        assert "bối cảnh: chợ đêm" in build_prompt(state).text

    def test_refuses_a_backend_that_returns_the_wrong_count(self):
        states = [VideoState("A", make_shots("A", 2))]
        with pytest.raises(RuntimeError, match="tra ve"):
            run_recap(states, lambda prompts: [], batch_videos=2)

    def test_on_shot_fires_after_every_shot_not_every_video(self):
        """The checkpoint hook. This stage is not reproducible — re-running a
        video yields different captions because `M_t` took another path — so a
        lost round is not re-doable work, it is different data."""
        states = [VideoState("A", make_shots("A", 4))]
        seen = []
        run_recap(
            states,
            lambda ps: [answer("c", "m") for _ in ps],
            on_shot=lambda s: seen.append(s.shot_index),
        )
        assert seen == [0, 1, 2, 3]


class TestResume:
    def test_picks_up_where_the_checkpoint_left_off(self):
        shots = make_shots("A", 5)
        first = VideoState("A", shots)
        run_recap([first], lambda ps: [answer("c", "m") for _ in ps])
        checkpoint = first.to_checkpoint()

        # New session: same shots, memory and position restored.
        resumed = VideoState.resume("A", shots, checkpoint)
        assert resumed.done
        assert resumed.memory == "m"
        assert len(resumed.captions) == 10

    def test_a_half_finished_video_continues_rather_than_restarts(self):
        shots = make_shots("A", 5)
        partial = VideoState("A", shots)
        run_recap([partial], lambda ps: [answer("c1", "m1") for _ in ps][:1], batch_videos=1)

        stopped = VideoState.resume(
            "A", shots, {"shot_index": 1, "memory": "m1", "captions": {}}
        )
        asked = []
        run_recap([stopped], lambda ps: (asked.extend(p.shot_index for p in ps),
                                         [answer("c2", "m2") for _ in ps])[1])
        assert asked == [2, 3, 4]      # khong lam lai shot 0 va 1


class TestSummary:
    def test_reports_the_failure_rate(self):
        state = VideoState("A", make_shots("A", 4))
        for text in (answer("c", "m"), "rac", answer("c", "m"), "rac"):
            advance(state, text)
        report = summarise([state])
        assert report["parse_failures"] == 2
        assert report["parse_failure_rate"] == 0.5
