"""Stage C — mfe.online.rerank.

No real checkpoint here: SigLIP is ~1 GB and 116 ms/image, which belongs in a
measurement run, not a unit test. What is worth pinning down is the *contract*
around the model — when stage C must refuse to reorder, what the budget does,
and what happens to a candidate whose image will not load. Those are the paths
that decide whether a bad day degrades or corrupts.
"""

from __future__ import annotations

import io

import pytest

from mfe.contracts.tools import Candidate
from mfe.online.rerank import Reranker


def _cand(index: int, score: float, ordinal: int) -> Candidate:
    return Candidate(
        keyframe_id=f"L23_V001#{index:03d}",
        video_id="L23_V001",
        score=score,
        evidence_id=f"vector_search:t:{ordinal}",
    )


def _jpeg() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 120, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeAssetStore:
    """Returns a tiny JPEG for every keyframe, except those in `broken`."""

    def __init__(self, broken: set[str] | None = None) -> None:
        self.broken = broken or set()
        self.asked: list[str] = []

    def get_asset(self, keyframe_id: str, *args, **kwargs) -> bytes:
        self.asked.append(keyframe_id)
        if keyframe_id in self.broken:
            raise FileNotFoundError(keyframe_id)
        return _jpeg()


def _wire_fake_model(reranker: Reranker, scores: list[float]) -> None:
    """Give the reranker a model that hands back `scores`, in call order.

    The reranker feeds images to the processor in candidate order, batch by
    batch, so consuming a queue reproduces "image i gets score i" without the
    fake having to recognise any pixels.
    """
    import torch

    queue = list(scores)

    class FakeProcessor:
        def __call__(self, images=None, text=None, **kwargs):
            count = len(images) if images is not None else len(text)
            return {"_n": count}

    class FakeModel:
        def get_image_features(self, **kwargs):
            taken = [queue.pop(0) for _ in range(kwargs["_n"])]
            # Unit vectors whose cosine with the text vector below is `taken`.
            return torch.tensor(
                [[s, (max(0.0, 1.0 - s * s)) ** 0.5] for s in taken], dtype=torch.float32
            )

        def get_text_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    reranker._processor = FakeProcessor()
    reranker._model = FakeModel()


# --- refusals: stage C must not reorder on nothing --------------------------


def test_empty_candidate_list() -> None:
    assert Reranker(asset_store=FakeAssetStore()).rerank("q", []) == []


def test_without_an_asset_store_the_order_is_untouched() -> None:
    """No pixels means no early fusion. Returning the input is the honest move.

    Scoring a keyframe without looking at its image is what the old hash stub
    did, and it demoted correct answers with total confidence.
    """
    candidates = [_cand(1, 9.0, 0), _cand(2, 8.0, 1)]

    out = Reranker(asset_store=None).rerank("q", candidates)

    assert [c.keyframe_id for c in out] == [c.keyframe_id for c in candidates]
    assert all(c.evidence_id.startswith("vector_search:") for c in out)


def test_unavailable_checkpoint_degrades_to_coarse_order() -> None:
    """A missing model must not take down a run that had coarse results."""
    candidates = [_cand(1, 9.0, 0), _cand(2, 8.0, 1)]
    reranker = Reranker(asset_store=FakeAssetStore(), model_name="no/such-model-xyz")

    out = reranker.rerank("q", candidates)

    assert [c.keyframe_id for c in out] == [c.keyframe_id for c in candidates]


# --- the happy path ---------------------------------------------------------


def test_candidates_are_reordered_by_image_score() -> None:
    """The point of the stage: coarse order in, image-scored order out."""
    candidates = [_cand(1, 9.0, 0), _cand(2, 8.0, 1), _cand(3, 7.0, 2)]
    reranker = Reranker(asset_store=FakeAssetStore())
    _wire_fake_model(reranker, [0.1, 0.9, 0.5])

    out = reranker.rerank("q", candidates, run_id="t")

    assert [c.keyframe_id for c in out] == [
        "L23_V001#002",
        "L23_V001#003",
        "L23_V001#001",
    ]
    assert [round(c.score, 3) for c in out] == [0.9, 0.5, 0.1]


def test_reranked_candidates_carry_a_rerank_evidence_id() -> None:
    """Nhật Bảo's validator traces a submission back to the stage that made it."""
    reranker = Reranker(asset_store=FakeAssetStore())
    _wire_fake_model(reranker, [0.5, 0.4])

    out = reranker.rerank("q", [_cand(1, 9.0, 0), _cand(2, 8.0, 1)], run_id="run7")

    assert all(c.evidence_id.startswith("rerank:run7:") for c in out)


def test_the_image_is_actually_read() -> None:
    """Guards against a future refactor that scores ids instead of pixels."""
    store = FakeAssetStore()
    reranker = Reranker(asset_store=store)
    _wire_fake_model(reranker, [0.5, 0.4])

    reranker.rerank("q", [_cand(1, 9.0, 0), _cand(2, 8.0, 1)])

    assert store.asked == ["L23_V001#001", "L23_V001#002"]


# --- degradation ------------------------------------------------------------


def test_a_missing_image_keeps_its_coarse_position() -> None:
    """Not scored 0.0 — a missing file says nothing about relevance.

    Scoring it zero would bury a keyframe for a storage problem, and the
    operator would read that as "the system judged it irrelevant".
    """
    candidates = [_cand(1, 9.0, 0), _cand(2, 8.0, 1), _cand(3, 7.0, 2)]
    reranker = Reranker(asset_store=FakeAssetStore(broken={"L23_V001#002"}))
    _wire_fake_model(reranker, [0.1, 0.9])  # only #001 and #003 reach the model

    out = reranker.rerank("q", candidates)

    assert len(out) == 3
    assert {c.keyframe_id for c in out} == {
        "L23_V001#001",
        "L23_V001#002",
        "L23_V001#003",
    }
    assert out[-1].keyframe_id == "L23_V001#002"


def test_zero_budget_scores_nothing_and_raises_nothing() -> None:
    """Truncate, never raise (CHỐT). Zero budget is the extreme of that."""
    candidates = [_cand(1, 9.0, 0), _cand(2, 8.0, 1)]
    reranker = Reranker(asset_store=FakeAssetStore(), budget_ms=0)
    _wire_fake_model(reranker, [0.1, 0.9])

    out = reranker.rerank("q", candidates)

    assert [c.keyframe_id for c in out] == [c.keyframe_id for c in candidates]


def test_every_candidate_survives_the_stage() -> None:
    """Reranking reorders; it must never drop an answer we are allowed to submit."""
    candidates = [_cand(i, 10.0 - i, i) for i in range(1, 13)]
    reranker = Reranker(asset_store=FakeAssetStore())
    _wire_fake_model(reranker, [i / 20 for i in range(12)])

    out = reranker.rerank("q", candidates)

    assert len(out) == len(candidates)
    assert {c.keyframe_id for c in out} == {c.keyframe_id for c in candidates}


def test_scores_stay_in_the_unit_interval() -> None:
    """`final_score.r_at_k` rejects anything outside [0,1]; cosine belongs there."""
    reranker = Reranker(asset_store=FakeAssetStore())
    _wire_fake_model(reranker, [0.0, 1.0, 0.33])

    out = reranker.rerank("q", [_cand(i, 9.0, i) for i in range(1, 4)])

    assert all(0.0 <= c.score <= 1.0 for c in out)


@pytest.mark.parametrize("count", [1, 8, 9, 17])
def test_batching_does_not_lose_or_duplicate(count: int) -> None:
    """Batch boundaries are where an off-by-one would silently drop a candidate."""
    candidates = [_cand(i, 10.0, i) for i in range(1, count + 1)]
    reranker = Reranker(asset_store=FakeAssetStore())
    _wire_fake_model(reranker, [i / 100 for i in range(count)])

    out = reranker.rerank("q", candidates)

    assert len(out) == count
    assert len({c.keyframe_id for c in out}) == count
