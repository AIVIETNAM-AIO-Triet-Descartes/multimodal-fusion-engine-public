"""Early-fusion rerank — stage C of docs/04 §1, §3.3.

What makes this stage worth having
----------------------------------
Stage A ranks by comparing the query to **stored** CLIP vectors. Stage C goes
back to the **pixels** and scores them with a *second, independent* model. The
independence is the whole point: re-scoring with `clip-ViT-B-32` would reproduce
stage A's ranking exactly, because those are the very vectors BTC shipped and
the vector index already holds them. A second opinion is only a second opinion
if it comes from a different model.

The model here is `google/siglip-base-patch16-256-multilingual`, chosen because
it reads Vietnamese natively — the same requirement that forced the multilingual
text tower in stage A (`online/encoder.py`). A stronger English-only model would
need the query translated first, adding a failure mode between the operator and
the score.

**Budget policy: truncate, never raise** (CHỐT — Vỹ Triết). Over budget, the
reranker stops scoring and returns what it has, with the unscored tail appended
in its incoming (coarse) order. It does not raise, and callers therefore do not
wrap it in a try/except pretending to catch a timeout — an except branch that
nothing can reach reads as insurance while providing none.

That policy is safe precisely because the tail keeps coarse order: the result is
never worse *ordered* than stage B's, which is what "fallback bắt buộc" asks for.

How deep to rerank — and why "just the top 20" is wrong
-------------------------------------------------------
Measured on the project's CPU (no GPU): **116 ms per image**, so 100 images take
about 12 seconds.

The tempting argument is that a shallow rerank suffices: `Final Score` is the
mean of R@{1,5,20,50,100}, and R@20/50/100 are all satisfied by anything correct
in the top 20, so why score past there? That argument is **wrong**, and measuring
is what showed it. It quietly assumes the answer is already near the top — but
promoting a buried answer is the entire job of this stage. On the 2026 sample,
reranking the full 100 moved the correct keyframe:

    "đoàn xe mô tô ba bánh màu cam"      rank 92 -> 3      (Final 0.0 -> 0.8)
    "cổng hơi màu cam đỏ ở vạch đích"    rank 56 -> 10     (Final 0.0 -> 0.6)
    "người phụ nữ mặc áo dài vàng…"      rank 65 -> 50

A depth-25 rerank reaches none of those and changes nothing. The depth has to
cover where the answer actually sits, which on this data is deep.

So `retrieval.rerank_depth` defaults to the full submission depth, and
`rerank_budget_ms` is sized to let it finish rather than to cut it off. The cost
is real — roughly 12 s per query on CPU — and it is charged to TTCA, the metric
docs/context/buoi-2.md §2 says to optimise. It is worth it here only because a
human operator spends far longer than 12 s reading a result grid; on a GPU the
question disappears.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

from mfe.contracts.tools import Candidate
from mfe.online.tools.evidence import create_evidence_id

logger = logging.getLogger(__name__)

#: Independent of the stage A towers on purpose — see the module docstring.
DEFAULT_RERANK_CHECKPOINT = "google/siglip-base-patch16-256-multilingual"

#: Images per forward pass. Larger batches amortise the fixed cost, but the
#: budget is only checked between batches, so a huge batch would overshoot it.
_BATCH_SIZE = 8


class Reranker:
    """Stage C: score the query against the real keyframe images.

    Needs an `AssetStore` — that is what turns a `keyframe_id` into pixels, and
    is the difference between early fusion and a second text pass. Without one
    the reranker refuses to reorder anything rather than inventing a score.
    """

    def __init__(
        self,
        asset_store: Any = None,
        model_name: str = DEFAULT_RERANK_CHECKPOINT,
        budget_ms: int = 3000,
    ) -> None:
        self.asset_store = asset_store
        self.model_name = model_name
        self.budget_ms = budget_ms
        self._model: Any = None
        self._processor: Any = None

    # --- model plumbing ----------------------------------------------------

    def _load(self) -> bool:
        """Load the checkpoint on first use. False when it is unavailable.

        Lazy because importing torch and pulling ~1 GB of weights is not
        something `Cascade.from_config` should do just to answer a BM25 query.
        A missing checkpoint degrades to "no stage C", logged once, rather than
        taking down a run that would otherwise have produced coarse results.
        """
        if self._model is not None:
            return True
        try:
            from transformers import AutoModel, AutoProcessor

            started = time.perf_counter()
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).eval()
            logger.info(
                "rerank checkpoint %s loaded in %.1fs",
                self.model_name,
                time.perf_counter() - started,
            )
            return True
        except Exception as err:  # noqa: BLE001 — any load failure means "no stage C"
            logger.warning(
                "rerank checkpoint %s unavailable (%s); stage C is off for this run",
                self.model_name,
                err,
            )
            return False

    def _images(self, candidates: list[Candidate]) -> list[Any]:
        """Decode keyframe images. Missing ones come back as None."""
        from PIL import Image

        images: list[Any] = []
        for candidate in candidates:
            try:
                raw = self.asset_store.get_asset(candidate.keyframe_id)
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception as err:  # noqa: BLE001 — one bad asset is not a run
                logger.debug("no image for %s (%s)", candidate.keyframe_id, err)
                images.append(None)
        return images

    # --- the stage ---------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        run_id: str = "rerank_run",
    ) -> list[Candidate]:
        """Re-score candidates against their images, within the budget.

        Returns rescored candidates first (best first), then whatever the budget
        did not reach, in the order it arrived. Never raises on timeout — see
        the module docstring.

        A candidate whose image will not decode keeps its coarse position rather
        than being scored 0.0: a missing file says nothing about whether the
        keyframe answers the query.
        """
        if not candidates:
            return []
        if self.asset_store is None:
            logger.warning("rerank has no AssetStore; returning coarse order unchanged")
            return list(candidates)
        if not self._load():
            return list(candidates)

        import torch

        started = time.perf_counter()
        query_vec = self._encode_text(query)
        if query_vec is None:
            return list(candidates)

        scored: list[Candidate] = []
        scored_ids: set[str] = set()

        for start in range(0, len(candidates), _BATCH_SIZE):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms >= self.budget_ms:
                break

            batch = candidates[start : start + _BATCH_SIZE]
            images = self._images(batch)
            usable = [(c, im) for c, im in zip(batch, images, strict=True) if im is not None]
            if not usable:
                continue

            with torch.no_grad():
                pixels = self._processor(
                    images=[im for _, im in usable], return_tensors="pt"
                )
                feats = _pooled(self._model.get_image_features(**pixels))
                feats = torch.nn.functional.normalize(feats, dim=-1)
                sims = (feats @ query_vec).tolist()

            for (candidate, _), sim in zip(usable, sims, strict=True):
                scored_ids.add(candidate.keyframe_id)
                scored.append(
                    Candidate(
                        keyframe_id=candidate.keyframe_id,
                        video_id=candidate.video_id,
                        score=float(sim),
                        evidence_id=create_evidence_id("rerank", run_id, len(scored)),
                        matched_terms=candidate.matched_terms,
                    )
                )

        scored.sort(key=lambda c: c.score, reverse=True)
        remaining = [c for c in candidates if c.keyframe_id not in scored_ids]
        logger.info(
            "stage C scored %d/%d candidate(s) in %.0fms (budget %dms)",
            len(scored),
            len(candidates),
            (time.perf_counter() - started) * 1000.0,
            self.budget_ms,
        )
        return scored + remaining

    def _encode_text(self, query: str) -> Any:
        """Encode the query into the image space. None on failure."""
        import torch

        try:
            with torch.no_grad():
                tokens = self._processor(
                    text=[query], padding="max_length", return_tensors="pt"
                )
                vec = _pooled(self._model.get_text_features(**tokens))
                return torch.nn.functional.normalize(vec, dim=-1)[0]
        except Exception as err:  # noqa: BLE001 — degrade to coarse order
            logger.warning("rerank could not encode the query (%s)", err)
            return None


def _pooled(output: Any) -> Any:
    """Take the embedding tensor out of whatever the model returned.

    `transformers` has returned both a bare tensor and a `BaseModelOutput` from
    `get_*_features` across versions; normalising here keeps the version
    difference out of the scoring code.
    """
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    return output
