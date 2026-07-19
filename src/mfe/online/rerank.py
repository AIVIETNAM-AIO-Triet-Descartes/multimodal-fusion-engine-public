"""Early-fusion rerank — stage C of docs/04 §1, §3.3.

A cross-encoder over the *original images* from the Asset Store (`get_asset`).
Expensive, so it only ever runs on a small top-k.

Why it is worth the cost (docs/context/buoi-2.md §7): a dual encoder computes
image features independently of the query, which makes it fast but weak on
detailed relations. A cross-encoder lets image and text interact — the thing
that separates "a man and a purple bag" from "a man holding a purple bag".

Why it must be optional (docs/04 §3.3): it has a latency budget, and over budget
means skip it and use the coarse ranking. docs/context/buoi-2.md §9: "tránh
timeout đúng lúc cần nhất".
"""

from __future__ import annotations

from mfe.contracts.tools import Candidate


class Reranker:
    """Owner: Phạm Trọng. TODO(Phạm Trọng): implement."""

    def __init__(self, model_name: str, budget_ms: int) -> None:
        self.model_name = model_name
        self.budget_ms = budget_ms

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Re-score a small candidate set. docs/04 §2 `rerank` tool.

        Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

        - Load images through `AssetStore.get_asset` — never read the path off
          the manifest and open it (docs/01 §3.2).
        - Enforce `budget_ms`. Check it *between* candidates and return what has
          been scored so far, merged with the coarse ranking for the rest —
          partial progress beats none. Do not check only at the start.
        - On timeout, raise/return TOOL_TIMEOUT so the cascade's fallback fires
          (docs/04 §1) and the log records the degradation.
        - Emit a fresh `evidence_id`; a reranked candidate is a new claim about
          a different model's judgement.
        """
        raise NotImplementedError("TODO(Phạm Trọng): cross-encoder rerank within budget_ms")

    def _score_pair(self, query: str, image_bytes: bytes) -> float:
        raise NotImplementedError("TODO(Phạm Trọng): cross-encoder forward pass")
