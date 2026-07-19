"""Vector Index — semantic search. docs/03 §1.

The coarse tier of the cascade (stage A in docs/04 §1): a cheap model run over
the whole store. Backend: Faiss.

Faiss knows only `internal_id -> embedding`. Translating a hit back to a
`keyframe_id` is the sidecar's job — see `sidecar.py`. That translation is where
this store is most likely to go silently wrong.
"""

from __future__ import annotations

from mfe.contracts.records import VectorRecord
from mfe.contracts.tools import Candidate


class VectorIndex:
    """Faiss index + sidecar, behind one object.

    Owner: Công Quang. TODO(Công Quang): implement.
    """

    def __init__(self, store_path: str, embed_dim: int) -> None:
        self.store_path = store_path
        self.embed_dim = embed_dim

    def add(self, records: list[VectorRecord]) -> None:
        """Add records. Called by Quốc Đạt's writers during ingest.

        TODO(Công Quang): implement. The order records are added in *is* the
        `internal_id` assignment — the sidecar must be appended in lockstep, in
        the same transaction-ish unit. An add that half-succeeds leaves the two
        permanently misaligned, and nothing downstream can detect it.
        """
        raise NotImplementedError("TODO(Công Quang): add embeddings to Faiss + sidecar")

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        video_id: str | None = None,
    ) -> list[Candidate]:
        """Cosine top-k. The `vector_search` API of docs/03 §1.

        TODO(Công Quang): implement.

        - Both sides must be L2-normalized for inner product to equal cosine.
          Records are checked at construction (docs/00 §4.1); the *query* vector
          arrives from Phạm Trọng and is not — normalize it here or the scores are
          silently wrong rather than absent.
        - `video_id` filters to one video (docs/03 §1).
        - Translate `internal_id -> keyframe_id` through the sidecar. Never
          return an `internal_id` to a caller; it is meaningless outside Faiss.
        - `evidence_id` on every Candidate (docs/00 §5).
        """
        raise NotImplementedError("TODO(Công Quang): faiss search -> Candidate list")

    def save(self) -> None:
        raise NotImplementedError("TODO(Công Quang): persist index + sidecar to store_path")

    @classmethod
    def load(cls, store_path: str, embed_dim: int) -> VectorIndex:
        raise NotImplementedError("TODO(Công Quang): load index + sidecar from store_path")

    def __len__(self) -> int:
        """Number of indexed keyframes. Must equal the sidecar length."""
        raise NotImplementedError("TODO(Công Quang)")
