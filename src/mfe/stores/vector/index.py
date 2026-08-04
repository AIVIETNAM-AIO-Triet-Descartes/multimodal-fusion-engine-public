"""Vector Index — semantic search. docs/03 §1.

The coarse tier of the cascade (stage A in docs/04 §1): a cheap model run over
the whole store. Backend: Faiss.

Faiss knows only `internal_id -> embedding`. Translating a hit back to a
`keyframe_id` is the sidecar's job — see `sidecar.py`. That translation is where
this store is most likely to go silently wrong.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from mfe.contracts.records import VectorRecord
from mfe.contracts.tools import Candidate
from mfe.stores.vector.sidecar import Sidecar

_FAISS_FILENAME = "index.faiss"
_MAX_TOP_K = 500


class VectorIndex:
    """Faiss index + sidecar, behind one object.

    Owner: Công Quang.
    """

    def __init__(self, store_path: str, embed_dim: int) -> None:
        self.store_path = store_path
        self.embed_dim = embed_dim
        self._index = faiss.IndexFlatIP(embed_dim)
        self._sidecar = Sidecar(store_path=store_path)

    def add(self, records: list[VectorRecord]) -> None:
        """Add records. Called by Quốc Đạt's writers during ingest.

        The order records are added in *is* the `internal_id` assignment. All
        records are validated before either store is mutated, then Faiss and the
        sidecar are updated in lockstep.
        """
        self._sidecar.verify_alignment(faiss_size=len(self))

        if not records:
            return

        first_internal_id = len(self)
        batch_keyframe_ids: set[str] = set()
        for offset, record in enumerate(records):
            expected_internal_id = first_internal_id + offset
            if record.internal_id != expected_internal_id:
                raise ValueError(
                    f"Expected internal_id {expected_internal_id}, "
                    f"got {record.internal_id}"
                )
            if len(record.embedding) != self.embed_dim:
                raise ValueError(
                    f"Embedding for {record.keyframe_id!r} has dimension "
                    f"{len(record.embedding)}; expected {self.embed_dim}"
                )
            if not np.isfinite(record.embedding).all():
                raise ValueError(
                    f"Embedding for {record.keyframe_id!r} contains NaN or infinity"
                )
            if record.keyframe_id in batch_keyframe_ids:
                raise ValueError(f"Duplicate keyframe_id in batch: {record.keyframe_id!r}")
            try:
                self._sidecar.to_internal_id(record.keyframe_id)
            except KeyError:
                pass
            else:
                raise ValueError(f"Duplicate keyframe_id: {record.keyframe_id!r}")
            batch_keyframe_ids.add(record.keyframe_id)

        embeddings = np.ascontiguousarray(
            [record.embedding for record in records],
            dtype=np.float32,
        )
        self._index.add(embeddings)

        for record in records:
            assigned_internal_id = self._sidecar.append(
                keyframe_id=record.keyframe_id,
                video_id=record.video_id,
            )
            if assigned_internal_id != record.internal_id:
                raise AssertionError(
                    f"Sidecar assigned internal_id {assigned_internal_id}; "
                    f"expected {record.internal_id}"
                )

        self._sidecar.verify_alignment(faiss_size=len(self))

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        video_id: str | None = None,
        run_id: str = "store",
    ) -> list[Candidate]:
        """Cosine top-k. The `vector_search` API of docs/03 §1.

        - Both sides must be L2-normalized for inner product to equal cosine.
          Records are checked at construction (docs/00 §4.1); the *query* vector
          arrives from Phạm Trọng and is not — normalize it here or the scores are
          silently wrong rather than absent.
        - `video_id` filters to one video (docs/03 §1).
        - Translate `internal_id -> keyframe_id` through the sidecar. Never
          return an `internal_id` to a caller; it is meaningless outside Faiss.
        - `evidence_id` on every Candidate (docs/00 §5).
        """
        if not 1 <= top_k <= _MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {_MAX_TOP_K}; got {top_k}")
        if len(query_embedding) != self.embed_dim:
            raise ValueError(
                f"Query embedding has dimension {len(query_embedding)}; "
                f"expected {self.embed_dim}"
            )

        query = np.asarray(query_embedding, dtype=np.float32)
        if not np.isfinite(query).all():
            raise ValueError("Query embedding contains NaN or infinity")
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            raise ValueError("Query embedding must not be the zero vector")
        query = np.ascontiguousarray((query / query_norm).reshape(1, -1))

        self._sidecar.verify_alignment(faiss_size=len(self))
        if len(self) == 0:
            return []

        search_k = len(self) if video_id is not None else min(top_k, len(self))
        scores, internal_ids = self._index.search(query, search_k)

        candidates: list[Candidate] = []
        for score, internal_id in zip(scores[0], internal_ids[0], strict=True):
            internal_id = int(internal_id)
            if internal_id < 0:
                continue

            candidate_video_id = self._sidecar.video_id_of(internal_id)
            if video_id is not None and candidate_video_id != video_id:
                continue

            candidates.append(
                Candidate(
                    keyframe_id=self._sidecar.to_keyframe_id(internal_id),
                    video_id=candidate_video_id,
                    score=float(score),
                    # "{tool_name}:{run_id}:{ordinal}" — ordinal is the rank in
                    # THIS call, not the internal_id (contracts/tools.py).
                    evidence_id=f"vector_search:{run_id}:{len(candidates)}",
                )
            )
            if len(candidates) == top_k:
                break

        return candidates

    def save(self) -> None:
        """Persist the aligned Faiss index and sidecar."""
        self._sidecar.verify_alignment(faiss_size=len(self))
        store_dir = Path(self.store_path)
        store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(store_dir / _FAISS_FILENAME))
        self._sidecar.save()

    @classmethod
    def load(cls, store_path: str, embed_dim: int) -> VectorIndex:
        """Restore a persisted index and reject a mismatched dimension."""
        loaded_index = faiss.read_index(str(Path(store_path) / _FAISS_FILENAME))
        if loaded_index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                f"Stored Faiss metric {loaded_index.metric_type} is not inner product"
            )
        if loaded_index.d != embed_dim:
            raise ValueError(
                f"Stored Faiss dimension {loaded_index.d} does not match "
                f"requested dimension {embed_dim}"
            )

        vector_index = cls(store_path=store_path, embed_dim=embed_dim)
        vector_index._index = loaded_index
        vector_index._sidecar = Sidecar.load(store_path=store_path)
        vector_index._sidecar.verify_alignment(faiss_size=len(vector_index))
        return vector_index

    def __len__(self) -> int:
        """Number of indexed keyframes. Must equal the sidecar length."""
        return self._index.ntotal
