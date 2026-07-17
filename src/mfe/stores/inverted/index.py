"""Inverted Index — exact match & text search. docs/03 §2.

Covers what CLIP misses: proper nouns, numbers, signage, exact labels
(docs/context/buoi-1.md §8). Backend: BM25 or an inverted structure.

Two views of the same data (docs/03 §2):
  - document view : keyframe -> all its text        (InvertedRecord)
  - inverted view : term -> [{keyframe_id, weight}] (how it is stored)

TV1 writes the document view; this module owns the inversion.
"""

from __future__ import annotations

from mfe.contracts.records import InvertedRecord
from mfe.contracts.tools import Candidate


class InvertedIndex:
    """Owner: TV2. TODO(TV2): implement."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path

    def add(self, records: list[InvertedRecord]) -> None:
        """Index the document view. Called by TV1's writers during ingest.

        TODO(TV2): implement.

        Normalization is the contract that breaks quietly: TV1 emits
        "BẢN TIN 18H" and docs/03 §2.2 shows the term stored as "18h". Whatever
        casefolding and diacritic handling you choose must be applied
        identically at query time — agree it with TV1 (OCR) and TV3 (queries)
        rather than each side normalizing its own way.
        """
        raise NotImplementedError("TODO(TV2): invert documents -> term postings")

    def search(
        self,
        query_text: str,
        fields: list[str] | None = None,
        top_k: int = 100,
    ) -> list[Candidate]:
        """The `text_search` API of docs/03 §2.

        TODO(TV2): implement.

        - `fields` restricts to object / ocr / metadata (docs/03 §2.2).
        - Populate `Candidate.matched_terms` — TV3 needs to show the operator
          *why* a result matched, and the fusion step needs it to weigh a hit.
        - BM25 scores are unbounded and corpus-dependent. They are not
          comparable to CLIP's cosine; TV3 normalizes before fusing
          (docs/04 §3.1), but do not pre-normalize here — that would destroy
          information TV3 needs.
        """
        raise NotImplementedError("TODO(TV2): BM25 search -> Candidate list")

    def filter(self, constraints: dict, candidate_ids: list[str] | None = None) -> list[str]:
        """The `filter_metadata` API of docs/03 §2.

        TODO(TV2): implement. Returns keyframe_ids, not Candidates — this is a
        set operation, not a ranking. When `candidate_ids` is given, filter
        within it (TV3 narrows an existing candidate set); when None, filter the
        whole store.
        """
        raise NotImplementedError("TODO(TV2): constraint filter -> keyframe_id list")

    def save(self) -> None:
        raise NotImplementedError("TODO(TV2): persist to store_path")

    @classmethod
    def load(cls, store_path: str) -> InvertedIndex:
        raise NotImplementedError("TODO(TV2): load from store_path")
