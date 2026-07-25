"""Inverted Index — exact match & text search. docs/03 §2.

Covers what CLIP misses: proper nouns, numbers, signage, exact labels
(docs/context/buoi-1.md §8). Backend: BM25 or an inverted structure.

Two views of the same data (docs/03 §2):
  - document view : keyframe -> all its text        (InvertedRecord)
  - inverted view : term -> [{keyframe_id, weight}] (how it is stored)

Quốc Đạt writes the document view; this module owns the inversion.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from mfe.contracts.records import InvertedRecord
from mfe.contracts.tools import Candidate

_INDEX_FILENAME = "inverted_records.json"
_MAX_TOP_K = 500
_FIELD_ALIASES = {
    "object": "object_labels",
    "object_labels": "object_labels",
    "ocr": "ocr_text",
    "ocr_text": "ocr_text",
    "metadata": "metadata_text",
    "metadata_text": "metadata_text",
    "asr": "asr_text",
    "asr_text": "asr_text",
}
_DEFAULT_FIELDS = ("object_labels", "ocr_text", "metadata_text", "asr_text")


def _normalize_text(text: str) -> str:
    """Apply the same Unicode and case normalization at ingest and query time."""
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", _normalize_text(text), flags=re.UNICODE)


class InvertedIndex:
    """BM25-backed document index. Owner: Công Quang."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._records: dict[str, InvertedRecord] = {}
        self._bm25_cache: dict[
            tuple[str, ...],
            tuple[list[str], list[list[str]], BM25Okapi | None],
        ] = {}

    def add(self, records: list[InvertedRecord]) -> None:
        """Index the document view. Called by Quốc Đạt's writers during ingest.

        Normalization is the contract that breaks quietly: Quốc Đạt emits
        "BẢN TIN 18H" and docs/03 §2.2 shows the term stored as "18h". Whatever
        casefolding and diacritic handling you choose must be applied
        identically at query time — agree it with Quốc Đạt (OCR) and Phạm Trọng (queries)
        rather than each side normalizing its own way.
        """
        for record in records:
            self._records[record.keyframe_id] = record
        self._bm25_cache.clear()

    def search(
        self,
        query_text: str,
        fields: list[str] | None = None,
        top_k: int = 100,
    ) -> list[Candidate]:
        """The `text_search` API of docs/03 §2.

        - `fields` restricts to object / ocr / metadata (docs/03 §2.2).
        - Populate `Candidate.matched_terms` — Phạm Trọng needs to show the operator
          *why* a result matched, and the fusion step needs it to weigh a hit.
        - BM25 scores are unbounded and corpus-dependent. They are not
          comparable to CLIP's cosine; Phạm Trọng normalizes before fusing
          (docs/04 §3.1), but do not pre-normalize here — that would destroy
        information Phạm Trọng needs.
        """
        if not 1 <= top_k <= _MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {_MAX_TOP_K}; got {top_k}")

        selected_fields = self._resolve_fields(fields)
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            raise ValueError("query_text must contain at least one searchable term")
        if not self._records:
            return []

        keyframe_ids, documents, bm25 = self._get_bm25(selected_fields)
        if bm25 is None:
            return []

        scores = bm25.get_scores(query_tokens)
        ranked_positions = np.argsort(-scores, kind="stable")

        candidates: list[Candidate] = []
        for position in ranked_positions:
            position = int(position)
            document_terms = set(documents[position])
            matched_terms = list(
                dict.fromkeys(term for term in query_tokens if term in document_terms)
            )
            if not matched_terms:
                continue

            record = self._records[keyframe_ids[position]]
            candidates.append(
                Candidate(
                    keyframe_id=record.keyframe_id,
                    video_id=record.video_id,
                    score=float(scores[position]),
                    evidence_id=f"text_search:store:{record.keyframe_id}",
                    matched_terms=matched_terms,
                )
            )
            if len(candidates) == top_k:
                break

        return candidates

    def filter(
        self,
        constraints: dict[str, Any],
        candidate_ids: list[str] | None = None,
    ) -> list[str]:
        """The `filter_metadata` API of docs/03 §2.

        Returns keyframe_ids, not Candidates — this is a set operation, not a
        ranking. When `candidate_ids` is given, filter within it; when None,
        filter the whole store.
        """
        self._validate_constraints(constraints)
        source_ids = candidate_ids if candidate_ids is not None else list(self._records)
        return [
            keyframe_id
            for keyframe_id in source_ids
            if keyframe_id in self._records
            and self._matches_constraints(self._records[keyframe_id], constraints)
        ]

    def save(self) -> None:
        store_dir = Path(self.store_path)
        store_dir.mkdir(parents=True, exist_ok=True)
        target = store_dir / _INDEX_FILENAME
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        payload = [
            record.model_dump(mode="json")
            for record in self._records.values()
        ]
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)

    @classmethod
    def load(cls, store_path: str) -> InvertedIndex:
        source = Path(store_path) / _INDEX_FILENAME
        raw_records = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw_records, list):
            raise ValueError("Persisted inverted index must contain a JSON list")

        index = cls(store_path=store_path)
        index.add([InvertedRecord.model_validate(item) for item in raw_records])
        return index

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _resolve_fields(fields: list[str] | None) -> tuple[str, ...]:
        if fields is None:
            return _DEFAULT_FIELDS

        resolved: list[str] = []
        for field in fields:
            try:
                canonical = _FIELD_ALIASES[field]
            except KeyError:
                allowed = ", ".join(sorted(_FIELD_ALIASES))
                raise ValueError(
                    f"Unknown search field {field!r}; expected one of {allowed}"
                ) from None
            if canonical not in resolved:
                resolved.append(canonical)
        if not resolved:
            raise ValueError("fields must contain at least one search field")
        return tuple(resolved)

    @staticmethod
    def _record_text(record: InvertedRecord, fields: tuple[str, ...]) -> str:
        parts: list[str] = []
        for field in fields:
            value = getattr(record, field)
            if isinstance(value, list):
                parts.extend(value)
            else:
                parts.append(value)
        return " ".join(parts)

    def _get_bm25(
        self,
        fields: tuple[str, ...],
    ) -> tuple[list[str], list[list[str]], BM25Okapi | None]:
        cached = self._bm25_cache.get(fields)
        if cached is not None:
            return cached

        keyframe_ids = list(self._records)
        documents = [
            _tokenize(self._record_text(self._records[keyframe_id], fields))
            for keyframe_id in keyframe_ids
        ]
        bm25 = BM25Okapi(documents) if any(documents) else None
        result = (keyframe_ids, documents, bm25)
        self._bm25_cache[fields] = result
        return result

    @staticmethod
    def _constraint_values(value: Any) -> list[str]:
        if isinstance(value, dict):
            unknown_operators = set(value) - {"contains"}
            if unknown_operators:
                raise ValueError(
                    f"Unknown constraint operator(s): {sorted(unknown_operators)}"
                )
            value = value.get("contains", [])
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item) for item in value]
        else:
            raise ValueError(
                f"Constraint value must be a string or collection; got {value!r}"
            )
        if not values or any(not _normalize_text(item) for item in values):
            raise ValueError("Constraint values must contain non-empty text")
        return values

    @classmethod
    def _validate_constraints(cls, constraints: dict[str, Any]) -> None:
        allowed = {"video_id", *_FIELD_ALIASES}
        unknown = set(constraints) - allowed
        if unknown:
            raise ValueError(f"Unknown filter constraint(s): {sorted(unknown)}")
        for value in constraints.values():
            cls._constraint_values(value)

    @classmethod
    def _matches_constraints(
        cls,
        record: InvertedRecord,
        constraints: dict[str, Any],
    ) -> bool:
        for field, raw_value in constraints.items():
            expected_values = cls._constraint_values(raw_value)
            if field == "video_id":
                if record.video_id not in expected_values:
                    return False
                continue

            canonical = _FIELD_ALIASES[field]
            actual = getattr(record, canonical)
            if isinstance(actual, list):
                normalized_labels = {_normalize_text(label) for label in actual}
                if not all(
                    _normalize_text(value) in normalized_labels
                    for value in expected_values
                ):
                    return False
            else:
                normalized_text = _normalize_text(actual)
                if not all(_normalize_text(value) in normalized_text for value in expected_values):
                    return False
        return True
