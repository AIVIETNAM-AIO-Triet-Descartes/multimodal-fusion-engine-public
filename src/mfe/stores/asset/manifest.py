"""Asset Store — the actual bytes. docs/03 §3.

The other three stores hold *pointers*; this one holds *data*. Two consumers:
the early-fusion reranker, which runs a cross-encoder on original images
(docs/04 §3.3), and the browser grid.

Division of labour worth keeping straight (docs/03 §3): the Relational Map holds
what a keyframe *means* (its coordinates); the Asset Store holds what it *looks
like*. Neither should grow the other's columns.
"""

from __future__ import annotations

import json
from pathlib import Path

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.records import AssetRecord, AssetType
from mfe.core.storage import StorageBackend, get_backend

_MANIFEST_FILENAME = "manifest.json"


class AssetStore:
    """A manifest pointing at bytes. Owner: Công Quang.

    Bytes are read and written through `mfe.core.storage`, never `open()` —
    that is what makes local -> Modal a config change (docs/01 §3.2).
    """

    def __init__(
        self,
        store_path: str,
        backend: StorageBackend | None = None,
    ) -> None:
        self.store_path = store_path
        self._backend = backend or get_backend("local", root=".")
        self._records: dict[tuple[str, AssetType], AssetRecord] = {}

    def add(self, records: list[AssetRecord]) -> None:
        """Register assets. Called by Quốc Đạt's writers during ingest.

        A keyframe may have several rows — one `image`, one `video_segment`
        (docs/03 §3(c)) — so this is keyed by (keyframe_id, asset_type), not
        keyframe_id alone. Re-adding the same key updates its manifest row.
        """
        for record in records:
            key = (record.keyframe_id, record.asset_type)
            self._records[key] = record

    def get_asset(self, keyframe_id: str, asset_type: AssetType = AssetType.IMAGE) -> bytes:
        """Return original bytes through the configured storage backend.

        A missing manifest row raises KEYFRAME_NOT_FOUND. A registered path whose
        bytes are unavailable raises STORE_UNAVAILABLE from the backend.
        """
        record = self.get_record(keyframe_id=keyframe_id, asset_type=asset_type)
        return self._backend.read_bytes(record.asset_path)

    def get_record(self, keyframe_id: str, asset_type: AssetType) -> AssetRecord:
        """Manifest row without loading the bytes — for sizing and for the UI."""
        key = (keyframe_id, asset_type)
        try:
            return self._records[key]
        except KeyError:
            raise ContractError(
                ErrorCode.KEYFRAME_NOT_FOUND,
                f"No {asset_type.value} asset registered for {keyframe_id!r}",
            ) from None

    def total_bytes(self) -> int:
        """Sum of `byte_size`. Feeds the Modal volume estimate (docs/02 §5)."""
        return sum(record.byte_size for record in self._records.values())

    def save(self) -> None:
        """Persist the manifest, mirroring VectorIndex/InvertedIndex.

        Without this the manifest dies with the ingest process and the browser,
        which runs separately, has no way to resolve a keyframe_id to bytes.
        Written to a temp file then replaced, so an interrupted run cannot leave
        a half-written manifest that loads without error.
        """
        store_dir = Path(self.store_path)
        store_dir.mkdir(parents=True, exist_ok=True)
        target = store_dir / _MANIFEST_FILENAME
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        payload = [record.model_dump(mode="json") for record in self._records.values()]
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)

    @classmethod
    def load(cls, store_path: str, backend: StorageBackend | None = None) -> AssetStore:
        source = Path(store_path) / _MANIFEST_FILENAME
        raw_records = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw_records, list):
            raise ValueError("Persisted asset manifest must contain a JSON list")

        store = cls(store_path=store_path, backend=backend)
        store.add([AssetRecord.model_validate(item) for item in raw_records])
        return store

    def __len__(self) -> int:
        return len(self._records)
