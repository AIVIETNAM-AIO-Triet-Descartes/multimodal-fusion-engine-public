"""Asset Store — the actual bytes. docs/03 §3.

The other three stores hold *pointers*; this one holds *data*. Two consumers:
the early-fusion reranker, which runs a cross-encoder on original images
(docs/04 §3.3), and the browser grid.

Division of labour worth keeping straight (docs/03 §3): the Relational Map holds
what a keyframe *means* (its coordinates); the Asset Store holds what it *looks
like*. Neither should grow the other's columns.
"""

from __future__ import annotations

from mfe.contracts.records import AssetRecord, AssetType


class AssetStore:
    """A manifest pointing at bytes. Owner: Công Quang. TODO(Công Quang): implement.

    Bytes are read and written through `mfe.core.storage`, never `open()` —
    that is what makes local -> Modal a config change (docs/01 §3.2).
    """

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path

    def add(self, records: list[AssetRecord]) -> None:
        """Register assets. Called by Quốc Đạt's writers during ingest.

        TODO(Công Quang): implement. A keyframe may have several rows — one `image`,
        one `video_segment` (docs/03 §3(c)) — so this is keyed by
        (keyframe_id, asset_type), not keyframe_id alone.
        """
        raise NotImplementedError("TODO(Công Quang): register asset rows")

    def get_asset(self, keyframe_id: str, asset_type: AssetType = AssetType.IMAGE) -> bytes:
        """The `get_asset` API of docs/03 §3.

        TODO(Công Quang): implement. Raise ContractError(KEYFRAME_NOT_FOUND) on a miss
        — never return a placeholder image. A placeholder would silently feed
        the reranker garbage and the operator a lie.

        DECIDE WITH Phạm Trọng BEFORE IMPLEMENTING (docs/03 §3): does this return bytes
        or a path, and at what image size? Both the cross-encoder and the UI
        depend on the answer, and it is cheap to settle now and expensive later.
        The signature here says bytes — change it if Phạm Trọng needs otherwise.
        """
        raise NotImplementedError("TODO(Công Quang): keyframe_id -> asset bytes")

    def get_record(self, keyframe_id: str, asset_type: AssetType) -> AssetRecord:
        """Manifest row without loading the bytes — for sizing and for the UI."""
        raise NotImplementedError("TODO(Công Quang): keyframe_id -> AssetRecord")

    def total_bytes(self) -> int:
        """Sum of `byte_size`. Feeds the Modal volume estimate (docs/02 §5)."""
        raise NotImplementedError("TODO(Công Quang)")
