"""Write records into the 4 stores — step [4] of docs/02 §2.

The seam between Quốc Đạt and everyone else. docs/01 §1 lists Quốc Đạt<->Công Quang as a sync
point whose failure mode is "Công Quang đọc sai, phải chạy lại offline" — re-running
offline is the expensive kind of mistake, so validate here rather than trusting.

Mapping of step -> store (docs/02 §4):

    keyframe image      -> Asset Store      (AssetRecord)
    keyframe coordinates-> Relational Map   (KeyframeRow)  [Nhật Bảo owns the store]
    CLIP embedding      -> Vector Index     (VectorRecord)
    object/OCR/metadata -> Inverted Index   (InvertedRecord)

`StoreWriter` holds the four store handles so ingest never touches a store
constructor directly, and `verify_stores_agree` can compare all four before
anything is persisted.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.records import (
    AssetRecord,
    InvertedRecord,
    KeyframeRow,
    VectorRecord,
    VideoRow,
)
from mfe.mapping.relational import RelationalMap
from mfe.stores.asset.manifest import AssetStore
from mfe.stores.inverted.index import InvertedIndex
from mfe.stores.vector.index import VectorIndex

logger = logging.getLogger(__name__)


class StoreWriter:
    """The four stores, opened together and written through one object.

    Owner: Quốc Đạt (writes) / Công Quang and Nhật Bảo (own the stores).

    Keyframe ids seen by each store are tracked as they go, so
    `verify_stores_agree` is a set comparison rather than four re-reads.
    """

    def __init__(
        self,
        relational: RelationalMap,
        vector: VectorIndex,
        inverted: InvertedIndex,
        asset: AssetStore,
    ) -> None:
        self.relational = relational
        self.vector = vector
        self.inverted = inverted
        self.asset = asset
        self._seen: dict[str, set[str]] = {
            "relational": set(),
            "vector": set(),
            "inverted": set(),
            "asset": set(),
        }

    def write_videos(self, rows: Iterable[VideoRow]) -> int:
        """Write the `videos` table of the Relational Map (docs/05 §1.1).

        Must land before any KeyframeRow: keyframes carry a foreign key to it.
        """
        count = 0
        for row in rows:
            self.relational.upsert_video(row)
            count += 1
        return count

    def write_keyframes(self, rows: Iterable[KeyframeRow]) -> int:
        """Write the `keyframes` table of the Relational Map (docs/05 §1.2).

        This is the source of truth for every submission — a wrong row here is a
        wrong answer even when retrieval was perfect
        (docs/context/buoi-1.md §8).
        """
        batch = list(rows)
        written = self.relational.upsert_keyframes(batch)
        self._seen["relational"].update(row.keyframe_id for row in batch)
        return written

    def write_vectors(self, records: Iterable[VectorRecord]) -> int:
        """Write to the Vector Index + its sidecar (docs/03 §1).

        `internal_id` is Faiss's own position and must match the sidecar row
        order exactly; an off-by-one translates every search hit to the wrong
        keyframe. `VectorIndex.add` enforces that, which is why the records are
        handed over as one ordered batch rather than one at a time.
        """
        batch = list(records)
        self.vector.add(batch)
        self._seen["vector"].update(record.keyframe_id for record in batch)
        return len(batch)

    def write_inverted(self, records: Iterable[InvertedRecord]) -> int:
        """Write to the Inverted Index (docs/03 §2).

        A row is emitted per keyframe even when every text field is empty — an
        absent row and a row with no text mean different things to a filter.
        """
        batch = list(records)
        self.inverted.add(batch)
        self._seen["inverted"].update(record.keyframe_id for record in batch)
        return len(batch)

    def write_assets(self, records: Iterable[AssetRecord]) -> int:
        """Write the Asset Store manifest (docs/03 §3).

        Bytes go through `mfe.core.storage`, not `open()` — see core/README.md.
        The manifest only points at them; `AssetStore` reads through the backend.
        """
        batch = list(records)
        self.asset.add(batch)
        self._seen["asset"].update(record.keyframe_id for record in batch)
        return len(batch)

    def verify_stores_agree(self) -> None:
        """Cross-check that all 4 stores cover the same keyframe_id set.

        Cheap here, expensive later. docs/00 §1 requires mapping to be 100%
        correct on the sample, and docs/02 §6 requires ids to match across all
        four stores. A keyframe present in the Vector Index but missing from the
        Relational Map is a candidate the validator will refuse at submit time —
        better to catch it at ingest than during a timed run.
        """
        reference = self._seen["relational"]
        problems: list[str] = []
        for name in ("vector", "inverted", "asset"):
            missing = reference - self._seen[name]
            extra = self._seen[name] - reference
            if missing:
                problems.append(
                    f"{name} store is missing {len(missing)} keyframe_id(s) present in "
                    f"the relational map, e.g. {sorted(missing)[:3]}"
                )
            if extra:
                problems.append(
                    f"{name} store has {len(extra)} keyframe_id(s) absent from the "
                    f"relational map, e.g. {sorted(extra)[:3]}"
                )
        if problems:
            raise ContractError(
                ErrorCode.SCHEMA_MISMATCH,
                "stores disagree on their keyframe_id set: " + "; ".join(problems),
            )
        logger.info(
            "All 4 stores agree on %d keyframe_id(s).", len(reference)
        )

    def save(self) -> None:
        """Persist the three file-backed stores.

        The Relational Map commits per write (sqlite), so it is not listed here.
        Saving last, after `verify_stores_agree`, means a run that fails
        verification leaves the previous stores untouched rather than half
        replaced.
        """
        self.vector.save()
        self.inverted.save()
        self.asset.save()
