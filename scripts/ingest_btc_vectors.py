"""Full-ingest available BTC indexing artifacts into a local Milvus Lite database.

The source tree is read-only. Rows whose computed logical key collides are quarantined
deterministically (keep first row by CSV/n order); no source artifact is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mfe.stores.production.config import MilvusConfig
from mfe.stores.production.contracts import StorageRecord
from mfe.stores.production.milvus import MilvusVectorStore


def iter_records(
    root: Path,
    *,
    quarantine: list[dict[str, Any]],
) -> Iterator[StorageRecord]:
    """Yield one validated unique vector row and quarantine duplicate logical keys."""
    mapping_root = root / "map-keyframes"
    feature_root = root / "clip-features"
    seen: set[tuple[str, int]] = set()
    for csv_path in sorted(mapping_root.glob("*.csv")):
        video_id = csv_path.stem
        feature_path = feature_root / f"{video_id}.npy"
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing feature file for {video_id}: {feature_path}")
        embeddings = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if embeddings.shape != (len(rows), 512):
            raise ValueError(
                f"{video_id}: feature shape {embeddings.shape} does not match {len(rows)} CSV rows"
            )
        for row_number, row in enumerate(rows, start=2):
            try:
                pts_time = float(row["pts_time"])
                fps = float(row["fps"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{row_number}: invalid pts_time/fps") from exc
            frame_id = math.floor(pts_time * fps)
            provided = (row.get("frame_idx") or "").strip()
            # Cột `frame_idx` của CSV là bản chuẩn DUY NHẤT, `floor(pts*fps)` chỉ
            # là phép đối chiếu. Ô trống mà vẫn đi tiếp nghĩa là ta lặng lẽ dùng
            # con số suy ra — đúng thứ CLAUDE.md §2 cấm, và nó hỏng im lặng: dòng
            # đó vẫn vào kho, vẫn tra cứu được, chỉ trỏ sai khung.
            if not provided:
                raise ValueError(
                    f"{csv_path}:{row_number}: canonical frame_idx is blank"
                )
            if int(provided) != frame_id:
                raise ValueError(
                    f"{csv_path}:{row_number}: frame_idx {provided} != "
                    f"floor(pts_time*fps) {frame_id}"
                )
            logical_key = (video_id, frame_id)
            if logical_key in seen:
                quarantine.append(
                    {
                        "video_id": video_id,
                        "csv_row": row_number,
                        "frame_id": frame_id,
                        "reason": "duplicate_logical_key_keep_first",
                    }
                )
                continue
            seen.add(logical_key)
            yield StorageRecord.from_values(
                video_id=video_id,
                frame_id=frame_id,
                embedding=embeddings[row_number - 2],
            )


def ingest(root: Path, database: Path, *, batch_size: int) -> dict[str, Any]:
    if database.exists():
        raise FileExistsError(f"refusing to overwrite existing Milvus Lite database: {database}")
    database.parent.mkdir(parents=True, exist_ok=True)
    quarantine: list[dict[str, Any]] = []
    store = MilvusVectorStore(
        MilvusConfig(uri=str(database), collection="aic_btc_vectors_batch1_v1")
    )
    store.ensure_schema()
    attempted = 0
    inserted = 0
    batch: list[StorageRecord] = []
    for record in iter_records(root, quarantine=quarantine):
        batch.append(record)
        if len(batch) < batch_size:
            continue
        report = store.ingest(batch)
        attempted += report.attempted
        inserted += report.succeeded
        batch = []
    if batch:
        report = store.ingest(batch)
        attempted += report.attempted
        inserted += report.succeeded
    before_restart_count = store.count()
    probe = store.search([1.0] + [0.0] * 511, top_k=1)
    store.client.close()

    reopened = MilvusVectorStore(
        MilvusConfig(uri=str(database), collection="aic_btc_vectors_batch1_v1")
    )
    reopened.ensure_schema()
    after_restart_count = reopened.count()
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": str(root.resolve()),
        "database": str(database.resolve()),
        "collection": reopened.config.collection,
        "attempted_unique_rows": attempted,
        "inserted_rows": inserted,
        "quarantined_duplicate_rows": len(quarantine),
        "count_before_restart": before_restart_count,
        "count_after_restart": after_restart_count,
        "restart_count_preserved": before_restart_count == after_restart_count,
        "probe_hit": probe[0].model_dump(mode="json") if probe else None,
        "quarantine": quarantine,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 10_000:
        raise ValueError("batch-size must be between 1 and 10000")
    result = ingest(args.root, args.database, batch_size=args.batch_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
