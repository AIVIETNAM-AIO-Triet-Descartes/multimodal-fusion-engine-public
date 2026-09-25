#!/usr/bin/env python3
"""Prepare map/media-info and build the combined B1+B2 relational map.

The late B2 additions have complete map-keyframes and keyframe images but no
standalone media-info JSON.  Relational lookup needs only FPS, dimensions and
the map-derived frame bounds, so this script records a small, explicit fallback
metadata file for those three IDs instead of treating them as missing videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_btc_relational import rows_for  # noqa: E402

from mfe.contracts.ids import pts_time_to_timestamp_ms  # noqa: E402
from mfe.mapping.relational import RelationalMap  # noqa: E402

LATE_B2 = ("N025-V002", "N041-V001", "S01-V007")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch1-root",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframes-btc/bundle/aic_btc_b1_v1"
        ),
    )
    parser.add_argument(
        "--batch2-root",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2/bundle/aic_btc_b2_v1"
        ),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/data/multimodal-fusion-engine/runtime/btc_b12"),
    )
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copyfile(source, target)


def fallback_info(video_id: str, map_csv: Path, target: Path) -> None:
    rows = list(csv.DictReader(map_csv.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"{video_id}: map rỗng")
    fps = float(rows[0]["fps"])
    target.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "fps": fps,
                "width": 1920,
                "height": 1080,
                "source_path": f"btc://{video_id}",
                "metadata_source": "derived_from_map_keyframes_for_late_addition",
                "last_pts_time": float(rows[-1]["pts_time"]),
                "last_timestamp_ms": pts_time_to_timestamp_ms(float(rows[-1]["pts_time"])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    b1 = args.batch1_root
    b2 = args.batch2_root
    root = args.runtime_root
    maps = root / "relational" / "map-keyframes"
    infos = root / "relational" / "media-info"
    db_path = root / "relational" / "app.db"
    maps.mkdir(parents=True, exist_ok=True)
    infos.mkdir(parents=True, exist_ok=True)

    all_ids: list[str] = []
    for source_root in (b1, b2):
        source_maps = source_root / "map-keyframes"
        source_infos = source_root / "media-info"
        for map_csv in sorted(source_maps.glob("*.csv")):
            video_id = map_csv.stem
            link_or_copy(map_csv, maps / map_csv.name)
            info = source_infos / f"{video_id}.json"
            if info.is_file():
                link_or_copy(info, infos / info.name)
            elif source_root == b2 and video_id in LATE_B2:
                fallback_info(video_id, map_csv, infos / info.name)
            else:
                raise SystemExit(f"Thiếu media-info ngoài danh sách late additions: {video_id}")
            all_ids.append(video_id)

    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Trùng video id giữa B1 và B2")
    if db_path.exists():
        db_path.unlink()

    relational = RelationalMap(str(db_path))
    relational.init_schema()
    keyframes = 0
    collapsed = 0
    try:
        for index, video_id in enumerate(sorted(all_ids), 1):
            video, rows, csv_rows = rows_for(
                video_id, maps / f"{video_id}.csv", infos / f"{video_id}.json"
            )
            relational.upsert_video(video)
            relational.upsert_keyframes(rows)
            keyframes += len(rows)
            collapsed += csv_rows - len(rows)
            if index == 1 or index % 100 == 0 or index == len(all_ids):
                print(f"{index}/{len(all_ids)} videos, {keyframes:,} keyframes", flush=True)
        actual = relational.count_keyframes()
    finally:
        relational.close()

    report = {
        "videos": len(all_ids),
        "keyframes": actual,
        "csv_duplicate_rows_collapsed": collapsed,
        "late_additions_with_derived_media_info": list(LATE_B2),
        "db": str(db_path),
    }
    (root / "relational-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
