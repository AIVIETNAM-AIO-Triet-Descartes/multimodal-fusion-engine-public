"""Dựng Relational Map cho cả 873 video BTC — chỉ từ `map-keyframes` + `media-info`.

    python scripts/build_btc_relational.py --config config/btc_b1.yaml
    ... thêm --download để kéo hai thư mục đó về trước

Vì sao không dùng `mfe-ingest`
--------------------------------
`mfe.offline.ingest` dựng **cả bốn kho** và đọc một `ProvidedDataset` đầy đủ trên
đĩa: ảnh keyframe, `objects/`, và `clip-features/{video}.npy` **khớp từng dòng**
với `map-keyframes`. Với batch 1 đó là ~28 GB phải tải về, và ta **không có**
`clip-features` trên Volume.

Nhưng tầng truy vấn không cần cả bốn. Đã kiểm từng thứ:

* `mfe.eval.official.candidate_frame_id` -> `RelationalMap.get_keyframe` -> chỉ
  cần bảng `keyframes`.
* `SotaIdMap` (dùng khi nạp ES) chỉ gọi `read_map_keyframes` — không chạm ảnh.
* `KeyframeRow` / `VideoRow` chỉ cần cột trong `map-keyframes` và `media-info`.

Vector Index và Inverted Index **local** thì bỏ được: arm `clip` và `bm25` sẽ
ngồi ngoài, còn 10 arm kia đọc Elasticsearch + Milvus. Asset Store cũng bỏ được
vì `models.reranker` đang tắt.

Nên hai thư mục nhỏ (~2 MB) thay cho 28 GB.

Một chỗ dễ sai, và nó im lặng
-------------------------------
`map-keyframes` của BTC có những dòng **trùng `frame_idx`** — 614 cặp như vậy
trong batch 1. Hai dòng chỉ khác số thứ tự `n` sẽ sinh ra **cùng một
`keyframe_id`**, vì khoá là `frame_idx` chứ không phải `n` (`mfe.contracts.ids`).
Không gộp chúng thì `upsert_keyframes` ghi đè lẫn nhau và số hàng cuối cùng ít
hơn số dòng CSV — mà không ai đếm thì không ai biết.

Gộp bằng cách **giữ `n` nhỏ nhất**, đúng như `SotaIdMap._frame_to_ordinal`. Hai
chỗ phải cùng luật: nếu đây giữ `n` lớn nhất còn kia giữ nhỏ nhất thì một
`keyframe_id` do ES trả về sẽ trỏ tới hàng không tồn tại trong Relational Map.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.contracts.ids import build_keyframe_id, pts_time_to_timestamp_ms  # noqa: E402
from mfe.contracts.records import KeyframeRow, VideoRow  # noqa: E402
from mfe.core.cli import setup_console  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.mapping.relational import RelationalMap  # noqa: E402
from mfe.offline.bundle import BundleLayout, load_shard_plan  # noqa: E402
from mfe.offline.transport import build_transport  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_btc_relational", description=__doc__)
    parser.add_argument("--config", default="config/btc_b1.yaml")
    parser.add_argument("--shards", default="config/ingest_shards_batch1_btc.yaml")
    parser.add_argument("--bundle", default="aic_btc_b1_v1")
    parser.add_argument("--volume", default="keyframes-btc")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Keo map-keyframes/ va media-info/ tu Volume ve truoc khi dung.",
    )
    parser.add_argument("--workers", type=int, default=6)
    return parser


def download(transport, layout, videos, config, workers: int) -> None:
    """Kéo hai thư mục nhỏ về đúng chỗ `config.paths` trỏ tới."""
    jobs: list[tuple[str, Path]] = []
    map_dir = Path(config.paths.map_keyframes)
    info_dir = Path(config.paths.media_info)
    map_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    for video_id in videos:
        jobs.append((layout.map_keyframes(video_id), map_dir / f"{video_id}.csv"))
        jobs.append((layout.media_info(video_id), info_dir / f"{video_id}.json"))

    def grab(job, tries: int = 3):
        remote, local = job
        # Xoá trước: một tệp cũ hoặc 0 byte sẽ đi thẳng vào bảng mà không kêu.
        local.unlink(missing_ok=True)
        last = ""
        for attempt in range(tries):
            try:
                transport.get(remote, local)
                if local.is_file() and local.stat().st_size > 0:
                    return remote, None
                last = "tep 0 byte"
            except Exception as err:  # noqa: BLE001
                last = f"{type(err).__name__}: {err}"
            local.unlink(missing_ok=True)
            if attempt + 1 < tries:
                time.sleep(2 ** attempt)
        return remote, last

    print(f"tai {len(jobs)} tep ({workers} luong) …")
    failed: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for remote, err in pool.map(grab, jobs):
            done += 1
            if err:
                failed[remote] = err
            if done % 300 == 0 or done == len(jobs):
                print(f"  … {done}/{len(jobs)}")
    if failed:
        sample = list(failed.items())[:5]
        raise SystemExit(
            f"\nkhong tai duoc {len(failed)}/{len(jobs)} tep:\n"
            + "\n".join(f"  {Path(r).name}: {e}" for r, e in sample)
            + "\n\nKHONG dung bang tren so con lai — mot bang thieu video trong y "
            "het mot bang du, cho toi luc cham diem."
        )


def rows_for(video_id: str, map_csv: Path, info_json: Path):
    """`(VideoRow, [KeyframeRow], so dong CSV)` — hoặc ném, không đoán."""
    rows = list(csv.DictReader(map_csv.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError("map-keyframes rong")
    info = json.loads(info_json.read_text(encoding="utf-8"))

    fps = float(rows[0]["fps"])
    last = rows[-1]
    # `total_frames` đọc từ chính CSV, không suy từ duration × fps: hai cách cho
    # hai con số khác nhau ở video 29,97 fps, và cột này dùng để CHẶN frame
    # ngoài biên lúc nộp bài.
    total_frames = int(last["frame_idx"]) + 1
    # `pts_time_to_timestamp_ms` (làm tròn), KHÔNG `int(... * 1000)` (cắt cụt).
    # Hai phép này lệch nhau tới 1 ms, và `mfe.offline.keyframe_augmentation.
    # bundle` đã dùng hàm chuẩn — để nguyên phép cắt cụt ở đây là B0 và B1 ghi
    # hai `timestamp_ms` khác nhau cho CÙNG một keyframe.
    duration_ms = pts_time_to_timestamp_ms(float(last["pts_time"]))

    video = VideoRow(
        video_id=video_id,
        source_path=str(info.get("watch_url") or f"btc://{video_id}"),
        fps=fps,
        duration_ms=duration_ms,
        total_frames=total_frames,
        # BTC không phát kích thước trong media-info; 1280x720 là khung hình
        # chuẩn của bộ này, và cột này không tham gia phép nộp nào.
        width=int(info.get("width") or 1280),
        height=int(info.get("height") or 720),
    )

    # Gộp dòng trùng `frame_idx`, GIỮ `n` NHỎ NHẤT — cùng luật với SotaIdMap.
    best: dict[int, dict] = {}
    for row in rows:
        frame_idx = int(row["frame_idx"])
        n = int(row["n"])
        if frame_idx not in best or n < int(best[frame_idx]["n"]):
            best[frame_idx] = row

    keyframes = [
        KeyframeRow(
            keyframe_id=build_keyframe_id(video_id, frame_idx),
            video_id=video_id,
            keyframe_index=int(row["n"]),
            frame_idx=frame_idx,
            pts_time=float(row["pts_time"]),
            timestamp_ms=pts_time_to_timestamp_ms(float(row["pts_time"])),
        )
        for frame_idx, row in sorted(best.items())
    ]
    return video, keyframes, len(rows)


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    config = load_config(args.config)
    plan = load_shard_plan(args.shards)
    layout = BundleLayout(args.bundle)
    videos = plan.videos

    if args.download:
        transport = build_transport(
            "modal", volume=args.volume, local_root="bundle_local", version=args.version
        )
        download(transport, layout, videos, config, args.workers)

    map_dir = Path(config.paths.map_keyframes)
    info_dir = Path(config.paths.media_info)
    db_path = Path(config.paths.relational_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nkho    : {db_path}")
    print(f"nguon  : {map_dir} · {info_dir}")
    print(f"video  : {len(videos)}\n")

    relational = RelationalMap(str(db_path))
    relational.init_schema()
    stats = Counter()
    problems: list[str] = []
    try:
        for index, video_id in enumerate(videos, 1):
            map_csv = map_dir / f"{video_id}.csv"
            info_json = info_dir / f"{video_id}.json"
            if not map_csv.is_file() or not info_json.is_file():
                problems.append(f"{video_id}: thieu map-keyframes hoac media-info")
                continue
            try:
                video, keyframes, csv_rows = rows_for(video_id, map_csv, info_json)
            except Exception as err:  # noqa: BLE001
                problems.append(f"{video_id}: {type(err).__name__}: {err}")
                continue
            relational.upsert_video(video)
            relational.upsert_keyframes(keyframes)
            stats["videos"] += 1
            stats["keyframes"] += len(keyframes)
            stats["collapsed"] += csv_rows - len(keyframes)
            if index % 100 == 0 or index == len(videos):
                print(f"  … {index}/{len(videos)}")

        print(f"\nvideo    : {stats['videos']}/{len(videos)}")
        print(f"keyframe : {stats['keyframes']}")
        print(f"gop trung frame_idx: {stats['collapsed']} dong "
              "(hai dong khac `n` nhung cung frame -> cung mot keyframe_id)")
        if problems:
            print(f"\nVUONG ({len(problems)}):")
            for line in problems[:10]:
                print(f"  {line}")
            raise SystemExit(
                "\nBang KHONG day du. Dung dung no de cham diem — mot bang thieu "
                "video trong y het mot bang du."
            )
        print(f"\nDoc lai: {relational.count_keyframes()} keyframe trong kho")
    finally:
        relational.close()


if __name__ == "__main__":
    main()
