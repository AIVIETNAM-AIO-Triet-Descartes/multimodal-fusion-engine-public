"""repair_sample_2026.py — Vá lại các file bị thiếu trong data/sample_2026/.

    python scripts/repair_sample_2026.py

Vì sao cần: `prepare_sample_2026.py` tải hàng nghìn file nhỏ qua HTTP Range, và
server BTC thỉnh thoảng đóng kết nối đột ngột (`ConnectionResetError 10054`).
Một ảnh hụt nghe như chuyện nhỏ, nhưng CSV vẫn có dòng cho nó — nên ingest sẽ
dừng ngay ở video đó với "map-keyframes lists n=104 nhưng thiếu ảnh".

Script này lấy `map-keyframes/*.csv` làm nguồn sự thật: mỗi dòng `n` phải có
đúng một ảnh `{n:03d}.jpg`. Thiếu cái nào thì tải lại đúng cái đó.

Chạy được nhiều lần, không hại gì: đã đủ file thì nó chỉ báo "không thiếu".
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from remotezip import RemoteZip

BASE_URL = "https://aic-data.ledo.io.vn"
OUT = Path("data/sample_2026")


def missing_keyframes() -> dict[str, list[int]]:
    """So số dòng CSV với số ảnh thực có, trả về {video_id: [n bị thiếu]}."""
    gaps: dict[str, list[int]] = {}
    map_dir = OUT / "map-keyframes"
    if not map_dir.is_dir():
        print(f"Không thấy {map_dir} — chạy prepare_sample_2026.py trước.")
        sys.exit(1)

    for csv_path in sorted(map_dir.glob("*.csv")):
        video_id = csv_path.stem
        with csv_path.open(encoding="utf-8", newline="") as handle:
            wanted_n = [int(row["n"]) for row in csv.DictReader(handle)]
        image_dir = OUT / "keyframes" / video_id
        absent = [n for n in wanted_n if not (image_dir / f"{n:03d}.jpg").is_file()]
        if absent:
            gaps[video_id] = absent
    return gaps


def refetch(gaps: dict[str, list[int]]) -> int:
    """Tải lại từng ảnh thiếu, thử tối đa 3 lần mỗi ảnh.

    Nhóm theo group để mỗi zip chỉ mở một lần — đọc central directory của một
    zip 6GB không rẻ.
    """
    by_group: dict[str, dict[str, list[int]]] = {}
    for video_id, ns in gaps.items():
        by_group.setdefault(video_id.split("_")[0], {})[video_id] = ns

    still_missing = 0
    for group, videos in sorted(by_group.items()):
        url = f"{BASE_URL}/Keyframes_{group}.zip"
        print(f"\n[{group}] mở {url}")
        with RemoteZip(url) as zf:
            for video_id, ns in sorted(videos.items()):
                for n in ns:
                    member = f"keyframes/{video_id}/{n:03d}.jpg"
                    for attempt in range(3):
                        try:
                            zf.extract(member, path=OUT / "keyframes")
                            print(f"    OK  {member}")
                            break
                        except Exception as err:  # noqa: BLE001
                            if attempt == 2:
                                print(f"    HỎNG {member}: {type(err).__name__}")
                                still_missing += 1
                            else:
                                time.sleep(1.5 * (attempt + 1))
    return still_missing


def main() -> int:
    gaps = missing_keyframes()
    total = sum(len(v) for v in gaps.values())
    if not gaps:
        print("Không thiếu ảnh nào — số ảnh khớp đúng số dòng CSV ở mọi video.")
        return 0

    print(f"Thiếu {total} ảnh trên {len(gaps)} video:")
    for video_id, ns in sorted(gaps.items()):
        preview = ns[:8]
        more = f" ... (+{len(ns) - 8})" if len(ns) > 8 else ""
        print(f"  {video_id}: {preview}{more}")

    # Lớp lồng thừa do prefix trong zip — làm phẳng lại sau khi tải.
    failed = refetch(gaps)
    nested = OUT / "keyframes" / "keyframes"
    if nested.is_dir():
        for child in nested.iterdir():
            target = OUT / "keyframes" / child.name
            if target.exists():
                for item in child.iterdir():
                    dest = target / item.name
                    if not dest.exists():
                        item.rename(dest)
                if not any(child.iterdir()):
                    child.rmdir()
            else:
                child.rename(target)
        if not any(nested.iterdir()):
            nested.rmdir()

    remaining = sum(len(v) for v in missing_keyframes().values())
    print(f"\nCòn thiếu sau khi vá: {remaining} ảnh (lỗi không phục hồi: {failed})")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
