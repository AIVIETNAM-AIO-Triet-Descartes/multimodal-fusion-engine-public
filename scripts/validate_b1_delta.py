"""Chạy VALIDATOR THẬT của dự án lên delta B1 đã giao, không phải một phép kiểm tay.

Vì sao cần script này khi tôi đã kiểm tay một lượt
--------------------------------------------------
Kiểm tay hôm nay xác nhận: 5.301 id đúng khuôn, `build_keyframe_id` tái tạo
được, 0 va chạm với B0, fps khớp `map-keyframes` ở 688/688 video. Nhưng đó là
một phép kiểm **tôi tự nghĩ ra**, và nó không phải hợp đồng mà dự án đã viết.

Hợp đồng thật nằm ở hai lớp `pydantic`:

* `CandidateFrame._coordinate_roundtrip` — `floor(pts_time × fps)` phải bằng
  `frame_idx`. Đây là luật CLAUDE.md §2 viết ra thành mã.
* `ProvenanceRow._identity_agrees` — `build_keyframe_id(video_id, frame_idx)`
  phải bằng chính `keyframe_id` được ghi.

Lớp thứ nhất chạy ở nb12 lúc **chốt ứng viên**. Lớp thứ hai chỉ chạy trong
`bundle.materialize_bundle`, mà đường B1 (nb25→29, 33) đi thẳng ra `delta-flat`
**không qua** bước đó — nên nó chưa từng chạy trên artifact đã giao.

Script này đóng khoảng trống đó: dựng lại `CandidateFrame` từ chính
`b1-worklist.jsonl` mà lô phẳng được sinh ra, rồi đối chiếu tập id đó với tập id
thật sự nằm trong `delta-flat`. Kiểm mô hình mà không kiểm file là kiểm một thứ
không ai dùng.

    python scripts/validate_b1_delta.py \\
      --worklist <b1-worklist.jsonl> --delta stores/flat_b1_delta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id  # noqa: E402
from mfe.offline.keyframe_augmentation.models import CandidateFrame  # noqa: E402

#: `priority` là `tuple[float, float, float, str, int]` trong `CandidateFrame`,
#: nhưng worklist đã giao KHÔNG mang nó — nó là thứ tự nội bộ lúc chọn, không
#: phải toạ độ. Đắp một giá trị trung tính để lớp toạ độ chạy được, và **chỉ**
#: lớp toạ độ mới là thứ script này khẳng định.
NEUTRAL_PRIORITY = (0.0, 0.0, 0.0, "", 0)


def ids_of(path: Path) -> set[str]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            key = "ids" if "ids" in data.files else data.files[0]
            return {str(x) for x in data[key]}
    return set(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--delta", type=Path, default=Path("stores/flat_b1_delta"))
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.worklist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"worklist: {len(rows):,} hang\n")

    problems: list[str] = []
    seen: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        try:
            frame = CandidateFrame(
                arm=(row.get("arms") or ["B1"])[0],
                video_id=row["video_id"],
                shot_id=(row.get("shot_ids") or [f"{row['video_id']}#s0000"])[0],
                frame_idx=int(row["frame_idx"]),
                pts_time=float(row["pts_time"]),
                fps=float(row["fps"]),
                selector=(row.get("selectors") or ["shot_center"])[0],
                coverage_class=(row.get("coverage_classes") or ["C0"])[0],
                priority=NEUTRAL_PRIORITY,
                detector_fingerprint=row.get("detector_fingerprint", ""),
                audit_status=row.get("audit_status", "PASS"),
            )
        except Exception as err:  # noqa: BLE001 - gom het roi bao mot the
            problems.append(f"hang {index} ({row.get('keyframe_id')}): {err}")
            continue

        # `keyframe_id` worklist ghi ra phai TRUNG voi cai model dung duoc.
        if frame.keyframe_id != row.get("keyframe_id"):
            problems.append(
                f"hang {index}: worklist ghi {row.get('keyframe_id')!r} nhung "
                f"toa do dung ra {frame.keyframe_id!r}"
            )
        if frame.keyframe_id in seen:
            problems.append(
                f"hang {index}: {frame.keyframe_id} trung voi {seen[frame.keyframe_id]}"
            )
        seen[frame.keyframe_id] = f"hang {index}"
        parse_keyframe_id(frame.keyframe_id)
        if build_keyframe_id(frame.video_id, frame.frame_idx) != frame.keyframe_id:
            problems.append(f"hang {index}: round-trip id that bai")

    bad = len([p for p in problems if p.startswith("hang ")])
    print(f"CandidateFrame dung duoc      {len(rows) - bad:>7,}/{len(rows):,}")
    print("  floor(pts x fps)==frame_idx  (luat CLAUDE.md §2, do boi pydantic)")
    print("  keyframe_id khop toa do")
    print(f"id doc nhat                   {len(seen):>7,}")

    base = args.delta / "data" / "vectors"
    print("\nDOI CHIEU voi delta-flat that:")
    for path in sorted(base.glob("*")):
        try:
            got = ids_of(path)
        except Exception as err:  # noqa: BLE001
            print(f"  {path.name:<24}khong doc duoc: {err}")
            continue
        stray = got - seen.keys()
        flag = "" if not stray else f"   <- {len(stray)} id NGOAI worklist"
        share = len(got & seen.keys()) / len(seen)
        print(f"  {path.name:<24}{len(got):>7,} id · phu {share:>6.1%}{flag}")
        if stray:
            problems.append(f"{path.name}: {len(stray)} id khong co trong worklist")

    if problems:
        print(f"\n{len(problems)} VAN DE:")
        for line in problems[:20]:
            print(f"  {line}")
        raise SystemExit(1)
    print("\nSACH. Moi hang qua duoc hop dong toa do cua du an, va delta-flat "
          "khong chua id nao ngoai worklist.")


if __name__ == "__main__":
    main()
