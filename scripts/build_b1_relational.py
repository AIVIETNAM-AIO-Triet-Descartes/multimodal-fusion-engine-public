"""Bản đồ quan hệ CÓ 5.301 keyframe B1 — chỉ để ĐO A/B, không phải bản nộp.

Vì sao phải có
---------------
Lượt A/B đầu tiên **vỡ giữa chừng**:

    ContractError: [keyframe_not_found] Keyframe L22_V005#030040 not found

`mfe.eval.official.candidate_frame_id` tra `frame_idx` qua `RelationalMap`, mà
`stores/btc_b1/relational/app.db` chỉ có hàng B0. Nghĩa là **mọi** đường đi qua
bản đồ đó — chấm điểm, xuất CSV nộp, tra ảnh — đều gãy ở keyframe B1. Đây là
cùng một gốc với chuyện lưới ảnh trống, chỉ lộ ra ở chỗ ồn ào hơn.

Ghi ra file MỚI, không sửa `app.db`
------------------------------------
`stores/btc_b1/relational/app.db` là bản đồ dùng cho **lượt nộp thật**. Thêm
hàng vào đó là đưa 5.301 keyframe chưa được chứng minh có ích vào đường nộp,
và `share_data.verify` sẽ báo "876 video / 182008 keyframe" thay vì con số
177.707 nó đang canh — tức phép kiểm đó mất luôn tác dụng.

`keyframe_index` của hàng B1 là số BỊA, và nó cố ý bịa cho lộ
-------------------------------------------------------------
Cột đó là **số thứ tự `n`, tức tên file ảnh** của BTC. Keyframe B1 không có
`n`: chúng không nằm trong `map-keyframes`, và ảnh của chúng nằm trong tar trên
volume nghiên cứu chứ không phải `keyframes-btc`.

Nên hàng B1 nhận `100000 + i`. BTC có `n` lớn nhất khoảng một nghìn, nên bất kỳ
giá trị nào ≥ 100000 **không thể** bị nhầm là `n` thật — thấy nó là biết ngay
đang cầm bản đo. Đặt `n` nối tiếp sau `n` lớn nhất của video thì ngược lại: nó
trông hợp lệ, và một ngày nào đó ai đó sẽ đi tra ảnh `251.jpg` không tồn tại.

Điều đó an toàn ở đây vì cấu hình A/B (`config/btc_b1_ab.yaml`) đặt
`scene_order_weight: 0` — `keyframe_index` chỉ được dùng làm khoảng cách cho
phép phạt của DANTE, và tầng đó tắt. Bật thứ tự cảnh trên bản đồ này là đo một
thứ vô nghĩa.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfe.contracts.ids import (  # noqa: E402
    build_keyframe_id,
    parse_keyframe_id,
    pts_time_to_timestamp_ms,
)

#: Mốc bịa cho `keyframe_index`. Xem docstring — nó phải KHÔNG THỂ bị nhầm là `n`.
SYNTHETIC_BASE = 100_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("stores/btc_b1/relational/app.db"))
    parser.add_argument("--out", type=Path, default=Path("stores/ab_b1/relational/app.db"))
    parser.add_argument(
        "--worklist",
        type=Path,
        required=True,
        help="b1-worklist.jsonl tu gate1-b1-feature-aggregate-v1 (co pts_time that)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.source.is_file():
        raise SystemExit(f"khong thay {args.source}")
    if args.out.exists():
        raise SystemExit(f"{args.out} da ton tai — xoa tay neu that su muon dung lai")

    rows = []
    for line in args.worklist.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    print(f"worklist: {len(rows):,} keyframe B1")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.out)

    con = sqlite3.connect(args.out)
    before = con.execute("select count(*) from keyframes").fetchone()[0]

    per_video: dict[str, int] = defaultdict(int)
    payload = []
    for row in rows:
        video_id = row["video_id"]
        frame_idx = int(row["frame_idx"])
        keyframe_id = build_keyframe_id(video_id, frame_idx)
        # Kiem lai chinh id trong worklist thay vi tin no: hai nua phai cung mot
        # luat dung id, va lech o day la lech dia chi giua kho phang va ban do.
        if keyframe_id != row["keyframe_id"]:
            raise SystemExit(
                f"{row['keyframe_id']}: build_keyframe_id ra {keyframe_id} — "
                "worklist va contracts.ids KHONG cung mot luat"
            )
        parse_keyframe_id(keyframe_id)
        pts_time = float(row["pts_time"])
        payload.append((
            keyframe_id, video_id, SYNTHETIC_BASE + per_video[video_id],
            frame_idx, pts_time, pts_time_to_timestamp_ms(pts_time), None,
        ))
        per_video[video_id] += 1

    con.executemany(
        "insert into keyframes (keyframe_id, video_id, keyframe_index, frame_idx, "
        "pts_time, timestamp_ms, shot_id) values (?,?,?,?,?,?,?)",
        payload,
    )
    con.commit()
    after = con.execute("select count(*) from keyframes").fetchone()[0]
    videos = con.execute("select count(*) from videos").fetchone()[0]
    con.close()

    print(f"{args.out}")
    print(f"  keyframe {before:,} -> {after:,}  (+{after - before:,})")
    print(f"  video    {videos:,}")
    if after - before != len(rows):
        raise SystemExit("so hang them vao KHONG khop worklist — dung lai")


if __name__ == "__main__":
    main()
