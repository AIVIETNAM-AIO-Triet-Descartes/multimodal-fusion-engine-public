"""Contact sheet có đóng dấu GIÂY THẬT — công cụ viết ground truth cho bộ dev.

    python scripts/contact_sheet.py L23_V002 --from 0 --to 600 --step 5
    python scripts/contact_sheet.py L23_V002 --from 371 --to 377 --step 0.2

Vì sao không dùng `dake.decode_frames`
--------------------------------------
`decode_frames` **cố ý không seek**: nó đếm frame bằng vòng lặp, và một lượt seek
trên H.264 long-GOP rơi về keyframe trước đó nên số đếm sẽ sai. Đó là hợp đồng
đúng cho DAKE, và là hợp đồng sai cho việc này — soi một khoảnh khắc ở phút thứ
14 mà phải giải mã từ đầu là 21.000 frame cho 30 tấm ảnh.

Ở đây ta seek, và bù lại **không đếm frame**: mỗi ô lấy `pts_time` thật của frame
đã giải mã (`frame.pts * time_base`) và đóng dấu con số đó lên ảnh. Đổi ra frame
là việc của `scripts/time_to_frame.py`, bằng đúng công thức BTC dùng —
`floor(giây × fps)` với fps đọc từ `map-keyframes`.

Vì sao con số phải NẰM TRÊN ẢNH
--------------------------------
Ground truth của bộ dev được viết bằng cách "đọc mốc giây rồi đổi ra frame". Một
lưới ảnh không dấu thì mốc giây nằm trong đầu người xem, và một ô đếm lệch là một
cửa sổ ground truth lệch — sai lặng lẽ, và mọi phép đo sau đó thừa hưởng nó.

Ba mức quét, theo đúng cách bộ dev hiện tại được viết
------------------------------------------------------
1. `--step 5..10` quét cả video, tìm đoạn.
2. `--step 1` khoanh lấy ranh giới shot.
3. `--step 0.2` soi đúng khoảnh khắc, đặt cửa sổ hẹp.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DATA = Path("data/sample_2026")
DEFAULT_OUT = Path(".cache/contact")


#: Relational Map — nguồn dự phòng khi thiếu `map-keyframes/{video}.csv`.
#:
#: `data/sample_2026/map-keyframes/` chỉ có **16** file, đúng bằng bộ video mẫu
#: ban đầu; kho phẳng thì có cả **873** video. Không có đường dự phòng này thì
#: contact sheet — công cụ viết ground truth — chỉ dùng được cho 1,8% số video,
#: và bộ dev vĩnh viễn bị nhốt trong đúng 16 video ấy.
#:
#: **Đã kiểm trước khi tin:** fps trong bảng `videos` khớp CSV ở **16/16** video
#: đối chiếu được, và phân bố toàn kho (25,0 × 781 · 30,0 × 61 · 29,97 × 30 ·
#: 26,44 × 1) trùng khít con số CLAUDE.md §2 ghi. Tức DB lấy fps **từ** CSV chứ
#: không từ container — đúng thứ §2 đòi.
RELATIONAL_DB = Path("stores/btc_b1/relational/app.db")


def _from_db(video_id: str) -> tuple[float, list[tuple[int, float, int]]] | None:
    """`(fps, [(n, pts_time, frame_idx)])` từ Relational Map, hoặc `None`."""
    if not RELATIONAL_DB.is_file():
        return None
    import sqlite3  # noqa: PLC0415 - chi can khi thieu CSV

    with sqlite3.connect(RELATIONAL_DB) as con:
        row = con.execute(
            "select fps from videos where video_id = ?", (video_id,)
        ).fetchone()
        if row is None or not row[0]:
            return None
        rows = con.execute(
            "select keyframe_index, pts_time, frame_idx from keyframes "
            "where video_id = ? order by keyframe_index",
            (video_id,),
        ).fetchall()
    if not rows:
        return None
    return float(row[0]), [(int(n), float(t), int(f)) for n, t, f in rows]


def fps_of(video_id: str) -> float:
    """fps từ `map-keyframes`, KHÔNG từ container.

    CSV mang đúng con số BTC dùng để sinh `frame_idx` — container "đúng hơn" mà
    lệch CSV thì frame mình lệch frame họ (CLAUDE.md §2). Thiếu CSV thì rơi về
    Relational Map, vốn cũng chép từ CSV — xem `RELATIONAL_DB`.
    """
    path = DATA / "map-keyframes" / f"{video_id}.csv"
    if not path.is_file():
        found = _from_db(video_id)
        if found is None:
            raise SystemExit(
                f"khong thay {path}, va Relational Map cung khong co {video_id}"
            )
        return found[0]
    with path.open(encoding="utf-8", newline="") as handle:
        values = {float(row["fps"]) for row in csv.DictReader(handle)}
    if len(values) != 1:
        raise SystemExit(f"{path}: {len(values)} gia tri fps khac nhau — {sorted(values)}")
    return values.pop()


def keyframe_times(video_id: str) -> list[tuple[int, float, int]]:
    """`[(n, pts_time, frame_idx)]` — để đánh dấu ô nào có keyframe của kho."""
    path = DATA / "map-keyframes" / f"{video_id}.csv"
    if not path.is_file():
        found = _from_db(video_id)
        return found[1] if found else []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            (int(r["n"]), float(r["pts_time"]), int(r["frame_idx"]))
            for r in csv.DictReader(handle)
        ]


def grab(video_path: Path, wanted: list[float], max_side: int):
    """`[(pts_time thật, ảnh PIL)]` — mỗi mốc mong muốn lấy frame gần nhất sau nó.

    Trả `pts_time` **thật** chứ không trả mốc đã yêu cầu: seek rơi về keyframe
    trước đó và frame đầu tiên sau khi tua hiếm khi đúng mốc. Đóng dấu con số
    mình mong muốn lên một khung khác là cách chắc chắn nhất để viết một cửa sổ
    ground truth lệch mà không ai thấy.
    """
    import av
    from PIL import Image

    out: list[tuple[float, Image.Image]] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        for target in wanted:
            container.seek(int(target / time_base), stream=stream)
            picked = None
            for frame in container.decode(stream):
                when = float(frame.pts) * time_base
                if when >= target - 1e-6:
                    picked = (when, frame.to_image())
                    break
            if picked is None:
                continue
            when, image = picked
            image.thumbnail((max_side, max_side))
            out.append((when, image))
    return out


def sheet(tiles, columns: int, fps: float, marks: set[int]):
    """Ghép lưới, đóng dấu `giây (frame) [KF]` lên từng ô."""
    from PIL import Image, ImageDraw

    if not tiles:
        raise SystemExit("khong lay duoc frame nao — kiem lai --from/--to")
    width = max(image.width for _, image in tiles)
    height = max(image.height for _, image in tiles)
    band = 18
    rows = math.ceil(len(tiles) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + band)), "black")
    draw = ImageDraw.Draw(canvas)

    for index, (when, image) in enumerate(tiles):
        x = (index % columns) * width
        y = (index // columns) * (height + band)
        canvas.paste(image, (x, y))
        frame_idx = math.floor(when * fps)
        label = f"{when:8.2f}s  f{frame_idx}"
        if frame_idx in marks:
            label += "  KF"
        draw.text((x + 3, y + height + 3), label, fill="white")
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contact_sheet", description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("--from", dest="start", type=float, default=0.0)
    parser.add_argument("--to", dest="end", type=float, required=True)
    parser.add_argument("--step", type=float, default=5.0, help="Giay giua hai o")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--max-side", type=int, default=280)
    parser.add_argument("--out", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video = DATA / "videos" / f"{args.video_id}.mp4"
    if not video.is_file():
        raise SystemExit(f"khong thay {video}")

    fps = fps_of(args.video_id)
    marks = {frame_idx for _, _, frame_idx in keyframe_times(args.video_id)}

    wanted: list[float] = []
    current = args.start
    while current < args.end - 1e-9:
        wanted.append(round(current, 3))
        current += args.step
    if len(wanted) > 240:
        raise SystemExit(
            f"{len(wanted)} o — qua nhieu de nhin trong mot tam. Thu --step lon hon "
            f"hoac thu hep --from/--to."
        )

    tiles = grab(video, wanted, args.max_side)
    canvas = sheet(tiles, args.cols, fps, marks)

    out = Path(args.out) if args.out else (
        DEFAULT_OUT / f"{args.video_id}_{args.start:g}-{args.end:g}_s{args.step:g}.jpg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)
    print(
        f"{out}  ·  {len(tiles)} o  ·  fps {fps}  ·  "
        f"{tiles[0][0]:.2f}s -> {tiles[-1][0]:.2f}s  ·  "
        f"nhan 'KF' = frame nay co trong map-keyframes"
    )


if __name__ == "__main__":
    main()
