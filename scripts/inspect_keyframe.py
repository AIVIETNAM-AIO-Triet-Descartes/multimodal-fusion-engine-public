"""Đọc MỌI thứ hệ biết về một keyframe — và về các khung LÁNG GIỀNG của nó.

Ba chỗ tắc lúc thi mà lưới ảnh không gỡ được, và cả ba đều gỡ được bằng CHỮ:

1. **Câu nhiều mệnh đề.** Lưới cho một khung; câu hỏi hỏi về một *đoạn*. Không
   xem được bối cảnh quanh khung đó thì không xác nhận được. `--around N` in
   caption của N khung trước và sau **trong cùng video**, tức là kể lại đoạn
   phim bằng lời.
2. **Chữ trong ảnh bị vỡ khi phóng to.** Ảnh keyframe nén mạnh, nhưng OCR đã
   chạy trên bản gốc lúc offline — nên `qwen_ocr` / `viet_ocr` / `paddle_ocr`
   thường đọc được thứ mắt không đọc nổi trên giao diện. Ba bộ đọc độc lập, và
   chỗ chúng **bất đồng** chính là chỗ đáng ngờ.
3. **TRAKE.** Không xem được tập ứng viên thì không xếp được thứ tự. Đưa vài
   `keyframe_id` vào đây là có ngay mô tả từng mốc, đọc một lượt.

    python scripts/inspect_keyframe.py L21_V001#000483 --around 4
    python scripts/inspect_keyframe.py --video L21_V003 --from 26000 --to 27000
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id  # noqa: E402

BASE = Path("stores/flat/data/vectors")

#: Thứ tự in ra. `qwen_captions` trước vì nó là thứ kể được *cảnh*; ba bộ OCR
#: sau vì chúng chỉ nói *chữ trên màn hình*, và ta hỏi chúng khi mắt đọc không ra.
SOURCES = (
    ("caption", "qwen_captions.json"),
    ("ocr_qwen", "qwen_ocr.json"),
    ("ocr_viet", "viet_ocr.json"),
    ("ocr_paddle", "paddle_ocr.json"),
    ("ocr_easy", "ocr_text.json"),
    ("subtitle", "subtitles.json"),
    ("object", "objects_text.json"),
)


def load(root: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for label, name in SOURCES:
        path = root / name
        if path.is_file():
            out[label] = json.loads(path.read_text(encoding="utf-8"))
    return out


def index_by_video(caption: dict[str, str]) -> dict[str, list[int]]:
    frames: dict[str, list[int]] = {}
    for key in caption:
        video_id, frame_idx = parse_keyframe_id(key)
        frames.setdefault(video_id, []).append(frame_idx)
    for video_id in frames:
        frames[video_id].sort()
    return frames


def show(key: str, data: dict[str, dict[str, str]], *, brief: bool = False) -> None:
    video_id, frame_idx = parse_keyframe_id(key)
    print(f"\n─── {key}   (video {video_id}, frame {frame_idx:,})")
    for label, _ in SOURCES:
        text = (data.get(label) or {}).get(key, "").strip()
        if not text:
            continue
        if brief and label != "caption":
            continue
        body = text.replace("\n", " ⏎ ")
        print(f"  {label:<11}{body}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keyframes", nargs="*", help="vd L21_V001#000483")
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument(
        "--around", type=int, default=0,
        help="In them N khung TRUOC va SAU, trong cung video — bo canh dang chu.",
    )
    parser.add_argument("--video", help="Quet ca mot khoang frame thay vi mot khung.")
    parser.add_argument("--from", dest="start", type=int, default=0)
    parser.add_argument("--to", dest="end", type=int, default=10**9)
    parser.add_argument(
        "--brief", action="store_true", help="Chi in caption, bo qua OCR/subtitle.",
    )
    args = parser.parse_args()

    data = load(args.root)
    if "caption" not in data:
        raise SystemExit(f"khong doc duoc qwen_captions.json trong {args.root}")
    by_video = index_by_video(data["caption"])

    if args.video:
        frames = by_video.get(args.video)
        if not frames:
            raise SystemExit(f"khong co video {args.video} trong kho")
        lo = bisect.bisect_left(frames, args.start)
        hi = bisect.bisect_right(frames, args.end)
        print(f"{args.video}: {hi - lo} khung trong [{args.start:,}, {args.end:,}]")
        for frame_idx in frames[lo:hi]:
            show(build_keyframe_id(args.video, frame_idx), data, brief=args.brief)
        return

    for key in args.keyframes:
        video_id, frame_idx = parse_keyframe_id(key)
        frames = by_video.get(video_id) or []
        pos = bisect.bisect_left(frames, frame_idx)
        lo = max(0, pos - args.around)
        hi = min(len(frames), pos + args.around + 1)
        for neighbour in frames[lo:hi]:
            mark = " ★" if neighbour == frame_idx else ""
            key_n = build_keyframe_id(video_id, neighbour)
            show(key_n, data, brief=args.brief and neighbour != frame_idx)
            if mark:
                print("       ↑ khung được hỏi")


if __name__ == "__main__":
    main()
