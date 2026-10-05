"""Truy hồi một câu rồi in ứng viên KÈM CHỮ — để duyệt bằng đọc, không bằng nhìn.

Vì sao cần cái này bên cạnh `query_flat.py`
--------------------------------------------
`query_flat` in `keyframe_id` và điểm. Lúc thi thì đó chưa đủ: ba chỗ tắc trên
bàn nộp đều là chỗ **lưới ảnh không trả lời được**, và cả ba đều trả lời được
bằng chữ đã có sẵn trong kho.

* Câu nhiều mệnh đề — lưới cho một khung, đề hỏi một *đoạn*. `--around` in
  caption của các khung liền kề, tức kể lại đoạn phim.
* Chữ trong ảnh vỡ khi phóng to — OCR chạy trên ảnh gốc lúc offline nên nó
  thường đọc được thứ mắt không đọc nổi.
* TRAKE — mỗi mốc là một truy vấn riêng; `--events` chạy từng mốc và in ra để
  xếp thứ tự.

`--grep` lọc theo chữ SAU khi đã truy hồi: khi đề nêu một chi tiết chỉ có trong
văn bản (một dòng chữ trên màn hình, một cái tên), lọc rẻ hơn nhiều so với hy
vọng arm dense tự xếp nó lên đầu.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfe.contracts.ids import parse_keyframe_id  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle  # noqa: E402

BASE = Path("stores/flat/data/vectors")
SOURCES = (
    ("caption", "qwen_captions.json"),
    ("ocr_qwen", "qwen_ocr.json"),
    ("ocr_viet", "viet_ocr.json"),
    ("subtitle", "subtitles.json"),
)


def texts() -> dict[str, dict[str, str]]:
    return {
        label: json.loads((BASE / name).read_text(encoding="utf-8"))
        for label, name in SOURCES
        if (BASE / name).is_file()
    }


def line(key: str, data, *, rank: int | None = None, score: float | None = None) -> None:
    video_id, frame_idx = parse_keyframe_id(key)
    head = f"{rank:>3}. " if rank is not None else "     "
    tail = f"  [{score:.4f}]" if score is not None else ""
    print(f"{head}{key}   {video_id} frame {frame_idx:,}{tail}")
    for label, _ in SOURCES:
        body = (data.get(label) or {}).get(key, "").strip()
        if body:
            print(f"       {label:<9}{body.replace(chr(10), ' ⏎ ')[:400]}")


def run(cascade, query: str, *, top: int, en: str | None):
    found = cascade.search(query, top_k=top, query_text_en=en)
    return found[0] if isinstance(found, tuple) else found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--en", help="Ban dich tieng Anh, dua tay — tot nhat.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--events", nargs="*", default=None,
        help="TRAKE: moi mot chuoi la mot moc, chay rieng tung moc.",
    )
    parser.add_argument(
        "--grep", help="Chi giu ung vien co CHU khop bieu thuc nay (khong phan biet hoa).",
    )
    parser.add_argument("--config", default="config/btc_b1_ab.yaml")
    parser.add_argument("--root", default="stores/flat")
    args = parser.parse_args()

    data = texts()
    config = load_config(args.config)
    bundle = load_flat_bundle(args.root)
    cascade = build_flat_cascade(config, args.root, flat_bundle=bundle)

    jobs = [(args.query, args.en)]
    if args.events:
        jobs = [(text, None) for text in args.events]

    for index, (text, en) in enumerate(jobs, start=1):
        label = f"MỐC {index}" if args.events else "TRUY VẤN"
        print(f"\n{'=' * 78}\n{label}: {text}\n{'=' * 78}")
        hits = run(cascade, text, top=args.top * (4 if args.grep else 1), en=en)
        if args.grep:
            pattern = re.compile(args.grep, re.IGNORECASE)
            hits = [
                c for c in hits
                if any(
                    pattern.search((data.get(lbl) or {}).get(c.keyframe_id, ""))
                    for lbl, _ in SOURCES
                )
            ]
        for rank, candidate in enumerate(hits[: args.top], start=1):
            line(candidate.keyframe_id, data, rank=rank, score=candidate.score)


if __name__ == "__main__":
    main()
