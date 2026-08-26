"""Đối chiếu caption/OCR của ta với bản Bảo đã chạy — CHẠY TRƯỚC KHI SCALE.

    python scripts/compare_qwen_captions.py \
        --ours .cache/qwen25/probe.json \
        --theirs notebooks/reference/dense_captions_qwen.json \
        --maps data/btc_b1/map-keyframes

Vì sao bước này tồn tại
------------------------
Lượt chạy đầy đủ là 177.321 ảnh trên một model 7B. Phát hiện lệch **sau** lượt đó
là mất cả lượt. Bảo đã chạy 855 keyframe của `L21_V001..V003` và để kết quả lại;
đó là bản đối chiếu **duy nhất** ta có, nên nó phải được dùng trước khi mở rộng.

Cái được so **không phải** "hai chuỗi có giống hệt nhau không". Qwen sinh có
`temperature=0.1`, chạy trên hai GPU khác nhau, vLLM khác bản — hai lượt chạy sẽ
không bao giờ trùng từng ký tự. Thứ so được, và thứ thật sự hỏng được:

* **Phủ** — có đúng bấy nhiêu khoá không, và chúng có khớp nhau không. Lệch ở đây
  là lỗi **địa chỉ**, tức lỗi đắt nhất trong dự án này: `n` (tên file ảnh) và
  `frame_idx` (khoá của kho) là hai con số khác nhau, và nhầm chúng thì mọi thứ
  vẫn chạy, vẫn có kết quả, chỉ trỏ vào khung hình khác.
* **Tỉ lệ có nhãn** — bao nhiêu câu trả lời có đủ `[CAPTION]` và `[TEXT]`. Đây là
  cái duy nhất phân biệt "prompt còn hoạt động" với "prompt đã hỏng thầm lặng":
  mất nhãn thì cả câu trả lời chảy vào caption và bảng kết quả trông vẫn bình thường.
* **Độ dài** — trung vị token. Lệch lớn nghĩa là `max_tokens` khác, hoặc model khác.
* **Chồng lấn từ vựng** — Jaccard trên tập token. Hai mô tả cùng một khung hình
  chia sẻ danh từ; chồng lấn thấp nghĩa là **đang mô tả hai ảnh khác nhau**, và
  đó lại là lỗi địa chỉ ở trên.

Hai định dạng khoá
-------------------
Bảo đặt khoá `{video}_{n}` với `n` là **tên file ảnh** (`L21_V003_208`). Kho của
ta đặt khoá `{video}#{frame_idx:06d}`. Script nhận cả hai và quy về khoá của kho
bằng `map-keyframes` — chỗ **duy nhất** nối `n` với `frame_idx` (CLAUDE.md §2).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from mfe.contracts.ids import build_keyframe_id
from mfe.core.cli import setup_console
from mfe.offline.caption_audit import audit, format_audit

#: `{video}_{n}` của Bảo. `n` không được zero-pad cố định, nên bắt bằng `\d+`.
_ORDINAL_KEY = re.compile(r"^(?P<video>L\d+_V\d+)_(?P<n>\d+)$")
#: `{video}#{frame_idx:06d}` của kho.
_FRAME_KEY = re.compile(r"^(?P<video>L\d+_V\d+)#(?P<frame>\d+)$")

_TOKEN = re.compile(r"\w+", re.UNICODE)


def ordinal_to_frame(maps: Path) -> dict[tuple[str, int], int]:
    """`{(video_id, n): frame_idx}` đọc từ map-keyframes.

    Đọc từ CSV, **không** tính lại từ `pts_time × fps`: sai 83–93% số dòng ở
    video 29,97fps (CLAUDE.md §2).
    """
    table: dict[tuple[str, int], int] = {}
    for path in sorted(maps.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                table[(path.stem, int(row["n"]))] = int(row["frame_idx"])
    if not table:
        raise SystemExit(f"{maps}: khong co file map-keyframes nao")
    return table


def normalise(payload: dict[str, str], table, label: str) -> tuple[dict[str, str], int]:
    """`{keyframe_id: text}` — quy mọi khoá về khoá của kho. Trả kèm số khoá bỏ."""
    out: dict[str, str] = {}
    dropped = 0
    for key, value in payload.items():
        frame_match = _FRAME_KEY.match(key)
        if frame_match:
            out[build_keyframe_id(frame_match["video"], int(frame_match["frame"]))] = value
            continue
        ordinal_match = _ORDINAL_KEY.match(key)
        if ordinal_match is None:
            dropped += 1
            continue
        frame_idx = table.get((ordinal_match["video"], int(ordinal_match["n"])))
        if frame_idx is None:
            # KHONG doan. Mot `n` khong co trong map-keyframes nghia la file anh
            # do khong thuoc bo keyframe cua BTC, va gan bua mot frame_idx la
            # dung thu ma §2 goi la "cach de nhat de nop sai ma moi thu trong dung".
            dropped += 1
            continue
        out[build_keyframe_id(ordinal_match["video"], frame_idx)] = value
    if dropped:
        print(f"  ! {label}: bo {dropped} khoa khong doc duoc / khong co trong map-keyframes")
    return out, dropped


def split_fields(text: str) -> tuple[str, str, bool]:
    from mfe.offline.stages import parse_qwen25_caption_ocr

    return parse_qwen25_caption_ocr(text)


def jaccard(left: str, right: str) -> float:
    a = set(_TOKEN.findall(left.lower()))
    b = set(_TOKEN.findall(right.lower()))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _describe(label: str, values: list[float], fmt: str = "{:.3f}") -> str:
    if not values:
        return f"{label:<26} —"
    quantiles = statistics.quantiles(values, n=10) if len(values) >= 10 else [min(values)]
    return (
        f"{label:<26} trung vi {fmt.format(statistics.median(values))}  "
        f"p10 {fmt.format(quantiles[0])}  min {fmt.format(min(values))}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compare_qwen_captions", description=__doc__)
    parser.add_argument("--ours", required=True, help="JSON cua ta (tho, hoac da tach truong)")
    parser.add_argument("--theirs", default="notebooks/reference/dense_captions_qwen.json")
    parser.add_argument("--maps", default="data/btc_b1/map-keyframes")
    parser.add_argument(
        "--min-overlap", type=float, default=0.25,
        help="Chong lan tu vung CAPTION trung vi toi thieu de coi la DAT.",
    )
    parser.add_argument(
        "--caption-gate", choices=("overlap", "none"), default="overlap",
        help=(
            "`none` khi hai ben chay HAI PROMPT khac nhau (vd bien the `aic` viet "
            "tieng Viet con ban cua Bao viet tieng Anh). Luc do chong lan tu vung "
            "gan bang 0 du ca hai deu mo ta dung anh — no do su khac prompt, khong "
            "do chat luong. Cac truc con lai (phu khoa, nhan, chong lan CHU) van gac."
        ),
    )
    parser.add_argument(
        "--min-marked", type=float, default=0.95,
        help="Ti le cau tra loi co DU hai nhan toi thieu (mac dinh 0,95).",
    )
    parser.add_argument("--examples", type=int, default=3, help="So vi du lech in ra.")
    return parser


def main() -> None:  # noqa: PLR0915 — mot bao cao tuyen tinh, tach ra chi kho doc
    setup_console()
    args = build_parser().parse_args()

    table = ordinal_to_frame(Path(args.maps))
    ours_raw = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    theirs_raw = json.loads(Path(args.theirs).read_text(encoding="utf-8"))
    ours, _ = normalise(ours_raw, table, "ta")
    theirs, _ = normalise(theirs_raw, table, "Bao")

    shared = sorted(set(ours) & set(theirs))
    only_ours = sorted(set(ours) - set(theirs))
    only_theirs = sorted(set(theirs) - set(ours))

    print(f"\nta   : {len(ours):>6} khoa   ({Path(args.ours).name})")
    print(f"Bao  : {len(theirs):>6} khoa   ({Path(args.theirs).name})")
    print(
        f"chung: {len(shared):>6} khoa   ·  rieng ta {len(only_ours)}"
        f"  ·  rieng Bao {len(only_theirs)}"
    )

    if not shared:
        raise SystemExit(
            "\nKHONG CO KHOA CHUNG NAO. Day gan nhu chac chan la loi DIA CHI, khong\n"
            "phai loi model: kiem lai `n` (ten file anh) da duoc doi sang `frame_idx`\n"
            "qua map-keyframes chua. Xem CLAUDE.md §2."
        )

    caption_overlap: list[float] = []
    ocr_overlap: list[float] = []
    ours_caption_tokens: list[float] = []
    theirs_caption_tokens: list[float] = []
    ours_marked = theirs_marked = 0
    ours_with_ocr = theirs_with_ocr = 0
    worst: list[tuple[float, str]] = []

    for key in shared:
        our_caption, our_ocr, our_marked_ok = split_fields(ours[key])
        their_caption, their_ocr, their_marked_ok = split_fields(theirs[key])
        ours_marked += int(our_marked_ok)
        theirs_marked += int(their_marked_ok)
        ours_with_ocr += int(bool(our_ocr))
        theirs_with_ocr += int(bool(their_ocr))
        ours_caption_tokens.append(len(_TOKEN.findall(our_caption)))
        theirs_caption_tokens.append(len(_TOKEN.findall(their_caption)))
        score = jaccard(our_caption, their_caption)
        caption_overlap.append(score)
        if our_ocr or their_ocr:
            ocr_overlap.append(jaccard(our_ocr, their_ocr))
        worst.append((score, key))

    n = len(shared)
    print()
    print(f"{'':<26}{'ta':>12}{'Bao':>12}")
    print("-" * 50)
    print(f"{'co du [CAPTION]+[TEXT]':<26}{ours_marked / n:>11.1%}{theirs_marked / n:>12.1%}")
    print(f"{'co chu doc duoc':<26}{ours_with_ocr / n:>11.1%}{theirs_with_ocr / n:>12.1%}")
    print(
        f"{'do dai caption (token)':<26}"
        f"{statistics.median(ours_caption_tokens):>11.0f}"
        f"{statistics.median(theirs_caption_tokens):>12.0f}"
    )
    print()
    print(_describe("chong lan caption", caption_overlap))
    print(_describe("chong lan chu (OCR)", ocr_overlap))

    # Phep do THAT SU quan trong khi hai prompt khac nhau: caption co neu ra
    # nhung nang luc ma 49 de that doi khong. Chong lan tu vung voi ban cua Bao
    # khong tra loi duoc cau do — no chi noi hai prompt khac nhau.
    print()
    print(format_audit(
        audit(split_fields(ours[k])[:2] for k in shared), "TA"
    ))
    print()
    print(format_audit(
        audit(split_fields(theirs[k])[:2] for k in shared), "BAO"
    ))

    worst.sort()
    if args.examples:
        print(f"\n{args.examples} cap lech nhat:")
        for score, key in worst[: args.examples]:
            our_caption, _, _ = split_fields(ours[key])
            their_caption, _, _ = split_fields(theirs[key])
            print(f"\n  [{key}]  chong lan {score:.3f}")
            print(f"    ta : {our_caption[:150]}")
            print(f"    Bao: {their_caption[:150]}")

    median_overlap = statistics.median(caption_overlap)
    problems: list[str] = []
    if only_theirs:
        problems.append(
            f"{len(only_theirs)} khoa Bao co ma ta khong — ta chay THIEU anh "
            f"(vd {', '.join(only_theirs[:3])})"
        )
    if args.caption_gate == "overlap" and median_overlap < args.min_overlap:
        problems.append(
            f"chong lan caption trung vi {median_overlap:.3f} < {args.min_overlap} — "
            "hai ben rat co the dang mo ta HAI ANH KHAC NHAU, kiem lai anh xa khoa"
        )
    if args.caption_gate == "none" and ocr_overlap:
        # Voi hai prompt khac ngon ngu, nua `[TEXT]` VAN so duoc: chep chu tren
        # man hinh la cung mot viec o ca hai prompt. Neu ca truc nay cung sap thi
        # do khong con la chuyen prompt — do la dia chi hoac dong co.
        median_ocr = statistics.median(ocr_overlap)
        if median_ocr < args.min_overlap:
            problems.append(
                f"chong lan CHU trung vi {median_ocr:.3f} < {args.min_overlap}. "
                "Nua `[TEXT]` la cung mot viec o ca hai prompt, nen no PHAI khop — "
                "sap o day nghia la loi dia chi hoac dong co, khong phai loi prompt."
            )
    if ours_marked / n < args.min_marked:
        problems.append(
            f"chi {ours_marked / n:.1%} cau tra loi cua ta co du hai nhan "
            f"(< {args.min_marked:.0%}) — prompt hoac max_tokens dang cat mat `[TEXT]`"
        )

    print()
    if problems:
        print("CHUA DAT — dung scale:")
        for problem in problems:
            print(f"  ✗ {problem}")
        raise SystemExit(1)
    print("DAT. Ba truc (phu, nhan, chong lan tu vung) deu khop — scale duoc.")
    print(
        "Luu y: chuoi KHONG trung tung ky tu la binh thuong (temperature 0,1, "
        "GPU khac, ban vLLM khac). Thu phai khop la dia chi va hinh dang."
    )


if __name__ == "__main__":
    main()
