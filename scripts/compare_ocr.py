"""So một artifact văn bản của bundle với file phẳng của SOTA_AIC, theo từng keyframe.

    python scripts/compare_ocr.py --video L21_V001 \
        --sota ../SOTA_AIC/data/vectors/sviptr_ocr.json

Vì sao cần một script chứ không phải `diff`: **hai bên đánh khoá khác nhau.** Bảo
dùng số thứ tự keyframe (`L21_V001_302`), ta dùng frame của video gốc
(`L21_V001#037011`). Không có `map-keyframes` làm cầu thì hai file không so được
dòng nào với dòng nào, và so bằng thứ tự xuất hiện là cách chắc chắn nhất để đọc
sai (§1.2).

Điều script này trả lời: **keyframe nào Bảo bắt được chữ mà ta không.** Đó là câu
hỏi có hướng — thiếu chữ ở tầng OCR không ném lỗi, nó chỉ làm arm đó im lặng.

Cột `ocr_text` (easyocr) có mặt vì nó là nhân chứng độc lập **của chính ta** về
việc ảnh đó có chữ hay không. Khi Bảo có chữ, easyocr có chữ, mà `sviptr_ocr`
trống, thì lỗi nằm ở tầng Qwen — không phải ở ảnh.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.contracts.ids import build_keyframe_id  # noqa: E402
from mfe.core.cli import setup_console  # noqa: E402
from mfe.offline.adapters.sota_ids import parse_sota_id  # noqa: E402
from mfe.offline.bundle import BundleLayout  # noqa: E402
from mfe.offline.stages import read_keyframes  # noqa: E402
from mfe.offline.transport import build_transport  # noqa: E402

#: Chỉ có logo/đồng hồ của đài. Đọc đúng, nhưng nói rất ít về nội dung — nên nó
#: được đếm riêng: một nửa số khoá của Bảo là loại này, và trộn nó vào "bắt được
#: chữ" làm chênh lệch trông lớn hơn thực tế.
_WATERMARK = re.compile(r"^(HTV\d?\s*(HD)?[\s\d:]*)$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compare_ocr", description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--sota", required=True, help="File phang cua SOTA_AIC")
    parser.add_argument("--artifact", default="sviptr_ocr")
    parser.add_argument("--reference", default="ocr_text", help="Artifact lam nhan chung")
    parser.add_argument("--bundle", default="aic_btc_b1_v1")
    parser.add_argument("--transport", choices=("modal", "local"), default="modal")
    parser.add_argument("--volume", default="keyframes-btc")
    parser.add_argument("--local-root", default="bundle_local")
    parser.add_argument("--samples", type=int, default=12)
    return parser


def fetch_json(transport, remote: str, workdir: Path) -> dict[str, str]:
    """Tải một part về và đọc. Thiếu file là trạng thái hợp lệ, trả về rỗng."""
    local = workdir / Path(remote).name
    try:
        transport.get(remote, local)
    except Exception:  # noqa: BLE001 - chua chay tang do la binh thuong
        return {}
    return json.loads(local.read_text(encoding="utf-8"))


def kind(text: str) -> str:
    if not text.strip():
        return "rong"
    return "watermark" if _WATERMARK.match(text.strip()) else "chu that"


def main() -> None:
    setup_console()
    args = build_parser().parse_args()
    video = args.video

    layout = BundleLayout(args.bundle)
    transport = build_transport(
        args.transport, volume=args.volume, local_root=args.local_root
    )

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        # Cau noi giua hai cach danh khoa. `read_keyframes` doi chieu luon
        # frame_idx == floor(pts_time * fps), nen mot csv sai fps vo o day.
        map_csv = work / f"{video}.csv"
        transport.get(layout.map_keyframes(video), map_csv)
        keyframes = read_keyframes(map_csv, work / "images")
        n_to_key = {kf.n: build_keyframe_id(video, kf.frame_idx) for kf in keyframes}

        ours = fetch_json(transport, layout.part(args.artifact, video), work)
        reference = fetch_json(transport, layout.part(args.reference, video), work)

    sota_raw = json.loads(Path(args.sota).read_text(encoding="utf-8"))
    theirs: dict[str, str] = {}
    unmapped = 0
    for key, text in sota_raw.items():
        if not key.startswith(f"{video}_"):
            continue
        _, ordinal = parse_sota_id(key)
        target = n_to_key.get(ordinal)
        if target is None:
            unmapped += 1
            continue
        theirs[target] = text

    note = f"  (+{unmapped} khong map duoc)" if unmapped else ""
    print(f"video {video}: {len(keyframes)} keyframe trong map-keyframes")
    print(f"  ta   `{args.artifact}` : {len(ours)} khoa")
    print(f"  Bao  (da doi khoa)     : {len(theirs)} khoa{note}")
    print(f"  ta   `{args.reference}`: {len(reference)} khoa  <- nhan chung")
    print()

    buckets: dict[str, list[str]] = {
        "ca hai co chu": [],
        "chi Bao co": [],
        "chi ta co": [],
        "ca hai trong": [],
    }
    theirs_kind = {"watermark": 0, "chu that": 0}

    for kf in keyframes:
        key = build_keyframe_id(video, kf.frame_idx)
        mine = ours.get(key, "").strip()
        their = theirs.get(key, "").strip()
        if mine and their:
            buckets["ca hai co chu"].append(key)
        elif their:
            buckets["chi Bao co"].append(key)
            theirs_kind[kind(their)] = theirs_kind.get(kind(their), 0) + 1
        elif mine:
            buckets["chi ta co"].append(key)
        else:
            buckets["ca hai trong"].append(key)

    total = len(keyframes)
    for name, keys in buckets.items():
        print(f"  {name:<16} {len(keys):>4}  ({len(keys) / total:>4.0%})")
    print()

    missing = buckets["chi Bao co"]
    if missing:
        real = theirs_kind.get("chu that", 0)
        print(f"TA BO SOT {len(missing)} keyframe. Bao doc ra:")
        print(f"  - watermark/dong ho : {theirs_kind.get('watermark', 0)}")
        print(f"  - chu that          : {real}   <- phan dang lo")
        print()

        # Nhan chung: easyocr CUA TA co thay chu tren nhung anh do khong?
        seen_by_reference = sum(1 for k in missing if reference.get(k, "").strip())
        print(f"Trong {len(missing)} keyframe do, `{args.reference}` cua ta thay chu tren "
              f"{seen_by_reference} ({seen_by_reference / len(missing):.0%}).")
        if seen_by_reference > len(missing) * 0.5:
            print("  => Anh CO chu. Loi nam o tang "
                  f"`{args.artifact}`, khong o bo keyframe.")
        print()

        print(f"{min(args.samples, len(missing))} vi du (Bao doc duoc, ta trong):")
        ranked = sorted(missing, key=lambda k: -len(theirs[k]))
        for key in ranked[: args.samples]:
            their = " / ".join(theirs[key].splitlines())[:88]
            ref = " / ".join(reference.get(key, "").splitlines())[:44]
            print(f"  {key}")
            print(f"      Bao      : {their}")
            print(f"      easyocr  : {ref or '(khong co)'}")

    if buckets["chi ta co"]:
        print()
        print(f"{len(buckets['chi ta co'])} keyframe CHI TA co chu:")
        for key in buckets["chi ta co"][: args.samples]:
            print(f"  {key}: {' / '.join(ours[key].splitlines())[:88]}")


if __name__ == "__main__":
    main()
