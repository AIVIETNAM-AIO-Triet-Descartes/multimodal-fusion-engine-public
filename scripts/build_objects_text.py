"""Gói `objects/` của BTC -> `objects_text.json` cho arm `object`.

    python scripts/build_objects_text.py --zip downloads/objects-aic25-b1.zip

Vì sao đọc từ ZIP chứ không từ Modal Volume
--------------------------------------------
Volume có đủ 177.321 file `objects/{video}/{n:03d}.json` — nhưng đó là **177.321
lần đọc mạng**. Đo được ở phiên này: 873 file part mất 200–300 giây, nên 177 nghìn
file cùng cách sẽ mất hàng giờ. Gói zip là **một** lần tải 640 MB. Cùng dữ liệu,
khác hai bậc độ lớn về thời gian.

Ba quyết định trong phép chuyển
--------------------------------
**0. Giữ SỐ ĐẾM và VỊ TRÍ.** BTC cho sẵn `detection_boxes` (toạ độ chuẩn hoá) và
một dòng cho *mỗi* hộp. Bản đầu của script này khử trùng lặp nhãn, nên năm hộp
`Person` thành đúng một chữ "người" và toạ độ không được đọc tới. Đo trên 49 đề
chính thức: **51%** hỏi số lượng, **45%** có ràng buộc trái/phải/giữa — cả hai
trả lời được bằng dữ liệu đã nằm sẵn trong gói zip.

**1. Dịch nhãn sang tiếng Việt.** BTC đặt tên thực thể theo OpenImages bằng tiếng
Anh (`Person`, `Motorcycle`), còn truy vấn là tiếng Việt. BM25 khớp token, nên
không dịch thì giữa chỉ mục và truy vấn **không có một từ nào chung** — arm này
sẽ không bao giờ khớp ở mức keyframe. `labels_vi.translate_labels` gắn *nhiều*
đồng nghĩa cho mỗi nhãn ("xe máy", "mô tô", "xe gắn máy") vì người vận hành gõ
cách nào cũng phải trúng.

**2. Ngưỡng điểm lấy từ config**, không gõ tay ở đây. Nó phải bằng
`features.object_score_min` mà tầng offline dùng, nếu không hai bên đọc hai tập
nhãn khác nhau. File BTC mang ~100 hộp mỗi keyframe, phần đuôi dưới 0,05 là nhiễu;
đánh chỉ mục hết sẽ chôn nhãn thật, vì BM25 cân một từ theo độ hiếm của nó.

**3. `keyframe_id` xây từ `frame_idx`, không từ `n`.** File object đặt tên theo
`n` (số thứ tự), còn khoá của kho là `frame_idx`. `map-keyframes` là chỗ duy nhất
nối hai số đó, và nhầm chúng là cách dễ nhất để cả arm này lệch khỏi mọi arm khác.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import zipfile
from pathlib import Path

import yaml

from mfe.contracts.ids import build_keyframe_id
from mfe.core.cli import setup_console
from mfe.offline.labels_vi import missing_labels
from mfe.offline.object_text import read_detections, render

logger = logging.getLogger(__name__)

DEFAULT_OUT = "stores/flat/data/vectors/objects_text.json"
DEFAULT_MAPS = "data/btc_b1/map-keyframes"
DEFAULT_CONFIG = "config/btc_b1.yaml"


def load_ordinal_to_frame(map_dir: Path) -> dict[str, dict[int, int]]:
    """`{video_id: {n: frame_idx}}` đọc từ map-keyframes."""
    table: dict[str, dict[int, int]] = {}
    for path in sorted(map_dir.glob("*.csv")):
        with path.open(encoding="utf-8") as handle:
            table[path.stem] = {
                int(row["n"]): int(row["frame_idx"]) for row in csv.DictReader(handle)
            }
    return table


def open_keyframe(keyframe_dir: Path | None, video_id: str, ordinal: int):
    """Ảnh keyframe `{keyframe_dir}/{video}/{n:03d}.jpg`, hoặc `None`.

    Trả `None` **im lặng** khi thiếu ảnh, và đó là hành vi đúng: gói `objects`
    (640 MB) không kèm ảnh, còn ảnh keyframe là 28,7 GB. Một bản chỉ mục không
    màu vẫn đúng — nó chỉ thiếu nhóm token thứ tư. Ném ở đây sẽ biến một tính
    năng tuỳ chọn thành một điều kiện bắt buộc.

    Số ảnh có màu được ĐẾM và in ra cuối lượt, nên "thiếu im lặng" không có
    nghĩa là "thiếu mà không ai biết".
    """
    if keyframe_dir is None:
        return None
    path = keyframe_dir / video_id / f"{ordinal:03d}.jpg"
    if not path.is_file():
        return None
    try:
        from PIL import Image  # noqa: PLC0415

        return Image.open(path)
    except Exception:  # noqa: BLE001 - anh hong thi coi nhu khong co mau
        return None


def build(
    zip_path: Path,
    map_dir: Path,
    score_min: float,
    *,
    translate: bool = True,
    keyframe_dir: Path | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    ordinals = load_ordinal_to_frame(map_dir)
    logger.info("map-keyframes: %d video", len(ordinals))

    mapping: dict[str, str] = {}
    seen_labels: set[str] = set()
    counts = {"json": 0, "co_nhan": 0, "hop": 0, "gop_trung": 0,
              "khong_map": 0, "video_la": 0, "co_mau": 0}

    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".json")]
        logger.info("%s: %d file json", zip_path.name, len(names))
        for position, name in enumerate(names, start=1):
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            video_id, stem = parts[-2], Path(parts[-1]).stem
            if video_id not in ordinals:
                counts["video_la"] += 1
                continue
            try:
                ordinal = int(stem)
            except ValueError:
                continue
            frame_idx = ordinals[video_id].get(ordinal)
            if frame_idx is None:
                # Object co ma map-keyframes khong co: khong doan ra frame_idx
                # duoc, va doan sai thi arm nay tro toi mot khung khac han.
                counts["khong_map"] += 1
                continue

            counts["json"] += 1
            with archive.open(name) as handle:
                payload = json.load(handle)
            image = open_keyframe(keyframe_dir, video_id, ordinal)
            detections = read_detections(payload, score_min, image=image)
            if image is not None:
                counts["co_mau"] += 1
                image.close()
            if not detections:
                continue
            seen_labels.update(hit.label for hit in detections)
            counts["co_nhan"] += 1
            counts["hop"] += len(detections)
            key = build_keyframe_id(video_id, frame_idx)
            if key in mapping:
                # 614 cap keyframe cua batch 1 dung chung mot `frame_idx` (artefact
                # lam tron cua BTC o 25/30fps). `merge.py` gop chung bang cach GIU
                # BAN DAU TIEN; chi mot dict thi ban sau ghi de ban truoc, tuc giu
                # ban CUOI — nguoc lai. Hai artifact gop khac chieu la hai arm mo ta
                # hai khung khac nhau duoi cung mot khoa.
                counts["gop_trung"] += 1
                continue
            mapping[key] = render(detections, translate=translate)

            if position % 20000 == 0:
                logger.info("  %d/%d ...", position, len(names))

    unmapped = missing_labels(sorted(seen_labels))
    if unmapped:
        logger.warning(
            "%d nhan khong co trong labels_vi — truy van tieng Viet se khong khop "
            "chung: %s%s",
            len(unmapped),
            ", ".join(unmapped[:12]),
            " ..." if len(unmapped) > 12 else "",
        )
    else:
        logger.info("labels_vi phu het %d nhan xuat hien", len(seen_labels))
    return mapping, counts


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--zip", required=True, help="objects-aic25-b1.zip")
    parser.add_argument("--maps", default=DEFAULT_MAPS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--keyframes",
        help=(
            "Thu muc anh `{video}/{n:03d}.jpg` de tinh MAU trong tung hop. "
            "⚠️ DO NGAY 26-08-2026 TREN 6.441 HOP THAT: KHONG DUNG DUOC — "
            "mau ra `nau` 50%% + `hong` 27%% vi bang 11 mau dat MAU DA nguoi "
            "giua hai mau do, con vang/do/xanh chi 0-2%%. Xem "
            "`mfe.offline.object_text.dominant_colour`. Co nay giu lai de do "
            "lai khi co phep uoc luong khac, khong phai de dung."
        ),
    )
    parser.add_argument(
        "--no-translate", action="store_true",
        help="Giu nhan tieng Anh. Chi de doi chieu — truy van tieng Viet se truot.",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    score_min = float(raw.get("features", {}).get("object_score_min", 0.3))
    logger.info("score_min = %.2f (doc tu %s)", score_min, args.config)

    mapping, counts = build(
        Path(args.zip), Path(args.maps), score_min,
        translate=not args.no_translate,
        keyframe_dir=Path(args.keyframes) if args.keyframes else None,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "%s: %d keyframe co nhan / %d json doc duoc · %d hop tren nguong "
        "(%.1f hop/keyframe) · gop %d cap trung frame_idx · khong map %d, "
        "video la %d · %d keyframe co MAU · %.0f MB",
        out, len(mapping), counts["json"], counts["hop"],
        counts["hop"] / max(1, counts["co_nhan"]), counts["gop_trung"],
        counts["khong_map"], counts["video_la"], counts["co_mau"],
        out.stat().st_size / 2**20,
    )


if __name__ == "__main__":
    main()
