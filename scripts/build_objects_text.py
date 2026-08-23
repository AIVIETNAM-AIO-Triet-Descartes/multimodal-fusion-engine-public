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
from mfe.offline.labels_vi import missing_labels, translate_labels

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


def labels_from_payload(payload: dict, score_min: float) -> list[str]:
    """Nhãn trên ngưỡng, đã khử trùng lặp, giữ thứ tự điểm giảm dần."""
    entities = payload.get("detection_class_entities") or []
    scores = payload.get("detection_scores") or []
    labels: list[str] = []
    seen: set[str] = set()
    for entity, score in zip(entities, scores, strict=False):
        if float(score) < score_min:
            continue
        name = str(entity).strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            labels.append(name)
    return labels


def build(
    zip_path: Path,
    map_dir: Path,
    score_min: float,
    *,
    translate: bool = True,
) -> tuple[dict[str, str], dict[str, int]]:
    ordinals = load_ordinal_to_frame(map_dir)
    logger.info("map-keyframes: %d video", len(ordinals))

    mapping: dict[str, str] = {}
    seen_labels: set[str] = set()
    counts = {"json": 0, "co_nhan": 0, "khong_map": 0, "video_la": 0}

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
            labels = labels_from_payload(payload, score_min)
            if not labels:
                continue
            seen_labels.update(labels)
            counts["co_nhan"] += 1
            text = translate_labels(labels) if translate else labels
            mapping[build_keyframe_id(video_id, frame_idx)] = " ".join(text)

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
        "--no-translate", action="store_true",
        help="Giu nhan tieng Anh. Chi de doi chieu — truy van tieng Viet se truot.",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    score_min = float(raw.get("features", {}).get("object_score_min", 0.3))
    logger.info("score_min = %.2f (doc tu %s)", score_min, args.config)

    mapping, counts = build(
        Path(args.zip), Path(args.maps), score_min, translate=not args.no_translate
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "%s: %d keyframe co nhan / %d json doc duoc "
        "(khong map %d, video la %d) · %.0f MB",
        out, len(mapping), counts["json"], counts["khong_map"], counts["video_la"],
        out.stat().st_size / 2**20,
    )


if __name__ == "__main__":
    main()
