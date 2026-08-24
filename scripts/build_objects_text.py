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


#: Biên chia khung hình thành ba dải theo tâm hộp (toạ độ x đã chuẩn hoá).
#: Ba dải chứ không phải chín ô: đề nói "bên trái", "ở giữa", "bên phải" — không
#: đề nào nói "góc trên bên trái". Chia mịn hơn mức đề hỏi chỉ làm loãng BM25.
ZONE_EDGES = (1 / 3, 2 / 3)
ZONE_NAMES = ("trái", "giữa", "phải")

#: Số đếm viết bằng chữ tới ngưỡng này, trên ngưỡng thì viết "nhiều". Đề dùng
#: chữ ("ba người", "hơn 5 người") nên chỉ mục cũng phải dùng chữ — BM25 khớp
#: token, và "3" với "ba" là hai token khác nhau.
COUNT_WORDS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
MANY = "nhiều"


def count_word(n: int) -> str:
    return COUNT_WORDS[n] if n < len(COUNT_WORDS) else MANY


def zone_of(box: list) -> str:
    """Dải ngang chứa TÂM hộp. `box` là `[ymin, xmin, ymax, xmax]` chuẩn hoá.

    Lấy tâm chứ không lấy mép: một vật rộng chiếm cả khung sẽ có `xmin` ở dải
    trái và `xmax` ở dải phải, và gán nó vào cả hai thì "bên trái" mất hết sức
    phân biệt.
    """
    try:
        centre = (float(box[1]) + float(box[3])) / 2.0
    except (TypeError, ValueError, IndexError):
        return ZONE_NAMES[1]
    if centre < ZONE_EDGES[0]:
        return ZONE_NAMES[0]
    if centre < ZONE_EDGES[1]:
        return ZONE_NAMES[1]
    return ZONE_NAMES[2]


def read_detections(payload: dict, score_min: float) -> list[tuple[str, str]]:
    """`[(nhãn, dải)]` cho mọi hộp trên ngưỡng — **không khử trùng lặp**.

    Bản trước khử trùng lặp ở đây, và đó là chỗ mất trắng hai thứ BTC đã cho:
    năm hộp `Person` thành đúng một chữ "người", nên arm này không phân biệt nổi
    khung có 1 người với khung có 12. `detection_boxes` thì không hề được đọc.

    Đo trên 49 đề chính thức: **51%** hỏi số lượng, **45%** có ràng buộc trái/
    phải/giữa. Cả hai đều trả lời được bằng dữ liệu đã nằm sẵn trong gói zip.
    """
    entities = payload.get("detection_class_entities") or []
    scores = payload.get("detection_scores") or []
    boxes = payload.get("detection_boxes") or []
    out: list[tuple[str, str]] = []
    for index, (entity, score) in enumerate(zip(entities, scores, strict=False)):
        if float(score) < score_min:
            continue
        name = str(entity).strip()
        if not name:
            continue
        box = boxes[index] if index < len(boxes) else None
        out.append((name, zone_of(box) if box else ZONE_NAMES[1]))
    return out


def render(detections: list[tuple[str, str]], *, translate: bool) -> str:
    """Chuỗi đánh chỉ mục cho một keyframe: nhãn · số đếm · dải.

    Ba phần, và mỗi phần có một lý do riêng:

    1. **Nhãn (đã dịch)** lặp lại `min(n, 3)` lần. BM25 cân theo tần suất, nên
       lặp là cách gắn số đếm vào *điểm số* chứ không chỉ vào chữ. Chặn ở 3 để
       một khung đông người không nuốt cả bảng xếp hạng.
    2. **Số đếm viết bằng chữ** — "ba người" — vì đề viết bằng chữ.
    3. **Dải ngang** — "trái: người, xe máy" — cho 45% câu hỏi vị trí.

    Nhãn tiếng Anh KHÔNG bị bỏ: `translate_labels` gắn thêm đồng nghĩa tiếng
    Việt, và giữ cả hai nghĩa là một truy vấn tiếng Anh (hiếm, nhưng có) vẫn khớp.
    """
    if not detections:
        return ""

    counts: dict[str, int] = {}
    zones: dict[str, set[str]] = {}
    for name, zone in detections:
        key = name.casefold()
        counts[key] = counts.get(key, 0) + 1
        zones.setdefault(zone, set()).add(name)
    # Giữ cách viết hoa đầu tiên gặp cho mỗi nhãn.
    display: dict[str, str] = {}
    for name, _ in detections:
        display.setdefault(name.casefold(), name)

    parts: list[str] = []
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        label = display[key]
        words = translate_labels([label]) if translate else [label]
        parts.extend(words * min(n, 3))
        # Cum so dem dung tu TIENG VIET: truy van viet "ba nguoi", nen "ba Person"
        # chi dong gop mot nua so token can thiet. `translate_labels` dat nhan goc
        # o dau roi moi toi dong nghia, nen phan tu thu hai la tu tieng Viet dau.
        head = words[1] if len(words) > 1 else words[0]
        parts.append(f"{count_word(n)} {head}")

    for zone in ZONE_NAMES:
        if zones.get(zone):
            names = sorted(zones[zone])
            vi = translate_labels(names) if translate else names
            parts.append(f"{zone}: " + " ".join(vi))

    return " ".join(parts)


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
    counts = {"json": 0, "co_nhan": 0, "hop": 0, "gop_trung": 0,
              "khong_map": 0, "video_la": 0}

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
            detections = read_detections(payload, score_min)
            if not detections:
                continue
            seen_labels.update(label for label, _ in detections)
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
        "%s: %d keyframe co nhan / %d json doc duoc · %d hop tren nguong "
        "(%.1f hop/keyframe) · gop %d cap trung frame_idx · khong map %d, "
        "video la %d · %.0f MB",
        out, len(mapping), counts["json"], counts["hop"],
        counts["hop"] / max(1, counts["co_nhan"]), counts["gop_trung"],
        counts["khong_map"], counts["video_la"], out.stat().st_size / 2**20,
    )


if __name__ == "__main__":
    main()
