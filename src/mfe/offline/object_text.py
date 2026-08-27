"""Dựng chuỗi đánh chỉ mục cho arm `object` — thuộc tính GHÉP VÀO nhãn.

    docs/triet/huong-di-grid-vitri-mausac.md  ·  handoff §A.2, §C.2

Vì sao viết lại: arm cũ đo được **~0**
--------------------------------------
Bản trước rải thuộc tính ra thành token **tự do**: `"ba người"` sinh ra `ba` và
`người` là hai token rời, `"trái: người xe máy"` sinh ra `trái` rời. Đo idf trên
chính chỉ mục 168.009 tài liệu (handoff §A.2):

    `một`  có mặt ở 91,0% tài liệu  (idf 0,09)
    `giữa` có mặt ở 89,4% tài liệu  (idf 0,11)

Chúng là **stopword**, và stopword không xếp hạng được gì. Tệ hơn, token tự do
khớp bừa: `"bốn thức ăn"` tách thành `bốn` + `thức` + `ăn`, nên `bốn` khớp mọi
khung có đúng bốn *thứ gì đó*. Arm chạy một mình được 2/42 trên bộ dev, và trong
hợp nhất leo dốc chốt trọng số **0,0**.

Cách sửa: **một thuộc tính + một nhãn = MỘT token**
---------------------------------------------------
`ba_người`, `trái_người`, `đỏ_nón`. `_tokenize` dùng `re.findall(r"\\w+")` và `_`
là ký tự `\\w`, nên dấu gạch dưới **sống sót** qua bộ tách từ — đó là toàn bộ mẹo.

Token ghép thì hiếm, tức có idf, tức xếp hạng được. `ba_người` chỉ khớp khung có
đúng ba người, không khớp khung có ba cái ghế.

**Cái giá, và nó bắt buộc phải trả:** truy vấn cũng phải được viết lại theo đúng
quy tắc này, nếu không chỉ mục và truy vấn **không còn một token chung nào** —
đúng bài học `labels_vi`. Xem `mfe.online.object_query`.

Vì sao KHÔNG dùng lưới 7×7 làm token
-------------------------------------
Tài liệu hướng đi đề xuất lưới 7×7 và nêu ví dụ *"xe hơi ở góc trên bên phải"*.
Đếm trên 49 đề thật thì ví dụ đó **không tồn tại**:

    ngang (trái/phải/giữa)   8/49  16%
    dọc   (trên/dưới)        2/49   4%
    GÓC   (trên-trái…)       0/49   0%      <-- không đề nào
    trước/sau                8/49  16%      <-- lưới 2D không diễn đạt được

Chia mịn hơn mức đề hỏi chỉ làm loãng BM25: mỗi ô thêm vào là một token nữa
không bao giờ được truy vấn tới. Nên `grid_cell` **có** được cài (nó đúng, rẻ,
và tài liệu hướng đi đòi) và **có** đi vào artifact có cấu trúc, nhưng nó
**không** vào chuỗi BM25. Đổi ý thì đổi `EMIT_GRID_TOKEN` — và đo lại.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from mfe.offline.labels_vi import translate_labels

#: Ký tự nối thuộc tính với nhãn. PHẢI là `_`: `re.findall(r"\w+")` giữ nó lại,
#: còn `-` hay khoảng trắng thì bị cắt và token ghép vỡ trở lại thành token tự do.
JOIN = "_"

#: Biên chia khung hình thành ba dải theo tâm hộp (toạ độ x đã chuẩn hoá).
ZONE_EDGES = (1 / 3, 2 / 3)
ZONE_NAMES = ("trái", "giữa", "phải")

#: Số đếm viết bằng CHỮ. Đề viết bằng chữ ("ba người", "hơn 5 người"), và BM25
#: khớp token nên `"3"` với `"ba"` là hai token khác nhau.
COUNT_WORDS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
MANY = "nhiều"

#: Lưới cho `grid_cell`. Giữ đúng con số tài liệu hướng đi chốt.
GRID_SIZE = 7

#: Ô lưới KHÔNG đi vào chuỗi BM25 — xem docstring module. Đổi cờ này thì phải
#: đo lại trên bộ dev, đừng đổi vì nó nghe hợp lý.
EMIT_GRID_TOKEN = False

#: 11 màu phổ quát của tài liệu hướng đi §3.2, kèm tên tiếng Việt để token ghép
#: khớp được truy vấn tiếng Việt. Tên tiếng Việt là thứ đi vào token.
UNIVERSAL_COLOURS: dict[str, tuple[int, int, int]] = {
    "trắng": (255, 255, 255),
    "đen": (0, 0, 0),
    "đỏ": (255, 0, 0),
    "lá": (0, 128, 0),
    "vàng": (255, 255, 0),
    "dương": (0, 0, 255),
    "nâu": (139, 69, 19),
    "tím": (128, 0, 128),
    "hồng": (255, 192, 203),
    "cam": (255, 165, 0),
    "xám": (128, 128, 128),
}


@dataclass(frozen=True)
class Detection:
    """Một hộp đã qua ngưỡng điểm."""

    label: str
    zone: str
    cell: int
    colour: str | None = None


def count_word(n: int) -> str:
    return COUNT_WORDS[n] if n < len(COUNT_WORDS) else MANY


def zone_of(box) -> str:
    """Dải ngang chứa TÂM hộp. `box` là `[ymin, xmin, ymax, xmax]` chuẩn hoá.

    Thứ tự trục đã **xác minh trên dữ liệu thật** ngày 26-08-2026, ba cách độc
    lập trên `L21_V001` (20 keyframe có cả JSON lẫn ảnh):

    * tỉ lệ khung: dưới cách đọc này, `Man` cao hơn rộng ở **94%** số hộp,
      `Person` 81%, `Human face` 84%. Dưới cách đọc đảo trục thì **0%**.
    * cấu trúc: `Human face` nằm ở nửa TRÊN của hộp người chứa nó, **18/18**.
    * vẽ hộp lên ảnh và nhìn — hộp ôm khít người, không lệch trục.

    Lấy TÂM chứ không lấy mép: một vật rộng chiếm cả khung có `xmin` ở dải trái
    và `xmax` ở dải phải, gán vào cả hai thì "bên trái" mất hết sức phân biệt.
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


def grid_cell(box, grid_size: int = GRID_SIZE) -> int:
    """Ô lưới `grid_size × grid_size` chứa tâm hộp, đánh số 0 từ trái-trên.

    Chép đúng công thức `docs/triet/huong-di-grid-vitri-mausac.md` §3.1. Nó
    **không** đi vào chuỗi BM25 (0/49 đề hỏi theo góc) nhưng đi vào artifact có
    cấu trúc: rẻ, tất định, và là thứ một tầng lọc theo vùng sau này cần.
    """
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box[:4])
    except (TypeError, ValueError, IndexError):
        return (grid_size // 2) * grid_size + grid_size // 2
    col = min(int(((xmin + xmax) / 2) * grid_size), grid_size - 1)
    row = min(int(((ymin + ymax) / 2) * grid_size), grid_size - 1)
    return max(row, 0) * grid_size + max(col, 0)


def dominant_colour(pixels) -> str | None:
    """Màu phổ quát gần nhất với màu trung vị của vùng ảnh.

    ⚠️ **ĐO NGÀY 26-08-2026 TRÊN 6.441 HỘP THẬT: KHÔNG DÙNG ĐƯỢC.** Hàm này
    được giữ vì nó đúng như tài liệu hướng đi §3.2 mô tả, có test, và vì phép đo
    dưới đây phải còn chỗ để đọc. Nó **không** được nối vào bản dựng chỉ mục.

    Ba cách ước lượng, đo trên ảnh keyframe thật của 457 video:

        trung vị cả hộp        xám 52% · nâu 29% · đen 11%   (10 màu)
        mode từng pixel        xám 49% · nâu 24% · đen 11%   (11 màu)
        màu tươi nổi bật ≥15%  nâu 50% · hồng 27% · xám 12%  (11 màu)

    Hai cách đầu ra **xám** vì hộp chứa cả da, tóc, nền lẫn áo — trung bình của
    một mớ là bùn. Cách thứ ba tưởng sửa được, nhưng nó ra **nâu + hồng 77%**:
    bảng 11 màu phổ quát đặt **màu da người** đúng giữa nâu và hồng, mà phần lớn
    hộp là `Person` và `Clothing`.

    Trong cả ba, `vàng`/`đỏ`/`xanh` — đúng những màu 63% đề hỏi tới — đều ở mức
    0-2%. Một token `nâu_người` có mặt ở nửa số hộp là stopword mới, tức đúng
    cái bệnh vừa chữa xong.

    Muốn màu chạy được thì phải đổi bản chất phép đo, không phải chỉnh ngưỡng:
    lấy mẫu riêng vùng thân áo, hoặc loại pixel màu da trước. **Và trước khi làm
    thế, hãy nhớ caption Qwen2.5-VL đã nêu màu-gắn-vật ở 87,9%** — nó giải đúng
    năng lực này, tốt hơn nhiều, và lượt 5 đang chạy.

    `pixels` là mảng `(N, 3)` RGB (numpy) hoặc iterable bộ ba. Trả `None` khi
    vùng rỗng — **không** đoán bừa một màu, vì một màu sai gắn vào nhãn tạo ra
    một token ghép sai và nó sẽ khớp nhầm chứ không chỉ trượt.

    Dùng **trung vị** chứ không trung bình: trung bình của nửa đỏ nửa xanh là
    xám, một màu không hề có trong ảnh. Trung vị theo từng kênh vẫn lệch khi vật
    hai màu, nhưng nó luôn rơi gần một màu thật.
    """
    import numpy as np  # noqa: PLC0415 - chi can khi thuc su tinh mau

    array = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
    if array.size == 0:
        return None
    centre = np.median(array, axis=0)
    names = list(UNIVERSAL_COLOURS)
    table = np.asarray([UNIVERSAL_COLOURS[n] for n in names], dtype=np.float32)
    return names[int(np.argmin(((table - centre) ** 2).sum(axis=1)))]


def crop_pixels(image, box):
    """Pixel bên trong hộp, đọc từ một `PIL.Image`. `box` chuẩn hoá.

    Tách riêng khỏi `dominant_colour` để phép đo màu test được mà không cần ảnh.
    """
    import numpy as np  # noqa: PLC0415

    width, height = image.size
    ymin, xmin, ymax, xmax = (float(v) for v in box[:4])
    left, right = int(xmin * width), max(int(xmax * width), int(xmin * width) + 1)
    top, bottom = int(ymin * height), max(int(ymax * height), int(ymin * height) + 1)
    crop = image.convert("RGB").crop((left, top, right, bottom))
    return np.asarray(crop, dtype=np.uint8).reshape(-1, 3)


def canonical_vi(words: list[str], original: str) -> str:
    """Dạng tiếng Việt CHÍNH của một nhãn — thứ duy nhất được ghép thuộc tính.

    `translate_labels` trả `[nhãn gốc, dạng Việt 1, dạng Việt 2, …]`, nên dạng
    chính là phần tử **thứ hai**. Nhãn chưa có bản dịch thì rơi về chính nó.

    Tách thành hàm vì `mfe.online.object_query` phải chọn **đúng cùng** dạng
    này; hai nửa chọn khác nhau là chỉ mục và truy vấn không còn token chung.
    """
    for word in words:
        if word != original:
            return word
    return original


def compose(attribute: str, label: str) -> str:
    """`ba` + `người` -> `ba_người`. Khoảng trắng trong nhãn cũng thành `_`.

    `xe máy` phải thành `hai_xe_máy` chứ không phải `hai_xe máy`: cái sau tách
    thành `hai_xe` + `máy`, tức vừa mất nhãn vừa đẻ ra một token vô nghĩa.
    """
    slug = re.sub(r"\s+", JOIN, unicodedata.normalize("NFC", label).strip())
    return f"{attribute}{JOIN}{slug}" if slug else ""


def render(detections: list[Detection], *, translate: bool = True) -> str:
    """Chuỗi đánh chỉ mục cho MỘT keyframe.

    Bốn nhóm token, và mỗi nhóm trả lời một dạng đề đếm được trên 49 đề thật:

    1. **Nhãn trần** (đã dịch) — nền, cho đề chỉ nêu vật thể. Mỗi nhãn MỘT lần.
    2. **`{đếm}_{nhãn}`** — 51% đề hỏi số lượng.
    3. **`{dải}_{nhãn}`** — 16% đề hỏi trái/giữa/phải.
    4. **`{màu}_{nhãn}`** — 63% đề buộc màu vào vật, khi có dữ liệu màu.

    **Bỏ hẳn phép lặp nhãn `min(n, 3)` lần** của bản trước. Nó gắn số đếm vào
    *điểm số* thay vì vào *chữ*, nên nó không phân biệt được "ba người" với "ba
    cái ghế" — và nó làm tài liệu dài ra, mà BM25 phạt tài liệu dài. Token
    `ba_người` làm đúng việc đó bằng một token duy nhất, có idf.

    Nhãn tiếng Anh KHÔNG bị bỏ: `translate_labels` đặt nhãn gốc trước rồi mới
    tới đồng nghĩa tiếng Việt, nên một truy vấn tiếng Anh (hiếm, nhưng có) vẫn
    khớp ở nhóm 1.
    """
    if not detections:
        return ""

    counts: dict[str, int] = {}
    zones: dict[str, set[str]] = {}
    colours: dict[str, set[str]] = {}
    cells: dict[str, set[int]] = {}
    display: dict[str, str] = {}
    for hit in detections:
        key = hit.label.casefold()
        display.setdefault(key, hit.label)
        counts[key] = counts.get(key, 0) + 1
        zones.setdefault(key, set()).add(hit.zone)
        cells.setdefault(key, set()).add(hit.cell)
        if hit.colour:
            colours.setdefault(key, set()).add(hit.colour)

    parts: list[str] = []
    for key, total in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        words = translate_labels([display[key]]) if translate else [display[key]]
        parts.extend(words)
        # Token ghép CHỈ cho MỘT dạng — dạng tiếng Việt đầu tiên.
        #
        # Ghép cho cả ba đồng nghĩa của `Motorcycle` sẽ sinh 12 token cho một
        # loại vật (`một_xe_máy` `một_mô_tô` `một_xe_gắn_máy` × 4 thuộc tính),
        # và BM25 **phạt tài liệu dài** — đúng cái tội bản `min(n,3)` mắc phải,
        # chỉ đổi cơ chế. Đồng nghĩa vẫn còn ở token trần, nên phần thu hồi
        # không mất gì.
        #
        # `mfe.online.object_query` quy MỌI đồng nghĩa trong truy vấn về đúng
        # dạng này, nên "hai mô tô" vẫn ra `hai_xe_máy`. Hai nửa phải chọn cùng
        # một dạng chính, và đó là lý do cả hai đều lấy `translate_labels(...)[1]`.
        head = canonical_vi(words, display[key])
        parts.append(compose(count_word(total), head))
        parts.extend(compose(zone, head) for zone in sorted(zones[key]))
        parts.extend(compose(colour, head) for colour in sorted(colours.get(key, ())))
        if EMIT_GRID_TOKEN:
            parts.extend(compose(f"o{cell}", head) for cell in sorted(cells[key]))

    return " ".join(p for p in parts if p)


def read_detections(payload: dict, score_min: float, *, image=None) -> list[Detection]:
    """`Detection` cho mọi hộp trên ngưỡng — **không khử trùng lặp**.

    Khử trùng lặp là chỗ bản đầu tiên mất trắng số đếm: năm hộp `Person` thành
    đúng một chữ "người", nên arm không phân biệt nổi khung có 1 người với khung
    có 12.

    `image` (PIL) là **tuỳ chọn**: có ảnh thì tính màu, không có thì bỏ trống.
    Đó là một sự thật về chi phí chứ không phải sự lười — gói `objects` là một
    lần tải 640 MB, còn ảnh keyframe là 28,7 GB. Một bản chỉ mục không màu vẫn
    đúng, chỉ thiếu nhóm token thứ 4.
    """
    entities = payload.get("detection_class_entities") or []
    scores = payload.get("detection_scores") or []
    boxes = payload.get("detection_boxes") or []
    out: list[Detection] = []
    for index, (entity, score) in enumerate(zip(entities, scores, strict=False)):
        try:
            if float(score) < score_min:
                continue
        except (TypeError, ValueError):
            continue
        name = str(entity).strip()
        if not name:
            continue
        box = boxes[index] if index < len(boxes) else None
        colour = None
        if image is not None and box is not None:
            colour = dominant_colour(crop_pixels(image, box))
        out.append(
            Detection(
                label=name,
                zone=zone_of(box) if box else ZONE_NAMES[1],
                cell=grid_cell(box) if box else (GRID_SIZE**2) // 2,
                colour=colour,
            )
        )
    return out
