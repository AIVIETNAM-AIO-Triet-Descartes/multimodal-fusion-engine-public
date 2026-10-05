"""Viết lại truy vấn theo đúng quy tắc token ghép của arm `object`.

    mfe.offline.object_text  ·  handoff §A.2, §C.2

Đây là **nửa bắt buộc** của thay đổi kia, không phải phần thêm cho đẹp
------------------------------------------------------------------------
Chỉ mục giờ ghi `ba_người`, `trái_người`, `đỏ_nón` thay vì `ba` `người` `trái`
rời rạc. Truy vấn viết `"ba người đội nón đỏ"` sinh ra `ba` `người` `đội` `nón`
`đỏ` — và giữa hai bên **không còn một token chung nào**. Đó đúng là bài học
`labels_vi`, lần này tự ta gây ra.

Nên bộ viết lại này chạy cùng chỗ, cùng quy tắc, và mọi thay đổi ở
`object_text.compose` phải kéo theo ở đây. Một test khoá đúng chuyện đó.

Vì sao ghép được: trật tự tiếng Việt
-------------------------------------
Ba dạng, và cả ba đều là trật tự cố định:

    SỐ ĐẾM đứng TRƯỚC nhãn        "ba người"        -> ba_người
    MÀU     đứng SAU  nhãn        "nón đỏ"          -> đỏ_nón
    VỊ TRÍ  đứng SAU  nhãn        "người bên trái"  -> trái_người

Ngoại lệ đã tính tới: `"bên trái là một người"` đặt vị trí TRƯỚC. Nên vị trí
được ghép với nhãn **gần nhất** trong cửa sổ, hai chiều.

Token GỐC được giữ lại
-----------------------
Truy vấn viết lại **thêm** token ghép chứ không thay token trần. Không phải mọi
khung đều có màu tính được (gói `objects` không kèm ảnh), nên một truy vấn chỉ
có `đỏ_nón` sẽ trượt sạch phần chỉ mục không màu. Giữ cả hai là giữ cái sàn.
"""

from __future__ import annotations

import re
import unicodedata

from mfe.offline.labels_vi import LABELS_VI
from mfe.offline.object_text import COUNT_WORDS, MANY, UNIVERSAL_COLOURS, compose

#: Bao nhiêu token được phép nằm giữa thuộc tính và nhãn. 2 là chỗ đứng giữa hai
#: cái sai: 0 thì trượt `"ba chiếc xe máy"` (`chiếc` chen vào) và
#: `"người mặc áo đỏ"`; lớn hơn thì `"ba người đứng cạnh chiếc xe"` sẽ ghép
#: `ba` với `xe`, tức bịa ra một ràng buộc đề không hề nêu.
WINDOW = 2

#: Từ chỉ màu mà truy vấn dùng, kể cả dạng ghép mà `UNIVERSAL_COLOURS` không có.
#: `"xanh dương"` và `"xanh lá"` phải quy về đúng tên màu chỉ mục đang ghi, nếu
#: không thì hai bên lại lệch nhau ở đúng chỗ vừa nối được.
COLOUR_ALIASES: dict[str, str] = {
    "xanh dương": "dương",
    "xanh lam": "dương",
    "xanh lá": "lá",
    "xanh lục": "lá",
    "xanh": "lá",
    "trắng": "trắng",
    "đen": "đen",
    "đỏ": "đỏ",
    "vàng": "vàng",
    "nâu": "nâu",
    "tím": "tím",
    "hồng": "hồng",
    "cam": "cam",
    "xám": "xám",
    "ghi": "xám",
    "bạc": "xám",
}

#: Cụm chỉ vị trí trong truy vấn -> tên dải chỉ mục đang ghi.
ZONE_ALIASES: dict[str, str] = {
    "bên trái": "trái",
    "phía trái": "trái",
    "ngoài cùng bên trái": "trái",
    "trái": "trái",
    "bên phải": "phải",
    "phía phải": "phải",
    "ngoài cùng bên phải": "phải",
    "phải": "phải",
    "ở giữa": "giữa",
    "chính giữa": "giữa",
    "giữa": "giữa",
}

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


#: Từ ghép tiếng Việt mà một NHÃN là tiếng đầu, nhưng cả cụm nghĩa khác hẳn.
#:
#: `tay đua` là vận động viên, không phải bàn tay; `mặt trời` không phải khuôn
#: mặt. Không chặn thì `"một tay đua áo vàng"` sinh ra `một_tay` — một ràng buộc
#: đề **không hề nêu**, và nó khớp nhầm chứ không chỉ trượt.
#:
#: Danh sách này ngắn có chủ ý: nó chỉ chứa những cụm THẬT SỰ xuất hiện khi chạy
#: bộ viết lại trên 49 đề chính thức + bộ dev. Đoán thêm cụm chưa gặp là đoán,
#: và một cụm chặn nhầm thì lặng lẽ mất một token đúng.
BLOCKED_COMPOUNDS: frozenset[tuple[str, ...]] = frozenset({
    ("tay", "đua"),
    ("tay", "lái"),
    ("tay", "vợt"),
    ("mũi", "tên"),      # de dot-1/23: "hoa tiet dia cau va mui ten vang"
    ("mũi", "chân"),
    ("mặt", "trời"),
    ("mặt", "trăng"),
    ("mặt", "nước"),
    ("mặt", "đất"),
    ("mặt", "đường"),
    ("chân", "trời"),
    ("chân", "núi"),
    ("chân", "cầu"),
    ("đầu", "tiên"),
    ("con", "đường"),
})


def _label_vocabulary() -> dict[tuple[str, ...], str]:
    """`(token,…) -> dạng tiếng Việt CHÍNH` cho mọi nhãn đã dịch.

    Khoá là **bộ token** chứ không phải chuỗi: nhãn nhiều chữ (`xe máy`) phải
    khớp được khi nó nằm giữa câu, và so theo token là cách duy nhất không phụ
    thuộc dấu câu xung quanh.

    Giá trị là dạng **chính** (`translate_labels(...)[1]`), không phải dạng vừa
    khớp. Chỉ mục chỉ ghép thuộc tính vào dạng chính, nên `"hai mô tô"` phải ra
    `hai_xe_máy` mới khớp được. Hai nửa quy về cùng một dạng, hoặc không nửa nào
    gặp nửa nào.
    """
    vocab: dict[tuple[str, ...], str] = {}
    for forms in LABELS_VI.values():
        head = forms[0] if forms else ""
        canonical = " ".join(_TOKEN.findall(_fold(head)))
        if not canonical:
            continue
        for form in forms:
            key = tuple(_TOKEN.findall(_fold(form)))
            if key:
                vocab.setdefault(key, canonical)
    return vocab


_VOCAB = _label_vocabulary()
_MAX_LABEL_TOKENS = max((len(k) for k in _VOCAB), default=1)
#: `"không"` CỐ Ý bị loại khỏi phía TRUY VẤN dù nó là `COUNT_WORDS[0]`.
#:
#: Trong tiếng Việt nó gần như luôn là phủ định, không phải số không: đề
#: `dot-1/15` viết *"Không tính bảng chú giải…"* và bản đầu sinh ra
#: `không_bảng_trắng`. Chỉ mục không bao giờ ghi số đếm 0 (chỉ nhãn ĐƯỢC phát
#: hiện mới có dòng), nên token đó không khớp gì — nhưng nó là một token rác
#: sinh ra từ một phép đọc sai, và phép đọc sai đó sẽ còn sinh ra thứ khác.
_COUNTS = {*COUNT_WORDS, MANY, "vài", "đôi", "cặp"} - {"không"}
_COLOURS = {c for c in COLOUR_ALIASES} | set(UNIVERSAL_COLOURS)


def _find_labels(tokens: list[str]) -> list[tuple[int, int, str]]:
    """`[(đầu, cuối, dạng nhãn)]`, ưu tiên cụm DÀI nhất tại mỗi vị trí.

    Dài trước là bắt buộc: `xe máy` phải thắng `xe`, nếu không `hai_xe` được
    sinh ra trong khi chỉ mục ghi `hai_xe_máy`.
    """
    found: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        for size in range(min(_MAX_LABEL_TOKENS, len(tokens) - index), 0, -1):
            key = tuple(tokens[index : index + size])
            if key not in _VOCAB:
                continue
            # Nhãn khớp, nhưng nó có phải tiếng đầu của một từ ghép nghĩa khác?
            if tuple(tokens[index : index + size + 1]) in BLOCKED_COMPOUNDS:
                index += size
                break
            found.append((index, index + size, _VOCAB[key]))
            index += size
            break
        else:
            index += 1
    return found


def _nearest_label(spots: list[tuple[int, int, str]], position: int, *, after: bool) -> str | None:
    """Nhãn gần nhất về một phía, trong `WINDOW` token."""
    best: tuple[int, str] | None = None
    for start, end, form in spots:
        gap = start - position - 1 if after else position - end
        if 0 <= gap <= WINDOW and (best is None or gap < best[0]):
            best = (gap, form)
    return best[1] if best else None


def rewrite(text: str) -> str:
    """Truy vấn gốc **cộng thêm** các token ghép suy ra được.

    Trả về chuỗi để nó cắm thẳng vào chỗ `_tokenize(q)` của `flat_arms`, không
    đổi hình dạng gì của đường truy vấn.
    """
    tokens = _TOKEN.findall(_fold(text))
    if not tokens:
        return text
    spots = _find_labels(tokens)
    if not spots:
        return text

    extra: list[str] = []
    for position, token in enumerate(tokens):
        if token in _COUNTS:
            # Số đếm đứng TRƯỚC nhãn: "ba người".
            label = _nearest_label(spots, position, after=True)
            if label:
                extra.append(compose(token, label))
            continue

        # Màu và vị trí có thể là cụm nhiều chữ; thử cụm DÀI trước.
        # `xanh lá cây` phải thắng `xanh lá`, nếu không `cây` bị đọc thành nhãn.
        triple = " ".join(tokens[position : position + 3])
        pair = " ".join(tokens[position : position + 2])
        for phrase, table, backward_only in (
            (triple, COLOUR_ALIASES, True),
            (pair, COLOUR_ALIASES, True),
            (token, COLOUR_ALIASES, True),
            (pair, ZONE_ALIASES, False),
            (token, ZONE_ALIASES, False),
        ):
            if phrase not in table:
                continue
            span = len(phrase.split()) - 1
            # `position`, KHONG phai `position - 1`: `gap = position - end` va
            # `end` la chi so LOAI TRU. Nhan ngay sat truoc thuoc tinh co
            # `end == position`, tuc gap 0. Tru thêm mot lam gap thanh -1 va
            # nhan do bi loai — `"nguoi doi mu do"` ra `do_nguoi` thay vi
            # `do_mu`, tuc gan mau cho nham vat.
            label = _nearest_label(spots, position, after=False)
            # MÀU chỉ nhìn về SAU, và đó là một sửa lỗi có bằng chứng.
            #
            # Tiếng Việt đặt màu SAU danh từ ("áo đỏ"), nên khi không có nhãn nào
            # phía trước thì màu đó thuộc về một từ KHÔNG PHẢI nhãn. Cho phép
            # nhìn tới trước sinh ra ghép bịa: `"con lân trắng, mũi đỏ"` ra
            # `trắng_mũi` (trắng thuộc về con lân, không thuộc cái mũi), và
            # `"phông nền xanh lá cây"` ra `lá_cây`.
            #
            # VỊ TRÍ thì ngược lại, phải nhìn cả hai chiều: `"người bên trái"`
            # đặt sau, `"bên trái là một người"` đặt trước.
            if label is None and not backward_only:
                label = _nearest_label(spots, position + span, after=True)
            if label:
                extra.append(compose(table[phrase], label))
            break

    if not extra:
        return text
    # Giữ nguyên truy vấn gốc rồi mới nối: token trần vẫn là cái sàn khi chỉ mục
    # của một khung nào đó không có màu.
    return f"{text} " + " ".join(dict.fromkeys(extra))


#: `source` -> hàm viết lại. `flat_arms` tra bảng này trước khi `_tokenize`.
#: Chỉ `object` có mặt: các nguồn khác là văn bản tự nhiên, và ghép token vào
#: chúng sẽ tạo ra token không nguồn nào chứa.
QUERY_REWRITERS = {"object": rewrite}
