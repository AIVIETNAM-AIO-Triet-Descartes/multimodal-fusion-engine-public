"""Đếm xem caption sinh ra có nêu QUAN HỆ / MÀU / SỐ LƯỢNG / VỊ TRÍ không.

Vì sao đo cái này chứ không đo chồng lấn từ vựng với bản của Bảo
-----------------------------------------------------------------
Hai prompt khác nhau mô tả đúng cùng một khung hình vẫn ra hai chuỗi khác nhau,
nên chồng lấn từ vựng chỉ nói rằng prompt đã đổi — không nói prompt tốt lên hay
xấu đi. Thứ đo được là **tỉ lệ caption nêu ra từng năng lực mà 49 đề thật hỏi**
(`docs/triet/diem-yeu-pipeline-theo-49-de.md` §1, và `QUERY_DEMAND` ở dưới đo lại
bằng chính bộ dò này).

Một prompt bảo model nêu màu mà chỉ 20% caption có từ chỉ màu thì prompt đó
không hoạt động, và điều đó **đọc được ngay trên 307 ảnh probe** — không phải đợi
tới lúc đo Final Score trên bộ dev.

Phép đo này phải SONG NGỮ, và đó là một bài học phải trả giá
------------------------------------------------------------
Bản đầu chỉ có từ khoá tiếng Việt, vì lúc đó biến thể `aic` viết tiếng Việt. Đem
nó đo 855 caption **tiếng Anh** của Bảo thì ra `màu 1,2% · vị trí 0,1%`, và tôi
đã đọc con số đó như "caption của Bảo không nêu màu". **Sai.** Nó chỉ có nghĩa là
bộ dò không biết đọc tiếng Anh. Đo lại bằng danh sách song ngữ: `màu 78,6% ·
vị trí 59,2% · quan hệ 50,5%`.

Một phép đo mù trước ngôn ngữ của dữ liệu không trả về lỗi — nó trả về `0%`, và
`0%` trông y hệt một kết luận. Nên mọi danh sách từ khoá ở đây có cả hai thứ
tiếng, kể cả khi lượt chạy hiện tại chỉ sinh ra một thứ tiếng.

Hai cảnh báo về cách đọc con số ở đây
--------------------------------------
**Đây là phép đếm TỪ KHOÁ, không phải phép đo ĐÚNG SAI.** Một caption ghi "áo đỏ"
trong khi áo màu xanh vẫn được tính là "có màu". Nó trả lời "model có làm theo
chỉ dẫn không", không trả lời "model có nhìn đúng không". Câu thứ hai chỉ trả lời
được bằng cách soi ảnh — và đó là việc của người, trên vài chục mẫu.

**Ngưỡng không phải mục tiêu.** Không phải khung hình nào cũng có vật đếm được
hay có màu đáng nói: một cảnh trời mây thì không. Nên 100% là dấu hiệu model đang
*bịa cho đủ*, không phải dấu hiệu tốt.
"""

from __future__ import annotations

import re
import statistics
import unicodedata

#: Từ chỉ màu trong tiếng Việt. Gồm cả biến thể ghép (`xanh dương`, `xanh lá`) vì
#: `xanh` một mình vừa là lam vừa là lục — nó vẫn được tính, nhưng cụm ghép mới là
#: thứ đề dùng ("áo xanh dương quần đỏ").
#: `"be"` (màu be) CỐ Ý không có trong danh sách: nó trùng đúng từ `be` của tiếng
#: Anh, và một mình nó kéo tỉ lệ "có màu" của 855 caption TIẾNG ANH của Bảo lên
#: 58,1%. Màu be hiếm trong caption tiếng Việt; một từ khoá kéo theo hàng trăm
#: dương tính giả thì đắt hơn nhiều so với vài ca bỏ sót.
COLOUR_WORDS_VI = (
    "đỏ", "cam", "vàng", "xanh", "lam", "lục", "tím", "hồng", "nâu",
    "đen", "trắng", "xám", "bạc", "kem", "ngà", "sẫm",
)

#: Cùng danh sách cho TIẾNG ANH. Cần từ 25-08-2026, khi caption đổi sang tiếng Anh
#: để đi qua BGE-M3: phép đếm chỉ có từ khoá tiếng Việt sẽ đọc ra 0% trên mọi năng
#: lực và **cổng chấm probe trở nên mù** — nó vẫn in ra một bảng trông bình thường.
#: `"dark"`, `"light"`, `"bright"` CỐ Ý không có: chúng là độ sáng, không phải màu,
#: và chúng xuất hiện trong gần như mọi caption ("a dark suit", "light blue").
COLOUR_WORDS_EN = (
    "red", "orange", "yellow", "green", "blue", "navy", "purple", "violet",
    "pink", "brown", "black", "white", "grey", "gray", "silver", "gold",
    "golden", "beige", "khaki", "maroon", "turquoise", "cream", "ivory",
)

COLOUR_WORDS = COLOUR_WORDS_VI + COLOUR_WORDS_EN

#: Số đếm viết bằng CHỮ. Đề viết bằng chữ ("ba người", "hơn 5 người"), nên chỉ mục
#: cũng phải có chữ — và chữ số cũng được tính riêng ở dưới.
COUNT_WORDS_VI = (
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "nhiều", "vài", "đôi",
)

#: `"one"` CÓ trong danh sách dù nó cũng là đại từ ("the one on the left"): bỏ nó
#: là mù trước đúng dạng đề `dot-1/1` hỏi ("chỉ có **một** người đeo kính"). Nó
#: đắt — riêng nó thêm 15 điểm phần trăm trên 855 caption của Bảo (42,0% → 57,0%)
#: — nhưng tiếng Anh không dùng `one` làm mạo từ, nên phần lớn số đó là đếm thật.
#: Tiếng Việt thì ngược lại: `một` VỪA là số đếm VỪA là mạo từ, và đó là lý do
#: hàng `đếm` là hàng ít tin được nhất trong bảng (xem `format_audit`).
COUNT_WORDS_EN = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "several", "many", "few", "dozen", "dozens",
    "pair", "both", "group",
)

COUNT_WORDS = COUNT_WORDS_VI + COUNT_WORDS_EN

POSITION_WORDS_VI = (
    "bên trái", "bên phải", "ở giữa", "chính giữa", "phía sau", "phía trước",
    "phía trên", "phía dưới", "góc trên", "góc dưới", "trái", "phải", "giữa",
)

#: `"background"` và `"foreground"` CỐ Ý không có mặt. Đo trên 855 caption của Bảo
#: thì riêng `"background"` có mặt ở **65,1%** — cụm "in the background" là câu chữ
#: mặc định của mọi caption, không phải thông tin vị trí phân biệt được khung hình
#: này với khung hình khác. Giữ nó lại là để một stopword quyết định con số, đúng
#: cơ chế đã giết arm `object`. Đề thật hỏi **trái/giữa/phải**, và đó là thứ đo.
POSITION_WORDS_EN = (
    "left", "right", "middle", "centre", "center", "behind", "in front of",
    "top left", "top right", "bottom left", "bottom right", "leftmost",
    "rightmost", "beside", "next to", "above", "below", "between",
)

POSITION_WORDS = POSITION_WORDS_VI + POSITION_WORDS_EN

CAMERA_WORDS_VI = (
    "cận cảnh", "toàn cảnh", "trung cảnh", "từ trên cao", "từ dưới lên",
    "quay chậm", "flycam", "lia máy", "zoom",
)

CAMERA_WORDS_EN = (
    "close-up", "closeup", "close up", "wide shot", "wide-angle", "medium shot",
    "aerial", "overhead", "bird's-eye", "low angle", "high angle", "eye level",
    "from above", "from below", "slow motion", "zoom", "pan",
)

CAMERA_WORDS = CAMERA_WORDS_VI + CAMERA_WORDS_EN

#: ĐỘNG TỪ TƯƠNG TÁC — thứ phân biệt "một danh sách vật thể" với "một cảnh".
#:
#: Yêu cầu của Lead (25-08-2026): caption phải nói các vật thể đáng chú ý ĐANG LÀM
#: GÌ VỚI NHAU ("người đang tưới cây", "giáo viên chỉ bài cho học trò"), chứ không
#: chỉ liệt kê rời rạc. Phép đếm này là cách kiểm chỉ dẫn đó có được làm theo không.
#:
#: `đứng`, `ngồi`, `mặc`, `stand`, `sit`, `wear`, `be` CỐ Ý KHÔNG có mặt, và đó là
#: toàn bộ ý nghĩa của phép đo: "hai người đứng cạnh bàn" là liệt kê có tư thế, không
#: phải tương tác. Thêm chúng vào là biến phép đo thành hằng số 100%.
RELATION_VERBS_VI = (
    "cầm", "nắm", "bế", "cõng", "dắt", "dìu", "đẩy", "kéo", "chỉ", "đưa", "trao",
    "tưới", "rót", "cắt", "gọt", "rửa", "nấu", "lái", "chở", "ném", "đá", "nâng",
    "đặt", "chạm", "ôm", "bắt tay", "cho ăn", "trồng", "sửa", "xây", "bán", "mua",
    "khuân", "vác", "gánh", "quét", "đào", "leo", "hái", "gõ", "sơn", "hàn", "đo",
    "nhìn", "xem", "đọc", "viết", "treo", "buộc", "vẫy", "đuổi", "giúp", "trèo",
)

#: Dạng gốc tiếng Anh. `_word_pattern` tự sinh thêm `-s/-es/-ed/-ing/-d`, nên chỉ
#: những dạng BẤT QUY TẮC và những dạng gấp đôi phụ âm mới phải liệt kê riêng.
#: KHÔNG có: `build`, `hand`, `plant`, `place`, `type`, `wave`, `paint`, `measure`,
#: `stack`, `sort`. Với chúng, ngay cả dạng đã chia vẫn là danh từ thường gặp —
#: `buildings`, `hands`, `plants`, `places`, `painting`. Đo riêng trên 855 caption
#: của Bảo: `build` 10,2% và `hand` 6,7%, toàn bộ là toà nhà và bàn tay. Dạng động
#: từ của chúng được liệt kê tay ở `RELATION_IRREGULAR_EN`.
RELATION_STEMS_EN = (
    "hold", "carry", "push", "pull", "point", "pass", "give", "water",
    "pour", "cut", "peel", "wash", "cook", "ride", "drive", "steer", "throw",
    "catch", "kick", "lift", "touch", "hug", "shake", "feed",
    "harvest", "repair", "fix", "load", "unload", "sell", "buy", "serve",
    "stir", "sweep", "dig", "climb", "reach", "grab", "lean", "help", "hang",
    "tie", "chase", "follow", "talk", "speak", "read", "write",
    "weld", "pick", "wrap", "spray",
)

RELATION_IRREGULAR_EN = (
    "held", "carries", "carried", "carrying", "threw", "thrown", "caught", "fed",
    "built", "sold", "bought", "dug", "digging", "swept", "rode", "ridden",
    "driven", "drove", "gave", "shook", "wrote", "written", "stirring", "stirred",
    "cutting", "hanging", "hung", "hugging", "hugged", "grabbing", "grabbed",
    "wrapping", "wrapped", "tying", "tied",
    # Dang dong tu cua nhung tu bi loai o tren, liet ke tay de tranh danh tu.
    "handing", "handed", "planting", "planted", "placing", "placed",
    "stacking", "stacked", "sorting", "sorted", "measuring", "measured",
    "waving", "waved",
    # `painting`/`painted` KHONG co: mot cai la danh tu (buc tranh), cai kia la
    # tinh tu (buc tuong son mau). Ca hai deu khong phai quan he giua hai vat.
)

RELATION_WORDS = RELATION_VERBS_VI + RELATION_STEMS_EN + RELATION_IRREGULAR_EN

#: Mở đầu mà prompt CẤM. Không phải chuyện thẩm mỹ: mọi caption mở đầu giống nhau
#: thì cụm đó thành stopword trên toàn chỉ mục — đúng cơ chế đã giết arm `object`
#: (`một` có mặt ở 91,0% tài liệu, idf 0,09).
BANNED_OPENERS = (
    "hình ảnh cho thấy", "bức ảnh", "trong hình", "hình này", "bức hình",
    "the image shows", "the image depicts", "this image", "a photo of",
)

_TOKEN = re.compile(r"\w+", re.UNICODE)
#: Chữ cái tiếng Việt có dấu — dấu hiệu rẻ nhất cho "caption này là tiếng Việt".
_VIET_MARK = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩị"
                        r"òóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.IGNORECASE)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").casefold()


def _contains_word(folded: str, word: str) -> bool:
    """Khớp theo RANH GIỚI TỪ, không phải chuỗi con.

    Đây là một sửa lỗi, không phải sự cẩn thận thừa. Bản đầu dùng `word in text`
    và đo được **72,2%** caption của Bảo "có màu" — trong khi chúng viết bằng
    tiếng Anh và gần như không có từ chỉ màu tiếng Việt nào. Nguyên nhân: "be"
    nằm trong "before"/"between", "cam" nằm trong "camera". Con số sai đó trông
    hợp lý hơn con số đúng, nên nó sẽ không bao giờ tự lộ ra.

    Dùng lookaround thay cho \b để cụm nhiều từ ("bên trái") cũng đúng.
    """
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", folded) is not None


def _contains_verb(folded: str, stem: str) -> bool:
    """Khớp CHỈ các dạng ĐÃ CHIA của một động từ tiếng Anh, không khớp dạng gốc.

    Bỏ dạng gốc đi là một sửa lỗi có số đo, không phải sự cầu kỳ. Bản đầu khớp cả
    dạng gốc và đo được `water` 16,1% · `tie` 10,6% · `build` 10,4% · `hand` 8,9%
    trên 855 caption của Bảo — nhưng đó là các **danh từ**: nước lụt, cà vạt của
    người dẫn chương trình, toà nhà, bàn tay. Chúng thổi "quan hệ" lên 71,3% bằng
    những câu không có quan hệ nào.

    Caption tiếng Anh viết ở ngôi thứ ba nên động từ thật hầu như luôn đã chia
    (`holds`, `holding`, `watering`). Dạng gốc trần gần như chỉ xuất hiện khi từ
    đó là danh từ — nên bỏ nó lọc đúng thứ cần lọc.

    Dạng bất quy tắc (`held`, `threw`) không suy ra được nên nằm riêng ở
    `RELATION_IRREGULAR_EN`.
    """
    base = re.escape(stem)
    forms = [base + "s", base + "es", base + "ed", base + "ing"]
    if stem.endswith("e"):
        trimmed = re.escape(stem[:-1])
        forms += [trimmed + "ing", trimmed + "ed", base + "d"]
    return re.search(rf"(?<!\w)(?:{'|'.join(forms)})(?!\w)", folded) is not None


def has_colour(caption: str) -> bool:
    folded = _fold(caption)
    return any(_contains_word(folded, word) for word in COLOUR_WORDS)


def has_count(caption: str) -> bool:
    folded = _fold(caption)
    if re.search(r"\d", folded):
        return True
    return any(_contains_word(folded, word) for word in COUNT_WORDS)


def has_position(caption: str) -> bool:
    folded = _fold(caption)
    return any(_contains_word(folded, word) for word in POSITION_WORDS)


def has_camera(caption: str) -> bool:
    folded = _fold(caption)
    return any(_contains_word(folded, word) for word in CAMERA_WORDS)


def has_relation(caption: str) -> bool:
    """Caption có nói các vật thể ĐANG LÀM GÌ VỚI NHAU không.

    Đây là phép đếm TỪ KHOÁ trên động từ, cùng giới hạn như mọi phép đếm khác ở
    file này: nó trả lời "model có làm theo chỉ dẫn không", không trả lời "quan hệ
    model mô tả có đúng không". Một caption viết "người đàn ông tưới cây" trong khi
    ông ta đang cắt cỏ vẫn được tính.

    Nó vẫn đáng đo, vì thứ nó phân biệt là hai *dạng* caption khác hẳn nhau:
    "một người, một vòi nước, mấy chậu cây" so với "một người tưới mấy chậu cây
    bằng vòi nước xanh". Dạng đầu không trả lời được bất kỳ đề nào hỏi hành động.
    """
    folded = _fold(caption)
    if any(_contains_word(folded, word) for word in RELATION_VERBS_VI):
        return True
    if any(_contains_word(folded, word) for word in RELATION_IRREGULAR_EN):
        return True
    return any(_contains_verb(folded, stem) for stem in RELATION_STEMS_EN)


def has_banned_opener(caption: str) -> bool:
    folded = _fold(caption).lstrip("(\"' ")
    return any(folded.startswith(opener) for opener in BANNED_OPENERS)


def looks_vietnamese(caption: str) -> bool:
    """Có dấu tiếng Việt hay không.

    Thô có chủ ý. Một caption tiếng Anh thuần sẽ không có dấu nào, và đó là điều
    duy nhất cần phân biệt: biến thể `aic` yêu cầu tiếng Việt, còn `bao` yêu cầu
    tiếng Anh. Một câu tiếng Việt không dấu vẫn tính là Anh ở đây — chuyện đó
    không xảy ra với Qwen, và nếu xảy ra thì bản thân nó là lỗi cần thấy.
    """
    return bool(_VIET_MARK.search(caption or ""))


def audit(pairs) -> dict:
    """`pairs` = iterable of `(caption, ocr)`. Trả về bảng tỉ lệ.

    `ocr_lines` là số dòng trung vị trong nửa `[TEXT]` — thứ nói lên prompt có
    tách được các vùng chữ rời nhau trên màn hình hay không. Gộp cả biển hiệu,
    logo, đồng hồ và dòng chạy chân màn hình vào MỘT dòng là mất ranh giới giữa
    chúng, và BM25 sẽ khớp một cụm nằm vắt qua hai vùng chẳng liên quan gì nhau.
    """
    captions, ocrs = [], []
    for caption, ocr in pairs:
        captions.append(caption or "")
        ocrs.append(ocr or "")
    total = len(captions)
    if not total:
        return {"n": 0}

    line_counts = [len([ln for ln in o.splitlines() if ln.strip()]) for o in ocrs]
    with_text = [c for c in line_counts if c]
    return {
        "n": total,
        "mau": sum(map(has_colour, captions)) / total,
        "dem": sum(map(has_count, captions)) / total,
        "vi_tri": sum(map(has_position, captions)) / total,
        "goc_may": sum(map(has_camera, captions)) / total,
        "quan_he": sum(map(has_relation, captions)) / total,
        "mo_dau_cam": sum(map(has_banned_opener, captions)) / total,
        "tieng_viet": sum(map(looks_vietnamese, captions)) / total,
        "caption_token_trung_vi": statistics.median(
            len(_TOKEN.findall(c)) for c in captions
        ),
        "co_chu": sum(1 for c in line_counts if c) / total,
        "ocr_dong_trung_vi": statistics.median(with_text) if with_text else 0,
        "ocr_nhieu_dong": sum(1 for c in line_counts if c > 1) / total,
    }


#: Cùng bộ hàm ở file này, chạy trên 49 đề thật (`docs/context/**/query-*.txt`).
#:
#: Đo bằng CHÍNH bộ dò dùng cho caption là chuyện có chủ ý: hai cột trong bảng mới
#: so được với nhau. Con số người đọc tay trong `diem-yeu-pipeline-theo-49-de.md`
#: (63 · 51 · 45 · 16) khớp ở `màu` và `góc máy`, lệch ở hai hàng còn lại — và chỗ
#: lệch đó tự nó là thông tin, xem chú thích dưới bảng.
QUERY_DEMAND = {
    "mau": 0.633,
    "dem": 0.939,
    "vi_tri": 0.367,
    "goc_may": 0.163,
    "quan_he": 0.571,
}


def format_audit(report: dict, label: str = "") -> str:
    """Bảng người đọc được, kèm mốc đo trên 49 đề thật để so."""
    if not report.get("n"):
        return f"{label}: khong co du lieu"
    rows = (
        ("màu gắn vào vật", "mau"),
        ("quan hệ giữa vật", "quan_he"),
        ("đếm số lượng", "dem"),
        ("vị trí trái/phải", "vi_tri"),
        ("góc máy", "goc_may"),
    )
    lines = [
        f"{label}  ({report['n']} keyframe)",
        f"  {'năng lực':<22}{'caption có':>12}{'49 đề có':>12}",
        f"  {'-' * 46}",
    ]
    lines += [
        f"  {name:<22}{report[key]:>11.1%}{QUERY_DEMAND[key]:>11.1%}"
        for name, key in rows
    ]
    lines += [
        "",
        f"  tiếng Việt          {report['tieng_viet']:>11.1%}",
        f"  mở đầu bị cấm       {report['mo_dau_cam']:>11.1%}   (càng thấp càng tốt)",
        f"  caption (token)     {report['caption_token_trung_vi']:>11.0f}   trung vị",
        f"  có chữ đọc được     {report['co_chu']:>11.1%}",
        f"  [TEXT] nhiều dòng   {report['ocr_nhieu_dong']:>11.1%}   "
        f"(trung vị {report['ocr_dong_trung_vi']:.0f} dòng)",
        "",
        "  Hàng `đếm` là hàng ÍT TIN ĐƯỢC NHẤT: `một` trong tiếng Việt vừa là số",
        "  đếm vừa là mạo từ, nên 93,9% ở cột đề là con số thổi phồng (người đọc",
        "  tay đếm được 51%). Đọc nó như một sàn, đừng đọc như một mục tiêu.",
    ]
    return "\n".join(lines)
