"""Đếm xem caption sinh ra có nêu MÀU / SỐ LƯỢNG / VỊ TRÍ không.

Vì sao đo cái này chứ không đo chồng lấn từ vựng với bản của Bảo
-----------------------------------------------------------------
Biến thể `aic` viết tiếng Việt, biến thể `bao` viết tiếng Anh. Chồng lấn từ vựng
giữa hai bên gần bằng **không** dù cả hai đều mô tả đúng cùng một khung hình —
nên con số đó không nói được gì về chất lượng, chỉ nói rằng hai prompt khác nhau.

Thứ đo được, và thứ prompt mới sinh ra để làm, là **tỉ lệ caption nêu ra từng
năng lực mà 49 đề thật đòi** (`docs/triet/diem-yeu-pipeline-theo-49-de.md` §1):

    màu gắn vào vật  63%  ·  đếm số lượng  51%  ·  vị trí  45%  ·  góc máy  16%

Một prompt bảo model nêu màu mà chỉ 20% caption có từ chỉ màu thì prompt đó
không hoạt động, và điều đó **đọc được ngay trên 307 ảnh probe** — không phải đợi
tới lúc đo Final Score trên bộ dev.

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
COLOUR_WORDS = (
    "đỏ", "cam", "vàng", "xanh", "lam", "lục", "tím", "hồng", "nâu",
    "đen", "trắng", "xám", "bạc", "kem", "ngà", "sẫm",
)

#: Số đếm viết bằng CHỮ. Đề viết bằng chữ ("ba người", "hơn 5 người"), nên chỉ mục
#: cũng phải có chữ — và chữ số cũng được tính riêng ở dưới.
COUNT_WORDS = (
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "nhiều", "vài", "đôi",
)

POSITION_WORDS = (
    "bên trái", "bên phải", "ở giữa", "chính giữa", "phía sau", "phía trước",
    "phía trên", "phía dưới", "góc trên", "góc dưới", "trái", "phải", "giữa",
)

CAMERA_WORDS = (
    "cận cảnh", "toàn cảnh", "trung cảnh", "từ trên cao", "từ dưới lên",
    "quay chậm", "flycam", "lia máy", "zoom",
)

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
        "mo_dau_cam": sum(map(has_banned_opener, captions)) / total,
        "tieng_viet": sum(map(looks_vietnamese, captions)) / total,
        "caption_token_trung_vi": statistics.median(
            len(_TOKEN.findall(c)) for c in captions
        ),
        "co_chu": sum(1 for c in line_counts if c) / total,
        "ocr_dong_trung_vi": statistics.median(with_text) if with_text else 0,
        "ocr_nhieu_dong": sum(1 for c in line_counts if c > 1) / total,
    }


def format_audit(report: dict, label: str = "") -> str:
    """Bảng người đọc được, kèm mốc đề thật để so."""
    if not report.get("n"):
        return f"{label}: khong co du lieu"
    lines = [
        f"{label}  ({report['n']} keyframe)",
        f"  {'năng lực':<22}{'caption có':>12}{'đề thật đòi':>14}",
        f"  {'-' * 48}",
        f"  {'màu gắn vào vật':<22}{report['mau']:>11.1%}{'63%':>14}",
        f"  {'đếm số lượng':<22}{report['dem']:>11.1%}{'51%':>14}",
        f"  {'vị trí trái/phải':<22}{report['vi_tri']:>11.1%}{'45%':>14}",
        f"  {'góc máy':<22}{report['goc_may']:>11.1%}{'16%':>14}",
        "",
        f"  tiếng Việt          {report['tieng_viet']:>11.1%}",
        f"  mở đầu bị cấm       {report['mo_dau_cam']:>11.1%}   (càng thấp càng tốt)",
        f"  caption (token)     {report['caption_token_trung_vi']:>11.0f}   trung vị",
        f"  có chữ đọc được     {report['co_chu']:>11.1%}",
        f"  [TEXT] nhiều dòng   {report['ocr_nhieu_dong']:>11.1%}   "
        f"(trung vị {report['ocr_dong_trung_vi']:.0f} dòng)",
    ]
    return "\n".join(lines)
