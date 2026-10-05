"""Chuỗi đi TÌM khác chuỗi câu HỎI — và với câu QA, khác rất nhiều.

SigLIP huấn luyện trên **caption**, tức là *mô tả một tấm ảnh*. Câu đề QA của BTC
lại là một **câu hỏi**. Hai thứ khác thể loại, nên đưa nguyên văn câu hỏi vào tháp
text là đưa sai kiểu đầu vào.

Chỗ này lộ ra muộn vì hai đường dùng hai chuỗi khác nhau: `EvalQuery` có
`retrieval_text` viết tay (**48/48** câu QA dev và **15/15** heldout đều có, và
đều khác `text`), còn `PackagedQuery` — thứ đường nộp thật dùng — **không có**
trường đó, `searches()` trả thẳng `self.text`. Nên mọi con số QA từng công bố đều
đo trên một chuỗi mà lúc thi không ai dựng được.

Module này giữ các phép biến câu hỏi thành chuỗi tìm mà **không** cần người viết
tay từng câu.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Ngắt câu ở `.`, `?`, `!` rồi khoảng trắng. Không dùng thư viện tách câu: câu đề
#: BTC là một hai câu ngắn, và một phụ thuộc mới cho việc này là không xứng.
_SENTENCE = re.compile(r"(?<=[.?!])\s+")


def strip_questions(text: str) -> str:
    """Bỏ mọi câu kết thúc bằng `?`. Trả **nguyên văn** nếu bỏ xong thì rỗng.

    Câu đề QA gần như luôn có dạng *"<mô tả cảnh>. <câu hỏi>?"*. Phần mô tả là
    thứ tháp thị giác tìm được; phần hỏi thì không, và nó còn làm loãng vector.

    Đo trên `query-p1-9-qa` (đợt 1, đáp án `L21_V003#026220`): bỏ câu hỏi kéo hạng
    SigLIP từ 34 lên **12**; giữ nguyên cả câu mà sửa mỗi bản dịch thì chỉ tới 34.
    Hai nguyên nhân cộng dồn chứ không thay nhau.

    Trả nguyên văn khi cắt hết là có chủ ý: một câu đề toàn dấu hỏi không có phần
    mô tả nào để giữ, và tìm bằng chuỗi rỗng cho ra 0 ứng viên — tệ hơn hẳn tìm
    bằng chính câu hỏi. Nhiều câu QA dev thuộc loại này (*"Theo slide bài giảng
    Vật lí, hạt nhân nguyên tử được cấu tạo từ hai loại hạt nào?"* — một câu, và
    nó là câu hỏi), nên luật này **không chạm** tới chúng, và đó là hành vi đúng.
    """
    parts = [p.strip() for p in _SENTENCE.split(text.strip()) if p.strip()]
    kept = [p for p in parts if not p.endswith("?")]
    return " ".join(kept) if kept else text.strip()


def has_description(text: str) -> bool:
    """Câu đề có phần mô tả tách rời phần hỏi không?

    Giao diện cần biết để nói cho người vận hành: nút "bỏ câu hỏi" không đổi gì
    trên một câu đề chỉ có đúng một câu hỏi, và một nút bấm xong không đổi gì
    trông y hệt một nút hỏng.
    """
    return strip_questions(text) != text.strip()


@dataclass(frozen=True)
class Rewrite:
    """Câu hỏi đã được LLM viết lại thành chuỗi đi tìm.

    `ok=False` nghĩa là **chưa viết lại được** — chưa có khoá, API hỏng, hoặc
    model trả về thứ không đọc nổi. Lúc đó `vi`/`en` vẫn dùng được (đường luật
    tự cắt), nhưng giao diện phải nói rõ là nó đang cầm bản dự phòng.
    """

    vi: str
    en: str | None = None
    ok: bool = False
    reason: str | None = None


def quest_rewrite(
    question: str, decompose: Callable[[str], dict] | None = None
) -> Rewrite:
    """Viết lại một câu hỏi QA thành chuỗi tìm, qua `quest.quest_decompose`.

    Vì sao cần một lớp bọc thay vì gọi thẳng
    -----------------------------------------
    `quest_decompose` **rơi về chính câu hỏi gốc mà không báo gì** khi API hỏng.
    Một nút bấm xong không đổi gì trông y hệt một nút chạy rồi mà vô ích, nên
    lớp này so kết quả với đầu vào và trả `ok=False` kèm lý do.

    Phân vai hai chuỗi là điểm chính, không phải chi tiết: `search_hint` đã là
    **tiếng Anh** nên đi thẳng vào tháp SigLIP, bỏ qua envit5 — mà envit5 chính
    là một trong hai nguyên nhân làm `query-p1-9-qa` rơi khỏi pool. Còn
    `ocr_keywords` là tiếng Việt, dành cho các arm BM25.

    Đo trên 47 câu QA dev: pool 500 phủ **38/47** so với 36/47 của nguyên văn,
    hạng trung vị **14** so với 18 — nhưng `R@5` **giảm** 17 → 15. Nó cứu các câu
    hỏng nặng và làm hỏng vài câu đang tốt, nên đây là cần gạt **từng câu**,
    không phải mặc định cho cả gói.

    `decompose` tiêm được để test không cần mạng.
    """
    if decompose is None:
        from mfe.online.quest import quest_decompose  # noqa: PLC0415

        decompose = quest_decompose

    question = question.strip()
    try:
        parts = decompose(question)
    except Exception as err:  # noqa: BLE001 - chay trong vong ve giao dien
        logger.warning("quest_rewrite: %s: %s", type(err).__name__, err)
        return Rewrite(
            vi=strip_questions(question),
            reason=f"{type(err).__name__}: {str(err)[:160]}",
        )

    hint = str(parts.get("search_hint") or "").strip()
    keywords = str(parts.get("ocr_keywords") or "").strip()
    if not hint or hint == question:
        return Rewrite(
            vi=strip_questions(question),
            reason="quest tra ve chinh cau hoi — API hong hoac thieu khoa",
        )
    return Rewrite(vi=keywords or strip_questions(question), en=hint, ok=True)


def quest_apply(
    text: str,
    query_type: str,
    *,
    rewrite: Callable[[str], Rewrite] | None = None,
    expand: Callable[[str], list] | None = None,
) -> Rewrite:
    """QUEST cho MỘT câu, chọn phép đúng theo **dạng đề**.

    Ba việc của `mfe.online.quest` khác nhau, và dùng nhầm việc thì hại chứ
    không phải vô ích:

    * **QA** -> `quest_decompose` (qua `quest_rewrite`). Câu hỏi được tách thành
      `search_hint` tiếng Anh — đi thẳng vào tháp SigLIP, **bỏ qua envit5**, mà
      envit5 chính là một trong hai nguyên nhân làm `query-p1-9-qa` rơi khỏi
      pool 500 — cộng `ocr_keywords` tiếng Việt cho các arm BM25.
    * **KIS / TRAKE** -> `quest_expand`, và kết quả được **NỐI vào cuối** chứ
      không thay thế. Câu KIS đã là mô tả thị giác, tức đã đúng thể loại mà
      SigLIP ăn; đem `quest_decompose` vào đây là viết lại một câu vốn không cần
      viết lại. Cái KIS thiếu là **tên** của thứ đề cố ý không gọi tên
      (*"phim của đạo diễn Steven Spielberg 1975"* -> *"cá mập"*), và đó đúng là
      việc của `quest_expand`.

    **Rỗng là câu trả lời ĐÚNG, không phải lỗi.** 42/49 đề thật không cần thêm
    gì, nên `quest_expand` trả `[]` là bình thường; khi đó hàm này trả
    `ok=False` kèm lý do để giao diện nói ra, thay vì để người vận hành nhìn một
    nút bấm xong không đổi gì và đoán.

    Không bao giờ ném: nó chạy trong vòng vẽ Streamlit và giữa một lượt nộp.
    """
    text = (text or "").strip()
    if not text:
        return Rewrite(vi=text, reason="cau rong")

    if query_type == "QA":
        if rewrite is None:
            rewrite = quest_rewrite
        return rewrite(text)

    if expand is None:
        from mfe.online.quest import quest_expand  # noqa: PLC0415

        expand = quest_expand
    try:
        terms = expand(text)
    except Exception as err:  # noqa: BLE001 - mo rong hong thi van tim duoc
        logger.warning("quest_apply: %s: %s", type(err).__name__, err)
        return Rewrite(vi=text, reason=f"{type(err).__name__}: {str(err)[:160]}")

    if not terms:
        return Rewrite(vi=text, reason="quest khong goi ten them gi (42/49 de nhu vay)")

    from mfe.online.quest import expanded_queries  # noqa: PLC0415

    vi, _ = expanded_queries(text, None, terms)
    named = ", ".join(t.vi or t.en for t in terms)
    return Rewrite(vi=vi, ok=True, reason=f"goi ten: {named}")
