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

import re

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
