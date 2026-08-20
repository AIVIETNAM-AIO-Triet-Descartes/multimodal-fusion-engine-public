"""Bộ đọc câu trả lời Q&A chạy trên Gemini API.

    PDF §2.1.2 · `mfe.online.answering.AnswerReader`

BTC cho phép truy cập mạng và gọi API trong vòng sơ tuyển, nên đường này dùng
được. Nó cắm vào `answering.propose` và không cái nào trong `answering.py` biết
gì về Gemini — đổi nhà cung cấp là viết một lớp khác ở file này.

Vì sao `gemini-2.5-flash`
--------------------------
Cùng model `mfe.offline.stages.build_gemini_generate` đã dùng cho ReCap: một phụ
thuộc (`google-genai`), một khoá API, một hành vi phải hiểu. Flash chứ không Pro
vì đây là vòng lặp **tương tác** — người vận hành đợi câu trả lời hiện ra rồi mới
duyệt, nên độ trễ tính vào đúng chỗ đắt nhất (TTCA, `buoi-2.md` §2). Và vì có
người duyệt, chênh lệch chất lượng giữa Flash và Pro rẻ hơn chênh lệch độ trễ.

Khối lượng **không** phải vấn đề ở đây, khác hẳn ReCap. Lead từng bác Gemini cho
ReCap vì 116.028 shot vượt hạn mức free tier; Q&A gọi **một lần mỗi truy vấn**,
một vòng thi vài chục truy vấn. Cùng một model, hai bài toán khác nhau về bậc.

Ba ràng buộc đặt vào prompt, và cả ba đều có lý do đã trả giá
---------------------------------------------------------------
**1. Trả lời NGẮN.** BTC so `answer` theo ngữ nghĩa (§2.1.2) và ví dụ của họ là
`"5"` hoặc `"Năm"`. Một câu giải thích dài chứa đáp án đúng vẫn có thể trượt phép
so; một từ thì không.

**2. Chỉ dùng ảnh và bằng chứng được đưa.** Không có ràng buộc này thì model trả
lời từ kiến thức chung — nghe hợp lý, sai video, và người vận hành không phân
biệt được.

**3. Có đường thoát `KHONG_DOC_DUOC`.** Ngược hẳn với bài học ở `_QWEN_PROMPT`,
nơi bỏ đường thoát `None` là thứ sửa được tầng OCR: ở đó model bỏ cuộc trên
305/307 ảnh THẬT SỰ có chữ, và một chuỗi rác trong index chỉ là nhiễu. Ở đây
người vận hành **tin** câu trả lời và nộp nó, nên một câu bịa ra đắt hơn một ô
trống. Số lần bỏ cuộc được đếm để thấy được nếu nó lạm dụng.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mfe.online.answering import Answer, Evidence

logger = logging.getLogger(__name__)

#: Token model phát ra khi không đọc ra được. Ngắn, viết hoa, không dấu — để nó
#: không trùng với một câu trả lời thật.
BAIL = "KHONG_DOC_DUOC"

PROMPT = """Bạn đang trả lời một câu hỏi của cuộc thi truy hồi video.

Bạn nhận được: một hoặc vài khung hình ứng viên, và toàn bộ văn bản trích được \
từ chính những khung hình đó (OCR, phụ đề, nhãn vật thể, mô tả tự động).

QUY TẮC:
1. Chỉ trả lời dựa trên khung hình và văn bản được cung cấp. Không dùng kiến \
thức bên ngoài, không suy đoán.
2. Trả lời NGẮN NHẤT có thể — một từ, một con số, hoặc một cụm danh từ. Không \
giải thích, không viết thành câu.
3. Trả lời bằng đúng ngôn ngữ của câu hỏi.
4. Nếu không xác định được từ những gì được cung cấp, trả lời đúng một từ: {bail}

CÂU HỎI: {question}

VĂN BẢN TRÍCH ĐƯỢC:
{evidence}

TRẢ LỜI:"""


@dataclass
class GeminiAnswerReader:
    """`AnswerReader` chạy trên Gemini. Trả `None` khi model không đọc ra được.

    `images` là số khung hình đầu bảng gửi kèm. Nhiều hơn 1 là một phép phòng hờ
    có tính toán: xếp hạng 1 sai vẫn còn hạng 2, 3 — nhưng quá nhiều ảnh thì câu
    hỏi bị pha loãng và model bắt đầu trả lời về khung hình khác.
    """

    model: str = "gemini-2.5-flash"
    images: int = 3
    api_key_env: str = "GEMINI_API_KEY"
    name: str = "gemini"
    #: Đếm số lần model bỏ cuộc. Không phải để chặn — để thấy nếu nó lạm dụng.
    bails: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)
    _client: object | None = field(default=None, init=False, repr=False)

    def client(self):
        if self._client is None:
            from google import genai  # noqa: PLC0415

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"thieu {self.api_key_env} — dat bang bien moi truong, "
                    "khong viet vao code hay config (§9.3)"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def __call__(
        self,
        question: str,
        evidence: Sequence[Evidence],
        image_paths: Sequence[Path] = (),
    ) -> Answer | None:
        from google import genai  # noqa: PLC0415

        used = list(evidence)[: max(1, self.images)]
        blob = "\n\n".join(
            f"--- {item.keyframe_id} ---\n{item.as_prompt() or '(khong co van ban)'}"
            for item in used
        ) or "(khong co van ban nao)"

        parts: list[object] = []
        for path in list(image_paths)[: self.images]:
            try:
                parts.append(
                    genai.types.Part.from_bytes(
                        data=Path(path).read_bytes(), mime_type="image/jpeg"
                    )
                )
            except Exception:
                # Thiếu một ảnh thì đọc bằng phần còn lại, không bỏ cả câu.
                logger.warning("khong doc duoc anh %s", path)
        parts.append(PROMPT.format(bail=BAIL, question=question.strip(), evidence=blob))

        self.calls += 1
        response = self.client().models.generate_content(model=self.model, contents=parts)
        text = (getattr(response, "text", "") or "").strip()
        return self._parse(text, used)

    def _parse(self, text: str, used: Sequence[Evidence]) -> Answer | None:
        # Model đôi khi thêm dấu câu hoặc xuống dòng quanh đáp án ngắn.
        cleaned = text.strip().strip("`").strip()
        first = cleaned.splitlines()[0].strip() if cleaned else ""
        if not first or BAIL in cleaned.upper():
            self.bails += 1
            logger.info("gemini khong doc ra duoc (%d/%d lan)", self.bails, self.calls)
            return None
        anchor = used[0] if used else None
        return Answer(
            text=first,
            keyframe_id=anchor.keyframe_id if anchor else "",
            sources=tuple(source for item in used for source, _ in item.ordered()),
            reader=f"{self.name}:{self.model}",
        )
