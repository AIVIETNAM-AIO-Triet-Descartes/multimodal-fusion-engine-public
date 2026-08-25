"""QUEST: Question Evaluation & Structuring (Tầng 0).

Mô đun phục hồi kỹ thuật PBR (Personalize Before Retrieve) gốc của SOTA_V2_AIC.
Ba việc, và chúng **khác nhau** — trộn vào một prompt là hỏng cả ba:

1. `QuestKIS` — *viết lại theo văn phong*. Dịch câu truy vấn sang tiếng Anh mô tả
   thị giác. Luật số 1 của nó là **cấm thêm** thứ không có trong câu.
2. `quest_decompose` — *tách câu hỏi QA* thành `search_hint` / `ocr_keywords` /
   `logic_question`.
3. `quest_expand` — *gọi tên thứ đề cố ý không nêu tên*. Đây là việc duy nhất
   được phép **thêm**, và nó là việc mà (1) bị cấm làm.

Vì sao (3) phải là một hàm riêng
---------------------------------
Đọc 49 đề chính thức: **7 câu hỏi về thứ không nằm trong câu hỏi**
(`docs/triet/diem-yeu-pipeline-theo-49-de.md` §2.4). *"phim của đạo diễn Steven
Spielberg 1975"* phải thành *"cá mập"*; *"Đại học ở thành phố Lausanne"* phải
thành *"EPFL"*. Không phép truy hồi nào giải được, vì từ khoá cần tìm không có
trong câu.

Nhưng 42 câu còn lại **không** cần thêm gì, và một từ bịa thêm vào truy vấn là
recall mất đi. Nên `quest_expand` phải trả về **rỗng** được, và prompt của nó nói
thẳng điều đó. Nhét luật này vào `QUEST_KIS_PROMPT` sẽ đổi hành vi của cả 49 câu
để phục vụ 7 câu.

Nhà cung cấp — đã kiểm ENDPOINT, không chỉ đọc code
----------------------------------------------------
Đo ngày 24-08-2026 trên chính máy này:

* `gemini-2.5-flash` (giá trị cũ trong file này) trả **404** *"This model is not
  found"* — dù nó vẫn nằm trong `models.list()`. Liệt kê không có nghĩa là gọi được.
* `gemini-3.6-flash` **200**, 5,8 s. `gemini-flash-lite-latest` **200**, 1,0 s.
* Hai đường OpenRouter và NVIDIA cần `openai`, và gói đó **không có trong venv** —
  `QuestClient.__init__` nuốt `ImportError` nên cả hai âm thầm không tồn tại.

Nghĩa là trước hôm nay `quest` không chạy được một lượt nào trên máy này, và
`generate()` sẽ ném "Mọi Provider đều không khả dụng" ở lượt gọi đầu tiên.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Model Gemini dùng cho mọi lượt QUEST. **Không** phải `gemini-2.5-flash` — xem
#: docstring module: nó trả 404 cho khoá của nhóm. `-lite-latest` nhanh gấp ~6
#: lần nhưng đây là bước viết lại truy vấn, chạy một lần mỗi truy vấn, nên chất
#: lượng đáng giá hơn 5 giây.
GEMINI_MODEL = "gemini-3.6-flash"

QUEST_EXPAND_PROMPT = """You are a query analyst for a Vietnamese news-video search engine.

The user's query sometimes refers to something WITHOUT NAMING IT — a film, an
animal species, a place, an institution, an event. The search engine matches
words, so an unnamed thing can never be found.

Your ONLY job: name what the query refers to but does not say.

RULES:
1. If the query names everything it refers to, return an EMPTY list. This is the
   common case. Returning nothing is a correct and expected answer.
2. Only add a term when you are confident. A wrong term costs the user the query.
3. Give each term in BOTH Vietnamese and English, as the search engine has arms
   in both languages.
4. Do not restate what the query already says. Do not add adjectives, colours,
   scenery or mood.

Return ONLY this JSON, no markdown, no explanation:
{"terms": [{"vi": "...", "en": "...", "why": "..."}]}

EXAMPLE A (something is unnamed):
Query: "Đoạn clip về 1 thị trấn ven biển thu hút du khách nhờ 1 loài động vật biển
nguy hiểm. Loài này nổi tiếng trong 1 bộ phim của đạo diễn Steven Spielberg sản
xuất vào năm 1975"
{"terms": [{"vi": "cá mập", "en": "shark", "why": "phim Spielberg 1975 la Jaws (Ham ca map)"},
           {"vi": "cá mập trắng", "en": "great white shark", "why": "loai trong phim"}]}

EXAMPLE B (nothing is unnamed):
Query: "tay đua mặc áo vàng giơ hai tay lên ăn mừng sau khi về đích"
{"terms": []}

Query: "{query}"
"""

QUEST_VQA_PROMPT = """Bạn là bộ phân tích câu hỏi cho hệ thống tìm kiếm video thời sự Việt Nam.

Nhiệm vụ: Cho một câu hỏi VQA (Visual Question Answering), hãy tách thành 3 phần JSON:

1. "search_hint": Viết bằng TIẾNG ANH. Mô tả chi tiết hình ảnh/cảnh vật/bối cảnh 
   mà câu hỏi đang đề cập đến. Phải đủ chi tiết để một hệ thống tìm kiếm hình ảnh 
   (không hiểu ngôn ngữ tự nhiên) có thể tìm đúng frame. Bao gồm: đối tượng chính, 
   bối cảnh (trong nhà/ngoài trời, studio tin tức, v.v.), màu sắc nổi bật, text overlay 
   trên màn hình nếu được đề cập. Độ dài: 30-60 từ tiếng Anh. 
   KHÔNG chứa câu hỏi. CHỈ mô tả thị giác.

2. "ocr_keywords": Viết bằng TIẾNG VIỆT. Liệt kê CÁC TỪ KHÓA ĐẶC TẢ có khả năng cao hiển thị chính xác trên màn hình TV (ví dụ: tên riêng, địa danh, con số cụ thể, tiêu đề trích dẫn).
   [LUẬT THÉP CẤM KỴ - ĐỌC KỸ]:
   - Đối với các từ chỉ thời gian (giờ, phút, giây, ngày, tháng, năm) hoặc đại từ nghi vấn (ai, đâu, mấy, bao nhiêu): CHỈ ĐƯỢC phép trích xuất nếu chúng nằm trong một CỤM TỪ TRÍCH DẪN RÕ RÀNG (VD: dòng chữ 'bản tin 12 giờ').
   - NẾU chúng đóng vai trò là từ để hỏi trong câu (VD: 'là mấy giờ?', 'ngày nào?'), TUYỆT ĐỐI KHÔNG trích xuất.
   - GHI NHỚ: Nếu lấy sai từ khóa, hệ thống sẽ sụp đổ. Nếu bạn không chắc chắn 100% từ đó có CHỮ VIẾT trên màn hình, hãy để chuỗi rỗng "".

3. "logic_question": Giữ nguyên TIẾNG VIỆT. Đây là câu hỏi thực sự cần trả lời, 
   được diễn đạt lại ngắn gọn và rõ ràng. Bỏ phần mô tả gợi ý, chỉ giữ lại phần hỏi.

VÍ DỤ 1 (KHÔNG trích xuất từ để hỏi về thời gian):
Input: "Màn hình đang chiếu cảnh một chú rùa lớn đang bơi lội trong làn nước xanh. Ở phía trên cùng bên phải màn hình có logo của kênh truyền hình nào và hiển thị mốc thời gian là mấy giờ?"
Output:
{
  "search_hint": "A TV screen displaying a large turtle swimming in a greenish-blue body of water. The top right corner of the screen shows a TV channel logo and a time stamp.",
  "ocr_keywords": "",
  "logic_question": "Logo kênh truyền hình nào xuất hiện trên màn hình và mốc thời gian là mấy giờ?"
}

VÍ DỤ 2 (PHẢI trích xuất thời gian vì nó là text trích dẫn):
Input: "Biên tập viên nam mặc áo vest đang đứng trước phông nền xanh. Dưới màn hình có chạy dòng chữ 'Hưởng ứng chiến dịch Giờ Trái Đất'. Theo lời dẫn, chiến dịch này sẽ bắt đầu từ mấy giờ?"
Output:
{
  "search_hint": "A male news anchor wearing a suit standing in front of a blue background. There is a scrolling text banner at the bottom of the screen mentioning 'Giờ Trái Đất'.",
  "ocr_keywords": "Hưởng ứng chiến dịch Giờ Trái Đất",
  "logic_question": "Chiến dịch Giờ Trái Đất sẽ bắt đầu từ mấy giờ?"
}

Bây giờ, phân tích câu hỏi sau. Trả lời ĐÚNG format JSON, không giải thích gì thêm:
"{question}"
"""

QUEST_KIS_PROMPT = """You are an expert query rewriting system (LLM-QE) for a video retrieval engine.
Your task is to translate and expand the user's short query into a highly descriptive English visual statement that exactly matches the tone, style, and vocabulary of our dataset's dense captions.

Here are examples of how our system describes images (Notice the literal, physical descriptions without assumptions):
Example 1: "A construction site with a highway in the background. The highway has multiple lanes and is surrounded by greenery. There are several workers and construction vehicles on the site..."
Example 2: "Two news anchors are standing behind a desk in a studio with a cityscape background. The man is wearing a blue shirt and tie..."

Now, rewrite the following user query in that exact same visual description style. 
CRITICAL RULES:
1. DO NOT add any objects, colors, or details that are not explicitly mentioned in the User Query.
2. DO NOT guess the time of day or location unless specified.
3. Do not include conversational text or quotes. Only output the rewritten English text (max 50 words).

User Query: {query}"""


class QuestClient:
    """Xử lý gọi API chung cho NVIDIA NIM, OpenRouter và Gemini."""
    
    def __init__(self):
        self.nvidia_key = os.environ.get("NVIDIA_API_KEY")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        
        self.nvidia_client = None
        self.openrouter_client = None
        self.gemini_client = None
        
        try:
            from openai import OpenAI
            if self.nvidia_key:
                self.nvidia_client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=self.nvidia_key,
                    timeout=30
                )
            if self.openrouter_key:
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_key,
                    timeout=30
                )
        except ImportError:
            pass
            
        try:
            if self.gemini_key:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_key)
        except ImportError:
            pass

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Gửi prompt tới LLM. Thử OpenRouter, NIM, rồi fallback sang Gemini."""
        # Thử OpenRouter / NVIDIA NIM (Llama 3.1 / 3.3)
        clients = []
        if self.openrouter_client:
            clients.append((self.openrouter_client, "google/gemini-2.0-flash-lite-preview-02-05:free"))
            clients.append((self.openrouter_client, "meta-llama/llama-3.3-70b-instruct:free"))
        if self.nvidia_client:
            clients.append((self.nvidia_client, "meta/llama-3.1-70b-instruct"))
            
        for client, model in clients:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"QUEST LLM failed with {model}: {e}")
                
        # Thử Gemini SDK
        if self.gemini_client:
            try:
                from google.genai import types
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                return (response.text or "").strip()
            except Exception as e:
                logger.warning("QUEST Gemini (%s) failed: %s: %s", GEMINI_MODEL, type(e).__name__, e)

        raise RuntimeError(
            "QUEST: khong provider nao chay duoc. "
            f"openai={'co' if self.openrouter_client or self.nvidia_client else 'KHONG cai'}, "
            f"gemini={'co khoa' if self.gemini_key else 'KHONG co GEMINI_API_KEY'}"
        )

    def available(self) -> tuple[bool, str | None]:
        """Có đường nào gọi được không, và nếu không thì vì sao.

        Hỏi trước khi gọi, vì `generate()` chỉ ném ở lượt gọi đầu — nghĩa là một
        pipeline bật quest mà thiếu khoá sẽ chạy tới đúng truy vấn đầu tiên rồi
        mới đổ, sau khi đã nạp xong cả kho.
        """
        if self.openrouter_client or self.nvidia_client or self.gemini_client:
            return True, None
        if not (self.openrouter_key or self.nvidia_key or self.gemini_key):
            return False, "khong co OPENROUTER_API_KEY / NVIDIA_API_KEY / GEMINI_API_KEY"
        return False, "co khoa nhung thieu SDK (pip install openai / google-genai)"


class QuestKIS:
    """Backend đóng giả TranslationBackend, dùng cho dịch và mở rộng truy vấn KIS."""
    
    model_id: str = "quest-llama-3.1-70b-qe"
    
    def __init__(self):
        self.client = QuestClient()
        
    def translate(self, text: str) -> str:
        prompt = QUEST_KIS_PROMPT.replace("{query}", text)
        result = self.client.generate(prompt, max_tokens=150)
        return result


@dataclass(frozen=True)
class ExpansionTerm:
    """Một thứ đề nhắc tới mà không gọi tên, kèm lý do LLM đưa ra."""

    vi: str
    en: str
    why: str = ""


def quest_expand(query: str, client: QuestClient | None = None) -> list[ExpansionTerm]:
    """Gọi tên thứ câu truy vấn nhắc tới mà không nêu tên. Rỗng là câu trả lời ĐÚNG.

    Trả `[]` trong ba trường hợp, và cả ba đều **không** phải lỗi: câu đã nêu đủ
    tên, LLM không chạy được, hoặc trả về thứ không đọc được. Người gọi vẫn tìm
    bằng nguyên câu — mở rộng là phần **cộng thêm**, không phải phần thay thế.

    Đó là lý do không có `fallback` trả lại chính câu hỏi ở đây. `quest_decompose`
    có, và nó đúng cho việc của nó (tách một câu hỏi thì luôn còn lại câu hỏi);
    với mở rộng thì "trả lại chính câu" nghĩa là *nhân đôi* mọi token của truy vấn
    và làm lệch BM25 mà không thêm một chữ mới nào.
    """
    client = client or QuestClient()
    ok, reason = client.available()
    if not ok:
        logger.warning("quest_expand: khong mo rong duoc — %s", reason)
        return []

    try:
        text = client.generate(QUEST_EXPAND_PROMPT.replace("{query}", query), max_tokens=400)
    except Exception as err:  # noqa: BLE001 — không có mở rộng thì vẫn tìm được
        logger.warning("quest_expand: %s: %s", type(err).__name__, err)
        return []

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("quest_expand: khong doc duoc JSON tu %r", text[:120])
        return []
    try:
        payload = json.loads(match.group(0))
    except ValueError as err:
        logger.warning("quest_expand: JSON hong (%s): %r", err, text[:120])
        return []

    terms: list[ExpansionTerm] = []
    for item in payload.get("terms") or []:
        if not isinstance(item, dict):
            continue
        vi = str(item.get("vi") or "").strip()
        en = str(item.get("en") or "").strip()
        if not (vi or en):
            continue
        terms.append(ExpansionTerm(vi=vi, en=en, why=str(item.get("why") or "").strip()))
    return terms


def expanded_queries(
    query_vi: str, query_en: str | None, terms: list[ExpansionTerm]
) -> tuple[str, str | None]:
    """Ghép các từ đã gọi tên vào **cuối** truy vấn, mỗi ngôn ngữ một bản.

    Nối vào cuối chứ không thay thế: câu gốc vẫn mang phần mô tả thị giác mà
    `siglip` sống bằng, còn từ mới chỉ thêm cơ hội cho arm văn bản. Thay hẳn thì
    *"phim Spielberg 1975"* thành *"cá mập"* — mất luôn "thị trấn ven biển",
    "du khách", mọi thứ đã đúng.
    """
    vi_extra = " ".join(t.vi for t in terms if t.vi)
    en_extra = " ".join(t.en for t in terms if t.en)
    out_vi = f"{query_vi} {vi_extra}".strip() if vi_extra else query_vi
    out_en = f"{query_en} {en_extra}".strip() if (query_en and en_extra) else query_en
    return out_vi, out_en


def quest_decompose(question: str) -> dict[str, str]:
    """Phân rã câu hỏi VQA thành search_hint, ocr_keywords, logic_question."""
    prompt = QUEST_VQA_PROMPT.replace("{question}", question)
    client = QuestClient()
    
    try:
        text = client.generate(prompt, max_tokens=500)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            if all(k in result for k in ["search_hint", "ocr_keywords", "logic_question"]):
                return result
    except Exception as e:
        logger.error(f"QUEST Decompose failed: {e}")
        
    # Fallback
    return {
        "search_hint": question,
        "ocr_keywords": question,
        "logic_question": question
    }
