"""QUEST: Question Evaluation & Structuring (Tầng 0).

Mô đun phục hồi kỹ thuật PBR (Personalize Before Retrieve) gốc của SOTA_V2_AIC.
Sử dụng LLM (Llama-3.1-70B hoặc Gemini) để:
1. KIS Mode (QuestKIS): Dịch và phóng to câu truy vấn ngắn thành mô tả hình ảnh chi tiết.
2. QA Mode (QuestVQA): Tách câu hỏi VQA phức tạp thành search_hint, ocr_keywords, logic_question.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

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
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                return response.text.strip()
            except Exception as e:
                logger.warning(f"QUEST Gemini fallback failed: {e}")
                
        raise RuntimeError("QUEST: Mọi Provider đều không khả dụng hoặc lỗi.")


class QuestKIS:
    """Backend đóng giả TranslationBackend, dùng cho dịch và mở rộng truy vấn KIS."""
    
    model_id: str = "quest-llama-3.1-70b-qe"
    
    def __init__(self):
        self.client = QuestClient()
        
    def translate(self, text: str) -> str:
        prompt = QUEST_KIS_PROMPT.replace("{query}", text)
        result = self.client.generate(prompt, max_tokens=150)
        return result


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
