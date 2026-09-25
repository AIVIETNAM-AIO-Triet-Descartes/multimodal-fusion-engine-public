"""Cổng API tương thích OpenAI của Đạt (`llm-gateway.invalid`) — dùng chung cho mọi tầng gọi API.

Vì sao có file này thay vì mỗi tầng tự gọi
-------------------------------------------
Ba tầng cần LLM — `translate`, `quest`, `rerank_tiers.VlmTier` — và trước đây mỗi
tầng nối một nhà cung cấp riêng: NVIDIA NIM, OpenRouter, Gemini trực tiếp. Mỗi
nhà cung cấp là một hạn mức riêng, một dạng lỗi riêng, một chỗ hỏng im lặng riêng.
`gemini-3.6-flash` free tier cho **20 request MỖI NGÀY** và đó là lý do thật của
hai lượt đo `vlm` dở dang (§4 CLAUDE.md).

Cổng của Đạt xoay vòng usage qua nhiều tài khoản nên trần cao hơn hẳn. Đo 20-09
trên 800 lượt gọi liên tiếp: **0 lỗi**.

Ba cái bẫy đã trả giá khi nối cổng này — đừng gỡ ra
----------------------------------------------------
**1. KHÔNG có `User-Agent` thì Cloudflare chặn 403 `error code: 1010`.** Không
phải lỗi khoá. `urllib` trần không tự đặt UA nên bị chặn; `curl` thì qua. Đây là
lỗi theo *chữ ký client*, và nó không nói gì về API.

**2. `401 "Missing or invalid API key"` thường là khoá KHÔNG được nạp.**
`load_dotenv()` tìm `.env` đi lên từ **thư mục của file gọi**, không phải cwd —
một script ngoài cây repo sẽ nhận `None` và gửi đi header `Bearer None`. Thông
báo lỗi đổ tội cho khoá trong khi khoá vô can. Cùng lớp lỗi với
`mfe.eval.official` quên nạp `.env` (§5).

**3. `503` kèm `[403] ... (reset after 1m 52s)` là MỘT TÀI KHOẢN thượng nguồn
cạn, không phải cổng chết.** Pool sẽ xoay sang tài khoản khác. Phải thử lại chứ
đừng coi là hỏng — nếu không thì một lượt ingest sẽ bỏ khung một cách im lặng.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_KEY = "TRAMMAY_API_KEY"
ENV_BASE = "MFE_TRAMMAY_BASE_URL"
ENV_MODEL = "MFE_TRAMMAY_MODEL"

DEFAULT_BASE = "https://llm-gateway.invalid/v1"
#: Đo 20-09 trên 200 ảnh: caption có động từ ở 90,2%, 0 lỗi trên 800 lượt.
DEFAULT_MODEL = "ag/gemini-3.8-flash"

#: Cloudflare chặn client không có UA bằng `error code: 1010`. Giá trị cụ thể
#: không quan trọng — sự CÓ MẶT của header mới quan trọng.
USER_AGENT = "mfe/1.0 (+https://llm-gateway.invalid)"

#: Mã đáng thử lại, gộp hai nhóm rất khác nhau:
#:
#: **5xx thường** (500, 502, 503, 504) — `503` của cổng này kèm chuỗi
#: `[403] ... (reset after 1m 52s)`, tức một tài khoản thượng nguồn vừa cạn và
#: pool sẽ xoay sang tài khoản khác. Không phải cổng chết.
#:
#: **5xx của Cloudflare** (520–527) — `520 "error code: 520"` là lỗi gateway tạm
#: thời giữa Cloudflare và origin, **không** nói gì về khoá hay về request.
#: Thiếu nhóm này trong bảng đã làm một lượt dịch 163 câu chết ở câu ~92 sau khi
#: đã chạy trơn 800 lượt trước đó.
RETRY_CODES = frozenset({429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527})


class TrammayError(RuntimeError):
    """Gọi thất bại sau khi đã thử lại hết lượt."""


class QuotaError(TrammayError):
    """Hết hạn mức — thử lại KHÔNG giúp gì, phải đợi cửa sổ reset.

    Tách khỏi `TrammayError` vì hai loại `429` đòi hai cách xử ngược nhau:

        429 "rate limit"      -> chờ vài giây là qua, THỬ LẠI đúng
        429 "quota reached"   -> chờ 5 TIẾNG mới qua, thử lại là đốt thời gian

    Đo 23-09, một worker batch 1 chạm quota 5 tiếng của `antigravity/
    gemini-3.8-flash`: **1.419,9 giây cho MỘT video mà thu về 0 caption**
    (`[51/51] L26_V470: 0/171 · hỏng 171 · PARTIAL`). 171 khung × 4 lần thử ×
    giãn cách — gần 24 phút ném vào những lượt gọi không thể thành công. Nhân
    với số video còn lại là mất cả phiên Kaggle.

    Người gọi nên **dừng hẳn lượt chạy** khi gặp nó, chứ đừng đếm nó như một
    khung hỏng bình thường rồi đi tiếp.
    """


#: Dấu hiệu trong thân lỗi cho biết đây là hạn mức cạn, không phải nghẽn nhất
#: thời. Cổng và nhà cung cấp thượng nguồn diễn đạt khác nhau, nên dò theo chữ:
#:
#:   cổng     : "Daily token quota exceeded" · code "token_quota_exceeded"
#:   thượng nguồn: "Individual quota reached. Please upgrade your subscription"
_DAU_HIEU_QUOTA = ("quota", "upgrade your subscription", "insufficient_quota")


def _la_can_quota(than: str) -> bool:
    t = than.lower()
    return any(d in t for d in _DAU_HIEU_QUOTA)


@dataclass(frozen=True)
class TrammayClient:
    """Client tối giản cho cổng. Không phụ thuộc SDK `openai`.

    Cố ý dùng `urllib` thay vì `openai`: gói đó **không có trong venv** và chính
    sự vắng mặt của nó đã làm hai trong ba nhà cung cấp của `quest` biến mất im
    lặng (§5). Một phụ thuộc không bắt buộc là một tầng có thể biến mất.
    """

    api_key: str
    base_url: str = DEFAULT_BASE
    model: str = DEFAULT_MODEL
    timeout: int = 180
    #: 4 chứ không 3, và giãn cách dài hơn — xem `_post`.
    max_retries: int = 4
    #: Nền của giãn cách `backoff_base · 2^n`. Phải LỚN HƠN thời gian một lượt
    #: gọi, nếu không thử lại sẽ rơi vào đúng lúc các suất đồng thời vẫn bận.
    #:
    #: 2,0 hợp cho `chat` một ảnh (~8s/lượt). **Đường batch phải đặt 30–60**:
    #: đo 22-09, một batch 8 ảnh chạy 26–47s, nên 2s/4s/8s làm cả bốn lần thử
    #: đều trượt — 8/18 batch chết ở 4 luồng dù đã thử đủ lượt.
    backoff_base: float = 2.0

    @classmethod
    def from_env(cls, **kwargs) -> TrammayClient | None:
        """Client đọc từ `.env`, hoặc None khi chưa có khoá.

        Trả None chứ không ném: người gọi hỏi `available()` để biết **vì sao**.
        """
        key = os.getenv(ENV_KEY)
        if not key:
            return None
        return cls(
            api_key=key,
            base_url=os.getenv(ENV_BASE) or DEFAULT_BASE,
            model=os.getenv(ENV_MODEL) or DEFAULT_MODEL,
            **kwargs,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Bỏ dòng này là ăn 403 `error code: 1010` từ Cloudflare.
            "User-Agent": USER_AGENT,
        }

    def _post(self, payload: dict, *, endpoint: str = "/chat/completions") -> dict:
        body = json.dumps(payload).encode()
        cuoi: Exception | None = None
        for lan in range(self.max_retries):
            retry_after: float | None = None
            req = urllib.request.Request(
                f"{self.base_url}{endpoint}",
                data=body,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                than = e.read().decode("utf-8", "replace")
                chi_tiet = than[:200]
                cuoi = TrammayError(f"HTTP {e.code}: {chi_tiet}")
                if e.code not in RETRY_CODES:
                    raise cuoi from e
                if _la_can_quota(than):
                    raise QuotaError(f"HTTP {e.code}: {chi_tiet}") from e
                try:
                    retry_after = float((getattr(e, "headers", None) or {}).get("Retry-After", ""))
                except (TypeError, ValueError):
                    retry_after = None
                logger.warning(
                    "trammay HTTP %s (lần %d/%d), thử lại: %s",
                    e.code, lan + 1, self.max_retries, chi_tiet,
                )
            except Exception as e:  # noqa: BLE001 — timeout, DNS, reset
                cuoi = e
                logger.warning(
                    "trammay %s (lần %d/%d), thử lại", type(e).__name__, lan + 1, self.max_retries
                )
            if lan + 1 < self.max_retries:
                # Nền là 2·2^n (2s · 4s · 8s), KHÔNG phải 2^n (1s · 2s).
                #
                # Đo 21-09: ở 30 luồng, cổng trả `EOF occurred in violation of
                # protocol` và `RemoteDisconnected` hàng loạt — đứt ở tầng TLS
                # chứ không phải HTTP, tức bị chặn ở mức KẾT NỐI. Ở 16 luồng thì
                # 800/800 lượt sạch. Lùi 1s là cả 30 luồng cùng đập lại một lúc
                # vào đúng thứ vừa từ chối chúng.
                #
                # Trên nền đó: tôn trọng `Retry-After` của cổng khi nó lớn hơn,
                # và cộng jitter để sáu worker không cùng thức dậy một thời điểm.
                #
                # `backoff_base` phải khớp ĐỘ DÀI một lượt gọi của người dùng
                # nó. Trần đồng thời của cổng đếm request ĐANG BAY, nên lùi
                # ngắn hơn thời gian một request là quay lại đúng lúc chỗ vẫn
                # chưa trống — đo 22-09: nền 2s với batch 26–47s làm 8/18 batch
                # chết dù đã thử đủ 4 lần.
                delay = max(self.backoff_base * (2**lan), retry_after or 0.0)
                time.sleep(min(delay, 120.0) + random.uniform(0.0, 1.0))
        raise TrammayError(f"thất bại sau {self.max_retries} lần: {cuoi}")

    def chat(
        self,
        prompt: str,
        *,
        images: list[Path | str] | None = None,
        images_b64: list[str] | None = None,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Một lượt hỏi. `images` biến nó thành lượt hỏi thị giác.

        `images_b64` là cùng thứ đó nhưng **đã mã hoá sẵn** (base64 của JPEG,
        không kèm tiền tố `data:`). Có mặt vì người gọi đôi khi đã thu nhỏ ảnh
        rồi: `rerank_tiers.VlmTier` hạ cạnh dài về 512 px trước khi gửi, và bắt
        nó ghi ra file tạm chỉ để hàm này đọc lại là trả hai lần cùng một chi
        phí — trên 20 ảnh 1280×720 thì đó là vài MB base64 thừa mỗi truy vấn.
        """
        noi_dung: list[dict] = [{"type": "text", "text": prompt}]
        for p in images or []:
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            noi_dung.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        for b64 in images_b64 or []:
            noi_dung.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": noi_dung}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        d = self._post(payload)
        try:
            return d["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise TrammayError(f"hình dạng trả về lạ: {json.dumps(d)[:200]}") from e

    def chat_batch_detail(
        self,
        items: list[dict],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> dict[str, dict]:
        """Như `chat_batch` nhưng giữ cả phần bao quanh caption.

        Trả `{id: {"content": str, "finish_reason": str | None}}`.

        Vì sao cần `finish_reason` chứ không chỉ `content`: một caption bị cắt
        vì chạm trần `max_tokens` **vẫn là một chuỗi hợp lệ**. Nó đi thẳng vào
        BM25, vẫn có màu, vẫn có vị trí, vẫn đếm được động từ — không phép kiểm
        nội dung nào nhìn ra. `finish_reason == "length"` là tín hiệu DUY NHẤT
        phân biệt "model tả xong" với "model bị ngắt lời", và batch làm nguy cơ
        này nặng hơn hẳn vì một `max_tokens` phải chia cho nhiều ảnh.

        Cùng lớp bẫy §5 với "báo đã chấm 12 trong khi nó chấm 1": hình dạng trả
        về đúng, số lượng đúng, chỉ nội dung là thiếu.
        """
        if not items:
            return {}

        requests = []
        expected: list[str] = []
        seen: set[str] = set()
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"batch item {position} must be an object")
            item_id = str(item.get("id", "")).strip()
            prompt = item.get("prompt")
            if not item_id or item_id in seen or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"batch item {position} has invalid id/prompt")
            seen.add(item_id)
            expected.append(item_id)
            encoded_images = []
            for image in item.get("images", []) or []:
                raw = base64.b64encode(Path(image).read_bytes()).decode()
                encoded_images.append(f"data:image/jpeg;base64,{raw}")
            requests.append({"id": item_id, "prompt": prompt, "images": encoded_images})

        payload = {
            "model": model or self.model,
            "requests": requests,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self._post(payload, endpoint="/chat/completions/batch")
        returned = data.get("data") if isinstance(data, dict) else None
        if not isinstance(returned, list):
            raise TrammayError(f"batch response hình dạng lạ: {json.dumps(data)[:300]}")

        result: dict[str, dict] = {}
        for item in returned:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise TrammayError("batch response thiếu id")
            item_id = item["id"]
            content = item.get("content")
            if item_id in result or not isinstance(content, str):
                raise TrammayError("batch response có id trùng hoặc content không hợp lệ")
            ly_do = item.get("finish_reason")
            result[item_id] = {
                "content": content.strip(),
                "finish_reason": ly_do if isinstance(ly_do, str) else None,
            }
        if set(result) != set(expected):
            missing = [item_id for item_id in expected if item_id not in result]
            extra = [item_id for item_id in result if item_id not in expected]
            raise TrammayError(f"batch response lệch id (thiếu={missing[:3]}, thừa={extra[:3]})")

        # Cắt giữa chừng = vứt CẢ batch, không nhận phần lành. Nhận một nửa là
        # ghi vào kho một caption cụt mà sau này không cách nào tìm lại được nó.
        cut = [k for k, v in result.items() if v["finish_reason"] == "length"]
        if cut:
            raise TrammayError(
                f"{len(cut)}/{len(expected)} item chạm trần max_tokens "
                f"(finish_reason='length'): {cut[:3]} — nới max_tokens rồi gửi lại"
            )
        return {item_id: result[item_id] for item_id in expected}

    def chat_batch(
        self,
        items: list[dict],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> dict[str, str]:
        """Caption nhiều ảnh trong một request.

        ``items`` có dạng ``{"id": str, "prompt": str, "images": [Path]}``.
        Gateway trả ``data`` gồm đúng các id đó; kiểm tra nghiêm ngặt để không
        gắn nhầm caption sang frame khác khi model trả thiếu/trùng item.
        """
        chi_tiet = self.chat_batch_detail(
            items, model=model, max_tokens=max_tokens, temperature=temperature
        )
        return {k: v["content"] for k, v in chi_tiet.items()}

    def models(self) -> list[str]:
        req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return [m.get("id", "") for m in json.loads(r.read()).get("data", [])]


def available() -> tuple[bool, str]:
    """`(dùng được, vì sao không)` — bắt buộc với mọi lớp bọc API (§5).

    `QuestClient` từng bọc `from openai import OpenAI` trong `try/except
    ImportError: pass`, và vì `openai` không có trong venv nên hai trong ba nhà
    cung cấp **không tồn tại mà không dòng log nào nói ra**.
    """
    if not os.getenv(ENV_KEY):
        return False, f"chưa có {ENV_KEY} trong môi trường (kiểm .env)"
    return True, ""


#: Chỉ dẫn dịch. Ràng "CHỈ trả bản dịch" vì tầng gọi lấy nguyên chuỗi trả về làm
#: truy vấn — một câu rào đầu kiểu "Sure, here is..." sẽ đi thẳng vào BM25.
_TRANSLATE_PROMPT = (
    "Translate this Vietnamese video-search query into natural English.\n"
    "It describes a scene in a video. Keep every visual detail: colours, counts, "
    "positions, actions, and any on-screen text.\n"
    "Reply with ONLY the English translation — no quotes, no preamble, no notes.\n\n"
    "Vietnamese: {text}\nEnglish:"
)


def translate_backend(client: TrammayClient | None = None, model: str | None = None):
    """`TranslationBackend` cho tầng `llm` của `mfe.online.translate`.

    Nối vào qua `build_translator(config, extra_backends={"llm": ...})` — đúng
    điểm mở rộng mà `translate.py` đã chừa sẵn, nên module đó không phải biết
    một client API nào tồn tại.
    """
    from mfe.online.translate import callable_backend  # noqa: PLC0415

    c = client or TrammayClient.from_env()
    if c is None:
        raise TrammayError(available()[1])
    ten = model or c.model

    def _dich(text: str) -> str:
        return c.chat(_TRANSLATE_PROMPT.format(text=text), model=ten, max_tokens=256)

    return callable_backend(f"trammay:{ten}", _dich)
