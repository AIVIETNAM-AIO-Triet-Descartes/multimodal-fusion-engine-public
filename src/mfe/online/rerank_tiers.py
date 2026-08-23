"""Ba tầng rerank bổ sung, dựng lại từ `SOTA_AIC` của Nhật Bảo — bật/tắt từng cái.

    Tầng 3  ColPali MaxSim        late-interaction trên pixel
    Tầng 4  Cross-encoder HF API  trên text metadata
    Tầng 5  VLM xếp hạng          gửi ảnh cho một model thị giác

Chúng **không thay** `mfe.online.rerank.Reranker` (tầng của ta, SigLIP đa ngữ chấm
lại pixel). Ta giữ nó, và ba tầng này xếp thành chuỗi phía sau để đo được từng
cái đáng bao nhiêu — đó là điểm của việc dựng lại thay vì chọn bừa một cái.

Khác biệt đáng kể so với bản của Bảo
-------------------------------------
**ColPali mã hoá ảnh LÚC CHẠY, không precompute.** Bản gốc dựng sẵn ma trận token
cho toàn kho rồi giữ trong RAM, và nó đã bị tắt cứng trong code với ghi chú "RAM
thrashing (12GB)". Đo trên chính file của Bảo: một keyframe là `(1031, 128)`
float32 = **516 KB**, nên cả 176.707 keyframe là **87 GB** — float16 vẫn 43 GB.
Bảo trích được 855 ảnh, tức 0,48%.

Rerank chỉ chạm top 50–100. Mã hoá ngần ấy ảnh lúc cần rẻ hơn mã hoá cả kho
trước, đúng ba bậc độ lớn, và kho tụt về **không**. Đổi lại là độ trễ mỗi truy
vấn — đó là đánh đổi đúng cho một tầng chỉ chạy trên phần đuôi đã lọc.

Chính sách khi hỏng: **trả về thứ tự đang có, không bao giờ ném lỗi**
----------------------------------------------------------------------
Giống hệt `rerank.py`. Một tầng không chạy được (thiếu thư viện, hết hạn API,
model lạnh) trả lại đúng danh sách nó nhận, kèm `reason` viết ra được. Nghĩa là
bật một tầng hỏng **không bao giờ tệ hơn** tắt nó — và người vận hành nhìn thấy
lý do thay vì nhìn một thứ hạng đổi mà không hiểu vì sao.

Ba tầng này đều gọi dịch vụ ngoài hoặc tải model nặng, nên không cái nào bật mặc
định.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mfe.contracts.tools import Candidate

logger = logging.getLogger(__name__)

#: Ảnh của một keyframe_id, hoặc None nếu không lấy được.
ImageProvider = Callable[[str], "Path | None"]

#: KHONG phai `Alibaba-NLP/gte-multilingual-reranker-base` nhu ban cua Bao. Do
#: duoc 23-08: model do tra ve "Model not supported by provider hf-inference"
#: tren endpoint moi, va endpoint CU (`api-inference.huggingface.co`) da bi go
#: khoi DNS han. `bge-reranker-v2-m3` la ban da ngu, chay duoc, da thu that.
HF_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
HF_ROUTER = "https://router.huggingface.co/hf-inference/models"
COLPALI_MODEL = "vidore/colpali"
NVIDIA_VLM_MODEL = "meta/llama-3.2-90b-vision-instruct"
#: `gemini-2.5-flash` cua ban goc tra ve 404 "no longer available to new users".
#: Cung lop loi da gap o phien 21-08 voi bay diem goi Gemini khac.
GEMINI_VLM_MODEL = "gemini-3.6-flash"

#: NVIDIA NIM chỉ nhận MỘT ảnh mỗi request. Bản của Bảo có `if ... <= 1` cho
#: chuyện này, nên trên thực tế rerank nhiều ảnh của bạn ấy luôn rơi xuống
#: Gemini — cái tên "Llama-3.2-90B" trên giao diện gần như không được dùng.
NVIDIA_MAX_IMAGES = 1


@dataclass
class TierOutcome:
    """Một tầng đã làm gì. `ran=False` kèm `reason` là kết quả hợp lệ, không phải lỗi."""

    tier: str
    ran: bool
    reason: str | None = None
    scored: int = 0
    elapsed_s: float = 0.0
    #: keyframe_id -> hạng trước, hạng sau. Chỉ giữ những cái ĐỔI chỗ.
    moved: dict[str, tuple[int, int]] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.ran:
            return f"{self.tier}: khong chay — {self.reason}"
        return (
            f"{self.tier}: cham {self.scored} ung vien, {len(self.moved)} doi cho, "
            f"{self.elapsed_s:.1f}s"
        )


def _moved(before: Sequence[Candidate], after: Sequence[Candidate]) -> dict:
    rank_before = {c.keyframe_id: i for i, c in enumerate(before, start=1)}
    return {
        c.keyframe_id: (rank_before[c.keyframe_id], i)
        for i, c in enumerate(after, start=1)
        if c.keyframe_id in rank_before and rank_before[c.keyframe_id] != i
    }


def _reorder(candidates: Sequence[Candidate], order: Sequence[str]) -> list[Candidate]:
    """Xếp lại theo `order`; cái nào không có tên trong đó thì bám đuôi, giữ nguyên thứ tự.

    Bám đuôi chứ không bỏ: một model trả thiếu id là chuyện thường (nó tóm tắt,
    nó gõ sai), và bỏ ứng viên đi thì tầng rerank có thể làm kết quả **tệ hơn**
    lúc chưa rerank. Nó chỉ được phép đổi thứ tự.
    """
    by_id = {c.keyframe_id: c for c in candidates}
    ranked = [by_id[k] for k in order if k in by_id]
    seen = {c.keyframe_id for c in ranked}
    return ranked + [c for c in candidates if c.keyframe_id not in seen]


# ---------------------------------------------------------------------------
# Tầng 3 — ColPali MaxSim
# ---------------------------------------------------------------------------


class ColPaliTier:
    """Late-interaction: mỗi ảnh là một *ma trận* token, điểm = tổng max-similarity.

    Khác cross-encoder ở chỗ nó so **từng token truy vấn** với **từng vùng ảnh**
    rồi lấy cực đại — nên nó bắt được "chữ này nằm ở đâu đó trong ảnh" mà một
    vector gộp làm mất.

    Truy vấn phải là **tiếng Anh**: ColPali dựng trên PaliGemma. Đưa tiếng Việt
    vào vẫn ra điểm, vẫn xếp hạng, và không có nghĩa gì.

    Cảnh báo về miền dữ liệu, không nên bỏ qua: ColPali huấn luyện cho **tài
    liệu** (ViDoRe — trang PDF, slide, bảng biểu). Trên cảnh quay tự nhiên lợi
    thế của nó so với SigLIP là chưa được chứng minh. Kho của ta nghiêng về phía
    nó hơn thường lệ — bản tin đầy chữ chạy, slide bài giảng, bản đồ có chú giải
    — nhưng đó là một giả thuyết cần đo, không phải một điều đã biết.
    """

    name = "colpali"
    needs_english = True
    needs_images = True

    def __init__(self, model_name: str = COLPALI_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def available(self) -> tuple[bool, str | None]:
        try:
            import colpali_engine  # noqa: F401, PLC0415
        except ImportError:
            return False, "chua cai colpali-engine (pip install colpali-engine)"
        return True, None

    def _load(self):
        if self._model is None:
            import torch  # noqa: PLC0415
            from colpali_engine.models.paligemma import (  # noqa: PLC0415
                ColPali,
                ColPaliProcessor,
            )

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("nap ColPali %s tren %s (~6 GB)", self.model_name, device)
            self._model = ColPali.from_pretrained(
                self.model_name, torch_dtype=torch.float32
            ).to(device).eval()
            self._processor = ColPaliProcessor.from_pretrained(self.model_name)
            self.device = device
        return self._model, self._processor

    def rerank(
        self,
        candidates: Sequence[Candidate],
        *,
        query_vi: str,
        query_en: str | None,
        images: ImageProvider,
        depth: int = 50,
        **_,
    ) -> tuple[list[Candidate], TierOutcome]:
        if not query_en:
            return list(candidates), TierOutcome(
                self.name, False, "khong co ban dich tieng Anh — ColPali la model tieng Anh"
            )
        ok, reason = self.available()
        if not ok:
            return list(candidates), TierOutcome(self.name, False, reason)

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        started = time.perf_counter()
        head = list(candidates[:depth])
        paths = [(c, images(c.keyframe_id)) for c in head]
        usable = [(c, p) for c, p in paths if p is not None and p.is_file()]
        if not usable:
            return list(candidates), TierOutcome(
                self.name, False, "khong lay duoc anh nao cho top ung vien"
            )

        model, processor = self._load()
        batch_query = processor.process_queries([query_en]).to(self.device)
        with torch.no_grad():
            query_embedding = model(**batch_query)[0]

        scored: list[tuple[Candidate, float]] = []
        for candidate, path in usable:
            with Image.open(path) as raw:
                batch = processor.process_images([raw.convert("RGB")]).to(self.device)
            with torch.no_grad():
                doc = model(**batch)[0]
            # MaxSim: moi token truy van lay diem cao nhat tren moi vung anh,
            # roi cong lai. Do la dinh nghia cua late interaction.
            similarity = torch.einsum("qd,pd->qp", query_embedding, doc)
            scored.append((candidate, float(similarity.max(dim=1).values.sum())))

        scored.sort(key=lambda pair: -pair[1])
        order = [c.keyframe_id for c, _ in scored]
        after = _reorder(candidates, order)
        return after, TierOutcome(
            self.name, True, None, len(usable),
            time.perf_counter() - started, _moved(candidates, after),
        )


# ---------------------------------------------------------------------------
# Tầng 4 — cross-encoder qua HF Inference API
# ---------------------------------------------------------------------------


class HfCrossEncoderTier:
    """Cross-encoder đa ngữ trên **văn bản đi kèm**, không phải trên ảnh.

    Đây là giới hạn thật của tầng này và nó quyết định khi nào tầng này đáng bật:
    nó xếp lại theo OCR/phụ đề/nhãn vật thể, nên nó **không** sửa được một kết
    quả sai về mặt thị giác. Bù lại nó đọc tiếng Việt trực tiếp và không cần dịch.

    Ứng viên không có văn bản thì không gửi đi chấm — gửi chuỗi rỗng chỉ tốn một
    lượt API để nhận về một điểm vô nghĩa, rồi điểm đó vẫn xếp hạng nó.
    """

    name = "hf_cross"
    needs_english = False
    needs_images = False

    def __init__(self, model: str = HF_RERANK_MODEL, timeout_s: float = 30.0) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.url = f"{HF_ROUTER}/{model}"

    def available(self) -> tuple[bool, str | None]:
        if not os.getenv("HF_TOKEN"):
            return False, "chua co HF_TOKEN trong moi truong"
        return True, None

    def rerank(
        self,
        candidates: Sequence[Candidate],
        *,
        query_vi: str,
        query_en: str | None,
        texts: Callable[[str], str] | None = None,
        depth: int = 50,
        **_,
    ) -> tuple[list[Candidate], TierOutcome]:
        ok, reason = self.available()
        if not ok:
            return list(candidates), TierOutcome(self.name, False, reason)
        if texts is None:
            return list(candidates), TierOutcome(
                self.name, False, "khong co nguon van ban cho ung vien"
            )

        import requests  # noqa: PLC0415

        started = time.perf_counter()
        head = list(candidates[:depth])
        pairs = [(c, (texts(c.keyframe_id) or "").strip()[:512]) for c in head]
        usable = [(c, t) for c, t in pairs if t]
        if not usable:
            return list(candidates), TierOutcome(
                self.name, False, "khong ung vien nao co van ban de cham"
            )

        payload = {"inputs": [{"text": query_vi, "text_pair": t} for _, t in usable]}
        try:
            response = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
                json=payload,
                timeout=self.timeout_s,
            )
        except Exception as err:  # noqa: BLE001
            return list(candidates), TierOutcome(
                self.name, False, f"{type(err).__name__}: {err}"
            )

        if response.status_code == 503:
            return list(candidates), TierOutcome(
                self.name, False, "HF dang nap model (cold start) — thu lai sau"
            )
        if response.status_code != 200:
            return list(candidates), TierOutcome(
                self.name, False, f"HF tra ve {response.status_code}: {response.text[:120]}"
            )

        try:
            raw_scores = response.json()
        except ValueError as err:
            return list(candidates), TierOutcome(
                self.name, False, f"HF tra ve khong phai JSON: {err}"
            )

        flat = _unwrap_batch(raw_scores, len(usable))
        if flat is None:
            # KHONG cham diem mot phan roi bao la da cham het. Bang ket qua se
            # trong nhu binh thuong, va thu hang se la thu hang cu doi ten.
            return list(candidates), TierOutcome(
                self.name, False,
                f"HF tra ve {len(raw_scores)} diem cho {len(usable)} cap — hinh dang la",
            )
        scored = [
            (candidate, _score_of(entry))
            for (candidate, _), entry in zip(usable, flat, strict=True)
        ]
        scored.sort(key=lambda pair: -pair[1])
        after = _reorder(candidates, [c.keyframe_id for c, _ in scored])
        return after, TierOutcome(
            self.name, True, None, len(usable),
            time.perf_counter() - started, _moved(candidates, after),
        )


def _unwrap_batch(payload: object, expected: int) -> list | None:
    """Danh sách điểm PHẲNG, đúng `expected` phần tử — hoặc None nếu không khớp.

    HF router trả về `[[s1, s2, …]]`: một danh sách chứa **một** danh sách điểm,
    lồng sâu hơn một lớp so với thứ ai cũng đoán. Đọc thẳng `payload[i]` thì lấy
    được đúng phần tử đầu và 11 ứng viên còn lại giữ nguyên điểm cũ — tầng rerank
    báo "đã chấm 12" trong khi nó chấm 1. Bản gốc trong `SOTA_AIC` cũng vướng
    đúng chỗ này.

    Không khớp số lượng thì trả None: chấm một phần rồi gọi là chấm đủ là loại
    sai không nhìn thấy được từ bảng kết quả.
    """
    if not isinstance(payload, list):
        return None
    if len(payload) == expected:
        return payload
    if len(payload) == 1 and isinstance(payload[0], list) and len(payload[0]) == expected:
        return payload[0]
    return None


def _score_of(entry: object) -> float:
    """HF trả về vài hình dạng khác nhau tuỳ model. Đọc cả ba, không đoán."""
    if isinstance(entry, list) and entry:
        return float(entry[0].get("score", 0.0) if isinstance(entry[0], dict) else entry[0])
    if isinstance(entry, dict):
        return float(entry.get("score", 0.0))
    try:
        return float(entry)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Tầng 5 — VLM xếp hạng bằng ảnh
# ---------------------------------------------------------------------------


class VlmTier:
    """Gửi ảnh thật cho một model thị giác và bảo nó xếp hạng.

    Đây là tầng duy nhất **nhìn** ảnh bằng một model đủ lớn để suy luận về nội
    dung, nên nó là tầng duy nhất sửa được lỗi kiểu "đúng chủ thể, sai khoảnh
    khắc". Cũng là tầng đắt nhất và chậm nhất.

    Hai đường, và đường mặc định KHÔNG phải Llama: NVIDIA NIM chỉ nhận một ảnh
    mỗi request (`NVIDIA_MAX_IMAGES`), nên xếp hạng nhiều ảnh buộc phải đi Gemini.
    Bản của Bảo cũng vậy — cái tên "Llama-3.2-90B" trên giao diện của bạn ấy thực
    tế gần như không bao giờ chạy cho rerank.

    **Ảnh keyframe của BTC rời khỏi máy khi tầng này chạy.** Đó là dữ liệu cuộc
    thi gửi cho dịch vụ bên thứ ba. Không bật mặc định, và người bật cần biết.
    """

    name = "vlm"
    needs_english = True
    needs_images = True

    def __init__(self, max_side: int = 512, jpeg_quality: int = 85) -> None:
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality

    def available(self) -> tuple[bool, str | None]:
        if os.getenv("GEMINI_API_KEY"):
            return True, None
        if os.getenv("NVIDIA_API_KEY"):
            return True, None
        return False, "chua co GEMINI_API_KEY hay NVIDIA_API_KEY"

    def rerank(
        self,
        candidates: Sequence[Candidate],
        *,
        query_vi: str,
        query_en: str | None,
        images: ImageProvider,
        depth: int = 20,
        **_,
    ) -> tuple[list[Candidate], TierOutcome]:
        ok, reason = self.available()
        if not ok:
            return list(candidates), TierOutcome(self.name, False, reason)

        started = time.perf_counter()
        head = list(candidates[:depth])
        usable = [(c, images(c.keyframe_id)) for c in head]
        usable = [(c, p) for c, p in usable if p is not None and p.is_file()]
        if not usable:
            return list(candidates), TierOutcome(
                self.name, False, "khong lay duoc anh nao cho top ung vien"
            )

        prompt = self._prompt(query_en or query_vi, [c.keyframe_id for c, _ in usable])
        try:
            text = self._ask(prompt, [p for _, p in usable])
        except Exception as err:  # noqa: BLE001
            return list(candidates), TierOutcome(
                self.name, False, f"{type(err).__name__}: {str(err)[:140]}"
            )

        order = _parse_id_array(text)
        if not order:
            return list(candidates), TierOutcome(
                self.name, False, f"khong doc duoc mang JSON tu tra loi: {text[:120]!r}"
            )
        after = _reorder(candidates, order)
        return after, TierOutcome(
            self.name, True, None, len(usable),
            time.perf_counter() - started, _moved(candidates, after),
        )

    @staticmethod
    def _prompt(query: str, ids: list[str]) -> str:
        return (
            "You are an expert video retrieval judge. The user is looking for the "
            f"frame that best captures: '{query}'.\n"
            f"I am providing {len(ids)} candidate frames, in this exact order: "
            f"{json.dumps(ids)}.\n"
            "Rank them from MOST to LEAST relevant. Pay attention to the peak of the "
            "action and to any text visible in the frame.\n"
            "Return ONLY a JSON array of the frame ids in ranked order. No markdown, "
            "no explanation."
        )

    def _encode(self, path: Path) -> str:
        from io import BytesIO  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            image.thumbnail((self.max_side, self.max_side))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buffer.getvalue()).decode()

    def _ask(self, prompt: str, paths: list[Path]) -> str:
        if os.getenv("NVIDIA_API_KEY") and len(paths) <= NVIDIA_MAX_IMAGES:
            return self._ask_nvidia(prompt, paths)
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                f"{len(paths)} anh nhung NVIDIA NIM chi nhan {NVIDIA_MAX_IMAGES}, "
                "va khong co GEMINI_API_KEY de di duong con lai"
            )
        return self._ask_gemini(prompt, paths)

    def _ask_nvidia(self, prompt: str, paths: list[Path]) -> str:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for path in paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode(path)}"},
            })
        completion = client.chat.completions.create(
            model=NVIDIA_VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.2, top_p=0.7, max_tokens=1024,
        )
        return completion.choices[0].message.content.strip()

    def _ask_gemini(self, prompt: str, paths: list[Path]) -> str:
        from google import genai  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        contents: list[object] = [Image.open(p).convert("RGB") for p in paths]
        contents.append(prompt)
        response = client.models.generate_content(
            model=GEMINI_VLM_MODEL, contents=contents
        )
        return (response.text or "").strip()


def _parse_id_array(text: str) -> list[str]:
    """Mảng JSON các id trong một câu trả lời có thể kèm markdown, lời dẫn, rác."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


# ---------------------------------------------------------------------------
# Chuỗi
# ---------------------------------------------------------------------------


def apply_tiers(
    candidates: Sequence[Candidate],
    tiers: Sequence[object],
    *,
    query_vi: str,
    query_en: str | None = None,
    images: ImageProvider | None = None,
    texts: Callable[[str], str] | None = None,
    depth: int = 50,
) -> tuple[list[Candidate], list[TierOutcome]]:
    """Chạy các tầng **theo thứ tự đưa vào**, mỗi tầng ăn đầu ra của tầng trước.

    Thứ tự là một lựa chọn, không phải chi tiết vặt: tầng sau ghi đè tầng trước
    ở phần nó chạm tới. Đặt VLM (đắt, sâu 20) sau ColPali (rẻ hơn, sâu 50) nghĩa
    là VLM chỉ nói lời cuối trên 20 cái mà ColPali đã đẩy lên.

    Mọi tầng đều trả về đúng số ứng viên nó nhận. Không tầng nào được phép làm
    danh sách ngắn đi.
    """
    current = list(candidates)
    outcomes: list[TierOutcome] = []
    for tier in tiers:
        before = len(current)
        current, outcome = tier.rerank(
            current,
            query_vi=query_vi,
            query_en=query_en,
            images=images,
            texts=texts,
            depth=depth,
        )
        assert len(current) == before, f"{outcome.tier} lam mat ung vien"
        outcomes.append(outcome)
        logger.info("%s", outcome.summary())
    return current, outcomes
