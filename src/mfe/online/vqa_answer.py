"""Cho VLM **nhìn** top keyframe rồi đọc ra đáp án cho câu QA.

Câu QA nộp ba cột: `video_id, frame_id, answer` — và **`answer` giống nhau ở cả
100 dòng**. Nên đây không phải một phép xếp hạng, mà là đúng một chuỗi. Người vận
hành đang phải tự mở từng ảnh, đọc con số trên biển báo / dòng chữ trên slide, rồi
gõ lại. Với 49 đề thì đó là phần chậm nhất của cả bàn nộp.

Vì sao trả về rỗng phải là một kết quả HỢP LỆ
----------------------------------------------
Đáp án sai không mất một dòng — nó mất **cả 100 dòng**, vì BTC chấm cột thứ ba
trước. Một câu truy hồi hoàn hảo với đáp án sai ghi đúng **0** điểm, y hệt như
không nộp. Nên module này thà nói "không đọc được" còn hơn đoán: `Answer.ok`
False kèm lý do, và giao diện để ô đáp án trống cho người gõ.

Đó cũng là lý do prompt bắt model tự khai **đọc được ở khung nào**. Người vận hành
kiểm một tấm ảnh nhanh hơn kiểm hai mươi tấm, và một đáp án không chỉ được nguồn
là một đáp án bịa.

Vì sao ở đây chứ không trong `app/submit_lab.py`
------------------------------------------------
Nó quyết định **nội dung cột thứ ba của file nộp**. Script Streamlit chạy mã ở cấp
module ngay khi import nên không test nào với tới được — cùng lý do
`mfe.eval.submission_bundle` được tách ra.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from mfe.contracts.tools import Candidate
from mfe.online.answering import Evidence, gather_evidence, gather_with_context
from mfe.online.trammay import TrammayClient
from mfe.online.trammay import available as trammay_available

logger = logging.getLogger(__name__)

#: QA đi qua gateway OpenAI-compatible của Trammay, không gọi Google Gemini
#: trực tiếp. Model thực tế lấy từ `MFE_TRAMMAY_MODEL` hoặc default của
#: `TrammayClient`; như vậy khóa và quota đều dùng chung với các tầng online.

#: Nhìn bao nhiêu khung đầu. Đáp án gần như luôn nằm ở vài khung đầu nếu truy hồi
#: đúng; nhìn sâu hơn chỉ tăng tiền và tăng cơ hội model đọc nhầm khung của một
#: video khác.
DEFAULT_DEPTH = 15

#: Cạnh dài tối đa khi nén ảnh. Lớn hơn tầng rerank (512) **có chủ ý**: ở đây model
#: phải ĐỌC CHỮ — số trên biển báo, dòng trên slide — chứ không phải nhận ra cảnh.
#: 512 đủ để biết "có một biển báo", không đủ để biết nó ghi `2,15`.
MAX_SIDE = 1024
JPEG_QUALITY = 90

_PROMPT = """You are reading frames from a Vietnamese TV/news video to answer ONE question.

QUESTION (Vietnamese): {question}

I am giving you {count} candidate frames, in this exact order: {ids}

Evidence was assembled as {continuous_count} continuous frames around the top
retrieval anchor and {scattered_count} additional ranked frames. OCR and
subtitles are authoritative when present; generated captions are supporting
evidence only.

OCR keywords: {ocr_keywords}
Logic question: {logic_question}

TEXT EVIDENCE:
{evidence_text}

Your job: read the answer OFF THE FRAMES. The answer is almost always something
visible — a number on a sign, text on a slide, a channel logo, a timestamp, a
name in a caption — or something directly observable in the image.

RULES:
1. Answer in VIETNAMESE, copied EXACTLY as it appears on screen, including
   diacritics and punctuation. If the sign reads "2,15" answer "2,15" — not
   "2.15", not "2,15 m".
2. Keep it SHORT: the exact value asked for, nothing else. No sentence, no
   explanation, no unit unless the question asks for one.
3. If you cannot read the answer from these frames, set "answer" to "" and say
   why in "why". An empty answer is a CORRECT response. Do NOT guess: a wrong
   answer scores zero on all 100 submitted rows.
4. "frame" must be the id of the ONE frame you read the answer from, copied from
   the list above. If you did not read it from any frame, use "".
5. "confidence" is one of "cao", "vua", "thap".

Return ONLY this JSON, no markdown, no explanation:
{{"answer": "...", "frame": "...", "confidence": "...", "why": "..."}}"""


@dataclass(frozen=True)
class Answer:
    """Kết quả một lượt đọc. `ok=False` nghĩa là **chưa có đáp án**, không phải lỗi.

    Tách `ok` khỏi `text` rỗng vì hai thứ đó khác nhau và giao diện phải nói khác
    nhau: "model nhìn rồi mà không đọc được" cần người mở ảnh ra xem, còn "chưa
    có khoá API" cần người sửa `.env`.
    """

    text: str = ""
    #: `keyframe_id` model khai là đã đọc đáp án ở đó. Rỗng = không chỉ được nguồn.
    frame: str = ""
    confidence: str = ""
    why: str = ""
    ok: bool = False
    #: Vì sao không chạy được — chỉ có nghĩa khi `ok` False.
    reason: str | None = None
    looked_at: int = 0
    seconds: float = 0.0


def available() -> tuple[bool, str | None]:
    """Gọi được không, và nếu không thì vì sao.

    Hỏi trước khi gọi: một lớp bọc chỉ ném ở lượt gọi đầu sẽ khiến giao diện vẽ
    xong nút, người bấm, rồi mới báo thiếu khoá.
    """
    ok, reason = trammay_available()
    return (True, None) if ok else (False, reason)


def encode_image(path: Path, *, max_side: int = MAX_SIDE) -> object:
    """Ảnh đã thu nhỏ, sẵn sàng gửi đi."""
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as raw:
        image = raw.convert("RGB")
        image.thumbnail((max_side, max_side))
        return image


def read_answer(
    question: str,
    candidates: Sequence[Candidate],
    images: Callable[[str], Path | None],
    *,
    depth: int = DEFAULT_DEPTH,
    ask: Callable[[str, list[Path]], str] | None = None,
    evidence: Sequence[Evidence] = (),
    logic_question: str = "",
    ocr_keywords: str = "",
) -> Answer:
    """Đọc đáp án từ `depth` khung đầu. Không bao giờ ném — trả `Answer` kèm lý do.

    `ask(prompt, paths) -> str` tiêm được để test không cần mạng. Mặc định đi
    Gemini qua gateway Trammay.

    Không ném là có chủ ý: hàm này chạy trong vòng lặp vẽ giao diện, và một lượt
    gọi API hỏng không được phép làm trắng cả trang mà người vận hành đang xếp dở
    một trăm dòng.
    """
    ok, reason = available()
    if ask is None and not ok:
        return Answer(reason=reason)

    started = time.perf_counter()
    head = list(candidates[:depth])
    usable = [(c.keyframe_id, images(c.keyframe_id)) for c in head]
    usable = [(kid, p) for kid, p in usable if p is not None and p.is_file()]
    if not usable:
        return Answer(reason="khong lay duoc anh nao cho top ung vien")

    prompt = _PROMPT.format(
        question=question.strip(),
        count=len(usable),
        ids=json.dumps([kid for kid, _ in usable], ensure_ascii=False),
        continuous_count=0,
        scattered_count=len(usable),
        evidence_text=_evidence_text(evidence),
        logic_question=logic_question.strip() or "(khong co)",
        ocr_keywords=ocr_keywords.strip() or "(khong co)",
    )
    caller = ask or _ask_trammay
    try:
        raw = caller(prompt, [p for _, p in usable])
    except Exception as err:  # noqa: BLE001 - mot luot goi hong khong duoc lam trang trang
        logger.warning("vqa_answer: %s: %s", type(err).__name__, err)
        return Answer(
            reason=f"{type(err).__name__}: {str(err)[:160]}", looked_at=len(usable)
        )

    parsed = parse_answer(raw)
    if parsed is None:
        return Answer(
            reason=f"khong doc duoc JSON tu tra loi: {raw[:120]!r}",
            looked_at=len(usable),
        )

    text = str(parsed.get("answer") or "").strip()
    frame = str(parsed.get("frame") or "").strip()
    # Model tu bia ra mot id khong co trong danh sach nghia la no khong thuc su
    # doc tu khung nao — bo nguon di chu khong bo dap an, vi dap an van co the
    # dung; nhung nguoi van hanh phai thay la khong co nguon.
    known = {kid for kid, _ in usable}
    return Answer(
        text=text,
        frame=frame if frame in known else "",
        confidence=str(parsed.get("confidence") or "").strip(),
        why=str(parsed.get("why") or "").strip(),
        ok=bool(text),
        reason=None if text else "model nhin roi nhung khong doc duoc dap an",
        looked_at=len(usable),
        seconds=time.perf_counter() - started,
    )


def read_answer_hybrid(
    question: str,
    candidates: Sequence[Candidate],
    images: Callable[[str], Path | None],
    text_store: object,
    relational: object,
    *,
    logic_question: str = "",
    ocr_keywords: str = "",
    ask: Callable[[str, list[Path]], str] | None = None,
    window: int = 2,
    max_images: int = 15,
) -> Answer:
    """Read QA with continuous context plus scattered ranked evidence.

    The first candidate anchors a ±2 keyframe relational window. Remaining
    candidates are added in retrieval order, skipping IDs already present in the
    window, until the hard fifteen-image reader budget is reached.
    """
    head = list(candidates)
    if not head:
        return Answer(reason="khong co ung vien cho QA")
    anchor = head[0].keyframe_id
    try:
        continuous = gather_with_context(anchor, text_store, relational, window=window)
    except Exception as exc:  # noqa: BLE001 - QA must remain usable if context is stale
        logger.warning("qa context skipped: %s: %s", type(exc).__name__, exc)
        continuous = gather_evidence([anchor], text_store)
    continuous = continuous[:max_images]
    seen = {item.keyframe_id for item in continuous}
    scattered: list[Evidence] = []
    remaining = max(0, max_images - len(continuous))
    for candidate in head[1:]:
        if remaining <= 0:
            break
        if candidate.keyframe_id in seen:
            continue
        evidence = gather_evidence([candidate.keyframe_id], text_store)
        if evidence:
            scattered.extend(evidence[:1])
            seen.add(candidate.keyframe_id)
            remaining -= 1
    evidence = [*continuous, *scattered]
    items = [(item.keyframe_id, images(item.keyframe_id)) for item in evidence]
    return _read_answer_items(
        question,
        items,
        ask=ask,
        evidence=evidence,
        continuous_count=len(continuous),
        scattered_count=len(scattered),
        logic_question=logic_question,
        ocr_keywords=ocr_keywords,
    )


def _evidence_text(evidence: Sequence[Evidence]) -> str:
    return "\n\n".join(
        f"--- {item.keyframe_id} ---\n{item.as_prompt() or '(khong co van ban)'}"
        for item in evidence
    ) or "(khong co van ban nao)"


def _read_answer_items(
    question: str,
    items: Sequence[tuple[str, Path | None]],
    *,
    ask: Callable[[str, list[Path]], str] | None,
    evidence: Sequence[Evidence],
    continuous_count: int,
    scattered_count: int,
    logic_question: str,
    ocr_keywords: str,
) -> Answer:
    ok, reason = available()
    if ask is None and not ok:
        return Answer(reason=reason)
    started = time.perf_counter()
    usable = [(kid, path) for kid, path in items if path is not None and path.is_file()]
    if not usable:
        return Answer(reason="khong lay duoc anh nao cho top ung vien")
    prompt = _PROMPT.format(
        question=question.strip(),
        count=len(usable),
        ids=json.dumps([kid for kid, _ in usable], ensure_ascii=False),
        continuous_count=continuous_count,
        scattered_count=scattered_count,
        evidence_text=_evidence_text(evidence),
        logic_question=logic_question.strip() or "(khong co)",
        ocr_keywords=ocr_keywords.strip() or "(khong co)",
    )
    caller = ask or _ask_trammay
    try:
        raw = caller(prompt, [path for _, path in usable if path is not None])
    except Exception as err:  # noqa: BLE001
        logger.warning("vqa_answer: %s: %s", type(err).__name__, err)
        return Answer(reason=f"{type(err).__name__}: {str(err)[:160]}", looked_at=len(usable))
    parsed = parse_answer(raw)
    if parsed is None:
        return Answer(
            reason=f"khong doc duoc JSON tu tra loi: {raw[:120]!r}", looked_at=len(usable)
        )
    text = str(parsed.get("answer") or "").strip()
    frame = str(parsed.get("frame") or "").strip()
    known = {kid for kid, _ in usable}
    return Answer(
        text=text,
        frame=frame if frame in known else "",
        confidence=str(parsed.get("confidence") or "").strip(),
        why=str(parsed.get("why") or "").strip(),
        ok=bool(text),
        reason=None if text else "model nhin roi nhung khong doc duoc dap an",
        looked_at=len(usable),
        seconds=time.perf_counter() - started,
    )


def parse_answer(text: str) -> dict | None:
    """Đối tượng JSON đầu tiên trong một câu trả lời có thể kèm markdown, lời dẫn.

    Model được bảo "chỉ JSON" và phần lớn lượt nó nghe lời, nhưng phần lớn không
    phải tất cả — và một dấu ```json ở đầu làm `json.loads` hỏng cả lượt gọi đã
    trả tiền.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _ask_trammay(prompt: str, paths: list[Path]) -> str:
    client = TrammayClient.from_env()
    if client is None:
        raise RuntimeError("thieu TRAMMAY_API_KEY trong .env")
    encoded = []
    for path in paths:
        image = encode_image(path)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return client.chat(
        prompt,
        images_b64=encoded,
        model=client.model,
        max_tokens=512,
        temperature=0.0,
    )
