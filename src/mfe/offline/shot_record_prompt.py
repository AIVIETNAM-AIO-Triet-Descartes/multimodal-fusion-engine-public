"""Prompt sinh BẢN GHI SHOT từ caption keyframe — VĂN BẢN vào, văn bản ra.

Vì sao không gọi VLM với ảnh nữa (đổi so với thiết kế 17-09)
-------------------------------------------------------------
Thiết kế cũ giả định caption keyframe **thiếu diễn biến** (20,7% có cấu trúc
SVO), nên tầng shot phải tự nhìn ảnh mới sinh được hành động. Caption Gemini
`v5_bao` đã đổi tiền đề đó: **49,7% có ≥3 gốc động từ**, trung bình 2,56 gốc,
185 token — gấp 2,6 lần độ dài bản Qwen cũ.

Đo 24-09 trên gói audit 97.597 shot:

| | token/shot | tổng cho 87.508 shot có caption |
|---|---:|---:|
| gọi lại VLM với ảnh | ~4.485 | **392M** |
| chỉ đọc caption (bản này) | ~669 | **59M** |

Rẻ hơn **6,6 lần**, và quan trọng hơn: không phải tải ảnh, không phụ thuộc kho
keyframe, chạy được trên máy không GPU.

Vì sao MỘT CỤM shot trong một prompt, thay cho vòng hồi quy
-----------------------------------------------------------
ReCap của U-CESE truyền `M_{t-1}` qua từng shot vì VLM chỉ nhìn được ảnh của
MỘT shot mỗi lượt. Với đầu vào là văn bản, ràng buộc đó biến mất: nhét caption
của 10 shot liền nhau vào một prompt thì model thấy **cả chuỗi** và tự viết
được liên tục tường thuật, trong một lượt gọi.

Ba hệ quả:
  - 87.508 shot / 10 = **8.751 lượt gọi**, không phải 87.508.
  - Trần song song không còn là 873 video: cụm độc lập nhau nên song song được
    bên trong một video.
  - Vẫn giữ được chút hồi quy ở mức CỤM — `bo_nho_truoc` mang sang cụm sau —
    nên số bước tuần tự giảm 10 lần chứ không mất hẳn.

Nội dung bản ghi bám theo §8 của thiết kế, không bám theo Figure 8 của paper
-----------------------------------------------------------------------------
Đếm trên 115 đề thật: `T1` diễn biến **53,9%** · `T3` đếm tích luỹ 13,0% ·
`T4` biến đổi trạng thái 10,4% · `T5` bối cảnh chương trình 7,8% · `T2` đồng
nhất đối tượng **6,1%**. Figure 8 phô diễn sổ THỰC THỂ (`char:`, `prop:`) —
tức đất của `T2`, ô thấp nhất. Nên bộ nhớ ở đây là **sổ diễn biến + bộ đếm**.

Chống bịa: luật 2 và luật 3 là hàng rào chính. Model **chỉ được** dùng thông
tin có trong caption đưa vào; suy luận ngoài đó phải để trống. Đây là thứ đo
được — mọi thực thể trong bản ghi phải truy ngược được về một caption nguồn.
"""

from __future__ import annotations

#: Tiếng ANH, cùng lý do đã chốt 25-08 cho caption: bản ghi shot sẽ đi vào
#: `bge_m3` (đa ngữ) và một arm BM25 mới; tiếng Anh phục vụ cả hai, tiếng Việt
#: chỉ phục vụ một. Chữ trên màn hình thì GIỮ NGUYÊN tiếng Việt — cùng ngoại lệ
#: như nửa `[TEXT]` của caption.
PROMPT_BAN_GHI_SHOT = """You are building a SHOT-LEVEL INDEX for a Vietnamese video retrieval system.

You are given consecutive time segments from ONE video, in order. For each segment you get the dense captions of the keyframes that fall inside it, already written by a vision model. You will NOT see the images. A segment may contain one camera take or several.

Your job is to turn each segment into a compact record that captures what the keyframe captions CANNOT: what HAPPENS, what CHANGES, and how the segment connects to its neighbours.

Return a single JSON object, nothing else:

{
  "program": "<one short phrase naming the programme or topic of this run of segments, or null>",
  "segments": [
    {
      "segment_id": "<copy verbatim from the input>",
      "event": "<ONE sentence in the present tense, starting with a subject, saying what happens across this segment. Prefer an action verb; when the captions show no movement, use a verb of state or of framing (stands, shows, holds on) rather than inventing motion>",
      "entities": ["<short noun phrases for the people/objects that matter, e.g. 'male news anchor', 'collapsed concrete embankment'>"],
      "state_change": "<'X: before -> after' when the captions show the same thing in two different states, else null>",
      "onscreen_text": ["<text visible on screen, copied VERBATIM in its original language with diacritics>"],
      "link_to_previous": "<how this segment follows the one before: 'continues', 'cuts to new location', 'returns to anchor', 'same subject closer', or null>"
    }
  ],
  "counters": {"<name of a thing that recurs across these shots>": <how many distinct times it appears>},
  "memory": "<at most 3 lines carrying what the NEXT batch of segments needs to know: the programme, the 2-3 most recent events, and any running count>"
}

RULES:

1. EVENT IS THE POINT. The captions describe static frames; your job is the verb. Write "A male anchor introduces the segment from the studio", not "A male anchor stands in a studio". If a segment genuinely has no action, describe the change of view instead ("The camera holds on the damaged embankment"). Never write a sentence without a verb.

2. USE ONLY WHAT THE CAPTIONS SAY. Every entity, number, name and piece of text in your output must be traceable to a caption you were given. If you are unsure whether two shots show the same person or place, say so in "link_to_previous" and do NOT assert identity. NEVER add a fact from world knowledge — not a place name, not a person's name, not an event name.

3. NULL IS A VALID ANSWER. "state_change": null, "onscreen_text": [], "program": null are all correct when the captions do not support anything better. An invented state change is far worse than a missing one.

3b. CHECK THE CLOCK BEFORE CLAIMING A CHANGE. Each segment's duration in seconds is given to you. Before writing "state_change", ask whether the change could physically happen in that many seconds. Pouring a glass in 3 seconds: yes. Repairing a collapsed embankment, cooking a whole dish, a crowd assembling: no. Two keyframes taken 2 seconds apart are usually two VIEWS of one moment — the camera panned or zoomed — while two keyframes 10 seconds apart may well be two moments. If the captions describe different PARTS of one place rather than one thing in two conditions, that is not a state change; leave it null and put the second view in "entities" instead. Writing a change that never happened is the worst error you can make here, because later it is indistinguishable from a real one.

4. COUNTERS. If something appears in several of these segments — a number shown on screen, an ingredient added, a person entering — count the DISTINCT occurrences across these segments. Leave "counters" as {} if nothing recurs.

5. ON-SCREEN TEXT. Copy it exactly as the caption quotes it, keeping Vietnamese diacritics. Do NOT translate it. Skip channel logos and watermarks.

6. LENGTH. "event" is one sentence. "entities" is at most 5 short phrases. "memory" is at most 3 lines. This record sits NEXT TO the full keyframe captions in the index — repeating their detail wastes the budget that the verb needs.

7. ENGLISH for everything you write yourself; original language only inside "onscreen_text"."""  # noqa: E501


def dung_dau_vao(cum: list[dict], bo_nho_truoc: str | None = None) -> str:
    """Dựng phần dữ liệu của prompt cho MỘT cụm shot.

    `cum` = `[{shot_id, duration_seconds, captions: [str]}]`.

    `bo_nho_truoc` là trường `memory` của cụm ngay trước trong CÙNG video — chỗ
    duy nhất còn giữ tính hồi quy. Cụm đầu tiên của video truyền `None`.
    """
    phan = []
    if bo_nho_truoc:
        phan.append(
            f"CONTEXT FROM THE PRECEDING SEGMENTS OF THIS VIDEO:\n{bo_nho_truoc}\n")
    phan.append("SEGMENTS:")
    for s in cum:
        phan.append(f"\n--- {s['segment_id']}  ({s['duration_seconds']:.1f}s, "
                    f"{len(s['captions'])} keyframe) ---")
        for i, c in enumerate(s["captions"], 1):
            phan.append(f"[keyframe {i}] {c}")
    return "\n".join(phan)
