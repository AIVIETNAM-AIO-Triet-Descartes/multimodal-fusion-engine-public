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

Trường `prose` — và vì sao vẫn giữ các trường rời (Lead chốt 24-09)
--------------------------------------------------------------------
Chuỗi phẳng đưa vào BM25/`bge_m3` là **`prose`**, không phải bản nối các trường
rời. Lý do: `bge_m3` là bộ nhúng CÂU, mà `"… ground. . male news anchor . water
geyser …"` không phải câu. Và văn xuôi **nuốt được quan hệ liên đoạn vào nội
dung**: `link_to_previous: "same subject closer"` ở dạng nhãn là stopword vô
dụng cho BM25, còn *"In a closer view of the scene, the man crouches beside the
well pipe"* thì khớp được với đề thật.

Nhưng **không bỏ các trường rời**, vì ba thứ sẽ mất:
  - hàng rào chống bịa: `state_change: null` là một câu trả lời DỄ, còn văn xuôi
    thì ép viết trôi chảy — mà trôi chảy chính là thứ đẻ ra tình tiết nối;
  - mặt kiểm: `onscreen_text` phải ở dạng danh sách thì `loc_khoi_ocr` mới gỡ
    được mảnh logo, và mới đếm được tỉ lệ chép đúng nguyên văn;
  - quyền loại bỏ: nhét `link_to_previous` vào văn xuôi rồi thì không gỡ ra được.

Và có lợi ích phụ: **văn xuôi lệch với trường rời là tín hiệu bịa ĐO ĐƯỢC** —
văn xuôi kể một biến đổi mà `state_change` là `null` thì bắt được bằng máy.

⚠ Luật 5c sinh ra sau khi ĐO, không phải từ suy luận. Bản đầu có `prose` nhưng
thiếu 5c, và model viết `"No state changes occur across the segment."` ở **5/5**
cửa sổ — phần lớn cửa sổ không có biến đổi, nên câu phủ định đó sẽ nằm trong
~90% tài liệu, idf ~0,1, tức **stopword**. Đúng cơ chế đã giết arm `object`
(`một` 91,0% tài liệu). Cùng lượt đó nó cũng nhét mảnh logo `"60", "giây"` vào
văn xuôi — nơi `loc_khoi_ocr` không với tới. Thêm 5c: hết cả hai.

Phụ đề (24-09): nguồn THỨ HAI, và là nguồn duy nhất khác loại
---------------------------------------------------------------
Caption tả **thứ nhìn thấy**; phụ đề mang **thứ nghe được** — chủ đề, tên riêng,
con số, và nguyên nhân của cảnh đang chiếu. Đây là chỗ §1 của thiết kế chỉ ra là
thứ làm ReCap mạnh, và là nguồn duy nhất còn bỏ trống.

Nhưng nó vào prompt kèm **ba hàng rào**, vì cả ba nguy cơ đều đo được:

1. **Phụ đề là ASR, KHÔNG phải sự thật.** faster-whisper nghe sai tên riêng
   tiếng Việt một cách có hệ thống — ngay đoạn đầu `L21_V001`: *"sông Cổ Long"*
   (Cửu Long), *"sụt lũng"* (sụt lún). Chép nguyên tên riêng từ phụ đề vào bản
   ghi là **đóng dấu sai chính tả thành dữ liệu**, và tệ hơn caption bịa vì nó
   trông rất giống thật. Luật 8a: tên riêng chỉ lấy từ caption hoặc
   `onscreen_text`, không lấy từ thoại.
2. **Nghe ≠ thấy.** Bản tin đọc lời bình trên hình tư liệu: người dẫn nói về
   hạn mặn trong khi màn hình chiếu cánh đồng. Model gộp hai thứ là sinh ra
   khẳng định *"cánh đồng hạn mặn"* mà không khung nào chứng minh. Luật 8b bắt
   tách: `event`/`entities` chỉ tả thứ thấy; thứ nghe đi vào `speech_topic`.
3. **Không nhét thoại tiếng Việt vào chuỗi phẳng.** Đo 21-09: nối `vi + en`
   THUA `en` (`en→vi_en`: 17 lên / **85 xuống**) — token tiếng Việt không khớp
   gì trong kho tiếng Anh nhưng vẫn tham gia chuẩn hoá BM25. Và phụ đề **đã có
   ba arm riêng** ở cấp keyframe (`subtitle_exact`, `subtitle_partial`,
   `subtitle_bge_m3`); đổ lại vào đây là trả token cho một thứ đã được đánh chỉ
   mục. Nên `speech_topic` viết **tiếng Anh**, là chủ đề chứ không phải bản
   chép, và chuỗi phẳng vẫn chỉ gồm `prose` + `onscreen_text`.

Video câm thì `subtitles` rỗng và khối `[speech]` biến mất khỏi prompt — nhóm N
(camera giao thông) là video không có tiếng.
"""

from __future__ import annotations

#: Tiếng ANH, cùng lý do đã chốt 25-08 cho caption: bản ghi shot sẽ đi vào
#: `bge_m3` (đa ngữ) và một arm BM25 mới; tiếng Anh phục vụ cả hai, tiếng Việt
#: chỉ phục vụ một. Chữ trên màn hình thì GIỮ NGUYÊN tiếng Việt — cùng ngoại lệ
#: như nửa `[TEXT]` của caption.
PROMPT_BAN_GHI_SHOT = """You are building a SHOT-LEVEL INDEX for a Vietnamese video retrieval system.

You are given consecutive time segments from ONE video, in order. For each segment you get the dense captions of the keyframes that fall inside it, already written by a vision model. You will NOT see the images. A segment may contain one camera take or several.

Some segments also carry a [speech] block: the Vietnamese soundtrack of that segment, auto-transcribed. It tells you what is being SAID, which the captions cannot know. It is machine transcription and it is often wrong — see rule 8.

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
      "speech_topic": "<ENGLISH. One short phrase saying what the [speech] block of THIS segment is ABOUT — the subject being discussed, not a translation of it. null when there is no speech, or when the speech is only greetings/filler>",
      "link_to_previous": "<how this segment follows the one before: 'continues', 'cuts to new location', 'returns to anchor', 'same subject closer', or null>",
      "prose": "<2-3 sentences of ordinary English prose saying the same thing as the fields above, written as one flowing description a person could read. Weave in the programme name, what the segment shows, the transition from the previous segment, and any state change. Quote on-screen text verbatim inside the prose, in its original language. Add NOTHING that is not already in the fields above.>"
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

5b. THE "prose" FIELD IS A REWRITE, NOT A NEW CHANCE TO ADD THINGS. It must say exactly what the structured fields say, in ordinary sentences — nothing more. A reader comparing the two must find no fact in the prose that is missing from the fields. This field exists because a search index reads sentences better than a list of fragments, not because more detail is wanted.

5c. NEVER WRITE A SENTENCE ABOUT SOMETHING THAT IS ABSENT. If "state_change" is null, say NOTHING about state changes — do not write "no state changes occur". If "link_to_previous" is null, say nothing about transitions. Most segments have no state change, so a sentence denying one would appear in almost every record in the index and become noise that matches every query equally. Describe only what IS there. And do not name the channel or programme logo in the prose ("60 giây", "HTV"): a logo is on screen in every single segment, so it carries no information about this one.

6. LENGTH. "event" is one sentence. "prose" is 2-3 sentences. "entities" is at most 5 short phrases. "memory" is at most 3 lines. This record sits NEXT TO the full keyframe captions in the index — repeating their detail wastes the budget that the verb needs.

6b. "onscreen_text" IS AT MOST 5 ITEMS, AND THEY ARE THE IDENTIFYING ONES. A segment's captions may quote dozens of strings — every banner, every shop sign, every slogan on a wall, every label in the background. Do not copy them all. Keep only the text that would let someone recognise THIS segment: the headline or news banner, the name of the place or product being shown, a score, a price, a date, a caption naming a speaker. Drop decorative slogans, standing signage, and anything that merely says where you generally are. Every one of these strings is already stored with its own keyframe, so this field is a pointer, not an archive.

6c. DROP GARBLED READINGS OF A WATERMARK. The captions were written one frame at a time, so a station logo or camera watermark is read slightly differently each time: you will see a cluster of short strings that are obviously the same mark mangled in different ways ("UTMC", "UT/\\-G", "UTM-G", "UT A&G"). Keep NONE of them — a mark that is burned into every frame of the video says nothing about any one segment, and the misreadings are not even text. This does NOT apply to a fixed overlay that carries real information, such as the name of the junction a traffic camera watches or a running clock: keep those, in their most complete and correct spelling.

7. ENGLISH for everything you write yourself; original language only inside "onscreen_text".

8. THE [speech] BLOCK IS WHAT IS HEARD, NOT WHAT IS SEEN, AND IT IS NOT RELIABLE.

8a. NEVER take a proper noun from [speech]. The transcript is produced by a speech recogniser that mishears Vietnamese names constantly — place names, people's names, organisations and numbers in it are frequently wrong, and a wrong name written confidently is worse than no name. Names, numbers and quoted text may come ONLY from the captions or from "onscreen_text". If the speech says a name and nothing on screen confirms it, leave it out.

8b. DO NOT MERGE WHAT IS HEARD INTO WHAT IS SEEN. In a news programme the narrator talks about one thing while the pictures show another. "event", "entities", "state_change" and "onscreen_text" describe ONLY what the captions describe. What the soundtrack is about goes in "speech_topic" and nowhere else. Never write that something is visible because the speech mentioned it.

8c. "speech_topic" IS A TOPIC, NOT A TRANSLATION. Write a short English phrase — "drought in the Mekong Delta", "a new bus route opening" — not a rendering of the sentences. If the speech is only a greeting, a programme opening, or too garbled to be about anything, write null. In "prose" you may mention it only as reported speech ("the narration turns to ..."), so a reader can tell it apart from what the shot shows."""  # noqa: E501


def dung_dau_vao(cum: list[dict], bo_nho_truoc: str | None = None) -> str:
    """Dựng phần dữ liệu của prompt cho MỘT cụm shot.

    `cum` = `[{segment_id, duration_seconds, captions: [str], subtitles: [str]}]`.
    `subtitles` là tuỳ chọn: thiếu hoặc rỗng thì khối `[speech]` **không được
    in ra** — in một nhãn rỗng là dạy model rằng im lặng có nghĩa, trong khi
    video câm (camera giao thông nhóm N) chỉ đơn giản là không có tiếng.

    `bo_nho_truoc` là trường `memory` của cụm ngay trước trong CÙNG video — chỗ
    duy nhất còn giữ tính hồi quy. Cụm đầu tiên của video truyền `None`.
    """
    phan = []
    if bo_nho_truoc:
        phan.append(
            f"CONTEXT FROM THE PRECEDING SEGMENTS OF THIS VIDEO:\n{bo_nho_truoc}\n")
    phan.append("SEGMENTS:")
    # Một câu whisper vắt qua nhiều cửa sổ sẽ được `phu_de_cua` trả về ở TẤT CẢ
    # các cửa sổ nó chạm — đúng về mặt thời gian, nhưng trong một prompt thì là
    # chép lại nguyên văn. Đo trên `L26_V213`: một đoạn 700 ký tự xuất hiện ở 3
    # cửa sổ liền nhau. Giữ lần ĐẦU và bỏ các lần sau trong cùng cụm: model đọc
    # cả cụm theo thứ tự nên không mất gì, còn prompt thì không trả tiền ba lần.
    da_thay: set[str] = set()
    for s in cum:
        phan.append(f"\n--- {s['segment_id']}  ({s['duration_seconds']:.1f}s, "
                    f"{len(s['captions'])} keyframe) ---")
        for i, c in enumerate(s["captions"], 1):
            phan.append(f"[keyframe {i}] {c}")
        thoai = []
        for t in (s.get("subtitles") or []):
            t = (t or "").strip()
            if t and t not in da_thay:
                da_thay.add(t)
                thoai.append(t)
        if thoai:
            phan.append("[speech] " + " ".join(thoai))
    return "\n".join(phan)
