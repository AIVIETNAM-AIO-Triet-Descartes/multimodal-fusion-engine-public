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

import re

#: Boilerplate mà faster-whisper BỊA RA khi audio chỉ có nhạc hoặc tiếng ồn —
#: nó tuôn văn mẫu YouTube tiếng Việt từ dữ liệu huấn luyện.
#:
#: Đo 24-09 trên 11 video nhóm S (32.693 dòng): `"Hãy subscribe cho kênh Ghiền
#: Mì Gõ..."` có ở **8/11 video**, tới **×220** trong `S01-V009` (25,7% toàn bộ
#: dòng của video đó) và ×97 + ×84 trong `S01-V001` (32% khi cộng cả bốn mẫu).
#: Không riêng nhóm S: `M01_V003` cũng dính ×23.
#:
#: ⚠ **Lọc theo SỐ LẦN LẶP là sai, và số đo nói thẳng điều đó.** Ngưỡng "bỏ
#: dòng lặp ≥10 lần" sẽ ăn luôn `Thành phố Hồ Chí Minh` (×51), `Christoph
#: Janssen` (×15), `Hà Kiều Tấn Đại` (×14), `Lê Hải Đăng` (×14), `Của tỉnh Gia
#: Lai` (×10) — tức **tên tay đua và tên đội**, thứ giá trị nhất để truy hồi
#: một chặng đua. Bình luận thể thao lặp tên liên tục là chuyện bình thường.
#: Thứ phân biệt được là **mẫu câu**, không phải độ lặp.
_AO_GIAC_THOAI = re.compile(
    r"subscribe\s+cho\s+kênh"
    r"|đăng\s*ký\s+kênh"
    r"|không\s+bỏ\s+lỡ\s+những\s+video"
    r"|cảm\s+ơn\s+(các\s+bạn|quý\s+vị)\s+đã\s+(theo\s+dõi|xem|lắng\s+nghe)"
    # Neo cuối câu và cho tối đa ~25 ký tự đuôi ("… trong video sau."): bình
    # luận thật vẫn nói "hẹn gặp lại các bạn ở chặng đua ngày mai" giữa câu.
    r"|hẹn\s+gặp\s+lại[^.!?]{0,25}[.!?]?\s*$",
    re.I,
)


def loc_ao_giac_thoai(doan: list[str]) -> list[str]:
    """Bỏ các dòng phụ đề là ảo giác của Whisper. Giữ NGUYÊN thứ tự còn lại.

    Chạy ở chỗ gom cửa sổ chứ không sửa file gốc trên volume: `subtitles_segments`
    là dữ liệu thô của `n2_subtitles`, và ba arm phụ đề cấp keyframe cũng đọc nó.
    Sửa tại chỗ là đổi dữ liệu dưới chân người khác.

    Vì sao phải bỏ: một câu quảng cáo kênh YouTube không nói gì về cửa sổ nào,
    nhưng nó sẽ chiếm `speech_topic` và có thể trôi vào `story` — đúng cơ chế
    `loc_khoi_ocr` sinh ra để chặn ở phía chữ màn hình.
    """
    return [t for t in doan if t and not _AO_GIAC_THOAI.search(t)]

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
  "story": {
    "what": "<the event or report these segments are covering, in one sentence, or null>",
    "who": ["<people or organisations named by ANY source, each with their role, e.g. 'ông Đàm Xuân Hòa, the landowner'>"],
    "where": "<the place, as specifically as a source gives it, e.g. 'xã Yạc Ly, huyện Triều Vô Rông, tỉnh Gia Lai', or null>",
    "when": "<any date or time the sources give for the events, or null>",
    "background": "<up to 2 sentences of cause, history or consequence that the pictures cannot show but a source states — why this is happening, what led to it, what was done about it. null if no source says>"
  },
  "segments": [
    {
      "segment_id": "<copy verbatim from the input>",
      "event": "<ONE sentence in the present tense, starting with a subject, saying what happens across this segment. Prefer an action verb; when the captions show no movement, use a verb of state or of framing (stands, shows, holds on) rather than inventing motion>",
      "entities": ["<short noun phrases for the people/objects that matter, e.g. 'male news anchor', 'collapsed concrete embankment'>"],
      "state_change": "<'X: before -> after' when the captions show the same thing in two different states, else null>",
      "onscreen_text": ["<text visible on screen, copied VERBATIM in its original language with diacritics>"],
      "speech_topic": "<ENGLISH. One short phrase saying what the [speech] block of THIS segment is ABOUT — the subject being discussed, not a translation of it. null when there is no speech, or when the speech is only greetings/filler>",
      "link_to_previous": "<how this segment follows the one before: 'continues', 'cuts to new location', 'returns to anchor', 'same subject closer', or null>",
      "prose": "<3-5 sentences of ordinary English prose. Say what the segment SHOWS, the transition from the previous segment, any state change, and — when this segment belongs to the story — name the people and the place from \"story\" and give the one piece of background that makes this segment make sense. Quote on-screen text verbatim inside the prose, in its original language. Everything here must come from \"story\" or from this segment's own fields; add no fact of your own.>"
    }
  ],
  "counters": {"<name of a thing that recurs across these shots>": <how many distinct times it appears>},
  "memory": "<at most 5 lines carrying what the NEXT batch of segments needs to know. Restate \"story\" in full — the event, the names with their roles, the place, the date, the background — because the next batch will not see this transcript and cannot recover them. Then the 2-3 most recent events and any running count. If the last segments have moved on to a new report, say so explicitly here so the next batch does not keep using a finished story.>"
}

RULES:

1. EVENT IS THE POINT. The captions describe static frames; your job is the verb. Write "A male anchor introduces the segment from the studio", not "A male anchor stands in a studio". If a segment genuinely has no action, describe the change of view instead ("The camera holds on the damaged embankment"). Never write a sentence without a verb.

1b. KEEP THE NUMBERS THE CAPTIONS GIVE YOU. When a caption states how many of something there are — "approximately sixteen motorbikes", "over ten representatives", "five light blue signs", "three boats" — carry that quantity into "event" or "entities" instead of flattening it to "several" or "a group of". Keep the caption's own hedge: write "around sixteen motorbikes", not "sixteen motorbikes". Do NOT count anything yourself, and do not add a number the captions did not give. If two captions of the same segment give different counts for the same thing, use the larger one and keep it hedged, because the frames show different moments of one scene. Quantity is one of the most common things a search asks about, and it is the first thing lost when a record is compressed.

2. USE ONLY WHAT YOU WERE GIVEN. Every entity, number, name and piece of text in your output must be traceable to a caption, to a [speech] block, or to the context handed to you from the preceding segments. If you are unsure whether two shots show the same person or place, say so in "link_to_previous" and do NOT assert identity. NEVER add a fact from your own knowledge — not a place name, not a person's name, not an event name, and never a detail that "must" be true of a scene like this. The test is simple: for every fact in the record you must be able to point at the line you took it from.

3. NULL IS A VALID ANSWER. "state_change": null, "onscreen_text": [], "program": null are all correct when the captions do not support anything better. An invented state change is far worse than a missing one.

3b. CHECK THE CLOCK BEFORE CLAIMING A CHANGE. Each segment's duration in seconds is given to you. Before writing "state_change", ask whether the change could physically happen in that many seconds. Pouring a glass in 3 seconds: yes. Repairing a collapsed embankment, cooking a whole dish, a crowd assembling: no. Two keyframes taken 2 seconds apart are usually two VIEWS of one moment — the camera panned or zoomed — while two keyframes 10 seconds apart may well be two moments. If the captions describe different PARTS of one place rather than one thing in two conditions, that is not a state change; leave it null and put the second view in "entities" instead. Writing a change that never happened is the worst error you can make here, because later it is indistinguishable from a real one.

4. COUNTERS. If something appears in several of these segments — a number shown on screen, an ingredient added, a person entering — count the DISTINCT occurrences across these segments. Leave "counters" as {} if nothing recurs.

5. ON-SCREEN TEXT. Copy it exactly as the caption quotes it, keeping Vietnamese diacritics. Do NOT translate it. Skip channel logos and watermarks.

5b. "prose" DRAWS ON EXACTLY TWO PLACES: this segment's own fields, and "story". Nothing else may enter it. A reader comparing them must find no fact in the prose that is missing from both. This field is what a search actually reads, so it is the one place where what is SEEN in this segment and what is KNOWN about the report come together in sentences — but coming together is not licence to add. If you cannot point at the field or the story line a sentence came from, delete the sentence.

5d. "prose" IS A DESCRIPTION, NOT A LIST WITH A SENTENCE WRAPPED AROUND IT. Never write a sentence whose job is to enumerate a field. These openings are all forbidden: "Notable entities include ...", "The scene features ...", "Prominent elements include ...", "The shot captures ...", "Visible markings include ...", "Displayed signage includes ...", "Notable visible text includes ...", "Among the ... are ...". Never refer to the record's own structure either — do not write "a state change occurs", "the transition is marked by", "this segment shows". Write about the people and things DOING and BEING something: "a dark blue SUV waits at the stop line while a courier with a green delivery box edges forward beside it", not "notable entities include a dark blue SUV and a courier with a green delivery box". Quote on-screen text inside a sentence that says where it is or what it announces, not as a trailing inventory. This field exists because a search index reads sentences better than a list of fragments — a list with "include" in front of it is still a list.

5c. NEVER WRITE A SENTENCE ABOUT SOMETHING THAT IS ABSENT. If "state_change" is null, say NOTHING about state changes — do not write "no state changes occur". If "link_to_previous" is null, say nothing about transitions. Most segments have no state change, so a sentence denying one would appear in almost every record in the index and become noise that matches every query equally. Describe only what IS there. And do not name the channel or programme logo in the prose ("60 giây", "HTV"): a logo is on screen in every single segment, so it carries no information about this one.

6. LENGTH. "event" is one sentence. "prose" is 3-5 sentences. "entities" is at most 5 short phrases. "memory" is at most 5 lines. Spend the length on what the keyframe captions do NOT have — the verb, the change, the link between segments, and who and where this is. Do not spend it re-describing how people are dressed or what colour things are: this record sits NEXT TO the full keyframe captions in the index, and every clause that repeats them is a clause not spent on what only you can say.

6b. "onscreen_text" IS AT MOST 5 ITEMS, AND THEY ARE THE IDENTIFYING ONES. A segment's captions may quote dozens of strings — every banner, every shop sign, every slogan on a wall, every label in the background. Do not copy them all. Keep only the text that would let someone recognise THIS segment. Drop decorative slogans, standing signage, and anything that merely says where you generally are. Every one of these strings is already stored with its own keyframe, so this field is a pointer, not an archive.

When more than five strings qualify, rank them like this and keep the top five:
  (1) text carrying a NUMBER that means something — a price or amount, a date, a clock time, a score, a distance, a speed, a jersey or plate number;
  (2) the headline, news banner, or the step being demonstrated;
  (3) the name of the specific place, product or organisation being shown IN THIS SEGMENT;
  (4) a caption naming a person who is speaking;
  (5) everything else.
A number is the hardest thing for a searcher to describe any other way, so it outranks a slogan every time.

Rank (3) never covers the programme's own title, the channel ident, or a sponsor mark that sits on screen for the whole video ("MÓN NGON MỖI NGÀY", "AJINOMOTO", "Eat Well, Live Well.", "60 giây", "HTV"). Those belong in the chunk-level "program" field, once, and nowhere else: a mark that is burned into every segment cannot tell one segment from another. Rule 6c applies to them as well.

6c. DROP GARBLED READINGS OF A WATERMARK — BUT READ IT ONCE FIRST. The captions were written one frame at a time, so a station logo or camera watermark is read slightly differently each time: you will see a cluster of short strings that are obviously the same mark mangled in different ways ("UTMC", "UT/\\-G", "UTM-G", "UT A&G"). Keep NONE of them in "onscreen_text" — a mark burned into every frame says nothing about any ONE segment.

Before you discard it, use it: a channel ident or tournament watermark ("HTV", "TÔN ĐÔNG Á", "CÚP TRUYỀN HÌNH", "60 giây") is often the only thing naming the programme, and that is exactly what the chunk-level "program" field is for. Put it there once, in its most complete spelling, then leave it out of every segment. Losing the name entirely is worse than repeating it — mid-event footage may carry no other identification.

This rule does NOT apply to a fixed overlay carrying real information, such as the name of the junction a traffic camera watches, a lap counter, a speed readout or a running clock: keep those in "onscreen_text", in their most complete and correct spelling.

7. ENGLISH for everything you write yourself; original language only inside "onscreen_text".

8. THE [speech] BLOCK IS WHAT IS HEARD, NOT WHAT IS SEEN, AND IT IS NOT RELIABLE.

8a. NAMES, PLACES AND DATES FROM [speech] ARE WANTED — COPY THEM EXACTLY AS HEARD. The soundtrack is usually the ONLY source that says who someone is, where this is happening and when. Take them. But the transcript comes from a speech recogniser that mishears Vietnamese proper nouns often, so three things bind you:
  - Spell it EXACTLY as the transcript spells it. Never "fix" a name toward one you recognise, never expand an abbreviation, never supply a province you think it belongs to. A misspelt name simply fails to match a search; a name you corrected from your own knowledge is an invented fact.
  - When on-screen text and speech give the same name differently, use the ON-SCREEN spelling — it was read from the picture, not heard.
  - Attach a name to a person or place ONLY where a source attaches it. If the narration names a man and the captions show a man, you may say the segment shows him ONLY if the narration ties them; otherwise name the person in "story.who" and leave the segment's "entities" describing what is visible.

8b. DO NOT MERGE WHAT IS HEARD INTO WHAT IS SEEN. "event", "entities", "state_change" and "onscreen_text" describe ONLY what the captions describe. Never write that something is visible because the speech mentioned it — if the narration says a hospital and the captions show a field, the record shows a field.

8d. A NEWS PROGRAMME CHANGES STORY WITHOUT WARNING, AND "story" MUST NOT OUTLIVE ITS ITEM. The segments you are given may span the end of one report and the start of another; the memory handed to you may describe a report that has already finished. Before you attach "story" to a segment, check that this segment's captions still show that report. When the pictures move to a different subject, the old story stops there: do not carry a name, a place or a cause across the cut. If the segments cover two reports, fill "story" for the one that covers most of them and say nothing about it in the segments belonging to the other.

8c. "speech_topic" IS A TOPIC, NOT A TRANSLATION. Write a short English phrase — "drought in the Mekong Delta", "a new bus route opening" — not a rendering of the sentences. If the speech is only a greeting, a programme opening, or too garbled to be about anything, write null.

Do NOT spend a sentence of "prose" announcing that topic. A sentence like "The narration turns to gas and water eruption after an earthquake" repeats a field that is already stored beside it and says nothing a search can use. Put the substance in instead — who, where, what happened — and mark it as reported where a reader could otherwise mistake it for something visible: "According to the report, the well had been drilled to 170 meters without finding water". Attribution belongs inside the sentence that carries the fact, not in a sentence of its own."""  # noqa: E501


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
