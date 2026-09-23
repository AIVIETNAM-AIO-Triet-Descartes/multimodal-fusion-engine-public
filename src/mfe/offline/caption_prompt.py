"""Prompt sinh caption + OCR cho Qwen2.5-VL. **Dá»® LIá»†U, khÃ´ng pháº£i mÃ£.**

TÃ¡ch khá»i `stages.py` vÃ¬ hai lÃ½ do, vÃ  lÃ½ do thá»© hai má»›i lÃ  lÃ½ do tháº­t:

1. `stages.py` Ä‘Ã£ hÆ¡n 2.000 dÃ²ng, vÃ  prompt lÃ  thá»© ngÆ°á»i ta sá»­a nhiá»u nháº¥t.
2. **Xuá»‘ng dÃ²ng cho vá»«a 100 cá»™t lÃ  Ä‘á»•i Ä‘Ãºng chuá»—i kÃ½ tá»± model nháº­n Ä‘Æ°á»£c.** Má»™t
   prompt bá»‹ sá»­a vÃ¬ lÃ½ do Ä‘á»‹nh dáº¡ng lÃ  thá»© khÃ´ng ai nghÄ© tá»›i khi Ä‘i tÃ¬m nguyÃªn
   nhÃ¢n káº¿t quáº£ Ä‘á»•i. NÃªn file nÃ y Ä‘Æ°á»£c `pyproject.toml` tha `E501` â€” cÃ¹ng cÃ¡ch
   `mfe/online/quest.py` Ä‘Ã£ Ä‘Æ°á»£c tha, vÃ¬ cÃ¹ng má»™t lÃ½ do.

Hai biáº¿n thá»ƒ cÃ¹ng tá»“n táº¡i, vÃ  cáº£ hai Ä‘á»u Ä‘Æ°á»£c dÃ¹ng:

| biáº¿n thá»ƒ | dÃ¹ng khi | tiáº¿ng | max_tokens |
|---|---|---|---|
| `bao` | lÆ°á»£t **probe** Ä‘á»‘i chiáº¿u vá»›i báº£n Báº£o Ä‘Ã£ cháº¡y | Anh | 256 |
| `aic` | lÆ°á»£t cháº¡y **tháº­t** | Anh | 384 |

Giá»¯ `bao` khÃ´ng pháº£i Ä‘á»ƒ dá»± phÃ²ng: nÃ³ lÃ  phÃ©p kiá»ƒm **Ä‘Æ°á»ng á»‘ng** duy nháº¥t ta cÃ³.
`notebooks/reference/dense_captions_qwen.json` (855 keyframe) chá»‰ Ä‘á»‘i chiáº¿u Ä‘Æ°á»£c
vá»›i chÃ­nh prompt Ä‘Ã£ sinh ra nÃ³. Äá»•i prompt rá»“i so vá»›i nÃ³ lÃ  so hai thá»© khÃ¡c nhau,
vÃ  khi lá»‡ch sáº½ khÃ´ng biáº¿t lá»—i náº±m á»Ÿ Ä‘á»‹a chá»‰ khoÃ¡, á»Ÿ Ä‘á»™ng cÆ¡ vLLM, hay á»Ÿ prompt.

VÃ¬ sao caption viáº¿t TIáº¾NG ANH (quyáº¿t Ä‘á»‹nh cá»§a Lead, 25-08-2026)
---------------------------------------------------------------
Caption Ä‘i vÃ o **hai** chá»—, khÃ´ng pháº£i má»™t:

    n6_caption_qwen -> qwen_captions.json --+--> arm `caption_qwen`  (BM25, vÄƒn báº£n)
                                            +--> n6_bge_m3 -> arm `bge_m3` (embedding)

BGE-M3 lÃ  mÃ´ hÃ¬nh **Ä‘a ngá»¯**: truy váº¥n tiáº¿ng Viá»‡t khá»›p caption tiáº¿ng Anh mÃ  khÃ´ng
cáº§n bÆ°á»›c dá»‹ch nÃ o (`DENSE_ARM_SPECS["bge_m3"]` khai `query_language = "any"`). VÃ 
nhÃ¡nh BM25 cÅ©ng khÃ´ng máº¥t, vÃ¬ Ä‘Æ°á»ng truy váº¥n **Ä‘Ã£ cÃ³ sáºµn** táº§ng dá»‹ch sang tiáº¿ng
Anh phá»¥c vá»¥ arm `siglip` (`--en-source auto`).

NÃªn má»™t caption tiáº¿ng Anh phá»¥c vá»¥ Ä‘Æ°á»£c cáº£ hai arm, cÃ²n caption tiáº¿ng Viá»‡t chá»‰
phá»¥c vá»¥ Ä‘Æ°á»£c má»™t. ÄÃ³ lÃ  lÃ½ do, chá»© khÃ´ng pháº£i "tiáº¿ng Anh mÃ´ táº£ tá»‘t hÆ¡n".

Má»™t Ä‘iá»u luáº­t KHÃ”NG Ä‘á»•i theo: ná»­a `[TEXT]` **cáº¥m dá»‹ch**. NÃ³ lÃ  chá»¯ trÃªn mÃ n hÃ¬nh,
lÃ  nguá»“n cá»§a arm BM25 `ocr_qwen25`, vÃ  nÃ³ pháº£i giá»¯ nguyÃªn dáº¥u tiáº¿ng Viá»‡t.
"""

from __future__ import annotations

#: Prompt cua Bao, CHEP NGUYEN VAN tu `notebooks/reference/caption.ipynb`.
#:
#: GIU LAI du bien the `aic` moi la cai dem di chay that. Ly do: ban chay cua Bao
#: (`notebooks/reference/dense_captions_qwen.json`, 855 keyframe cua L21_V001..V003)
#: la thu DUY NHAT doi chieu duoc, va no chi doi chieu duoc voi CHINH prompt da
#: sinh ra no. Chay bien the nay o luot probe la phep kiem **duong ong**: dia chi
#: khoa, dinh dang, dong co vLLM. Doi prompt roi so voi no la so hai thu khac nhau.
_QWEN25_PROMPT_BAO = (
    "Task 1: Describe all visual details, actions, and objects in this image in English.\n"
    "Task 2: Transcribe ALL visible text in this image exactly as written "
    "(including Vietnamese if any). If no text, write 'None'.\n\n"
    "Format your output EXACTLY like this:\n"
    "[CAPTION] (your detailed description here)\n"
    "[TEXT] (your extracted text here)"
)

#: Phan CHI DAN cua bien the `aic`. Doc nguoc tu `diem-yeu-pipeline-theo-49-de.md`
#: Â§3, va moi dieu khoan tra loi mot con so dem duoc tren 49 de that (do lai bang
#: chinh `caption_audit.QUERY_DEMAND`):
#:
#:   mau gan vao TUNG vat 63,3%  Â·  quan he giua cac vat 57,1%  Â·  vi tri 36,7%
#:   goc may 16,3%               Â·  dem so luong (xem canh bao trong bang)
#:
#: NGON NGU: **tieng Anh**, giong Bao (quyet dinh cua Lead 25-08-2026). Ly do la
#: dich den cua caption: `n6_bge_m3` -> arm `bge_m3`, va BGE-M3 la mo hinh **da
#: ngu** â€” truy van tieng Viet khop caption tieng Anh khong can dich. Con duong
#: BM25 (`caption_qwen`) cung khong mat: duong truy van da co san ban dich tieng
#: Anh cho arm `siglip` (`--en-source auto`), nen mot caption tieng Anh phuc vu ca
#: hai arm, con caption tieng Viet chi phuc vu mot.
#:
#: Bon khac biet so voi prompt cua Bao, moi cai ung mot con so do duoc tren chinh
#: 855 keyframe ban ay da chay:
#:
#: 1. **QUAN HE giua cac vat la dieu khoan SO 1.** Bao do duoc 50,5% trong khi
#:    57,1% de that hoi mot hanh dong. Mot caption liet ke "mot nguoi, mot voi
#:    nuoc, may chau cay" khong tra loi duoc de "nguoi dang tuoi cay" â€” dung vat
#:    the, sai su viec. Day la yeu cau cua Lead ngay 25-08-2026.
#: 2. **Cam mo dau bang "The image shows".** Bao: **59,9%**. Khong phai chuyen
#:    tham my: moi caption mo dau giong nhau thi cum do thanh stopword tren toan
#:    chi muc â€” dung co che da giet arm `object` (`mot` 91,0% tai lieu, idf 0,09).
#: 3. **Bo duong thoat "None".** Prompt cua Bao cho san mot tu de bo cuoc, va mot
#:    model se dung no. `stage_n5_ocr_qwen` da tra gia cho bai hoc nay: cho san
#:    duong thoat thi 305/307 anh tra ve "None" trong khi easyocr tim thay chu
#:    tren ca 307. Khong co chu thi de TRONG.
#: 4. **[TEXT] cam DICH.** Prompt cua Bao chi noi "including Vietnamese if any".
#:    Nua `[TEXT]` la nguon cua arm BM25 `ocr_qwen25`, va no chi lam viec neu chu
#:    tren man hinh duoc chep dung tung dau thanh â€” mot ban dich sang tieng Anh o
#:    day la mot arm chet, khong phai mot caption kem.
_QWEN25_INSTRUCTION_AIC = """You are describing ONE frame taken from a Vietnamese television video.

Return EXACTLY two blocks, in this order, each starting with its label:

[CAPTION] a description in ENGLISH, 3-5 sentences, covering ONLY what is visible.
[TEXT] every readable piece of text in the frame, copied VERBATIM.

RULES FOR [CAPTION] â€” five things you MUST state whenever the frame shows them:

1. WHAT THE PEOPLE AND OBJECTS ARE DOING WITH EACH OTHER. Never list things side by side;
   join them with a verb that says what is happening.
   **Your FIRST sentence must be the action.** Say what is happening, then spend the rest of
   the caption on who and what: colours, counts, positions. A caption that opens by listing
   what is present has already used its best sentence on the least useful thing.
   GOOD: "a man in a blue shirt waters a row of potted plants with a green hose"
   GOOD: "an adult in a white shirt points at an open notebook while a boy in a red shirt
         follows the page with his finger"
   GOOD: "a woman hands a bowl of noodles across the counter to a seated customer"
   BAD:  "a man, a green hose, and several potted plants"
   BAD:  "there are two people and a notebook in the frame"
   Use a role word (teacher, farmer, vendor, doctor, referee) ONLY when the frame itself
   shows the evidence â€” a uniform, a blackboard, a market stall, a whistle. With no such
   evidence, describe the action and leave the role out.
2. MAIN EVENT FOCUS: Explicitly describe any accidents, damage to vehicles (e.g., a car damaged at the rear), crimes, or central actions happening in the scene. Do not let background elements distract you from the primary subject.
3. COLOUR attached to EACH object. not the overall mood of the scene.
   GOOD: "the rider in a yellow jersey and black shorts leads the rider in a blue jersey"
   BAD:  "the riders wear colourful outfits"
4. HOW MANY, whenever they can be counted: "three people", "two motorbikes", "five plastic
   stools". Count what you can see; write "several" only when they are too many to count.
5. WHERE things are, relative to each other: on the left / in the middle / on the right /
   in front of / behind. Do not settle for "in the background".
6. CAMERA ANGLE only when it is UNUSUAL: a close-up, an overhead or aerial view, a low angle
   looking up. An ordinary eye-level shot is the default â€” say nothing about it. Never end a
   caption with a stock phrase about the shot.

FORBIDDEN in [CAPTION]:
- Guessing what you cannot see: place names, personal names, the topic of the news story,
  emotions, intentions, what happened before or after.
  "a report about flooding" is a GUESS. "a car drives through brown water that reaches the
  middle of its wheels" is SEEN.
- Starting with "The image shows", "This image", "In the picture", "A photo of" â€” start
  with the content itself.
- Writing the caption in Vietnamese.

RULES FOR [TEXT]:
- Copy it VERBATIM. Keep Vietnamese diacritics, CAPITALS, digits and punctuation exactly
  as they appear.
- DO NOT translate it. [TEXT] is a transcription, not a description. Vietnamese text stays
  Vietnamese.
- EACH separate region of text on screen goes on ITS OWN LINE. The channel logo, the clock,
  the headline and the ticker along the bottom are DIFFERENT regions.
- Add no commentary of your own.
- If the frame has no readable text at all, leave the space after [TEXT] EMPTY. Do not
  write "None", "no text", or a dash."""

#: Few-shot cho bien the `aic`. **CHI VAN BAN, khong kem anh mau** â€” mot anh mau
#: ton hang tram token thi giac moi cai, va o quy mo 177.321 keyframe thi do la
#: hang chuc gio GPU chi de lap lai cung mot vi du.
#:
#: Vi du day dinh dang, khong day noi dung: model van nhin anh THAT de mo ta. Ba
#: vi du, moi cai gai mot thu khac nhau:
#:
#:   1. co nhieu vung chu roi nhau  -> moi vung MOT DONG
#:   2. dem + mau gan vat + trai/giua/phai, va KHONG co chu -> [TEXT] de TRONG
#:   3. slide co chu dai            -> chep nguyen van, giu chu HOA va dau
#:
#: Vi du 2 la vi du quan trong nhat: no la cho duy nhat model thay rang "khong co
#: chu" duoc the hien bang mot khoang TRONG chu khong phai bang mot tu.
#:
#: Ca ba vi du deu co caption TIENG ANH va `[TEXT]` TIENG VIET. Do la mot phan cua
#: bai day chu khong phai su tinh co: cap doi do the hien bang vi du dung dieu luat
#: kho noi nhat cua prompt â€” mo ta thi dich, chu tren man hinh thi khong.
_QWEN25_FEWSHOT_AIC = """Here are three examples of the correct output FORMAT (for other frames).
Learn the shape of the answer; do not copy the content.

### Example 1
[CAPTION] Two presenters read from a red folder and speak toward the camera from behind a curved white desk. The man on the left wears glasses, a dark navy shirt and a pale tie; the woman on the right has long hair and wears a light blue blouse. The man rests his free hand on the desk while the woman turns a page. Behind them a wide city skyline glows orange at sunset.
[TEXT] HTV9 HD
06:31:14
TIN CHÃNH
Äá»£t mÆ°a lÅ© vá»«a qua khiáº¿n 3 ngÆ°á»i thiá»‡t máº¡ng

### Example 2
[CAPTION] Five people peel bamboo shoots together on a concrete yard, sitting on low plastic stools around the heap. The woman on the far left, in a red and white checked shirt and yellow trousers, strips the husk from a shoot; the person in the middle in a green t-shirt passes a peeled shoot toward her; the man on the far right in camouflage-patterned clothing sorts the shoots into a yellow plastic basket. A row of blue plastic drums stands behind them.
[TEXT]

### Example 3
[CAPTION] Close-up of a lecture slide on a white background. Three rounded blue buttons run down the left half, each one pointing to an empty white box on its right. In the lower left corner an inset shows a male lecturer in a white shirt sitting in front of an open laptop and looking at its screen.
[TEXT] THANH NIÃŠN
HOA KÃŒ
NHáº¬T Báº¢N
TÃ‚Y Ã‚U
CHÆ¯Æ NG TRÃŒNH Ã”N THI THPT MÃ”N Lá»ŠCH Sá»¬

Now do exactly the same for the frame you are given."""

_QWEN25_PROMPT_AIC = f"{_QWEN25_INSTRUCTION_AIC}\n\n{_QWEN25_FEWSHOT_AIC}"

_QWEN25_INSTRUCTION_SOTA = """You are describing ONE frame taken from a Vietnamese television video.
Practice strict OBJECTIVE REALISM: Describe the precise physical and geometric state of every object exactly as it appears. Do not use semantic shortcuts, and do not guess what you cannot clearly see. If a component is unclear, distant or tiny, describe it generally (e.g., "a vehicle", "an object").

Return EXACTLY two blocks, in this order, each starting with its label:

[SCENE_GRAPH]
Exhaustively list ALL visible entities and interactions as (Subject) -[RELATION]-> (Object) triplets.
- Include key visual attributes (color, clothing) and precise physical condition.
- Define actions (e.g., "holding") and spatial geometry (left, right, behind).
- Group identical objects collectively. Do not list individually.

[CAPTION] 
Using your [SCENE_GRAPH] and direct visual observation of the image, write a 3-5 sentence description in ENGLISH covering ONLY what is visible.

RULES FOR [CAPTION] â€” five things you MUST state whenever the frame shows them:
1. WHAT THE PEOPLE AND OBJECTS ARE DOING WITH EACH OTHER. Never list things side by side; join them with a verb that says what is happening.
   **Your FIRST sentence must be the action.** Say what is happening, then spend the rest of the caption on who and what: colours, counts, positions.
   GOOD: "a man in a blue shirt waters a row of potted plants with a green hose"
   BAD:  "a man, a green hose, and several potted plants"
2. MAIN EVENT FOCUS: Explicitly describe any accidents, damage to vehicles (e.g., a car damaged at the rear), crimes, or central actions happening in the scene. Do not let background elements distract you from the primary subject.
3. COLOUR attached to EACH object.
4. HOW MANY, whenever they can be counted: "three people", "two motorbikes".
5. WHERE things are, relative to each other: on the left / in the middle / on the right / in front of / behind.
6. CAMERA ANGLE only when it is UNUSUAL (close-up, aerial).

FORBIDDEN in [CAPTION]:
- Guessing what you cannot see: place names, personal names, emotions, intentions, unseen structural damage.
- Starting with "The image shows", "This image", "In the picture", "A photo of" â€” start with the content itself.
- Writing the caption in Vietnamese.
"""

_QWEN25_FEWSHOT_SOTA = """Here are two examples of the correct output FORMAT.
Learn the shape of the answer; do not copy the content.

### Example 1
[SCENE_GRAPH]
1. Presenter in dark navy shirt and pale tie -[STANDING BEHIND]-> Curved white desk
2. Presenter in light blue blouse -[STANDING BEHIND]-> Curved white desk
3. Presenter in dark navy shirt -[RESTING HAND ON]-> Curved white desk
4. Presenter in light blue blouse -[TURNING PAGE OF]-> Red folder
5. City skyline glowing orange -[LOCATED BEHIND]-> Presenters

[CAPTION]
Two presenters read from a red folder and speak toward the camera from behind a curved white desk. The man on the left wears glasses, a dark navy shirt and a pale tie; the woman on the right has long hair and wears a light blue blouse. The man rests his free hand on the desk while the woman turns a page. Behind them a wide city skyline glows orange at sunset.

### Example 2
[SCENE_GRAPH]
1. Five people -[SITTING ON]-> Low plastic stools
2. Five people -[SURROUNDING]-> Heap of bamboo shoots
3. Woman in red and white checked shirt and yellow trousers -[STRIPPING HUSK FROM]-> Bamboo shoot
4. Person in green t-shirt -[PASSING]-> Peeled shoot
5. Man in camouflage clothing -[SORTING]-> Shoots
6. Shoots -[INTO]-> Yellow plastic basket
7. Row of blue plastic drums -[LOCATED BEHIND]-> People

[CAPTION]
Five people peel bamboo shoots together on a concrete yard, sitting on low plastic stools around the heap. The woman on the far left, in a red and white checked shirt and yellow trousers, strips the husk from a shoot; the person in the middle in a green t-shirt passes a peeled shoot toward her; the man on the far right in camouflage-patterned clothing sorts the shoots into a yellow plastic basket. A row of blue plastic drums stands behind them.

Now do exactly the same for the frame you are given."""

_QWEN25_PROMPT_SOTA = f"{_QWEN25_INSTRUCTION_SOTA}\n\n{_QWEN25_FEWSHOT_SOTA}"

_QWEN25_INSTRUCTION_SOTA_V2 = """You are describing ONE frame taken from a Vietnamese television video.
Practice strict OBJECTIVE REALISM: Describe the precise physical and geometric state of every object exactly as it appears. Do not use semantic shortcuts, and do not guess what you cannot clearly see. If a component is unclear, distant or tiny, describe it generally (e.g., "a vehicle", "an object").

Return EXACTLY one block, starting with its label:

[CAPTION] 
Write a highly detailed and comprehensive description in ENGLISH covering ONLY what is visible. Focus heavily on how entities interact spatially and actively. Every major entity must be described in relation to another entity. Do not artificialy limit the length of your description; analyze everything accurately.

RULES FOR [CAPTION] â€” things you MUST state whenever the frame shows them:
1. WHAT THE PEOPLE AND OBJECTS ARE DOING WITH EACH OTHER. Never list things side by side; join them with a verb that says what is happening.
   **Your FIRST sentence must be the action.** Say what is happening, then spend the rest of the caption on who and what: colours, counts, positions.
   GOOD: "a man in a blue shirt waters a row of potted plants with a green hose"
   BAD:  "a man, a green hose, and several potted plants"
2. MAIN EVENT FOCUS: Explicitly describe any accidents, damage to vehicles (e.g., a car damaged at the rear), crimes, or central actions happening in the scene. Do not let background elements distract you from the primary subject.
3. COLOUR attached to EACH object.
4. HOW MANY, whenever they can be counted: "three people", "two motorbikes".
5. WHERE things are, relative to each other: on the left / in the middle / on the right / in front of / behind.
6. CAMERA ANGLE only when it is UNUSUAL (close-up, aerial).

FORBIDDEN in [CAPTION]:
- Transcribing irrelevant on-screen text: Do NOT read TV broadcast overlays, channel logos, timestamps, or background signs unless a person is actively interacting with them. We do not need OCR for irrelevant text.
- Do NOT generate exhaustive lists of triplet relations or identical objects. Write in natural prose.
- Guessing what you cannot see: place names, personal names, emotions, intentions, unseen structural damage.
- Starting with "The image shows", "This image", "In the picture", "A photo of" â€” start with the content itself.
- Writing the caption in Vietnamese.
"""

_QWEN25_FEWSHOT_SOTA_V2 = """Here are two examples of the correct output FORMAT.
Learn the shape of the answer; do not copy the content.

### Example 1
[CAPTION]
Two presenters read from a red folder and speak toward the camera from behind a curved white desk. The man on the left wears glasses, a dark navy shirt and a pale tie; the woman on the right has long hair and wears a light blue blouse. The man rests his free hand on the desk while the woman turns a page. Behind them a wide city skyline glows orange at sunset.

### Example 2
[CAPTION]
Five people peel bamboo shoots together on a concrete yard, sitting on low plastic stools around the heap. The woman on the far left, in a red and white checked shirt and yellow trousers, strips the husk from a shoot; the person in the middle in a green t-shirt passes a peeled shoot toward her; the man on the far right in camouflage-patterned clothing sorts the shoots into a yellow plastic basket. A row of blue plastic drums stands behind them.

Now do exactly the same for the frame you are given."""

_QWEN25_PROMPT_SOTA_V2 = f"{_QWEN25_INSTRUCTION_SOTA_V2}\n\n{_QWEN25_FEWSHOT_SOTA_V2}"

#: `ten bien the -> (prompt, max_tokens mac dinh)`.
#: Prompt "SOTA V5" — Bảo tinh chỉnh trên Gemini 3.8 qua cổng Trammay, chép
#: nguyên văn từ `notebooks/reference/Final_caption_frame.ipynb` (20-09-2026).
#:
#: Đo 20-09 trên 200 keyframe, cùng thước `caption_audit`, so với ba bản cũ:
#:
#:                     >=3 goc   so goc TB   goc/100 token   do dai
#:   Qwen+aic             5,0%        0,97            1,28       72
#:   Gemini+aic          10,0%        1,12            1,27       86
#:   Gemini+sota_v2      19,5%        1,48            1,04      132
#:   Gemini+V5           49,7%        2,56            1,32      185
#:
#: McNemar trên `>=3 gốc`, V5 so với cả ba: **p = 0,0000**. Và nó là bản duy nhất
#: mà mật độ (gốc/100 token) cũng CAO NHẤT — `sota_v2` được nhiều động từ nhờ
#: viết dài (mật độ tụt còn 1,04), V5 thì vừa dài vừa đặc.
#:
#: MỘT KHỐI, không tách `[TEXT]`. Đó là chủ ý của Bảo: chữ trên màn hình chỉ được
#: chép khi nó **liên quan tới bối cảnh khung hình** — bảng nguyên liệu, công
#: thức, giá tiền, bảng điểm — còn logo kênh và dòng chạy tin thì bỏ. Đo lại
#: phần chữ đó: nó chỉ chiếm **trung vị 1,8% token** của caption, và bỏ hết đi
#: thì `>=3 gốc` chỉ tụt 49,7% -> 46,4%. Tức phần mô tả thị giác tự nó đã mạnh.
#:
#: ⚠ `max_tokens` 1258 của bản gốc làm **8,5% lượt gọi timeout** (17/200
#: `URLError`). Mặc định ở đây hạ xuống 768; đo lại tỉ lệ hỏng trước khi nâng.
_V5_BAO = """You are an expert visual describer. Create an EXHAUSTIVELY DETAILED Dense Caption for this single frame from a Vietnamese television video.

Return EXACTLY ONE block:
[FINAL_CAPTION]

RULES:
1. EXHAUSTIVE DETAIL & RELATIONSHIPS: Write as many sentences as needed. Identify every visible object with precise quantities, exact colors, attire, and physical traits. Describe spatial and action relationships using strong verbs and prepositions (e.g., "A man in a blue shirt is handing a brown package to a woman standing behind a wooden counter"). Do not just list objects.

2. CAMERA ANGLES: Mention the camera angle (aerial, low angle, POV from inside vehicle, close-up, etc.) ONLY if it is unusual or noteworthy. Do not mention standard eye-level shots.

3. IDENTIFYING LOCATIONS/LANDMARKS: Name famous landmarks ONLY if you are 100% certain (e.g., Landmark 81, Bitexco Tower). Otherwise, describe what you see without guessing.

4. ON-SCREEN TEXT HANDLING (CRITICAL):
   - IGNORE ONLY these two categories: (a) TV channel logos/watermarks (e.g., "HTV9", "HTV", "VTV", "Online") and (b) scrolling news ticker bars running along the bottom edge of the screen that contain unrelated headlines.
   - READ AND TRANSCRIBE EVERYTHING ELSE that appears as text or numbers on-screen. This includes but is not limited to: ingredient lists, recipe measurements, menu charts, instructional captions, price tags, scoreboards, subtitles describing the scene, street signs, vehicle labels, banners, documents, T-shirt prints, and any other informational text overlay that is relevant to the content of the frame. Copy this text verbatim.

5. FORMAT: Do not start with "The image shows" or similar. Start directly with the subject. Write in ENGLISH."""  # noqa: E501

#: Bản `v5_bao` có thêm khối `[OCR]` — cho **batch 2**, nơi không có OCR nào khác.
#:
#: Vì sao cần: batch 1 có `qwen_ocr.json` (từ Qwen2.5-VL) và `vietocr`, nên hai
#: arm `ocr_qwen25` + `ocr_vietocr` có dữ liệu. Batch 2 **không có cả hai**. Nếu
#: chỉ sinh caption một khối thì với mọi keyframe batch 2, hai arm đó trả về
#: rỗng — không nổ, không cảnh báo, chỉ là khung batch 2 nhận ít phiếu hơn khung
#: batch 1 trong hợp nhất. Đó là **thiên lệch xếp hạng giữa hai nửa kho**, đúng
#: lớp lỗi §2 "sai một chỗ là hỏng im lặng".
#:
#: ⚠ Chữ trên màn hình được yêu cầu xuất HAI LẦN — vừa trong caption vừa trong
#: khối `[OCR]`. Trùng lặp là CỐ Ý. Nếu rút hẳn OCR ra khỏi caption thì caption
#: batch 2 khác cấu trúc caption batch 1 (bản sinh 22-09 có OCR nhúng), và hai
#: nửa kho lại lệch nhau ở một chỗ khác. Trùng lặp tốn thêm ~2% token (đo: phần
#: chữ chiếm trung vị 1,8% token caption), đổi lại hai nửa đồng dạng ở CẢ HAI
#: trường.
#:
#: ⚠ CHƯA ĐO. Thêm một khối là đổi prompt, mà đổi prompt là mọi con số của
#: `v5_bao` (`≥3 gốc` 49,7%, độ dài 185 token) không còn mô tả bản này. Phải
#: chạy lại trên đúng mẫu cũ trước khi dùng cho cả kho.
#: ⚠ SỬA LUẬT TICKER, và đây không phải chuyện chữ nghĩa.
#:
#: `v5_bao` dặn bỏ qua *"scrolling news ticker bars running along the bottom
#: edge"*. Với tin tức (nhóm M, **100 giờ** — nhóm lớn nhất batch 2) luật đó
#: ĐÚNG: dải chạy chân màn hình là headline không liên quan tới khung hình.
#:
#: Nhưng batch 2 có thêm **50 giờ đua xe đạp** (nhóm S, đo được là phát sóng
#: truyền hình thật: 1920×1080, 30 fps, ~2,8 giờ mỗi video). Trong phát sóng
#: thể thao, dải dưới màn hình chính là chỗ mang **thông tin phân biệt duy
#: nhất**: số hiệu VĐV, tên đội, khoảng cách thời gian, số km còn lại. Hàng
#: nghìn khung "đoàn đua trên đường nhựa" gần như giống hệt nhau — mô tả thị
#: giác không tách được chúng, chỉ chữ trên màn hình mới tách được.
#:
#: Nên luật phải phân biệt **theo NỘI DUNG, không theo loại video**: một biến
#: thể phục vụ cả hai nhóm, model tự phán đoán từng khung. Dán một câu "luôn
#: chép ticker" vào đây là làm hỏng đúng nhóm lớn nhất.
_LUAT_TICKER_CU = (
    "(b) scrolling news ticker bars running along the bottom edge of the screen "
    "that contain unrelated headlines."
)
_LUAT_TICKER_MOI = (
    "(b) scrolling ticker bars along the bottom edge ONLY when they carry "
    "headlines unrelated to what is visible in the frame.\n"
    "   - ⚠ A bottom bar is NOT automatically a ticker. In sports coverage the "
    "lower band carries the information that identifies this exact moment — "
    "rider or player numbers, team names, time gaps, distance remaining, lap or "
    "stage number, clocks, scores. TRANSCRIBE all of that. When in doubt about "
    "whether a bottom bar is an unrelated headline or live event data, "
    "TRANSCRIBE it."
)

_V5_BAO_OCR = (
    _V5_BAO.replace(
        "Return EXACTLY ONE block:\n[FINAL_CAPTION]",
        "Return EXACTLY TWO blocks, in this order:\n[FINAL_CAPTION]\n[OCR]",
    ).replace(_LUAT_TICKER_CU, _LUAT_TICKER_MOI)
    + """

6. [OCR] BLOCK: after the caption, output the marker [OCR] on its own line, then
transcribe VERBATIM every piece of on-screen text and numbers you kept under
rule 4 — one region per line, in reading order (top to bottom, left to right).
Keep the original language and diacritics exactly as shown; do NOT translate.
Apply the SAME exclusions as rule 4: no channel logos or watermarks, and no
ticker bars carrying unrelated headlines. If the frame carries no such text,
write exactly NO_TEXT. This repeats text you already wrote inside the caption —
that repetition is intentional, do not omit it from either place."""
)
if _LUAT_TICKER_MOI not in _V5_BAO_OCR:      # neo trôi thì vỡ ngay, đừng sinh âm thầm
    raise RuntimeError("luật ticker không vá được — `_V5_BAO` đã đổi, đọc lại")

#: Biến thể cho **camera giao thông** (nhóm `N`, ~50 giờ / 296 video batch 2).
#:
#: Vì sao VIẾT MỚI chứ không vá `v5_bao`: bản gốc mở đầu bằng *"this single
#: frame from a Vietnamese television video"* và dành nguyên luật 4 cho logo
#: kênh với dải chạy tin. Camera giao thông **không phải truyền hình** — không
#: logo, không ticker, không người dẫn. Vá một prompt truyền hình để tả cảnh
#: đường phố là bắt model đi tìm thứ không tồn tại, rồi im lặng bỏ qua thứ thật
#: sự có mặt.
#:
#: Bốn khác biệt so với `v5_bao`, mỗi cái có lý do riêng:
#:
#: **1. Góc máy: CẤM nhắc.** `v5_bao` bảo nêu góc máy khi nó bất thường. Camera
#: giao thông là camera TĨNH — mọi khung của một video đều cùng một góc, nên
#: nhắc nó là thêm ~10 token giống hệt nhau vào mọi tài liệu BM25. Đó đúng là
#: cơ chế đã biến `một` và `giữa` thành stopword ở arm `object` (idf 0,09).
#:
#: **2. Đếm theo TỪNG LOẠI xe.** "nhiều xe" vô dụng cho truy hồi; "4 xe máy,
#: 1 xe buýt xanh, 2 ô tô con trắng" thì phân biệt được. 51% đề thật hỏi số
#: lượng (§4).
#:
#: **3. Chiều di chuyển và làn.** Đây là thứ thay thế cho "hành động" của
#: `v5_bao` — điểm yếu lớn nhất của caption hiện tại là tả trạng thái tĩnh chứ
#: không tả biến cố (chỉ 20,7% có SVO). Cảnh giao thông có biến cố thật: rẽ,
#: vượt, dừng đèn đỏ, đi ngược chiều.
#:
#: **4. Biển số: CHỈ chép khi đọc được.** Camera giao thông gần như luôn có
#: biển số mờ, và model rất sẵn lòng bịa ra một dãy trông hợp lý. `caption_audit`
#: **không bắt được chuyện bịa** — nó chỉ đếm *có nhắc*, không đếm *nhắc có
#: đúng*. Nên hàng rào phải nằm trong prompt, và phải soi tay khi đo.
#:
#: ⚠ CHƯA ĐO, và mẫu 140 khung cũ **không dùng để đo bản này được** — nó toàn
#: ảnh truyền hình batch 1. Phải có mẫu camera giao thông thật.
_V5_BAO_TRAFFIC = """You are an expert visual describer. Create an EXHAUSTIVELY DETAILED Dense Caption for this single frame from a fixed Vietnamese road-traffic surveillance camera.

Return EXACTLY TWO blocks, in this order:
[FINAL_CAPTION]
[OCR]

RULES:
1. SCENE FIRST: open with the road layout and viewpoint content — intersection, straight road, bridge, roundabout, toll gate, number of lanes, dividers, crosswalks, pavements, and what sits alongside the road (shophouses, trees, billboards, bus stop). Then the traffic.

2. COUNT BY VEHICLE TYPE, never in bulk. Write "three motorbikes, one white sedan, one blue city bus" — never "several vehicles". Give each notable vehicle its colour and type (motorbike, scooter, sedan, SUV, pickup, van, minibus, coach, box truck, dump truck, tanker, bicycle, cyclo, ambulance, police car). Say which lane each occupies and whether it is near, mid or far in the frame.

3. MOVEMENT AND DIRECTION — this is the most important row. State where each vehicle or group is heading (left to right, towards the camera, away from the camera, turning left/right, crossing, merging, overtaking, reversing, stopped, queuing). Describe events when present: congestion, a vehicle stopped in a live lane, a collision, a vehicle crossing the centre line or running a red light, roadworks, flooding, a broken-down vehicle.

4. PEOPLE: count pedestrians and riders. Note helmets, whether a motorbike carries one, two or three people, cargo being carried, people crossing outside a crosswalk, street vendors, traffic police.

5. CONDITIONS: time of day (daylight, dusk, night with street lighting, headlights on), weather (dry, raining, wet road, standing water, fog), and road surface state. Say if the image is dark, low-contrast, motion-blurred, or partly obscured by rain on the lens.

6. TRAFFIC CONTROL: state the colour currently shown by any traffic light, and describe road signs, lane arrows painted on the tarmac, barriers, cones, and overhead gantries.

7. ON-SCREEN TEXT (CRITICAL):
   - Surveillance overlays — the burnt-in timestamp, date, camera ID or channel name, location label — ARE meaningful here and must be transcribed. Do NOT treat them as watermarks to ignore.
   - Transcribe road signs, shop signs, banners, billboards, bus route boards, and any painted road text, verbatim and in the original language with full diacritics.
   - LICENCE PLATES: transcribe a plate ONLY if you can actually read the characters. If it is blurred, angled, too small or partly hidden, say "plate not legible" and write nothing else about it. NEVER guess or complete a plate — a plausible invented plate is worse than no plate.

8. DO NOT mention the camera angle or shot type. This camera is fixed, so the angle is identical in every frame of the video and repeating it adds nothing that distinguishes this frame.

9. NEVER invent detail. If something is unclear, describe what is actually visible and say it is unclear. Do not name a landmark, street or district unless a sign in the frame states it.

10. FORMAT: do not start with "The image shows" or similar — start directly with the scene. Write in ENGLISH. Write as many sentences as needed.

11. [OCR] BLOCK: after the caption, output the marker [OCR] on its own line, then transcribe VERBATIM every piece of on-screen text and numbers from rule 7 — one region per line, in reading order (top to bottom, left to right). Keep the original language and diacritics exactly as shown; do NOT translate. Include the timestamp and camera ID. Include a licence plate ONLY if it was legible. If the frame carries no readable text at all, write exactly NO_TEXT. This repeats text you already wrote inside the caption — that repetition is intentional, do not omit it from either place."""  # noqa: E501

#: `ten bien the -> (prompt, max_tokens mac dinh)`.
PROMPT_VARIANTS: dict[str, tuple[str, int]] = {
    "bao": (_QWEN25_PROMPT_BAO, 256),
    "aic": (_QWEN25_PROMPT_AIC, 384),
    "sota": (_QWEN25_PROMPT_SOTA, 768),
    "sota_v2": (_QWEN25_PROMPT_SOTA_V2, 768),
    "v5_bao": (_V5_BAO, 768),
    "v5_bao_ocr": (_V5_BAO_OCR, 1024),
    "v5_bao_traffic": (_V5_BAO_TRAFFIC, 1024),
}

#: Tiền tố `video_id` -> biến thể prompt cho BATCH 2.
#:
#: Để ở đây chứ không để trong notebook vì cùng lý do `OCR_MARKER` ở đây: người
#: sinh và người đọc phải dùng chung một bảng. Notebook chỉ tra, không tự quyết.
#: Marker PHẢI ghi lại biến thể đã dùng — không truy được khung nào sinh bằng
#: prompt nào là đúng lớp lỗi mà `prompt_suffix` sinh ra để tránh.
BIEN_THE_THEO_NHOM_B2: dict[str, str] = {
    "N": "v5_bao_traffic",   # camera giao thông, ~50 giờ
    "M": "v5_bao_ocr",       # tin tức truyền hình, ~100 giờ
    "S": "v5_bao_ocr",       # đua xe đạp — ĐO được là phát sóng TV, không phải cam thô
}


def bien_the_cho(video_id: str) -> str:
    """Biến thể prompt cho một `video_id` batch 2, hoặc ném nếu không biết.

    Ném chứ không trả mặc định: một tiền tố lạ nghĩa là batch 2 có nhóm nội dung
    ta chưa xem qua, và im lặng dùng prompt truyền hình cho nó là đúng kiểu hỏng
    mà không ai thấy.
    """
    try:
        return BIEN_THE_THEO_NHOM_B2[video_id[:1].upper()]
    except KeyError:
        raise ValueError(
            f"{video_id!r}: tiền tố lạ, chưa có biến thể prompt. "
            f"Đã biết: {sorted(BIEN_THE_THEO_NHOM_B2)}"
        ) from None

#: Tên khối mà `v5_bao_ocr` sinh ra. Tách ở MỘT chỗ để hai nửa — người sinh và
#: người đọc — không bao giờ lệch quy tắc (bài học `object_text`/`object_query`).
OCR_MARKER = "[OCR]"
CAPTION_MARKER = "[FINAL_CAPTION]"
NO_TEXT = "NO_TEXT"


def tach_caption_ocr(raw: str) -> tuple[str, str]:
    """Tách chuỗi `v5_bao_ocr` thành `(caption, ocr)`.

    Khoan dung có chủ ý: model đôi khi bỏ `[FINAL_CAPTION]`, đôi khi thêm dấu
    câu quanh `[OCR]`. Thiếu khối `[OCR]` thì trả `ocr = ""` chứ KHÔNG ném —
    người gọi đếm số khung rỗng và tự quyết, vì một lượt sinh cả kho không được
    chết vì một khung lạ. Nhưng `NO_TEXT` và "không có khối" là HAI chuyện khác
    nhau, và chỗ này cố tình gộp chúng thành `""`: cả hai đều nghĩa là không có
    gì để đưa vào chỉ mục.
    """
    phan = raw.split(OCR_MARKER, 1)
    caption = phan[0].replace(CAPTION_MARKER, "").strip()
    if len(phan) == 1:
        return caption, ""
    ocr = phan[1].strip()
    return caption, "" if ocr.upper().startswith(NO_TEXT) else ocr
