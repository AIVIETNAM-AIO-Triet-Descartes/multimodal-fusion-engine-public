"""Prompt sinh caption + OCR cho Qwen2.5-VL. **DỮ LIỆU, không phải mã.**

Tách khỏi `stages.py` vì hai lý do, và lý do thứ hai mới là lý do thật:

1. `stages.py` đã hơn 2.000 dòng, và prompt là thứ người ta sửa nhiều nhất.
2. **Xuống dòng cho vừa 100 cột là đổi đúng chuỗi ký tự model nhận được.** Một
   prompt bị sửa vì lý do định dạng là thứ không ai nghĩ tới khi đi tìm nguyên
   nhân kết quả đổi. Nên file này được `pyproject.toml` tha `E501` — cùng cách
   `mfe/online/quest.py` đã được tha, vì cùng một lý do.

Hai biến thể cùng tồn tại, và cả hai đều được dùng:

| biến thể | dùng khi | tiếng | max_tokens |
|---|---|---|---|
| `bao` | lượt **probe** đối chiếu với bản Bảo đã chạy | Anh | 256 |
| `aic` | lượt chạy **thật** | Anh | 384 |

Giữ `bao` không phải để dự phòng: nó là phép kiểm **đường ống** duy nhất ta có.
`notebooks/reference/dense_captions_qwen.json` (855 keyframe) chỉ đối chiếu được
với chính prompt đã sinh ra nó. Đổi prompt rồi so với nó là so hai thứ khác nhau,
và khi lệch sẽ không biết lỗi nằm ở địa chỉ khoá, ở động cơ vLLM, hay ở prompt.

Vì sao caption viết TIẾNG ANH (quyết định của Lead, 25-08-2026)
---------------------------------------------------------------
Caption đi vào **hai** chỗ, không phải một:

    n6_caption_qwen -> qwen_captions.json --+--> arm `caption_qwen`  (BM25, văn bản)
                                            +--> n6_bge_m3 -> arm `bge_m3` (embedding)

BGE-M3 là mô hình **đa ngữ**: truy vấn tiếng Việt khớp caption tiếng Anh mà không
cần bước dịch nào (`DENSE_ARM_SPECS["bge_m3"]` khai `query_language = "any"`). Và
nhánh BM25 cũng không mất, vì đường truy vấn **đã có sẵn** tầng dịch sang tiếng
Anh phục vụ arm `siglip` (`--en-source auto`).

Nên một caption tiếng Anh phục vụ được cả hai arm, còn caption tiếng Việt chỉ
phục vụ được một. Đó là lý do, chứ không phải "tiếng Anh mô tả tốt hơn".

Một điều luật KHÔNG đổi theo: nửa `[TEXT]` **cấm dịch**. Nó là chữ trên màn hình,
là nguồn của arm BM25 `ocr_qwen25`, và nó phải giữ nguyên dấu tiếng Việt.
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
#: §3, va moi dieu khoan tra loi mot con so dem duoc tren 49 de that (do lai bang
#: chinh `caption_audit.QUERY_DEMAND`):
#:
#:   mau gan vao TUNG vat 63,3%  ·  quan he giua cac vat 57,1%  ·  vi tri 36,7%
#:   goc may 16,3%               ·  dem so luong (xem canh bao trong bang)
#:
#: NGON NGU: **tieng Anh**, giong Bao (quyet dinh cua Lead 25-08-2026). Ly do la
#: dich den cua caption: `n6_bge_m3` -> arm `bge_m3`, va BGE-M3 la mo hinh **da
#: ngu** — truy van tieng Viet khop caption tieng Anh khong can dich. Con duong
#: BM25 (`caption_qwen`) cung khong mat: duong truy van da co san ban dich tieng
#: Anh cho arm `siglip` (`--en-source auto`), nen mot caption tieng Anh phuc vu ca
#: hai arm, con caption tieng Viet chi phuc vu mot.
#:
#: Bon khac biet so voi prompt cua Bao, moi cai ung mot con so do duoc tren chinh
#: 855 keyframe ban ay da chay:
#:
#: 1. **QUAN HE giua cac vat la dieu khoan SO 1.** Bao do duoc 50,5% trong khi
#:    57,1% de that hoi mot hanh dong. Mot caption liet ke "mot nguoi, mot voi
#:    nuoc, may chau cay" khong tra loi duoc de "nguoi dang tuoi cay" — dung vat
#:    the, sai su viec. Day la yeu cau cua Lead ngay 25-08-2026.
#: 2. **Cam mo dau bang "The image shows".** Bao: **59,9%**. Khong phai chuyen
#:    tham my: moi caption mo dau giong nhau thi cum do thanh stopword tren toan
#:    chi muc — dung co che da giet arm `object` (`mot` 91,0% tai lieu, idf 0,09).
#: 3. **Bo duong thoat "None".** Prompt cua Bao cho san mot tu de bo cuoc, va mot
#:    model se dung no. `stage_n5_ocr_qwen` da tra gia cho bai hoc nay: cho san
#:    duong thoat thi 305/307 anh tra ve "None" trong khi easyocr tim thay chu
#:    tren ca 307. Khong co chu thi de TRONG.
#: 4. **[TEXT] cam DICH.** Prompt cua Bao chi noi "including Vietnamese if any".
#:    Nua `[TEXT]` la nguon cua arm BM25 `ocr_qwen25`, va no chi lam viec neu chu
#:    tren man hinh duoc chep dung tung dau thanh — mot ban dich sang tieng Anh o
#:    day la mot arm chet, khong phai mot caption kem.
_QWEN25_INSTRUCTION_AIC = """You are describing ONE frame taken from a Vietnamese television video.

Return EXACTLY two blocks, in this order, each starting with its label:

[CAPTION] a description in ENGLISH, 3-5 sentences, covering ONLY what is visible.
[TEXT] every readable piece of text in the frame, copied VERBATIM.

RULES FOR [CAPTION] — five things you MUST state whenever the frame shows them:

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
   shows the evidence — a uniform, a blackboard, a market stall, a whistle. With no such
   evidence, describe the action and leave the role out.
2. COLOUR attached to EACH object, not the overall mood of the scene.
   GOOD: "the rider in a yellow jersey and black shorts leads the rider in a blue jersey"
   BAD:  "the riders wear colourful outfits"
3. HOW MANY, whenever they can be counted: "three people", "two motorbikes", "five plastic
   stools". Count what you can see; write "several" only when they are too many to count.
4. WHERE things are, relative to each other: on the left / in the middle / on the right /
   in front of / behind. Do not settle for "in the background".
5. CAMERA ANGLE only when it is UNUSUAL: a close-up, an overhead or aerial view, a low angle
   looking up. An ordinary eye-level shot is the default — say nothing about it. Never end a
   caption with a stock phrase about the shot.

FORBIDDEN in [CAPTION]:
- Guessing what you cannot see: place names, personal names, the topic of the news story,
  emotions, intentions, what happened before or after.
  "a report about flooding" is a GUESS. "a car drives through brown water that reaches the
  middle of its wheels" is SEEN.
- Starting with "The image shows", "This image", "In the picture", "A photo of" — start
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

#: Few-shot cho bien the `aic`. **CHI VAN BAN, khong kem anh mau** — mot anh mau
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
#: kho noi nhat cua prompt — mo ta thi dich, chu tren man hinh thi khong.
_QWEN25_FEWSHOT_AIC = """Here are three examples of the correct output FORMAT (for other frames).
Learn the shape of the answer; do not copy the content.

### Example 1
[CAPTION] Two presenters read from a red folder and speak toward the camera from behind a curved white desk. The man on the left wears glasses, a dark navy shirt and a pale tie; the woman on the right has long hair and wears a light blue blouse. The man rests his free hand on the desk while the woman turns a page. Behind them a wide city skyline glows orange at sunset.
[TEXT] HTV9 HD
06:31:14
TIN CHÍNH
Đợt mưa lũ vừa qua khiến 3 người thiệt mạng

### Example 2
[CAPTION] Five people peel bamboo shoots together on a concrete yard, sitting on low plastic stools around the heap. The woman on the far left, in a red and white checked shirt and yellow trousers, strips the husk from a shoot; the person in the middle in a green t-shirt passes a peeled shoot toward her; the man on the far right in camouflage-patterned clothing sorts the shoots into a yellow plastic basket. A row of blue plastic drums stands behind them.
[TEXT]

### Example 3
[CAPTION] Close-up of a lecture slide on a white background. Three rounded blue buttons run down the left half, each one pointing to an empty white box on its right. In the lower left corner an inset shows a male lecturer in a white shirt sitting in front of an open laptop and looking at its screen.
[TEXT] THANH NIÊN
HOA KÌ
NHẬT BẢN
TÂY ÂU
CHƯƠNG TRÌNH ÔN THI THPT MÔN LỊCH SỬ

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

RULES FOR [CAPTION] — five things you MUST state whenever the frame shows them:
1. WHAT THE PEOPLE AND OBJECTS ARE DOING WITH EACH OTHER. Never list things side by side; join them with a verb that says what is happening.
   **Your FIRST sentence must be the action.** Say what is happening, then spend the rest of the caption on who and what: colours, counts, positions.
   GOOD: "a man in a blue shirt waters a row of potted plants with a green hose"
   BAD:  "a man, a green hose, and several potted plants"
2. COLOUR attached to EACH object.
3. HOW MANY, whenever they can be counted: "three people", "two motorbikes".
4. WHERE things are, relative to each other: on the left / in the middle / on the right / in front of / behind.
5. CAMERA ANGLE only when it is UNUSUAL (close-up, aerial).

FORBIDDEN in [CAPTION]:
- Guessing what you cannot see: place names, personal names, emotions, intentions, unseen structural damage.
- Starting with "The image shows", "This image", "In the picture", "A photo of" — start with the content itself.
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

#: `ten bien the -> (prompt, max_tokens mac dinh)`.
PROMPT_VARIANTS: dict[str, tuple[str, int]] = {
    "bao": (_QWEN25_PROMPT_BAO, 256),
    "aic": (_QWEN25_PROMPT_AIC, 384),
    "sota": (_QWEN25_PROMPT_SOTA, 768),
}
