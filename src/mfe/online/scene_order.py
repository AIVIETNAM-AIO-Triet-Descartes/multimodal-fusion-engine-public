"""Thứ tự cảnh cho KIS — tách câu thành mệnh đề, đòi chúng đúng trình tự.

**Vì sao tầng này tồn tại.** Đề KIS thật của BTC không tả một khung, nó tả một
*đoạn*: `query-p1-12-kis.txt` viết "có 4 tài xế xe ôm công nghệ trong trạm
xăng… **Trước đó là cảnh** một người đậy nắp bình xăng". Ta đánh chỉ mục từng
khung rời, nên cấu trúc đó bị vứt: hai mệnh đề đi vào cùng một túi từ, và một
video chỉ có *một* trong hai cảnh ghi điểm y như video có **cả hai, đúng thứ
tự**.

Đếm trên 49 đề thật: **21/38 câu KIS** (55,3%) mang từ nối trình tự. Đếm trên bộ
dev 189 câu của nhóm: **10/115** (8,7%). Hai con số đó là cùng một phép đếm trên
hai bộ câu, và khoảng cách 6× là **đặc tính của bộ dev**, không phải của đề thi —
xem §"Đo ở đâu" cuối tệp.

Cơ chế
------
1. Lấy danh sách mệnh đề, **theo trình tự tường thuật** chứ không theo thứ tự
   chữ: "Trước đó là cảnh X" đẩy X lên *trước* mệnh đề chính. Hai nguồn:
   `clauses_from_texts` (người gõ tay ở giao diện, hoặc LLM tách hộ — chính xác
   hơn) và `split_clauses` (bộ luật, chạy khi không có ai). Đọc docstring của
   `clauses_from_texts` về việc vì sao bộ luật chỉ là đường dự phòng.
2. Phần gọi truy hồi **từng mệnh đề một** qua đúng bộ arm đang có.
3. `video_prior` gọi `dante.align` tìm phép gán tăng dần trong từng video — đúng
   bài toán TRAKE giải, chỉ khác đầu ra.
4. `apply_prior` nhân hệ số đó vào điểm gộp của những khung **nằm gần vị trí
   DANTE đã ghép** — không phải mọi khung của video.

Điểm khác TRAKE: TRAKE trả **N khung** làm câu trả lời, KIS trả một danh sách
khung. Ta dùng cả `Alignment.score` (video nào kể đúng câu chuyện) lẫn
`Alignment.hits` (khoảnh khắc nào trong video đó) — và lượt đo đầu, bỏ `hits` đi
để thưởng cả video, đã hỏng đúng vì thiếu nửa thứ hai. Xem `apply_prior`.

Tách phải DÈ DẶT — đây là chỗ dễ làm hỏng nhất
-----------------------------------------------
Không phải từ nối nào cũng báo một cảnh mới. Trong `triet_kis_01`, "tay đua mặc
áo vàng giơ hai tay lên ăn mừng **sau khi** về đích" là **một** khoảnh khắc: cắt
nó thành hai mệnh đề rồi đòi hai cảnh cách nhau là bịa ra một ràng buộc mà video
không có, và câu đang đúng sẽ hỏng.

Ba cái chốt, mỗi cái chặn một kiểu hỏng:

* `MIN_CLAUSE_WORDS` — hai bên phải đủ dài. "sau khi về đích" chỉ 3 từ, không
  đứng thành cảnh.
* `SCENE_CUES` tách khỏi `WEAK_CUES` — "bắt đầu bằng cảnh", "cảnh cuối cùng",
  "tiếp đến là" nêu thẳng một cảnh mới; "rồi", "sau khi" thì không. Từ nối yếu
  **chỉ** được cắt khi câu đã có ít nhất một từ nối mạnh, nghĩa là ngữ cảnh đã
  xác nhận đây là một câu kể trình tự.
* `MAX_CLAUSES` — quá nhiều mảnh thì mỗi mảnh còn quá ít chữ để truy hồi, mà
  DANTE lại đòi đủ ngần ấy vị trí tăng dần.

Đo ở đâu
--------
Bộ dev có **11** câu KIS chạm từ nối, nhưng chỉ **3** (`triet_order_01..03`) là
nhiều cảnh thật. 3/115 thì Final tổng **không** đọc được gì: một thay đổi hoàn
hảo trên 3 câu cũng nằm dưới sàn nhiễu ~0,03 của phép quét trọng số. Nên tầng
này phải đo bằng **thứ hạng từng câu trên đúng nhóm câu nó chạm**, cộng một phép
kiểm "không làm hỏng" trên 104 câu KIS còn lại — y như cách arm `object` đã được
đo và đã bị chốt trọng số 0,0.

`scripts/measure_scene_order.py` chạy đúng giao thức đó.

**Chưa nối vào `Cascade`, và đó là chủ ý.** DANTE cần `keyframe_index`, thứ chỉ
`RelationalMap` biết, mà `Cascade` không cầm bản đồ nào — nối vào là sửa một lớp
lõi cho một tính năng chưa có số đo. Cũng không có cần gạt trong config: một
trường không ai đọc sẽ nhận `0.5` từ YAML rồi im lặng không làm gì, và trông y
hệt như đã bật. Cả hai việc đó chờ số đo, và chờ cùng một lúc.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.tools import Candidate
from mfe.online import dante
from mfe.online.dante import EventHit

#: Số từ tối thiểu mỗi bên một chỗ cắt. "sau khi về đích" (3 từ) là mệnh đề phụ
#: của cùng một khoảnh khắc, không phải một cảnh.
MIN_CLAUSE_WORDS = 5

#: Cắt quá nhỏ thì mỗi mảnh không còn đủ chữ để truy hồi, mà DANTE lại đòi đủ
#: ngần ấy vị trí tăng dần trong một video — hai cái cùng siết vào một chỗ.
MAX_CLAUSES = 4

#: Từ nối **mạnh**: chúng nêu thẳng một cảnh/đoạn mới. Khoá viết không dấu vì
#: `_strip` chuẩn hoá cả hai vế — đề thật có câu gõ thiếu dấu.
#:
#: Giá trị `True` nghĩa là mệnh đề đi SAU từ nối xảy ra TRƯỚC mệnh đề đứng trước
#: nó ("Trước đó là cảnh…"). Trình tự tường thuật khi đó ngược thứ tự chữ và
#: `_to_narrative_order` đảo lại. Không đảo thì DANTE đi tìm một chuỗi tăng dần
#: theo đúng chiều sai, và câu tụt hạng thay vì lên.
SCENE_CUES: dict[str, bool] = {
    "bat dau bang canh": False,
    "bat dau bang": False,
    "mo dau bang": False,
    "canh cuoi cung la": False,
    "canh cuoi la": False,
    "canh cuoi": False,
    "ket thuc bang": False,
    "tiep den la": False,
    "tiep den": False,
    "tiep theo la": False,
    "tiep theo": False,
    "ke tiep": False,
    "ngay sau do": False,
    "sau do": False,
    "truoc do la canh": True,
    "truoc do la": True,
    "truoc do": True,
    "chuyen sang canh": False,
    "chuyen sang": False,
}

#: Từ nối **yếu**: đúng ngữ pháp thì chúng nối thời gian, nhưng phần lớn lần xuất
#: hiện nằm gọn trong *một* khoảnh khắc ("giơ tay ăn mừng sau khi về đích"). Chỉ
#: cắt khi câu đã có một từ nối mạnh — lúc đó ngữ cảnh đã xác nhận đây là câu kể
#: trình tự, và từ nối yếu nhiều khả năng cũng đang kể tiếp.
#: "lần lượt" KHÔNG có ở đây dù nó nghe như một từ nối trình tự. Đo trên đề
#: thật, cả ba lần nó xuất hiện đều liệt kê **vật thể trong cùng một cảnh**, chứ
#: không kể cảnh nối cảnh: "về đích theo thứ tự nhất, nhì, ba lần lượt là…"
#: (`query-p1-11-kis`), "ba ô… lần lượt ghi HOA KÌ, NHẬT BẢN và TÂY ÂU"
#: (`triet_order_02`). Cắt ở đó là xé một cảnh làm đôi.
WEAK_CUES: dict[str, bool] = {
    "truoc khi": True,
    "sau khi": False,
    "tiep tuc": False,
    "roi": False,
}

#: Số mệnh đề tối thiểu phải **tự** tìm được vị trí của chính nó. DANTE cho phép
#: một sự kiện "đi nhờ" vị trí do sự kiện khác tìm ra (S = 0) — với TRAKE điều đó
#: có ích, nhưng ở đây một mệnh đề đi nhờ không chứng minh được trình tự nào cả,
#: mà trình tự đúng là thứ duy nhất tầng này thưởng.
#:
#: `None` = **đòi mọi mệnh đề**, và đó là mặc định vì một hằng số cố định có
#: nghĩa khác nhau tuỳ số mệnh đề:
#:
#:     2 mệnh đề · MIN_SUPPORT=2  ->  2/2, không ai đi nhờ
#:     3 mệnh đề · MIN_SUPPORT=2  ->  2/3, MỘT mệnh đề đi nhờ miễn phí
#:     4 mệnh đề · MIN_SUPPORT=2  ->  2/4, HAI mệnh đề đi nhờ
#:
#: Nghịch lý: câu 3 mệnh đề — đúng loại câu duy nhất mà ràng buộc trình tự có
#: nghĩa — lại bị nới lỏng nhất. Đo trên bộ dev cũ, câu 3 mệnh đề **không bao
#: giờ tệ đi** (3→1, 2→2 ở cả ba cấu hình) còn câu 2 mệnh đề **luôn** tệ đi, và
#: lý do là số học: với 2 mệnh đề, hai vị trí bất kỳ trong một video xếp đúng
#: thứ tự với xác suất ~½, nên ràng buộc gần như không lọc gì.
MIN_SUPPORT: int | None = None

#: Số mệnh đề tối thiểu để tầng này được chạy. Dưới ngưỡng thì `video_prior` trả
#: rỗng và `apply_prior` thành phép không-làm-gì — xem `MIN_SUPPORT` về việc vì
#: sao 2 mệnh đề không đủ để ràng buộc tăng dần nói được điều gì.
MIN_CLAUSES_FOR_PRIOR = 3

#: Bán kính thưởng, tính bằng **keyframe** quanh mỗi vị trí DANTE đã ghép.
#: `None` = thưởng **cả video**, và đó là mặc định vì nó **thắng ở cả hai bộ đo**:
#:
#:     bộ dò 14 câu 3 cảnh   cả video: hạng TB 4,2 · 4 lên / 0 xuống
#:                           window 3: hạng TB 8,8 · 3 lên / 1 xuống
#:     8 câu dev tách tay    cả video: hạng TB 26,2 (nền 22,2)
#:                           window 3: hạng TB 27,6
#:
#: Trực giác ban đầu của tôi ngược lại — "bằng chứng về một khoảnh khắc thì đừng
#: thưởng cả video" — và nó SAI, đo hai lần đều sai. Lý do: `Alignment.hits` chỉ
#: là *một* cách ghép chuỗi tăng dần, không phải toàn bộ vùng cảnh; siết quanh ba
#: vị trí ấy thì bỏ lỡ chính khung mà người chấm chọn. Còn thưởng cả video thì
#: nhân đều nên **giữ nguyên thứ hạng bên trong video** — các arm vẫn quyết định
#: khung nào lên trước, trình tự chỉ quyết định video nào lên trước.
#:
#: Giữ tham số lại để đo hai bản cạnh nhau, không phải để dùng.
WINDOW: int | None = None

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_LEAD_PUNCT = re.compile(r"^[\s,.;:–—-]+")
_TRAIL_PUNCT = re.compile(r"[\s,.;:–—-]+$")
_LEAD_COPULA = re.compile(r"^(?:la|là)\s+", re.IGNORECASE)


def _strip(text: str) -> str:
    """Bỏ dấu, hạ chữ thường — để khoá từ nối viết được bằng ASCII.

    Trả về chuỗi **cùng độ dài** với đầu vào (NFD rồi bỏ dấu kết hợp không đổi số
    ký tự cơ sở), nên vị trí tìm thấy trên bản không dấu dùng thẳng được để cắt
    bản gốc. Nếu chuẩn hoá làm lệch độ dài thì mọi chỗ cắt sẽ lệch theo.
    """
    decomposed = unicodedata.normalize("NFD", text)
    plain = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return plain.replace("đ", "d").replace("Đ", "D").lower()


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


@dataclass(frozen=True)
class Clause:
    """Một mệnh đề, kèm chỗ nó đứng trong **trình tự tường thuật**.

    `order` là thứ tự thời gian, không phải thứ tự chữ: "Trước đó là cảnh X" cho
    X một `order` nhỏ hơn mệnh đề đứng trước nó trong câu.
    """

    text: str
    order: int
    cue: str | None = None


def _cue_positions(
    plain: str, cues: dict[str, bool]
) -> list[tuple[int, str, bool]]:
    """`(vị trí, từ nối, before)` cho mọi từ nối tìm thấy, không chồng nhau.

    Duyệt từ nối **dài trước** nên "truoc do la canh" thắng "truoc do", và
    "bat dau bang canh" thắng "bat dau bang". Không có thứ tự đó thì mảnh cắt ra
    còn dính lại chữ "la canh" ở đầu.
    """
    found: list[tuple[int, str, bool]] = []
    taken: list[tuple[int, int]] = []
    for cue in sorted(cues, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(cue)}(?!\w)")
        for match in pattern.finditer(plain):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            taken.append((start, end))
            found.append((start, cue, cues[cue]))
    found.sort()
    return found


def _tidy(body: str) -> str:
    """Bỏ dấu câu và hệ từ rơi lại ở hai đầu chỗ cắt."""
    body = _LEAD_PUNCT.sub("", body.strip())
    body = _TRAIL_PUNCT.sub("", body)
    return _LEAD_COPULA.sub("", body).strip()


def _to_narrative_order(
    items: Sequence[tuple[str, str | None, bool]],
) -> list[Clause]:
    """Đổi thứ tự chữ thành thứ tự thời gian.

    Mảnh mang từ nối `before=True` ("trước đó là cảnh…") xảy ra **trước** mảnh
    liền trước nó trong câu, nên hai mảnh đổi chỗ. Đây là chỗ duy nhất trong tệp
    biết rằng thứ tự chữ và thứ tự thời gian có thể ngược nhau.
    """
    bodies: list[str] = []
    cues: list[str | None] = []
    for body, cue, before in items:
        if before and bodies:
            bodies.insert(len(bodies) - 1, body)
            cues.insert(len(cues) - 1, cue)
        else:
            bodies.append(body)
            cues.append(cue)
    return [
        Clause(text=body, order=index, cue=cue)
        for index, (body, cue) in enumerate(zip(bodies, cues, strict=True))
    ]


def clauses_from_texts(texts: Sequence[str]) -> list[Clause]:
    """Dựng `Clause` từ một danh sách mệnh đề **đã tách sẵn**, đúng trình tự.

    Đây là đường vào cho người gõ tay ở giao diện, hoặc cho một LLM tách hộ —
    và nó là đường **tốt hơn** `split_clauses` chứ không phải đường phụ.

    Bộ luật của `split_clauses` dò tiếng Việt bằng biểu thức chính quy, và số đo
    của chính nó nói ra giới hạn đó: trên 38 đề KIS thật có **21 câu mang từ nối
    trình tự**, luật tách được **14**. Bảy câu còn lại bị bỏ vì `MIN_CLAUSE_WORDS`
    và quy tắc "từ nối yếu cần một từ nối mạnh" — hai cái chốt phải có, nếu
    không luật sẽ xé "giơ tay ăn mừng sau khi về đích" thành hai cảnh. Nới ra để
    bắt thêm bảy câu là mở cửa cho kiểu hỏng đắt hơn.

    Người đọc câu thì không có đánh đổi đó. Lúc thi có người ngồi ở bàn phím,
    nên mệnh đề gõ tay là nguồn chính xác nhất đang có; `split_clauses` là thứ
    chạy khi **không** có ai — đo hàng loạt trên dev/heldout, và lượt nộp tự
    động.

    `order` gán theo đúng thứ tự truyền vào: phần gọi chịu trách nhiệm đưa mệnh
    đề theo **trình tự thời gian**, không phải thứ tự chúng xuất hiện trong câu.
    Với "Trước đó là cảnh X" thì X phải đứng đầu danh sách.
    """
    bodies = [_tidy(t) for t in texts]
    bodies = [b for b in bodies if b]
    return [Clause(text=body, order=index) for index, body in enumerate(bodies)]


def split_clauses(text: str) -> list[Clause]:
    """Tách câu KIS thành các mệnh đề, xếp theo **trình tự tường thuật**.

    Trả về **một** phần tử khi câu không kể trình tự — và đó là đường ra thường
    gặp nhất, có chủ đích. Phần gọi kiểm `len(...) < 2` để bỏ qua cả tầng.

    Ba cái chốt dè dặt (`MIN_CLAUSE_WORDS`, `SCENE_CUES` vs `WEAK_CUES`,
    `MAX_CLAUSES`) nằm ở docstring đầu tệp cùng lý do từng cái.
    """
    # NFC trước khi làm gì khác: `_strip` chỉ giữ được độ dài khi đầu vào đã
    # dựng sẵn. Với đầu vào tách sẵn (NFD), mọi vị trí tìm thấy trên bản không
    # dấu sẽ lệch so với bản gốc và mọi chỗ cắt lệch theo.
    text = unicodedata.normalize("NFC", text)
    whole = [Clause(text=text.strip(), order=0)]
    plain = _strip(text)
    if len(plain) != len(text):  # pragma: no cover - canh gac cho `_strip`
        return whole

    strong = _cue_positions(plain, SCENE_CUES)
    if not strong:
        return whole

    # HAI LƯỢT, mạnh trước — và thứ tự đó là bắt buộc, không phải tối ưu hoá.
    # Duyệt chung một danh sách theo vị trí thì một nhát cắt yếu **sai** ăn mất
    # một nhát mạnh **đúng** đứng ngay sau: ở `query-p1-13-kis`, cắt tại "rồi"
    # để lại "đèn." trước "Tiếp theo là cảnh…", và mảnh 1 từ đó làm nhát "tiếp
    # theo" trượt `MIN_CLAUSE_WORDS`. Kết quả là mệnh đề đầu còn đúng sáu chữ vô
    # nghĩa. Cắt hết chỗ mạnh trước rồi mới chen chỗ yếu vào các khoảng còn lại
    # thì từ nối yếu không bao giờ hi sinh được một ranh giới cảnh thật.
    accepted: list[tuple[int, int, str, bool]] = []
    for marks in (strong, _cue_positions(plain, WEAK_CUES)):
        for start, cue, before in marks:
            end = start + len(cue)
            if any(start < c_end and c_start < end for c_start, c_end, _, _ in accepted):
                continue
            left = max((c_end for _, c_end, _, _ in accepted if c_end <= start), default=0)
            right = min(
                (c_start for c_start, _, _, _ in accepted if c_start >= end),
                default=len(text),
            )
            # Hai bên của nhát cắt này, TRONG mảnh nó rơi vào — không phải toàn
            # bộ phần còn lại của câu.
            if _word_count(text[left:start]) < MIN_CLAUSE_WORDS:
                continue
            if _word_count(text[end:right]) < MIN_CLAUSE_WORDS:
                continue
            accepted.append((start, end, cue, before))
    accepted.sort()

    pieces: list[tuple[str, str | None, bool]] = []
    pending: tuple[str | None, bool] = (None, False)
    cursor = 0
    for start, end, cue, before in accepted:
        pieces.append((text[cursor:start], pending[0], pending[1]))
        pending = (cue, before)
        cursor = end
    pieces.append((text[cursor:], pending[0], pending[1]))

    cleaned = [
        (_tidy(body), cue, before)
        for body, cue, before in pieces
        if _word_count(body) >= MIN_CLAUSE_WORDS
    ]
    if len(cleaned) < 2 or len(cleaned) > MAX_CLAUSES:
        return whole
    return _to_narrative_order(cleaned)


@dataclass(frozen=True)
class ScenePrior:
    """Bằng chứng trình tự của một video: điểm, **và chỗ nó nằm**.

    `ordinals` là `keyframe_index` của các vị trí DANTE đã ghép — tức là chính
    những khoảnh khắc mà mệnh đề mô tả. Giữ chúng lại là chỗ khác biệt giữa bản
    này và bản đầu: bản đầu chỉ giữ `score` rồi nhân vào **mọi** khung của video,
    và đo được rằng đó là cách áp sai (xem `apply_prior`).
    """

    score: float
    ordinals: tuple[int, ...]


def video_prior(
    per_clause: Sequence[Sequence[EventHit]],
    *,
    penalty: float = dante.PENALTY,
    min_support: int | None = MIN_SUPPORT,
    min_clauses: int = MIN_CLAUSES_FOR_PRIOR,
) -> dict[str, ScenePrior]:
    """`{video_id: ScenePrior}` cho video ghép được chuỗi tăng dần.

    Video **không** có mặt trong dict khi nó không kể được câu chuyện theo đúng
    thứ tự. Đó là hình dạng đúng: `apply_prior` chỉ **thưởng**, không phạt, nên
    một video vắng mặt giữ nguyên điểm gộp của nó.

    Vì sao chỉ thưởng: bằng chứng trình tự chỉ chạy trên các mệnh đề *ta tách
    được*, và tách sai là chuyện có thật (xem ba cái chốt ở đầu tệp). Phạt một
    video vì nó không khớp một cấu trúc **ta có thể đã bịa ra** thì mỗi lần tách
    hỏng là một lần đánh tụt chính câu trả lời đúng.

    Chia cho điểm cao nhất để prior nằm trong (0, 1] bất kể thang điểm arm — cùng
    lý do `dante.normalise_events` chuẩn hoá, và nó làm `weight` của
    `apply_prior` đọc được giữa các cấu hình.

    Dưới `min_clauses` mệnh đề thì trả rỗng ngay, không gọi DANTE: với 2 mệnh đề
    ràng buộc tăng dần lọc được ~½ số video, tức gần như không lọc gì, và mọi
    lần đo đều cho thấy nó **hại**.
    """
    if len(per_clause) < max(2, min_clauses):
        return {}
    alignments = dante.align(
        list(per_clause),
        penalty=penalty,
        # `None` = doi MOI menh de co bang chung that. Xem `MIN_SUPPORT`.
        min_support=len(per_clause) if min_support is None else min_support,
    )
    if not alignments:
        return {}
    top = max(item.score for item in alignments)
    if top <= 0:
        return {}
    return {
        item.video_id: ScenePrior(
            score=item.score / top,
            ordinals=tuple(hit.ordinal for hit in item.hits),
        )
        for item in alignments
    }


def apply_prior(
    candidates: Sequence[Candidate],
    prior: dict[str, ScenePrior],
    *,
    ordinal_of: Callable[[str], int] | None = None,
    weight: float,
    window: int | None = WINDOW,
) -> list[Candidate]:
    """`score * (1 + weight · prior)`, chỉ cho khung **gần vị trí đã ghép**.

    Nhân chứ không cộng: cộng một hằng số đẩy mọi khung được thưởng lên bằng
    nhau, xoá thứ hạng mà các arm đã dựng bên trong cụm đó.

    **Mặc định thưởng CẢ VIDEO** (`window=None`), và đó là kết quả đo chứ không
    phải mặc định cho tiện — `WINDOW` ghi cả hai bảng số. Trực giác ngược lại
    ("bằng chứng về một khoảnh khắc thì đừng thưởng cả video") đã được thử và
    **thua ở cả hai bộ đo**: `hits` của DANTE chỉ là *một* cách ghép chuỗi tăng
    dần, không phải toàn bộ vùng cảnh, nên siết quanh ba vị trí ấy làm bỏ lỡ
    chính khung người chấm chọn. Nhân đều lên cả video thì giữ nguyên thứ hạng
    *bên trong* video — arm vẫn quyết khung nào lên trước, trình tự chỉ quyết
    video nào lên trước.

    `window` là một số thì thưởng gói trong ngần ấy keyframe quanh mỗi vị trí đã
    ghép. Khi ấy cần `ordinal_of` để tra `keyframe_index` từ `keyframe_id` — chỉ
    `RelationalMap` biết số đó (`frame_idx` KHÁC `n`, xem `mfe.contracts.ids`).
    Thiếu nó thì **ném**, chứ không lặng lẽ bỏ qua `window`: bỏ qua trong im
    lặng sẽ cho ra một phép đo của cấu hình khác với cấu hình người đo tưởng
    mình đang đo.

    `weight <= 0` hoặc `prior` rỗng trả lại đúng danh sách cũ, **cùng thứ tự** —
    đường mặc định của mọi câu không kể trình tự phải là một phép không-làm-gì
    thật sự, không phải một lần xếp lại "gần như cũ".
    """
    if weight <= 0 or not prior:
        return list(candidates)
    if window is not None and ordinal_of is None:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            "window can `ordinal_of` de tra keyframe_index; "
            "truyen window=None neu that su muon thuong ca video",
        )

    boosted: list[Candidate] = []
    for candidate in candidates:
        item = prior.get(candidate.video_id)
        if item is None:
            boosted.append(candidate)
            continue
        if window is not None:
            ordinal = ordinal_of(candidate.keyframe_id)  # type: ignore[misc]
            if min((abs(ordinal - o) for o in item.ordinals), default=window + 1) > window:
                boosted.append(candidate)
                continue
        boosted.append(
            candidate.model_copy(
                update={"score": candidate.score * (1.0 + weight * item.score)}
            )
        )
    boosted.sort(key=lambda c: (-c.score, c.keyframe_id))
    return boosted
