"""DANTE — Dynamic Alignment of Narrative Temporal Events, for TRAKE.

    Luu, Nguyen-Dinh, Bui, Tran, Le, Nguyen-Nhu (2025), arXiv:2512.13169 §4.3
    `docs/triet/2512.13169v1.pdf`

TRAKE asks for **one video and N frames in narrative order** (PDF §2.1.3). Sai
video là 0 điểm ngay; đúng video thì điểm là tỉ lệ khoảnh khắc rơi đúng cửa sổ.
Nên bài toán không phải "xếp hạng keyframe" mà là **gán N sự kiện vào N vị trí
tăng dần trong cùng một video**, và đó là quy hoạch động.

Công thức của paper (Eq. 2):

    DP[i, t] = S[i, t] + max ( DP[i-1, τ] − λ·(t − τ) )
                        τ ∈ [s_v, t−1]

`S[i, t]` là độ tương đồng của truy vấn sự kiện `i` với keyframe `t`. `λ` phạt
khoảng cách thời gian: nó **ưu tiên** τ gần t mà **không cấm** τ xa, nên một
chuỗi sự kiện thưa vẫn ghép được nếu độ tương đồng đủ mạnh. Paper hiệu chỉnh
λ ∈ [0,001, 0,01] ở vòng chung kết AIC 2025: 0,001 hợp khi khoảng cách sự kiện
là 3–15 keyframe, 0,01 khi 1–3.

Chạy được O(N·T) nhờ đưa `λ·τ` sang vế running max (Eq. 3, 4):

    running_max = max( running_max, DP[i-1, t-1] + λ·(t-1) )
    DP[i, t]    = S[i, t] + running_max − λ·t

Ba chỗ ta CHỆCH khỏi paper, và lý do
--------------------------------------
**1. `t` chạy trên keyframe ĐÃ TRUY HỒI, không trên mọi keyframe của video.**
Paper tính `S[i, t]` cho *toàn bộ* T keyframe bằng cách quét Milvus. Ta đã có
một tầng gộp nhiều arm (`Cascade`) trả về danh sách xếp hạng cho mỗi sự kiện,
nên `S` của ta thưa: keyframe nào không ai truy hồi thì không có điểm.

Điều đó **không** làm hỏng phép phạt, vì `λ` vẫn nhân với **số thứ tự thật**
(`keyframe_index`) chứ không phải chỉ số trong danh sách thưa. Nếu tính phạt
theo chỉ số danh sách thì hai keyframe cách nhau 200 frame và hai keyframe liền
nhau sẽ bị phạt như nhau — sai hẳn ý nghĩa của λ.

**2. Vị trí không được sự kiện `i` truy hồi vẫn dùng được, với `S = 0`.** Paper
cho phép mọi `t`; giữ điều đó nghĩa là một sự kiện yếu vẫn có thể "đi nhờ" qua
một vị trí do sự kiện khác tìm ra, thay vì làm cả video mất khả năng ghép.

**3. Chuẩn hoá điểm theo từng sự kiện.** `S` của paper là cosine ∈ [−1, 1], nên
λ của họ đọc được. Điểm của ta là điểm gộp RRF, thang tuỳ số arm và `rrf_k` —
cùng một λ sẽ có nghĩa khác nhau giữa hai cấu hình. `normalize=True` chia mỗi
danh sách cho điểm cao nhất của chính nó, đưa về (0, 1] và làm λ so sánh được
giữa các lần chạy. Chia cho max chứ không min-max: min-max để một ngoại lai nén
phần còn lại về gần 0 (xem `fusion.normalize_scores`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from mfe.contracts.errors import ContractError, ErrorCode

logger = logging.getLogger(__name__)

#: Giá trị paper hiệu chỉnh ở chung kết AIC 2025 (§4.3): 0,001 khi khoảng cách
#: sự kiện là 3–15 keyframe, 0,01 khi 1–3. Mặc định lấy đầu thưa — TRAKE của
#: BTC mô tả một chuỗi hành động (chạy đà → giậm nhảy → qua xà → tiếp đất), và
#: những khoảnh khắc đó cách nhau nhiều hơn một keyframe.
#:
#: **Chỉ đọc được cùng `normalize=True`.** Không chuẩn hoá thì λ đứng cạnh điểm
#: RRF thô, và cùng con số sẽ phạt mạnh yếu khác nhau tuỳ số arm đang bật.
PENALTY = 0.001


@dataclass(frozen=True)
class EventHit:
    """Một keyframe ứng viên cho một sự kiện, kèm toạ độ để xếp thứ tự.

    `ordinal` là `keyframe_index` — số thứ tự keyframe **trong video**, thứ mà
    phép phạt của DANTE đo khoảng cách trên. `frame_id` là frame của video gốc,
    thứ BTC chấm. Hai cái này khác nhau và `mfe.contracts.ids` giải thích dài
    về việc nhầm chúng; ở đây cả hai đều cần nên cả hai đều được mang theo.
    """

    keyframe_id: str
    video_id: str
    ordinal: int
    frame_id: int
    score: float


@dataclass(frozen=True)
class Alignment:
    """Một lời giải TRAKE cho một video: N keyframe, thứ tự tăng dần."""

    video_id: str
    score: float
    hits: tuple[EventHit, ...]
    #: Số sự kiện thực sự **truy hồi được** vị trí đã gán cho nó. Phần còn lại là
    #: những sự kiện "đi nhờ" một vị trí do sự kiện khác tìm ra (S = 0).
    #:
    #: Cần đếm riêng vì `score` không phân biệt được hai chuyện: một chuỗi khớp
    #: yếu và một chuỗi **không có bằng chứng nào** đều cho điểm gần 0. Cái thứ
    #: hai là một lần đoán không có gì đỡ, và `align` loại nó bằng `min_support`.
    support: int = 0

    @property
    def frame_ids(self) -> list[int]:
        return [hit.frame_id for hit in self.hits]

    @property
    def keyframe_ids(self) -> list[str]:
        return [hit.keyframe_id for hit in self.hits]


def normalise_events(
    per_event: Sequence[Sequence[EventHit]],
) -> list[list[EventHit]]:
    """Chia điểm mỗi sự kiện cho điểm cao nhất của **chính sự kiện đó, trên MỌI
    video**. Đưa về (0, 1] và làm `penalty` so sánh được giữa các lần chạy.

    **Phải chuẩn hoá toàn cục, không theo từng video** — và đó là một lỗi đã mắc
    rồi sửa. Chuẩn hoá bên trong từng video khiến video nào cũng có ứng viên tốt
    nhất bằng 1.0, nên `Alignment.score` của mọi video bằng nhau và **thứ hạng
    video biến mất**. Với TRAKE đó là hỏng đúng thứ quan trọng nhất: sai video là
    0 điểm ngay lập tức (PDF §2.1.3), nên chọn đúng video mới là phần khó.

    Chia cho max chứ không min-max: min-max để một ngoại lai nén phần còn lại về
    gần 0 (xem `fusion.normalize_scores`).
    """
    out: list[list[EventHit]] = []
    for hits in per_event:
        top = max((hit.score for hit in hits), default=0.0)
        if top <= 0:
            # Mọi điểm <= 0: chia sẽ đổi dấu hoặc chia cho 0. Giữ nguyên và để
            # phần gọi thấy — bất thường, không phải chỗ cần vá.
            out.append(list(hits))
            continue
        out.append([
            EventHit(h.keyframe_id, h.video_id, h.ordinal, h.frame_id, h.score / top)
            for h in hits
        ])
    return out


def _best_per_ordinal(hits: Sequence[EventHit]) -> dict[int, float]:
    """`{ordinal: điểm cao nhất}`. Hai arm có thể trả về cùng keyframe."""
    best: dict[int, float] = {}
    for hit in hits:
        if hit.score > best.get(hit.ordinal, -math.inf):
            best[hit.ordinal] = hit.score
    return best


def align_video(
    per_event: Sequence[Sequence[EventHit]],
    *,
    penalty: float = PENALTY,
) -> Alignment | None:
    """DANTE trên MỘT video. `None` khi không có phép gán tăng dần nào.

    `per_event[i]` là danh sách ứng viên của sự kiện thứ `i`, **cùng một video**.
    Trả về `N` keyframe với `ordinal` **tăng ngặt** — sự kiện `i+1` không thể xảy
    ra trước sự kiện `i`, và đó là ràng buộc làm TRAKE khác một phép xếp hạng.

    **Dùng điểm y như được đưa vào, không chuẩn hoá.** Chuẩn hoá là việc của
    `normalise_events`, chạy MỘT lần trên toàn bộ ứng viên trước khi gom theo
    video — đọc docstring của nó về việc vì sao làm ở đây là xoá mất thứ hạng
    video.
    """
    if not per_event:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, "per_event rong")
    if penalty < 0:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, f"penalty phai >= 0, nhan {penalty}"
        )

    videos = {hit.video_id for hits in per_event for hit in hits}
    if len(videos) > 1:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"align_video nhan nhieu video: {sorted(videos)}. Gom theo video truoc.",
        )
    if not videos:
        return None
    video_id = videos.pop()

    scores: list[dict[int, float]] = [_best_per_ordinal(hits) for hits in per_event]
    by_ordinal: dict[int, EventHit] = {}
    for hits in per_event:
        for hit in hits:
            by_ordinal.setdefault(hit.ordinal, hit)

    # Miền của `t`: hợp mọi vị trí ĐƯỢC truy hồi, sắp tăng dần. Sự kiện không
    # truy hồi một vị trí vẫn dùng được vị trí đó với S = 0 — xem docstring §2.
    positions = sorted(by_ordinal)
    n_events, n_pos = len(per_event), len(positions)
    if n_pos < n_events:
        return None

    neg_inf = -math.inf
    dp = [scores[0].get(t, 0.0) for t in positions]
    back: list[list[int]] = [[-1] * n_pos for _ in range(n_events)]

    for event in range(1, n_events):
        nxt = [neg_inf] * n_pos
        running, running_at = neg_inf, -1
        for index in range(1, n_pos):
            previous = index - 1
            if dp[previous] > neg_inf:
                # Eq. (3): day `λ·τ` sang running max de khong phai quet lai τ.
                offer = dp[previous] + penalty * positions[previous]
                if offer > running:
                    running, running_at = offer, previous
            if running == neg_inf:
                continue  # chua co vi tri nao dung duoc cho su kien truoc
            # Eq. (4)
            nxt[index] = (
                scores[event].get(positions[index], 0.0)
                + running
                - penalty * positions[index]
            )
            back[event][index] = running_at
        dp = nxt

    best_index, best_score = -1, neg_inf
    for index, value in enumerate(dp):
        if value > best_score:
            best_index, best_score = index, value
    if best_index < 0 or best_score == neg_inf:
        return None

    chain: list[int] = [best_index]
    for event in range(n_events - 1, 0, -1):
        chain.append(back[event][chain[-1]])
    chain.reverse()

    hits = tuple(by_ordinal[positions[index]] for index in chain)
    support = sum(
        1 for event, index in enumerate(chain) if positions[index] in scores[event]
    )
    return Alignment(
        video_id=video_id, score=best_score, hits=hits, support=support
    )


def align(
    per_event: Sequence[Sequence[EventHit]],
    *,
    penalty: float = PENALTY,
    normalize: bool = True,
    limit: int | None = None,
    min_support: int = 1,
) -> list[Alignment]:
    """DANTE trên mọi video xuất hiện trong `per_event`, xếp hạng giảm dần.

    Đây là hình dạng TRAKE cần: mỗi video cho **nhiều nhất một** câu trả lời (một
    câu trả lời TRAKE nêu đúng một video), và các video được xếp theo điểm.

    Video nào không ghép được chuỗi tăng dần thì bị **bỏ**, không phải độn thêm
    frame cho đủ N. Một frame độn ghi điểm như một lần đoán sai và không phân
    biệt được với một lần thử thật.

    `min_support` là cùng một nguyên tắc, ở một dạng tinh hơn. DANTE cho phép một
    sự kiện đi nhờ một vị trí do sự kiện khác tìm ra (S = 0) — đó là tính chất
    của paper và nó có ích. Nhưng nó cũng khiến một video **có bằng chứng mâu
    thuẫn về thời gian** vẫn sinh ra câu trả lời: sự kiện 1 tìm thấy frame 900,
    sự kiện 2 tìm thấy frame 20, và lời giải tăng dần duy nhất gán cả hai vào
    đúng chỗ **của nhau**, cho tổng điểm 0.

    Câu trả lời đó ghi 0 điểm y như không nộp gì, nhưng nó **chiếm một trong 100
    ô** và làm bẩn thứ hạng video. `min_support=1` đòi ít nhất một sự kiện đứng
    trên bằng chứng thật của chính nó.
    """
    # Chuẩn hoá TRƯỚC khi gom theo video — xem `normalise_events`.
    scaled = normalise_events(per_event) if normalize else [list(h) for h in per_event]

    grouped: dict[str, list[list[EventHit]]] = {}
    for event, hits in enumerate(scaled):
        for hit in hits:
            slot = grouped.setdefault(hit.video_id, [[] for _ in per_event])
            slot[event].append(hit)

    out: list[Alignment] = []
    for video_id, buckets in grouped.items():
        aligned = align_video(buckets, penalty=penalty)
        if aligned is None:
            logger.debug("%s: khong co phep gan tang dan", video_id)
            continue
        if aligned.support < min_support:
            logger.debug(
                "%s: chi %d/%d su kien co bang chung that, bo",
                video_id, aligned.support, len(per_event),
            )
            continue
        out.append(aligned)

    out.sort(key=lambda a: (-a.score, a.video_id))
    return out[:limit] if limit else out
