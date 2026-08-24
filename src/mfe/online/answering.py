"""Q&A — gom bằng chứng quanh một keyframe, và chỗ cắm bộ đọc câu trả lời.

    PDF §2.1.2 · `docs/context/Thong-tin-vong-So-tuyen-AIC2026.pdf`

Truy vấn Q&A nộp **ba** thứ: `video_id`, `frame_id`, và `answer`. Hai thứ đầu là
việc của tầng truy hồi và nó đã chạy. Thứ ba là thứ codebase chưa hề có — bộ
chấm hiện nộp thẳng `gt_answer` và `official.py` ghi rõ rằng con số Q&A vì thế
là **chặn trên**, không phải điểm thật.

Module này lấp phần đó, và nó lấp bằng **bằng chứng** trước, **model** sau.

Vì sao mặc định KHÔNG có model
--------------------------------
Vòng sơ tuyển là interactive track: người vận hành ngồi trước máy và gõ câu trả
lời. Thứ họ cần không phải một câu trả lời sinh tự động — thứ họ cần là **đọc
được chữ trên khung hình đó trong một giây**, rồi tự gõ. `gather_evidence` gom
đúng thứ ấy: mọi văn bản ta đang giữ về keyframe đó, nhóm theo nguồn.

Một bộ đọc tự động là **tuỳ chọn**, cắm qua `AnswerReader`. Không có nó thì
`propose` trả về `None` chứ không đoán bừa — một câu trả lời bịa ra trông y hệt
một câu trả lời đúng cho tới lúc chấm điểm, và §9.1 nói rõ không tự thay model.

Bằng chứng gom từ ĐÂU
----------------------
Từ text store, đúng những nguồn đã ingest — hiện là 7: `subtitle`, `ocr`,
`ocr_paddle`, `ocr_vietocr`, `object`, `caption`. Không phải từ
ảnh: đọc lại ảnh bằng một model nữa ở thời điểm truy vấn là trả lại chi phí của
cả tầng offline, ngay trong 5 phút đồng hồ của một lượt thi.

`window` mở rộng sang keyframe lân cận vì câu trả lời hay nằm **cạnh** khoảnh
khắc chứ không đúng trên nó: phụ đề của câu đang nói bắt đầu trước khi hình
xuất hiện, và biển hiệu thì còn trên màn hình vài giây.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id

logger = logging.getLogger(__name__)

#: Thứ tự hiển thị bằng chứng, đặt theo **mức trực tiếp trả lời được câu hỏi**,
#: không theo thứ tự bảng chữ cái. Chữ trên màn hình trả lời thẳng phần lớn câu
#: hỏi Q&A của BTC ("biển ghi gì", "mấy giờ", "tên chương trình"); caption là
#: câu sinh tự động nên đứng cuối.
SOURCE_ORDER: tuple[str, ...] = (
    "ocr_vietocr",
    "ocr_paddle",
    "ocr",
    "subtitle",
    "object",
    "caption",
)


@dataclass(frozen=True)
class Evidence:
    """Mọi văn bản ta giữ về MỘT keyframe, nhóm theo nguồn."""

    keyframe_id: str
    video_id: str
    frame_id: int
    #: `{source: text}`. Nguồn không có gì thì **không có khoá** — khác hẳn với
    #: khoá rỗng: "OCR im lặng" và "chưa chạy OCR" là hai chuyện.
    texts: dict[str, str] = field(default_factory=dict)

    def ordered(self) -> list[tuple[str, str]]:
        """`[(source, text)]` theo `SOURCE_ORDER`; nguồn lạ xếp cuối, theo tên."""
        known = [(s, self.texts[s]) for s in SOURCE_ORDER if s in self.texts]
        extra = sorted((s, t) for s, t in self.texts.items() if s not in SOURCE_ORDER)
        return known + extra

    def as_prompt(self) -> str:
        """Bằng chứng dưới dạng văn bản phẳng, cho một bộ đọc hoặc cho mắt người."""
        lines = [f"[{source}] {text}" for source, text in self.ordered()]
        return "\n".join(lines)


@dataclass(frozen=True)
class Answer:
    """Một câu trả lời đề xuất, kèm chỗ nó được đọc ra."""

    text: str
    #: `keyframe_id` mà câu trả lời dựa vào. Bắt buộc — `docs/context/buoi-3.md`
    #: §7: mọi câu trả lời phải trỏ về bằng chứng, không được trả lời từ trí nhớ.
    keyframe_id: str
    #: Nguồn nào trong `Evidence` đã dùng, để người vận hành kiểm lại nhanh.
    sources: tuple[str, ...] = ()
    reader: str = ""


class AnswerReader(Protocol):
    """Bộ đọc câu trả lời. Nhận câu hỏi + bằng chứng, trả câu trả lời hoặc `None`.

    `None` nghĩa là **không đọc ra được**, và nó phải được tôn trọng: nộp một câu
    trả lời bịa ra ghi 0 điểm y như bỏ trống, nhưng nó làm người vận hành tin là
    đã xong (`docs/context/buoi-3.md` §7 — evidence discipline).
    """

    name: str

    def __call__(
        self,
        question: str,
        evidence: Sequence[Evidence],
        image_paths: Sequence[Path] = (),
    ) -> Answer | None:
        ...


#: Nơi tìm ảnh keyframe, theo thứ tự. `{video}` và `{n}` được thay vào.
IMAGE_ROOTS: tuple[str, ...] = (
    "data/keyframes/{video}/{n:03d}.jpg",
    "data/sample_2026/keyframes/{video}/{n:03d}.jpg",
    "data/keyframes/{video}/{n:03d}.png",
)


def keyframe_image_path(keyframe_id: str, relational) -> Path | None:
    """Đường dẫn ảnh của một keyframe, hoặc `None`.

    **Tên file theo SỐ THỨ TỰ, `keyframe_id` theo FRAME GỐC.** BTC đặt tên ảnh là
    `{n:03d}.jpg` với `n` là số thứ tự keyframe trong video, còn `keyframe_id` là
    `{video}#{frame_idx:06d}`. Hai con số đó **không bằng nhau**:
    `L21_V001#000090` là frame 90 và nằm trong file `002.jpg`.

    Giao diện từng tách thẳng phần sau dấu `#` ra làm tên file. Nó không tìm thấy
    ảnh nào cả — và vì có sẵn ô giữ chỗ nên nó **không báo lỗi**, chỉ hiện một ô
    màu trên mọi thẻ. Cầu nối duy nhất giữa hai trục là Relational Map.
    """
    if relational is None:
        return None
    try:
        row = relational.get_keyframe(keyframe_id)
    except Exception:
        return None
    for template in IMAGE_ROOTS:
        candidate = Path(template.format(video=row.video_id, n=row.keyframe_index))
        if candidate.is_file():
            return candidate
    return None


@dataclass
class KeyframeImages:
    """Tìm ảnh keyframe: đĩa trước, Modal Volume sau, có cache.

    Máy này chỉ giữ **16 video** — phần được giao cho Lead. Bộ 60 truy vấn dev
    nhắc tới **36 video**, nên tra thẳng trên đĩa chỉ phủ được 12/36 (đo
    20-08-2026: KIS 11/27, Q&A 9/21, TRAKE 6/12). Volume giữ đủ cả batch 1, nên
    đường lui này là thứ biến một phép đo "43% số câu" thành một phép đo đủ.

    **Kéo từng ảnh từ Volume ở đây là an toàn, khác hẳn cảnh báo ở §6.2 handoff.**
    Cảnh báo đó nói về việc kéo *cả bộ keyframe* về cho tầng N4/N5 — ~160.000 lần
    `read_file` cho một người. Ở đây là 3 ảnh mỗi truy vấn, vài trăm ảnh cho cả bộ
    dev, và có cache nên chạy lại không tốn gì. Cùng một thao tác, hai bậc chi phí.
    """

    relational: object
    transport: object | None = None
    layout: object | None = None
    cache_dir: Path = Path(".cache/keyframes")
    #: Đếm để phân biệt "không có ảnh" với "chưa cấu hình Volume" — hai chuyện
    #: dẫn tới hai phản ứng khác nhau.
    from_disk: int = field(default=0, init=False)
    from_volume: int = field(default=0, init=False)
    missing: int = field(default=0, init=False)

    def path(self, keyframe_id: str) -> Path | None:
        local = keyframe_image_path(keyframe_id, self.relational)
        if local is not None:
            self.from_disk += 1
            return local
        if self.transport is None or self.layout is None:
            self.missing += 1
            return None
        try:
            row = self.relational.get_keyframe(keyframe_id)
        except Exception:
            self.missing += 1
            return None

        cached = self.cache_dir / row.video_id / f"{row.keyframe_index:03d}.jpg"
        if cached.is_file():
            self.from_disk += 1
            return cached
        cached.parent.mkdir(parents=True, exist_ok=True)
        remote = f"{self.layout.keyframes_dir(row.video_id)}/{row.keyframe_index:03d}.jpg"
        try:
            self.transport.get(remote, cached)
        except Exception:
            logger.warning("khong keo duoc %s tu Volume", remote)
            self.missing += 1
            return None
        self.from_volume += 1
        return cached

    def report(self) -> str:
        return (
            f"anh: {self.from_disk} tu dia/cache, {self.from_volume} tu Volume, "
            f"{self.missing} khong co"
        )


def gather_evidence(
    keyframe_ids: Sequence[str],
    text_store,
    *,
    sources: Iterable[str] | None = None,
) -> list[Evidence]:
    """Mọi văn bản đang giữ về những keyframe này, giữ nguyên thứ tự đầu vào.

    Thứ tự vào = thứ tự ra, vì đầu vào là **xếp hạng của tầng truy hồi** và người
    vận hành đọc từ trên xuống. Sắp lại ở đây là ném đi thông tin đắt nhất mà
    tầng gộp vừa tạo ra.
    """
    if text_store is None:
        return []
    wanted = set(sources) if sources is not None else None

    keys: list[tuple[str, int]] = []
    for keyframe_id in keyframe_ids:
        video_id, frame_id = parse_keyframe_id(keyframe_id)
        keys.append((video_id, frame_id))
    if not keys:
        return []

    hits = text_store.fetch(keys)
    grouped: dict[tuple[str, int], dict[str, str]] = {}
    for hit in hits:
        if hit.source is None or not (hit.text or "").strip():
            continue
        if wanted is not None and hit.source not in wanted:
            continue
        grouped.setdefault((hit.video_id, hit.frame_id), {})[hit.source] = hit.text

    out: list[Evidence] = []
    for video_id, frame_id in keys:
        out.append(
            Evidence(
                keyframe_id=build_keyframe_id(video_id, frame_id),
                video_id=video_id,
                frame_id=frame_id,
                texts=grouped.get((video_id, frame_id), {}),
            )
        )
    return out


def gather_with_context(
    keyframe_id: str,
    text_store,
    relational,
    *,
    window: int = 2,
) -> list[Evidence]:
    """Bằng chứng của một keyframe **và** các keyframe lân cận.

    Câu trả lời hay nằm cạnh khoảnh khắc chứ không đúng trên nó: phụ đề của câu
    đang nói bắt đầu trước khi hình xuất hiện, biển hiệu còn trên màn hình vài
    giây. `window=2` là ±2 keyframe.

    Lân cận đọc từ Relational Map chứ không suy ra: `frame_idx ± k` **không phải**
    keyframe lân cận — khoảng cách giữa hai keyframe là hàng chục tới hàng trăm
    frame và không đều (`mfe.contracts.ids`).
    """
    if window < 0:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"window phai >= 0, nhan {window}")
    rows = relational.get_neighbors(keyframe_id, window=window)
    ids = [row.keyframe_id for row in rows] if rows else [keyframe_id]
    if keyframe_id not in ids:
        ids.append(keyframe_id)
    return gather_evidence(ids, text_store)


def propose(
    question: str,
    evidence: Sequence[Evidence],
    reader: AnswerReader | None = None,
    image_paths: Sequence[Path] = (),
) -> Answer | None:
    """Chạy bộ đọc nếu có. Không có thì trả `None` — **không đoán**.

    Đây là nơi cám dỗ nhất của cả module: một heuristic kiểu "lấy chuỗi số đầu
    tiên trong OCR" sẽ trả lời đúng vài câu và sai phần còn lại, mà người vận
    hành không phân biệt được hai trường hợp. Trả `None` thì họ đọc bằng chứng và
    tự gõ — chậm hơn vài giây, đúng hơn nhiều.
    """
    if reader is None:
        return None
    if not question.strip():
        raise ContractError(ErrorCode.INVALID_ARGUMENT, "cau hoi rong")
    try:
        return reader(question, evidence, image_paths)
    except Exception:
        logger.exception(
            "bo doc %r ném loi; coi nhu khong doc ra duoc",
            getattr(reader, "name", "?"),
        )
        return None
