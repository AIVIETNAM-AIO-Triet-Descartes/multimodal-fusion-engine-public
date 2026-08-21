"""Đọc một **gói truy vấn** BTC — thư mục `query-*.txt` — thành thứ chạy được.

    docs/context/the-le-cuoc-thi.md · docs/context/bo-de-thu-nghiem/

BTC phát đề dưới dạng file text trần, một truy vấn một file, và **dạng truy vấn
nằm trong tên file**:

    query-p1-1-kis.txt      -> Textual KIS
    query-p1-15-qa.txt      -> Q&A
    query-p1-16-trake.txt   -> TRAKE

Tên file cũng quyết định tên file nộp (`query-p1-1-kis.csv`), nên nó được giữ
nguyên vẹn suốt cả đường đi thay vì tách ra rồi ghép lại — ghép lại là chỗ một
`p1` bị rơi và cả gói bị chấm 0 vì lệch tên.

Vì sao không tái dùng `mfe.eval.official.load_official_queryset`
-----------------------------------------------------------------
Định dạng ở đó là JSONL **có đáp án**: `gt_video_id`, `gt_frame_range`,
`gt_answer`. Nó là bộ đo nội bộ. Đề thật không có đáp án — đó là toàn bộ điểm
khác nhau — nên ép đề thật vào cùng một model Pydantic sẽ đòi những trường không
tồn tại, và cách duy nhất để thoả mãn nó là bịa ra chúng.

Tách sự kiện của TRAKE
-----------------------
Đề TRAKE viết mỗi sự kiện một dòng `E<n>: …`, đôi khi có một câu mở đầu tả cả
đoạn video:

    Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:
    E1: Lân quay vòng trên cột số 4 …
    E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.

Câu mở đầu **được ghép vào từng sự kiện** khi tìm kiếm. Hai câu hỏi khác nhau
đang chồng lên nhau trong một truy vấn TRAKE — *video nào* và *khoảnh khắc nào* —
và câu mở đầu trả lời câu thứ nhất còn dòng `E<n>` trả lời câu thứ hai. Tìm bằng
riêng "Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên" thì mọi video có con vật
nào tiếp đất đều ứng cử; DANTE chỉ xếp được các sự kiện *trong* một video, nên nó
không sửa được việc chọn nhầm video.

Không có câu mở đầu (như `query-p1-4-trake.txt`, mở thẳng bằng `E1:`) thì mỗi sự
kiện đi một mình. Đó là mất mát thật và được ghi lại ở `preamble is None`, không
được bù bằng cách đoán ra một câu mô tả.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mfe.contracts.errors import ContractError, ErrorCode

logger = logging.getLogger(__name__)

QueryType = Literal["KIS", "QA", "TRAKE"]

#: Hậu tố tên file -> dạng truy vấn (thể lệ §"Quy ước tên file truy vấn").
SUFFIXES: dict[str, QueryType] = {"kis": "KIS", "qa": "QA", "trake": "TRAKE"}

#: `E1:`, `E 2 :`, `e3.`, và `E1 ` — BTC gõ tay nên dấu ngăn không cố định.
#:
#: Dấu ngăn là TUỲ CHỌN vì gói đề ngày 21-08-2026 viết `E1 Khoảnh khắc…` không
#: có hai chấm, và bản regex đòi `[:.\-]` đã làm hỏng cả lượt chạy với thông báo
#: "không có dòng nào dạng E<n>" — trong khi file có đủ ba dòng, ngay trước mắt.
#: Một phép nhận dạng định dạng do người gõ phải chịu được cách gõ của người đó.
#:
#: Nhưng vẫn phải có KHOẢNG TRẮNG rồi mới tới nội dung: không thế thì `E1` trong
#: `E1000` hay một mã hiệu như `E5A` cũng thành mốc sự kiện.
#:
#: Neo `^` với `re.MULTILINE` để một chữ "E1" giữa câu không bị hiểu thành mốc.
_EVENT = re.compile(r"^\s*E\s*(\d+)\s*[:.\-]?\s+(.+)$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class PackagedQuery:
    """Một file đề, đã đọc. Không có đáp án — đây là đề thật."""

    #: `query-p1-16-trake`. Tên file nộp là chính nó cộng `.csv`.
    query_id: str
    type: QueryType
    #: Nguyên văn file, giữ cả xuống dòng.
    text: str
    source_path: Path
    #: Chỉ TRAKE: câu mở đầu tả cả đoạn video, hoặc `None` khi đề mở thẳng bằng `E1:`.
    preamble: str | None = None
    #: Chỉ TRAKE: nội dung từng `E<n>`, theo đúng thứ tự thời gian.
    events: tuple[str, ...] = ()

    @property
    def csv_name(self) -> str:
        return f"{self.query_id}.csv"

    @property
    def n_events(self) -> int:
        """Số frame mỗi dòng TRAKE phải có. `0` với KIS/QA."""
        return len(self.events)

    def searches(self) -> list[str]:
        """Chuỗi tìm kiếm — một với KIS/QA, một **mỗi sự kiện** với TRAKE."""
        if self.type != "TRAKE":
            return [self.text]
        if self.preamble:
            return [f"{self.preamble} {event}" for event in self.events]
        return list(self.events)


def parse_query_type(path: str | Path) -> QueryType:
    """Dạng truy vấn đọc từ hậu tố tên file. Hậu tố lạ là lỗi, không phải KIS.

    Mặc định về KIS sẽ ghi một file QA thiếu hẳn cột đáp án, và file đó nộp được:
    BTC đọc nó, chấm 0, và không có gì trong quá trình đó phát ra tiếng.
    """
    stem = Path(path).stem
    suffix = stem.rsplit("-", 1)[-1].lower()
    if suffix not in SUFFIXES:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{stem}: hau to {suffix!r} khong phai kis/qa/trake. "
            "The le quy dinh dang truy van nam o hau to ten file.",
        )
    return SUFFIXES[suffix]


def split_events(text: str) -> tuple[str | None, tuple[str, ...]]:
    """`(câu mở đầu, các sự kiện)` từ nội dung một đề TRAKE.

    **Thứ tự là vị trí dòng, không phải con số trên nhãn.** Đề thật có lỗi đánh
    máy: `query-p1-18-trake.txt` của bộ thử nghiệm ghi E1, E2, **E2**, E4 cho bốn
    dòng. Bốn dòng vẫn là bốn sự kiện theo đúng thứ tự thời gian; chỉ cái nhãn
    sai. Từ chối cả đề vì một chữ số gõ nhầm là tự bỏ một câu.

    Nhưng có một trường hợp **không** đọc bằng vị trí được: nhãn lớn nhất vượt
    quá số dòng, ví dụ E1, E2, E4 — lúc đó thiếu hẳn một dòng, và số cột mọi dòng
    nộp đều sai. Thể lệ ghi rõ "Số lượng Frame ID phải khớp chính xác với số
    events yêu cầu", nên chỗ đó dừng.

    Trả về cả cảnh báo? Không — người gọi tự so `n_events` với đề trên màn hình,
    còn chỗ này ghi log để cái sai không đi qua trong im lặng.
    """
    matches = list(_EVENT.finditer(text))
    if not matches:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            "de TRAKE khong co dong nao dang 'E<n>: ...' — khong biet phai tra ve "
            "may frame moi dong",
        )

    numbers = [int(m.group(1)) for m in matches]
    if max(numbers) > len(numbers):
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"nhan su kien chay den E{max(numbers)} nhung chi doc duoc "
            f"{len(numbers)} dong ({numbers}) — thieu han mot su kien, khong phai "
            "loi go nhan. Doc lai file de.",
        )
    if numbers != list(range(1, len(numbers) + 1)):
        logger.warning(
            "nhan su kien khong lien tuc (%s) nhung du %d dong — doc theo THU TU "
            "DONG. Kiem lai voi de goc truoc khi nop.",
            numbers,
            len(numbers),
        )

    head = text[: matches[0].start()].strip()
    events = tuple(" ".join(m.group(2).split()) for m in matches)
    return (head or None), events


def load_query(path: str | Path) -> PackagedQuery:
    """Đọc một file đề."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"{path.name}: file rong")

    query_type = parse_query_type(path)
    preamble: str | None = None
    events: tuple[str, ...] = ()
    if query_type == "TRAKE":
        preamble, events = split_events(text)

    return PackagedQuery(
        query_id=path.stem,
        type=query_type,
        text=text,
        source_path=path,
        preamble=preamble,
        events=events,
    )


def load_package(directory: str | Path) -> list[PackagedQuery]:
    """Cả một gói. Sắp theo **số thứ tự trong tên**, không theo thứ tự chuỗi.

    Sắp theo chuỗi đặt `query-p1-10` trước `query-p1-2`, và thứ tự đó là thứ tự
    người vận hành đối chiếu với đề trên màn hình. Không đổi điểm, đổi khả năng
    phát hiện mình đang nhìn nhầm câu.
    """
    directory = Path(directory)
    paths = sorted(directory.glob("query-*.txt"), key=_natural_key)
    if not paths:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, f"{directory}: khong co file query-*.txt nao"
        )
    return [load_query(path) for path in paths]


def _natural_key(path: Path) -> tuple:
    return tuple(
        int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", path.stem)
    )
