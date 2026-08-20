"""Viết file nộp đúng định dạng BTC — PDF §2.1, §2.2.

    docs/context/Thong-tin-vong-So-tuyen-AIC2026.pdf

Ba dạng truy vấn, ba hình dạng dòng, và **không dòng tiêu đề**:

    KIS    <video_id>,<frame_id>
    QA     <video_id>,<frame_id>,<answer>
    TRAKE  <video_id>,<frame_id>,<frame_id>,…      (N frame, đúng thứ tự sự kiện)

Mỗi truy vấn nộp **tối đa 100 câu trả lời** (§2). Thứ tự dòng **có tính điểm**:
Final Score là trung bình của R@k tại k ∈ {1, 5, 20, 50, 100}, nên một đáp án
đúng ở dòng 1 ăn điểm cao hơn hẳn cùng đáp án đó ở dòng 30. Module này giữ
nguyên thứ tự nó nhận được — xếp hạng là việc của tầng truy hồi, không phải của
tầng ghi file.

Vì sao là một module riêng chứ không phải vài dòng `csv.writer` trong UI
------------------------------------------------------------------------
`docs/context/buoi-1.md` §8: *"tìm đúng hình nhưng submit sai ID/frame là lỗi
trắng"*. Ba chỗ dễ sai, và cả ba đều im lặng:

* **`frame_id` phải là frame của video gốc**, không phải số thứ tự keyframe.
  `L21_V001#031124` là frame 31124 — tình cờ là keyframe thứ 250, và hai con số
  đó cách nhau 17 phút video (`mfe.contracts.ids`).
* **Dấu phẩy trong câu trả lời QA.** Một đáp án như "5, hoặc 6" tự tách thành
  hai cột và đẩy mọi thứ sau nó sang sai vị trí. `csv` chuẩn xử lý đúng bằng
  trích dẫn; `",".join(...)` thì không.
* **Quá 100 dòng** không được báo, chỉ bị cắt ở phía BTC.
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Sequence
from pathlib import Path

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY
from mfe.eval.metrics.kis import KISSubmission
from mfe.eval.metrics.qa import QASubmission
from mfe.eval.metrics.trake import TRAKESubmission

logger = logging.getLogger(__name__)

AnySubmission = KISSubmission | QASubmission | TRAKESubmission


def _row(item: AnySubmission) -> list[str]:
    if isinstance(item, QASubmission):
        return [item.video_id, str(item.frame_id), item.answer]
    if isinstance(item, TRAKESubmission):
        return [item.video_id, *(str(frame) for frame in item.frame_ids)]
    if isinstance(item, KISSubmission):
        return [item.video_id, str(item.frame_id)]
    raise ContractError(
        ErrorCode.INVALID_ARGUMENT, f"khong biet dang bai nop: {type(item).__name__}"
    )


def to_rows(
    submissions: Sequence[AnySubmission], *, limit: int = MAX_ANSWERS_PER_QUERY
) -> list[list[str]]:
    """`[[cot, …], …]` theo đúng thứ tự đã nhận, cắt còn `limit` dòng.

    Cắt **im lặng ở đây là cố ý không xảy ra**: quá hạn thì ghi log cảnh báo, vì
    "nộp 140 dòng" và "nộp 100 dòng" trông giống hệt nhau trong một file đã ghi.
    """
    if limit < 1:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"limit phai >= 1, nhan {limit}")
    if len(submissions) > limit:
        logger.warning(
            "cat %d -> %d dong: BTC chi cham %d cau tra loi moi truy van",
            len(submissions), limit, limit,
        )
    kinds = {type(item).__name__ for item in submissions}
    if len(kinds) > 1:
        # Một file nộp là một truy vấn, và một truy vấn có đúng một dạng. Trộn
        # hai dạng cho ra một file có số cột không đều — BTC đọc được, và đọc sai.
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, f"tron nhieu dang trong mot file: {sorted(kinds)}"
        )
    return [_row(item) for item in submissions[:limit]]


def write_csv(
    submissions: Sequence[AnySubmission],
    path: str | Path,
    *,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> Path:
    """Ghi một file nộp. Trả về đường dẫn đã ghi."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # `newline=""` là bắt buộc của `csv`: thiếu nó, trên Windows mỗi dòng ra hai
    # ký tự xuống dòng và file có một dòng trống xen kẽ.
    with target.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(to_rows(submissions, limit=limit))
    return target


def to_csv_text(
    submissions: Sequence[AnySubmission], *, limit: int = MAX_ANSWERS_PER_QUERY
) -> str:
    """Cùng nội dung, trả về chuỗi — cho nút tải trong giao diện."""
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(to_rows(submissions, limit=limit))
    return buffer.getvalue()
