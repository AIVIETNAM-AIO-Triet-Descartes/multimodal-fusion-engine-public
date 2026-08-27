"""Từ kết quả truy hồi tới **cả gói file nộp** — kể cả thứ tự người xếp tay.

`mfe.eval.submission` ghi *một* file CSV đúng định dạng. Module này là tầng trên
nó: nhận kết quả của cả một gói đề, áp thứ tự do người vận hành xếp lại, và trả
về hoặc một CSV hoặc một ZIP chứa đúng `{query_id}.csv` cho mọi câu.

Vì sao KHÔNG nằm trong `app/submit_lab.py`
------------------------------------------
Đây là phần dễ sai nhất của cả giao diện — sai ở đây nghĩa là **nộp sai**, và
`docs/context/buoi-1.md` §8 gọi đó là "lỗi trắng". Nhưng một script Streamlit
chạy mã ở cấp module ngay khi import, nên không test nào import được nó. Logic
nộp nằm trong script = logic nộp không có test.

Nên UI chỉ còn việc nối dây: nó lấy thứ tự tay ra khỏi `session_state` rồi đưa
vào đây như một danh sách chuỗi bình thường.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.tools import Candidate
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY
from mfe.eval.metrics.trake import TRAKESubmission
from mfe.eval.official import to_event_hits, to_kis_submissions, to_qa_submissions
from mfe.eval.query_package import PackagedQuery
from mfe.eval.submission import AnySubmission, to_csv_text
from mfe.mapping.relational import RelationalMap
from mfe.online import dante


def run_package(
    cascade,
    relational: RelationalMap,
    queries: Sequence[PackagedQuery],
    *,
    pool_k: int,
    penalty: float = dante.PENALTY,
    clauses: Mapping[str, Sequence[str]] | None = None,
    on_progress: Callable[[int, PackagedQuery], None] | None = None,
) -> dict[str, list]:
    """Chạy cả gói đề. `{query_id: kết quả}`.

    KIS/QA cho `list[Candidate]`; TRAKE cho `list[Alignment]` — hai hình dạng
    khác nhau, và `submissions_for` rẽ theo `query.type` chứ không đoán từ nội
    dung. Giữ nguyên hai hình dạng thay vì ép về một là có chủ ý: TRAKE trả lời
    bằng **một video kèm N khung**, và làm phẳng nó thành danh sách keyframe là
    đánh mất đúng thứ nó phải nộp.

    `cascade` nên đến từ `mfe.online.flat_arms.build_flat_cascade` — cùng đường
    mà `scripts/measure_rerank.py` và `mfe.eval.official` chạy. Tự ghép arm bằng
    tay sẽ cho một giao diện đo một hệ khác với hệ đem đi thi.

    `clauses` là `{query_id: [mệnh đề theo trình tự thời gian]}` do người vận
    hành tách tay — nó **thay** bộ luật `scene_order.split_clauses` cho đúng câu
    đó. Chỉ có nghĩa với KIS/QA; TRAKE đã có danh sách sự kiện tường minh.
    """
    clauses = clauses or {}
    out: dict[str, list] = {}
    for index, query in enumerate(queries, start=1):
        if on_progress is not None:
            on_progress(index, query)
        if query.type == "TRAKE":
            out[query.query_id] = _run_trake(
                cascade, relational, query, pool_k, penalty
            )
        else:
            out[query.query_id] = cascade.search(
                query_text=query.text, top_k=pool_k, run_id=query.query_id,
                clauses=clauses.get(query.query_id),
            )
    return out


def _run_trake(
    cascade, relational: RelationalMap, query: PackagedQuery,
    pool_k: int, penalty: float,
) -> list:
    """Truy hồi **từng sự kiện**, rồi ghép chuỗi tăng dần bằng DANTE.

    `run_id` dùng `-e{i}`, KHÔNG `#e{i}`: `run_id` đi thẳng vào `evidence_id`,
    và `Candidate` bắt nó khớp `^[a-zA-Z0-9_]+:[a-zA-Z0-9_\\-]+:\\d+$`. `#`
    trượt pattern, `ArmRegistry.run` nuốt `ValidationError` rồi chỉ ghi log
    "arm ... returned nothing" — mọi arm trả rỗng mà không có gì đỏ lên. Đã mất
    một lượt đo vì đúng chuyện này.
    """
    per_event = [
        to_event_hits(
            cascade.search(
                query_text=text, top_k=pool_k, run_id=f"{query.query_id}-e{index}"
            ),
            relational,
        )
        for index, text in enumerate(query.searches())
    ]
    return dante.align(per_event, penalty=penalty, limit=MAX_ANSWERS_PER_QUERY)


def apply_manual_order(
    candidates: Sequence[Candidate], manual: Sequence[str] | None
) -> list[Candidate]:
    """Xếp lại theo thứ tự người vận hành đã kéo thả.

    `manual` là danh sách **`keyframe_id`**, không phải chỉ số — và đó là một
    lựa chọn chứ không phải chi tiết. Chạy lại truy vấn với arm khác hoặc pool
    sâu khác sẽ đổi độ dài danh sách, và một thứ tự lưu bằng chỉ số sẽ âm thầm
    trỏ sang keyframe khác. Người vận hành thấy y nguyên giao diện, còn file nộp
    thì đã đổi.

    Hai đường rẽ, cả hai đều cố ý **không** báo lỗi:

    * id có trong `manual` mà không còn trong kết quả thì **rơi ra** — nó là tàn
      dư của một lượt chạy trước.
    * ứng viên mới chưa ai xếp thì **nối vào cuối**, giữ nguyên thứ tự máy. Bỏ
      chúng đi sẽ cắt ngắn bài nộp, mà R@100 chỉ tăng khi nộp thêm dòng.
    """
    if not manual:
        return list(candidates)
    by_id = {c.keyframe_id: c for c in candidates}
    head = [by_id[key] for key in manual if key in by_id]
    seen = {c.keyframe_id for c in head}
    return head + [c for c in candidates if c.keyframe_id not in seen]


def submissions_for(
    query: PackagedQuery,
    result: Sequence,
    relational: RelationalMap,
    *,
    limit: int = MAX_ANSWERS_PER_QUERY,
    manual_order: Sequence[str] | None = None,
    answer: str = "",
) -> list[AnySubmission]:
    """Kết quả một câu -> hình dạng nộp của **đúng dạng đề đó**.

    Ba dạng, ba hình dạng dòng, và dạng đọc từ hậu tố tên file (`query.type`) —
    không đoán từ nội dung kết quả. Mặc định về KIS sẽ ghi một file QA thiếu hẳn
    cột đáp án, và file đó nộp được.

    TRAKE **không nhận `manual_order`**: mỗi dòng của nó là một video kèm N khung
    tăng dần do `dante.align` ghép, nên "kéo một dòng lên đầu" không sửa được
    chuỗi bên trong dòng đó. `result` khi ấy là `list[Alignment]`, không phải
    `list[Candidate]`.

    QA **không có đáp án thì ném**, không ghi cột rỗng. `QASubmission` cấm chuỗi
    rỗng, và cái cấm đó đúng: một file QA đủ 100 dòng với cột đáp án trống trông
    y hệt một bài nộp hoàn chỉnh và ghi 0 điểm. Ném ở đây kèm tên câu để phần
    gọi biết **câu nào** còn thiếu — `build_zip` bắt lại và đếm ra.
    """
    if query.type == "TRAKE":
        return [
            TRAKESubmission(video_id=item.video_id, frame_ids=item.frame_ids)
            for item in result[:limit]
        ]

    ranked = apply_manual_order(result, manual_order)
    if query.type == "QA":
        if not answer.strip():
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"{query.query_id}: de QA nhung chua co dap an. "
                "Cot thu ba khong duoc de trong.",
            )
        return list(to_qa_submissions(ranked, relational, answer, limit=limit))
    return list(to_kis_submissions(ranked, relational, limit=limit))


@dataclass(frozen=True)
class Bundle:
    """ZIP đã dựng, **kèm danh sách câu bị bỏ và lý do**.

    Trả lý do ra ngoài chứ không chỉ ghi log: một ZIP thiếu file trông y hệt một
    ZIP đủ file cho tới lúc đếm, và lúc đếm thì đã nộp rồi.
    """

    payload: bytes
    #: `{query_id: lý do}` — câu không có file trong ZIP.
    skipped: dict[str, str]


def build_zip(
    queries: Sequence[PackagedQuery],
    results: Mapping[str, Sequence],
    relational: RelationalMap,
    *,
    limit: int = MAX_ANSWERS_PER_QUERY,
    manual_orders: Mapping[str, Sequence[str]] | None = None,
    answers: Mapping[str, str] | None = None,
) -> Bundle:
    """Một ZIP chứa `{query_id}.csv` cho mọi câu dựng được bài nộp.

    Câu chưa chạy, hoặc câu QA chưa có đáp án, thì **không có file** trong ZIP —
    thay vì có một file rỗng hay một file thiếu cột. Một CSV như thế nộp được và
    ghi 0 điểm y như không nộp, nhưng nó làm người vận hành tin là đã nộp đủ;
    một file *thiếu* thì đếm ra ngay, và `Bundle.skipped` nói luôn thiếu vì sao.

    Một câu hỏng **không** làm hỏng cả gói. Ném ra giữa chừng nghĩa là 48 câu
    đúng cũng không tải về được chỉ vì câu thứ 49 quên gõ đáp án.
    """
    manual_orders = manual_orders or {}
    answers = answers or {}
    skipped: dict[str, str] = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for query in queries:
            result = results.get(query.query_id)
            if not result:
                skipped[query.query_id] = "chua chay, hoac chay ra 0 ung vien"
                continue
            try:
                rows = submissions_for(
                    query, result, relational, limit=limit,
                    manual_order=manual_orders.get(query.query_id),
                    answer=answers.get(query.query_id, ""),
                )
            except ContractError as err:
                skipped[query.query_id] = str(err)
                continue
            archive.writestr(query.csv_name, to_csv_text(rows, limit=limit))
    return Bundle(payload=buffer.getvalue(), skipped=skipped)
