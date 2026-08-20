"""Chạy cả một gói đề BTC và ghi ra bộ file nộp.

    python scripts/run_submission.py --config config/pilot.yaml \
        --queries docs/context/bo-de-thu-nghiem --out eval/submissions/p1 --zip

Đây là mảnh nối cuối cùng giữa hai nửa đã có sẵn: `mfe.eval.query_package` đọc đề
thật (`query-*.txt`, không có đáp án), `mfe.online.cascade` tìm keyframe, và
`mfe.eval.submission` ghi CSV đúng định dạng. Trước file này, đường duy nhất đi
hết chặng đó là gõ tay từng câu trong giao diện Streamlit — 24 câu, mỗi câu tối
đa 100 dòng.

Khác `mfe.eval.official` ở đúng một chỗ, và đó là chỗ quan trọng
------------------------------------------------------------------
`official.py` **chấm điểm**: nó cần `gt_video_id`, `gt_frame_range`,
`gt_answer`, và nó in Final Score. File này **nộp**: không có đáp án nào để so,
nên nó không in con số nào giống điểm. Cái nó in là những thứ người vận hành phải
tự kiểm trước khi bấm nộp — arm nào đang chạy, mỗi câu ra bao nhiêu dòng, câu Q&A
nào chưa có câu trả lời.

Ba thứ nó từ chối làm im lặng
-------------------------------
**Arm tắt.** Trên máy chưa bật Elasticsearch/Milvus, 10 trong 12 arm nằm im và
hệ thống vẫn chạy, vẫn ra 100 dòng mỗi câu. Bảng arm được in **trước** khi chạy
câu đầu tiên, và số arm đang chạy đi vào báo cáo cùng với kết quả — một bộ file
nộp không nói được nó ra đời trong cấu hình nào là một bộ file không đối chiếu
được với lần chạy sau.

**Q&A không có câu trả lời.** Không có `GEMINI_API_KEY` thì bộ đọc không chạy.
File CSV vẫn được ghi — frame đã tìm được là công sức thật và điền tay được —
nhưng cột đáp án mang đúng chữ `CAN_DIEN_DAP_AN`, và câu đó được đếm riêng ở
bảng cuối. Ghi một câu trả lời đoán vào đó thì trông giống hệt một lần chạy thành
công, và đó là thứ duy nhất người vận hành không kiểm lại được bằng mắt.

**TRAKE ít dòng.** Mỗi video cho nhiều nhất một câu trả lời TRAKE, nên trên bộ 3
video trần của một câu TRAKE là **3 dòng**, không phải 100. Đó không phải lỗi, và
nó được in ra để không ai đi tìm lỗi.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.core.cli import setup_console  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.eval import submission as submission_io  # noqa: E402
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY  # noqa: E402
from mfe.eval.official import (  # noqa: E402
    to_kis_submissions,
    to_qa_submissions,
    to_trake_submissions,
)
from mfe.eval.query_package import PackagedQuery, load_package  # noqa: E402
from mfe.mapping.relational import RelationalMap  # noqa: E402
from mfe.online import answering  # noqa: E402
from mfe.online.cascade import Cascade  # noqa: E402

logger = logging.getLogger("run_submission")

#: Ghi vao cot dap an khi khong co bo doc va khong co dap an dien tay.
#: Thay duoc bang mat, grep duoc, va an dung so diem mot o trong an.
PLACEHOLDER = "CAN_DIEN_DAP_AN"

#: Vì sao một câu Q&A không có đáp án, và việc phải làm cho từng lý do. Khoá của
#: dict này cũng chính là tập `answer_source` được coi là "chưa có đáp án", nên
#: thêm một lý do mới mà quên nó ở đây thì câu đó biến mất khỏi báo cáo.
ADVICE: dict[str, str] = {
    "MODEL BO CUOC": (
        "model GOI DUOC va tra loi rang no khong doc ra dap an tu nhung khung hinh "
        "nay. Tren bo 3 video, day thuong la DUNG — dap an khong nam trong du lieu. "
        "Doc bang chung roi tu tra loi."
    ),
    "BO DOC HONG": (
        "bo doc NEM LOI (mang dut, het han muc, model khong goi duoc). Day KHONG "
        "phai mot ket luan ve du lieu — xem log roi chay lai."
    ),
    "KHONG CO BO DOC": "chua dat GEMINI_API_KEY, hoac dang chay --qa-reader none.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_submission", description=__doc__)
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument(
        "--queries",
        default="docs/context/bo-de-thu-nghiem",
        help="Thu muc chua query-*.txt cua BTC",
    )
    parser.add_argument("--out", default="eval/submissions/p1")
    parser.add_argument(
        "--zip",
        nargs="?",
        const="submission.zip",
        default=None,
        help="Nen thanh .zip dung cau truc BTC doi (thu muc submission/ ben trong)",
    )
    parser.add_argument(
        "--qa-reader",
        choices=("gemini", "none"),
        default="gemini",
        help="Bo doc dap an Q&A. 'none' = de trong de dien tay.",
    )
    parser.add_argument("--qa-model", default="gemini-3.6-flash")
    parser.add_argument(
        "--qa-images", type=int, default=3, help="So khung hinh dau bang gui kem cho bo doc"
    )
    parser.add_argument(
        "--answers",
        default=None,
        help="JSON {query_id: dap an} dien tay — DE len tren bo doc",
    )
    parser.add_argument("--only", default=None, help="Chi chay nhung query_id nay, ngan bang phay")
    parser.add_argument("--limit-rows", type=int, default=MAX_ANSWERS_PER_QUERY)
    return parser


# ---------------------------------------------------------------------------
# Mot truy van
# ---------------------------------------------------------------------------


def run_query(
    query: PackagedQuery,
    cascade: Cascade,
    relational: RelationalMap,
    top_k: int,
    *,
    reader=None,
    images=None,
    n_images: int = 3,
    manual_answer: str | None = None,
) -> tuple[list, dict]:
    """`(danh sach dong nop, ghi chu)` cho một câu.

    Ghi chú là thứ đi vào báo cáo và ra màn hình; nó cố ý ghi cả những chuyện
    **không** phải lỗi (TRAKE ít dòng, Q&A bỏ cuộc) vì chúng trông giống lỗi.
    """
    note: dict = {"query_id": query.query_id, "type": query.type}

    if query.type == "TRAKE":
        per_event = [
            cascade.search(query_text=text, top_k=top_k, run_id=query.query_id)
            for text in query.searches()
        ]
        note["n_events"] = query.n_events
        note["candidates_per_event"] = [len(hits) for hits in per_event]
        rows = list(to_trake_submissions(per_event, relational, limit=top_k))
        return rows, note

    candidates = cascade.search(query_text=query.text, top_k=top_k, run_id=query.query_id)
    note["candidates"] = len(candidates)

    if query.type == "KIS":
        return list(to_kis_submissions(candidates, relational, limit=top_k)), note

    # --- Q&A --------------------------------------------------------------
    #
    # Đáp án là thuộc tính của CÂU HỎI, không của từng khung hình: cả 100 dòng
    # mang cùng một câu trả lời, khác nhau ở frame. Nên bộ đọc chạy MỘT lần trên
    # vài khung hình đầu bảng, không phải 100 lần.
    answer = manual_answer
    note["answer_source"] = "dien tay" if manual_answer else "KHONG CO BO DOC"
    if answer is None and reader is not None and candidates:
        top = [c.keyframe_id for c in candidates[: max(1, n_images)]]
        evidence = answering.gather_evidence(top, cascade.text_store)
        paths = []
        if images is not None:
            for keyframe_id in top:
                path = images.path(keyframe_id)
                if path is not None:
                    paths.append(path)
        # `propose` bắt MỌI lỗi và trả None, nên từ ngoài nhìn vào "model nói nó
        # không đọc ra được" và "bộ đọc chết vì Google quá tải" giống hệt nhau.
        # `bails` là cái duy nhất phân biệt được — nó chỉ tăng ở nhánh model tự
        # bỏ cuộc. Gộp hai chuyện lại thì một lần 503 sẽ được đọc thành "đáp án
        # không có trong dữ liệu", và đó là kết luận sai về chính bộ dữ liệu.
        bails_before = getattr(reader, "bails", 0)
        proposed = answering.propose(query.text, evidence, reader=reader, image_paths=paths)
        note["images_sent"] = len(paths)
        if proposed is not None:
            answer = proposed.text
            note["answer_source"] = f"bo doc {getattr(reader, 'model', '?')}"
        elif getattr(reader, "bails", 0) > bails_before:
            note["answer_source"] = "MODEL BO CUOC"
        else:
            note["answer_source"] = "BO DOC HONG"

    if answer is None:
        # Vẫn ghi file: frame đã tìm được là công sức thật và điền tay được.
        #
        # Chỗ giữ chỗ là một CHUỖI THẤY ĐƯỢC chứ không phải ô trống, và không
        # phải vì `QASubmission` cấm chuỗi rỗng — cấm đúng. Một ô trống trong CSV
        # trông y hệt một câu trả lời hợp lệ đã bị cắt mất, còn `CAN_DIEN_DAP_AN`
        # thì không thể nhầm được, grep ra ngay, và nếu lỡ nộp thì ăn đúng số
        # điểm mà ô trống ăn: không.
        #
        # `answer_source` KHÔNG bị ghi đè ở đây: nó đã mang lý do cụ thể
        # (`MODEL BO CUOC` / `BO DOC HONG` / `KHONG CO BO DOC`), và đó là thứ
        # quyết định người vận hành phải làm gì tiếp.
        answer = PLACEHOLDER
    note["answer"] = answer
    return list(to_qa_submissions(candidates, relational, answer, limit=top_k)), note


# ---------------------------------------------------------------------------
# Ca goi
# ---------------------------------------------------------------------------


def main() -> None:
    setup_console()
    logging.basicConfig(level=logging.WARNING, format="  ! %(name)s: %(message)s")
    # Tai model qua HTTP la chi tiet trien khai; no khong thuoc ve ban ghi cua
    # mot luot nop, va no day moi thu dang doc ra khoi man hinh.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args()
    _load_env_file()

    config = load_config(args.config)
    queries = load_package(args.queries)
    if args.only:
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        queries = [q for q in queries if q.query_id in wanted]
        if not queries:
            raise SystemExit(f"--only {args.only!r} khong khop truy van nao")

    manual: dict[str, str] = {}
    if args.answers:
        manual = json.loads(Path(args.answers).read_text(encoding="utf-8"))

    top_k = min(args.limit_rows, config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)
    out_root = Path(args.out)
    # Thể lệ §"Cách chuẩn bị file nộp": bên trong zip PHẢI có thư mục `submission`.
    submit_dir = out_root / "submission"
    submit_dir.mkdir(parents=True, exist_ok=True)

    relational = RelationalMap(config.paths.relational_db)
    relational.init_schema()
    cascade = Cascade.from_config(config)

    _print_arms(cascade)
    videos = _known_videos(relational)
    print(f"\nkho dang giu : {len(videos)} video · {relational.count_keyframes()} keyframe")
    print(f"de           : {args.queries} — {len(queries)} truy van")
    print(f"toi da       : {top_k} dong/truy van\n")

    reader = images = None
    if args.qa_reader == "gemini" and any(q.type == "QA" for q in queries):
        reader, images = _build_reader(args, config, relational)

    notes: list[dict] = []
    try:
        for index, query in enumerate(queries, 1):
            rows, note = run_query(
                query,
                cascade,
                relational,
                top_k,
                reader=reader,
                images=images,
                n_images=args.qa_images,
                manual_answer=manual.get(query.query_id),
            )
            path = submission_io.write_csv(rows, submit_dir / query.csv_name, limit=top_k)
            note["rows"] = len(rows)
            note["file"] = path.name
            notes.append(note)
            print(f"[{index:>2}/{len(queries)}] {query.query_id:<22} {query.type:<5} "
                  f"{len(rows):>3} dong  -> {path.name}")
    finally:
        if cascade.translator is not None:
            cascade.translator.save_cache()
        relational.close()

    archive = None
    if args.zip:
        archive = _write_zip(out_root, submit_dir, args.zip)

    _report(notes, cascade, config, args, out_root, archive, reader)


def _load_env_file(path: str | Path = ".env") -> None:
    """Đưa `.env` vào môi trường, **không** đè lên biến đã đặt sẵn.

    `ElasticsearchConfig.from_env` và bộ đọc Gemini đều đọc thẳng `os.environ`,
    còn repo này không nạp `.env` ở đâu cả — nên một khoá đã dán đúng chỗ vẫn
    không tới nơi, và triệu chứng là "Q&A không có đáp án" chứ không phải một lỗi
    nói rằng thiếu khoá.

    Không đè biến có sẵn: một khoá đặt trong phiên là cố ý ghi đè file, và đảo
    thứ tự đó biến `$env:...` thành một câu lệnh không có tác dụng.
    """
    from mfe.offline.transport import load_dotenv

    for key, value in load_dotenv(path).items():
        os.environ.setdefault(key, value)


def _build_reader(args, config, relational):
    """Bộ đọc Q&A và nguồn ảnh, hoặc `(None, None)` kèm lý do đã in ra."""
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print(f"  ! KHONG co GEMINI_API_KEY — cot dap an Q&A se la {PLACEHOLDER!r}.\n"
              "    Dat khoa roi chay lai, hoac --answers <file.json> de dien tay.\n")
        return None, None
    from mfe.online.readers import GeminiAnswerReader

    return (
        GeminiAnswerReader(model=args.qa_model, images=args.qa_images),
        answering.KeyframeImages(relational),
    )


def _print_arms(cascade: Cascade) -> None:
    """Bảng arm, in TRƯỚC khi chạy — nó quyết định kết quả nhiều hơn mọi thứ khác."""
    rows = list(cascade.arms)
    on = [a for a in rows if a.enabled and a.available]
    print(f"arm: {len(on)}/{len(rows)} dang chay")
    for arm in rows:
        mark = "  ON " if (arm.enabled and arm.available) else "  --- "
        reason = "" if (arm.enabled and arm.available) else f"  ({arm.status()})"
        print(f"{mark}{arm.name:<18}{reason}")
    if len(on) < len(rows):
        print("\n  Arm tat vi thieu kho: bat Elasticsearch/Milvus roi nap lai —")
        print("  infra/search/docker-compose.yml, scripts/ingest_sota_*.py")


def _known_videos(relational: RelationalMap) -> list[str]:
    rows = relational._conn.execute("SELECT video_id FROM videos ORDER BY video_id")
    return [row["video_id"] for row in rows]


def _write_zip(out_root: Path, submit_dir: Path, name: str) -> Path:
    """Nén đúng cấu trúc BTC đòi: `submission/` nằm **trong** zip."""
    archive = out_root / (name if name.endswith(".zip") else f"{name}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(submit_dir.glob("*.csv")):
            bundle.write(path, arcname=f"submission/{path.name}")
    return archive


def _report(notes, cascade, config, args, out_root: Path, archive, reader) -> None:
    print("\n" + "=" * 72)
    empty = [n for n in notes if n["rows"] == 0]
    trake = [n for n in notes if n["type"] == "TRAKE"]
    # Ba lý do KHÁC NHAU cho cùng một ô trống, và chúng đòi ba việc khác nhau:
    # model bỏ cuộc -> đọc bằng chứng rồi tự trả lời; bộ đọc hỏng -> xem log rồi
    # chạy lại; không có bộ đọc -> đặt khoá. Gộp lại thành một dòng "chưa có đáp
    # án" là ném đi đúng phần thông tin người vận hành cần.
    by_reason: dict[str, list[dict]] = {}
    for note in notes:
        reason = note.get("answer_source")
        if note["type"] == "QA" and reason in ADVICE:
            by_reason.setdefault(reason, []).append(note)
    no_answer = [n for group in by_reason.values() for n in group]

    print(f"da ghi   : {len(notes)} file CSV -> {out_root / 'submission'}")
    if archive:
        print(f"zip      : {archive}")
    if trake:
        print(f"TRAKE    : {', '.join(str(n['rows']) for n in trake)} dong — "
              "moi video cho NHIEU NHAT mot dong, day la tran cua so video dang giu")
    if no_answer:
        total_qa = sum(1 for n in notes if n["type"] == "QA")
        print(f"\nCHUA CO DAP AN ({len(no_answer)}/{total_qa} cau Q&A) — cot dap an "
              f"dang la {PLACEHOLDER!r}:")
        for reason, group in by_reason.items():
            print(f"\n  [{reason}] {ADVICE[reason]}")
            for note in group:
                print(f"      {note['query_id']}  ({note['file']})")
    if empty:
        print(f"\nRONG ({len(empty)} cau) — khong ung vien nao:")
        for note in empty:
            print(f"  {note['query_id']}")

    on = [a.name for a in cascade.arms if a.enabled and a.available]
    payload = {
        "config": args.config,
        "queries": args.queries,
        "top_k": args.limit_rows,
        "arms_on": on,
        "arms_off": [a.name for a in cascade.arms if not (a.enabled and a.available)],
        "reranker": getattr(getattr(config, "models", None), "reranker", None),
        "qa_reader": getattr(reader, "model", None),
        "zip": str(archive) if archive else None,
        "queries_run": notes,
    }
    report = out_root / "run-report.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBao cao: {report}")
    print(f"arm da chay: {', '.join(on) if on else 'KHONG CO ARM NAO'}")


if __name__ == "__main__":
    main()
