"""Đo bộ đọc Q&A trên tập dev — TÁCH khỏi tầng truy hồi.

    python scripts/eval_qa_reader.py --queryset eval/dev --limit 5
    python scripts/eval_qa_reader.py --queryset eval/dev --volume keyframes-btc

Câu hỏi script này trả lời: **cho đúng khung hình, bộ đọc có đọc ra đáp án
không.** Nó KHÔNG đo truy hồi — nó đưa thẳng keyframe nằm trong `gt_frame_range`
cho bộ đọc.

Tách ra là cố ý. Một con số Q&A đầu-cuối thấp có hai nguyên nhân hoàn toàn khác
nhau — truy hồi đưa nhầm khung hình, hay bộ đọc nhìn đúng khung hình mà đọc sai —
và hai nguyên nhân đó đòi hai việc sửa ngược nhau. `mfe.eval.official` đã đo nửa
truy hồi; đây là nửa còn lại.

Hai điều script này TỪ CHỐI làm
--------------------------------
**Không in một con số phần trăm trần trụi.** Máy này giữ 16/36 video mà bộ dev
nhắc tới, nên "40% đúng" có thể là "bộ đọc tệ" hoặc "12 câu không có ảnh". Báo
cáo luôn tách **phủ** khỏi **đúng**.

**Không tự nhận phép so là đủ.** BTC so `answer` theo **ngữ nghĩa** (§2.1.2):
`"5"` và `"Năm"` đều đúng. Phép so tự động ở đây là so chuỗi đã chuẩn hoá, nên nó
**đếm thiếu** — mọi cặp (đoán, đáp án) đều được in ra để đọc tay. Con số tự động
là **chặn dưới**, và nhãn trong báo cáo nói đúng như vậy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.core.cli import setup_console  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.mapping.relational import RelationalMap  # noqa: E402
from mfe.online import answering  # noqa: E402
from mfe.online.readers import GeminiAnswerReader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval_qa_reader", description=__doc__)
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--queryset", default="eval/dev", help="Thu muc *.jsonl")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--images", type=int, default=1,
                        help="So khung hinh gui kem. 1 = dung khung hinh dap an.")
    parser.add_argument("--limit", type=int, default=0, help="0 = tat ca")
    parser.add_argument("--volume", default=None,
                        help="Ten Modal Volume de keo anh thieu, vd keyframes-btc")
    parser.add_argument("--bundle", default="aic_btc_b1_v1")
    parser.add_argument("--out", default="eval/results/qa_reader.json")
    return parser


def load_qa(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            body = json.loads(line)
            if body.get("type") == "QA":
                body["_file"] = path.name
                rows.append(body)
    return rows


def normalise(text: str) -> str:
    """Chuan hoa de so chuoi: NFC, thuong, bo dau cau va khoang trang thua."""
    text = unicodedata.normalize("NFC", (text or "").strip().lower())
    return re.sub(r"\s+", " ", re.sub(r"[.,;:!?\"'()\[\]]", " ", text)).strip()


def anchor_keyframe(query: dict, relational: RelationalMap) -> str | None:
    """Keyframe nam trong `gt_frame_range` cua truy van, hoac `None`.

    Đây là "đúng khung hình" theo nghĩa của phép đo này. Không có keyframe nào
    rơi vào khoảng đó là một phát hiện về **cách trích keyframe**, không phải về
    bộ đọc — và nó được đếm riêng.
    """
    start, end = query["gt_frame_range"]
    video_id = query["gt_video_id"]
    for frame_id in range(start, end + 1):
        keyframe_id = f"{video_id}#{frame_id:06d}"
        try:
            relational.get_keyframe(keyframe_id)
        except Exception:
            continue
        return keyframe_id
    return None


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    config = load_config(args.config)
    queries = load_qa(Path(args.queryset))
    if args.limit:
        queries = queries[: args.limit]

    transport = layout = None
    if args.volume:
        from mfe.offline.bundle import BundleLayout
        from mfe.offline.transport import build_transport

        transport = build_transport("modal", volume=args.volume, local_root="bundle_local")
        layout = BundleLayout(args.bundle)

    relational = RelationalMap(config.paths.relational_db)
    relational.init_schema()
    text_store = _open_text_store(config)
    images = answering.KeyframeImages(relational, transport=transport, layout=layout)
    reader = GeminiAnswerReader(model=args.model, images=args.images)

    print(f"{len(queries)} truy van Q&A · model {args.model}")
    print(f"text store: {'co' if text_store else 'KHONG — chi doc anh'}")
    print(f"Volume    : {args.volume or 'khong dung — chi anh tren dia'}\n")

    rows: list[dict] = []
    for index, query in enumerate(queries, 1):
        record = {
            "query_id": query.get("query_id"),
            "author": query.get("author"),
            "video_id": query["gt_video_id"],
            "question": query["text"],
            "gold": query["gt_answer"],
        }
        keyframe_id = anchor_keyframe(query, relational)
        if keyframe_id is None:
            record["skip"] = "khong co keyframe trong gt_frame_range"
            rows.append(record)
            print(f"[{index:>2}] {record['query_id']:<14} BO QUA — {record['skip']}")
            continue

        record["keyframe_id"] = keyframe_id
        path = images.path(keyframe_id)
        if path is None:
            record["skip"] = "khong co anh"
            rows.append(record)
            print(f"[{index:>2}] {record['query_id']:<14} BO QUA — khong co anh")
            continue

        evidence = (
            answering.gather_evidence([keyframe_id], text_store) if text_store else []
        )
        answer = answering.propose(
            query["text"], evidence, reader=reader, image_paths=[path]
        )
        record["predicted"] = answer.text if answer else None
        record["exact"] = bool(
            answer and normalise(answer.text) == normalise(query["gt_answer"])
        )
        rows.append(record)
        mark = "✓" if record["exact"] else ("·" if answer else "∅")
        print(
            f"[{index:>2}] {record['query_id']:<14} {mark} "
            f"doan={record['predicted']!r:<22} dap an={query['gt_answer']!r}"
        )

    _report(rows, images, reader, Path(args.out))
    relational.close()


def _open_text_store(config):
    try:
        from mfe.stores.production.elasticsearch import ElasticsearchTextStore

        store = ElasticsearchTextStore(config.stores.elasticsearch)
        store.count()
        return store
    except Exception as err:  # noqa: BLE001 - vang mat la trang thai hop le
        print(f"(khong mo duoc Elasticsearch: {type(err).__name__})")
        return None


def _report(rows: list[dict], images, reader, out: Path) -> None:
    scored = [r for r in rows if "skip" not in r]
    exact = [r for r in scored if r["exact"]]
    empty = [r for r in scored if r["predicted"] is None]
    # `bails` do chinh bo doc dem: model NOI khong doc ra duoc. Phan con lai cua
    # `empty` la bo doc CHET (thieu thu vien, het han muc, mang dut) —
    # `answering.propose` bat moi loi va tra None, nen hai chuyen do trong giong
    # het nhau tu ben ngoai. Gop chung lai thi mot lan cai thieu goi se doc ra la
    # "model kem", va do la ket luan sai hoan toan.
    crashed = len(empty) - reader.bails

    print("\n" + "=" * 72)
    print(f"PHU    : {len(scored)}/{len(rows)} truy van cham duoc")
    for reason in sorted({r["skip"] for r in rows if "skip" in r}):
        n = sum(1 for r in rows if r.get("skip") == reason)
        print(f"         bo qua {n}: {reason}")
    print(f"         {images.report()}")
    print(f"DUNG   : {len(exact)}/{len(scored)} khop chuoi sau chuan hoa")
    print(f"BO CUOC: {reader.bails}/{len(scored)} lan MODEL noi khong doc ra duoc")
    if crashed > 0:
        print(f"HONG   : {crashed}/{len(scored)} lan BO DOC CHET — xem log, "
              "day KHONG phai loi cua model")
    print()
    print("`DUNG` la CHAN DUOI, khong phai do chinh xac. BTC so theo NGU NGHIA")
    print("(§2.1.2) — '5' va 'Nam' deu dung — con phep so o day la so chuoi.")
    print("Doc tay cot doan/dap an o tren truoc khi ket luan bat cu dieu gi.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": reader.model,
                "queries": len(rows),
                "scored": len(scored),
                "exact_lower_bound": len(exact),
                "bailed": reader.bails,
                "crashed": crashed,
                "images": {
                    "from_disk": images.from_disk,
                    "from_volume": images.from_volume,
                    "missing": images.missing,
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nBao cao: {out}")


if __name__ == "__main__":
    main()
