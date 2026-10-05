"""Dựng ZIP nộp cho một gói đề: khung ĐÃ TÌM RA lên đầu, còn lại lấp bằng hệ.

Vì sao "lên đầu" là toàn bộ vấn đề
-----------------------------------
`Final Score = trung bình R@k` với k ∈ {1, 5, 20, 50, 100}. Một đáp án đúng ở
dòng 1 ăn cả năm số hạng; cùng đáp án đó ở dòng 30 chỉ ăn hai. Nên việc đưa
khung người vận hành đã xác nhận lên đầu **không phải chuyện sắp xếp cho đẹp** —
nó là phép nhân điểm lớn nhất trên bàn nộp, lớn hơn mọi cần gạt truy hồi.

Lấp phần còn lại bằng top của hệ chứ không bỏ trống: R@100 chỉ tăng khi có
thêm dòng, và một dòng sai không trừ điểm. Nộp thiếu dòng là tự bỏ cơ hội.

Sửa `PINNED` rồi chạy — đó là chỗ duy nhất cần đụng tay:

    python scripts/pack_submission.py --package eval/btc/bo-de-dot-3 \\
        --only 28,29,30,31,32,33,34,35,36 --out-dir nop-dot-3
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfe.contracts.tools import Candidate  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.eval.query_package import load_package  # noqa: E402
from mfe.eval.submission_bundle import build_zip, run_package  # noqa: E402
from mfe.mapping.relational import RelationalMap  # noqa: E402
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle  # noqa: E402

#: `{query_id: [keyframe_id, …]}` — đưa lên ĐẦU danh sách nộp, đúng thứ tự này.
#:
#: Id nào không còn trong kết quả truy hồi thì **rơi ra** (`apply_manual_order`),
#: nên ghim một id đã cũ không làm hỏng file — nó chỉ không có tác dụng. Ngược
#: lại, ghim một id sai *có* hại: nó chiếm dòng 1.
PINNED: dict[str, list[str]] = {
    # ── ĐÃ XÁC NHẬN ────────────────────────────────────────────────────────
    # Câu 35: nhóm 3 = đề QG-2018 (video duy nhất trong kho có nhãn "(NHÓM n)").
    "query-p2-35-qa": [
        "L25_V014#040413", "L25_V014#040499", "L25_V014#039982",
        "L25_V014#044699", "L25_V014#040350", "L25_V014#040199",
    ],
    # ── ỨNG VIÊN, chưa soi ảnh xác nhận ────────────────────────────────────
    # Câu 30: L23_V005 là video DUY NHẤT có phụ đề "Vòng quanh Hồ Hoàn Kiếm".
    #   #002167 phất cờ -> #002233/#002250 đoàn đua trên phố rợp cây (liền kề).
    "query-p2-30-kis": [
        "L23_V005#002233", "L23_V005#002250", "L23_V005#002167",
        "L23_V005#004625", "L23_V005#002364",
    ],
    # Câu 33: cụm nước rút tay đôi, ngay sau bảng "SPRINT 1" (#017256).
    "query-p2-33-kis": [
        "L23_V025#017750", "L23_V025#017441", "L23_V025#017500",
    ],
}

#: `{query_id: đáp án}` cho câu QA. Cột thứ ba KHÔNG được để trống — `QASubmission`
#: từ chối chuỗi rỗng, và cái từ chối đó đúng: một file QA đủ 100 dòng với cột
#: đáp án trống trông y hệt bài nộp hoàn chỉnh và ghi 0 điểm.
ANSWERS: dict[str, str] = {
    "query-p2-35-qa": "3",
}

#: TRAKE không nhận `manual_order` — mỗi dòng là một video kèm N khung do
#: `dante.align` ghép, nên "kéo một dòng lên đầu" không sửa được chuỗi bên trong.
#: Thứ ghim được là **video**: đẩy mọi lời giải của video này lên trước.
PINNED_TRAKE_VIDEO: dict[str, list[str]] = {
    "query-p2-34-trake": ["L24_V006", "L24_V013"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Thu muc nhan cac file `{query_id}.csv` — nop thang, khong nen zip.",
    )
    parser.add_argument(
        "--zip", type=Path, default=None, help="Ghi them mot ZIP (tuy chon).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Chi lam nhung so nay, vd 28,29,30. Mac dinh: ca goi.",
    )
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--config", default="config/btc_b1.yaml")
    parser.add_argument("--root", default="stores/flat")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queries = load_package(args.package)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        queries = [q for q in queries if q.query_id.split("-")[2] in wanted]
    if not queries:
        raise SystemExit("khong con cau nao sau khi loc --only")
    print(f"{len(queries)} câu: {', '.join(q.query_id for q in queries)}\n")

    config = load_config(args.config)
    bundle = load_flat_bundle(args.root)
    cascade = build_flat_cascade(config, args.root, flat_bundle=bundle)
    relational = RelationalMap(config.paths.relational_db)

    results = run_package(
        cascade, relational, queries, pool_k=args.pool,
        on_progress=lambda index, query: print(
            f"  [{index}/{len(queries)}] {query.query_id}", flush=True
        ),
    )

    # CHEN khung ghim vao pool neu he KHONG truy hoi ra no.
    #
    # `apply_manual_order` cố ý cho id lạ **rơi ra** — nó coi id không có trong
    # kết quả là tàn dư của một lượt chạy cũ. Đúng cho giao diện, SAI ở đây: chỗ
    # này ghim đúng những khung mà hệ **không** tìm ra, và đó chính là lý do
    # người vận hành phải đi tìm tay. Đo được ngay ở lượt đầu: 5/6 khung ghim
    # của câu 35 và 5/5 của câu 30 biến mất khỏi file nộp mà không có gì báo.
    for query in queries:
        rows = results.get(query.query_id)
        if rows is None or query.type == "TRAKE":
            continue
        known = {c.keyframe_id for c in rows}
        missing = [k for k in PINNED.get(query.query_id, []) if k not in known]
        if not missing:
            continue
        # Điểm cao hơn ứng viên đầu để thứ tự "tự nhiên" cũng đã đúng, nhưng
        # `apply_manual_order` mới là thứ quyết định — điểm chỉ để không có dòng
        # nào trông như hạng 0.
        top = max((c.score for c in rows), default=1.0)
        injected = [
            Candidate(
                keyframe_id=key,
                video_id=key.split("#")[0],
                score=top + 1.0,
                evidence_id=f"ghim_tay:{query.query_id}:{index + 1}",
            )
            for index, key in enumerate(missing)
        ]
        results[query.query_id] = injected + list(rows)
        print(f"  chèn {len(missing)} khung ghim mà hệ không tìm ra: {query.query_id}")

    # TRAKE: day cac loi giai cua video da ghim len truoc, GIU nguyen thu tu
    # tuong doi trong tung nhom — thu tu do la thu hang cua he, khong nen xao.
    for query_id, videos in PINNED_TRAKE_VIDEO.items():
        rows = results.get(query_id)
        if not rows:
            continue
        rank = {video: index for index, video in enumerate(videos)}
        results[query_id] = sorted(
            rows, key=lambda item: rank.get(item.video_id, len(rank))
        )

    zipped = build_zip(
        queries, results, relational,
        manual_orders=PINNED, answers=ANSWERS,
    )
    if args.zip:
        args.zip.write_bytes(zipped.payload)

    # Giai chinh ZIP vua dung ra thanh tung file, thay vi goi `submissions_for`
    # mot lan nua: hai duong sinh file nop la hai co hoi lech nhau, va lech o
    # day thi khong ai thay cho toi luc cham diem.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zipped.payload)) as archive:
        names = archive.namelist()
        for name in names:
            (args.out_dir / name).write_bytes(archive.read(name))

    print(f"\n{len(names)} file CSV trong {args.out_dir}")
    for query in queries:
        rows = results.get(query.query_id) or []
        pin = PINNED.get(query.query_id) or PINNED_TRAKE_VIDEO.get(query.query_id) or []
        tag = f"ghim {len(pin)}" if pin else "chỉ hệ"
        path = args.out_dir / query.csv_name
        lines = (
            len(path.read_text(encoding="utf-8").strip().splitlines())
            if path.is_file() else 0
        )
        print(
            f"  {query.csv_name:<26}{lines:>4} dòng nộp"
            f"   (pool {len(rows)})   {tag}"
        )
    if zipped.skipped:
        print("\nKHÔNG có file (và đây là lý do):")
        for query_id, why in zipped.skipped.items():
            print(f"  {query_id}: {why}")


if __name__ == "__main__":
    main()
