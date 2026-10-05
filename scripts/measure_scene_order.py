"""Thiên kiến thứ tự cảnh cho KIS đáng bao nhiêu — đo trên ĐÚNG nhóm câu nó chạm.

    python scripts/measure_scene_order.py --queryset eval/queryset_dev.jsonl \
        --flat-root stores/flat --evidence-cache .cache/tune/dev189-kisqa.json \
        --weight 0.5

Vì sao script này tồn tại thay vì một cột trong `tune_weights.py`
------------------------------------------------------------------
`tune_weights` đọc **Final tổng trên cả bộ**. Bộ dev 189 câu chỉ có **3** câu KIS
nhiều cảnh thật (`triet_order_01..03`); đề thật của BTC có 21/38. Một thay đổi
hoàn hảo trên 3/115 câu đổi Final tổng chừng 0,01 — nằm gọn dưới sàn nhiễu ~0,03
của chính phép quét trọng số. Đọc Final tổng ở đây là đọc nhiễu, và nó sẽ nói
"không có tác dụng" bất kể tầng này làm gì.

Nên bảng ở dưới chia câu làm **hai nhóm** và đọc chúng khác nhau:

* **nhóm CHẠM** — câu tách được ≥2 mệnh đề. Đọc `hạng` từng câu. Đây là chỗ tầng
  này được phép ghi điểm, và cũng là chỗ duy nhất nó có cơ hội.
* **nhóm KHÔNG CHẠM** — phần còn lại. `apply_prior` trên `prior` rỗng phải là một
  phép không-làm-gì, nên nhóm này **phải** giữ nguyên từng con số. Một thay đổi
  ở đây là lỗi, không phải kết quả.

Đây đúng là giao thức đã dùng cho arm `object`: nó nâng số câu tự tìm được từ
4 lên 6, và leo dốc vẫn chốt trọng số 0,0. Một tầng "có làm việc" và một tầng
"đáng bật" là hai câu hỏi khác nhau.

Nguồn mệnh đề
-------------
Mặc định dùng bộ luật `scene_order.split_clauses`. `--clauses` nhận một tệp JSON
`{query_id: ["mệnh đề 1", "mệnh đề 2", …]}` do người hoặc LLM tách — và đo được
**cả hai** trên cùng một bộ câu là mục đích của cờ đó: bộ luật tách được 14/38 đề
KIS thật trong khi 21/38 có từ nối, nên khoảng cách giữa hai cột là giá phải trả
cho việc tự động hoá phần tách.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from mfe.core.cli import setup_console
from mfe.core.config import load_config
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY, evaluate_query
from mfe.eval.official import (
    is_heldout,
    load_official_queryset,
    to_kis_submissions,
)
from mfe.eval.tuning import (
    Assignment,
    evidence_key,
    fuse_and_group,
    load_evidence_cache,
)
from mfe.mapping.relational import RelationalMap
from mfe.online.dante import EventHit
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle
from mfe.online.scene_order import (
    WINDOW,
    apply_prior,
    clauses_from_texts,
    split_clauses,
    video_prior,
)

logger = logging.getLogger("measure_scene_order")


def _score(query, candidates, relational, top_k: int) -> tuple[float, int | None]:
    """`(RỘNG, hạng đáp án đúng đầu tiên)` — cùng phép chấm `measure_rerank` dùng.

    Chỉ KIS, nên không có nhánh QA: `main` đã lọc bỏ mọi dạng khác.
    """
    submissions = list(to_kis_submissions(candidates, relational, limit=top_k))
    wide = evaluate_query(submissions, query.ground_truth(), query_id=query.query_id)
    return wide.final_score, wide.first_hit_rank


def _to_hits(candidates, relational) -> list[EventHit]:
    """`Candidate` -> `EventHit`, đọc cả hai toạ độ khỏi bản đồ.

    `ordinal` (`keyframe_index`) là thứ phép phạt của DANTE đo khoảng cách trên;
    `frame_id` là thứ BTC chấm. Nhầm hai cái là cách hỏng im lặng kinh điển của
    dự án này — `mfe.contracts.ids` nói dài về nó.
    """
    hits: list[EventHit] = []
    for candidate in candidates:
        row = relational.get_keyframe(candidate.keyframe_id)
        hits.append(
            EventHit(
                keyframe_id=candidate.keyframe_id,
                video_id=candidate.video_id,
                ordinal=row.keyframe_index,
                frame_id=row.frame_idx,
                score=float(candidate.score),
            )
        )
    return hits


def main() -> None:  # noqa: PLR0912, PLR0915 - mot script do, doc tuyen tinh
    setup_console()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/btc_b1.yaml")
    parser.add_argument("--queryset", default="eval/queryset_dev.jsonl")
    parser.add_argument("--flat-root", default="stores/flat")
    parser.add_argument("--evidence-cache", default=None)
    parser.add_argument("--en-source", choices=("manual", "auto", "none"), default="auto")
    parser.add_argument(
        "--weight", type=float, default=0.5,
        help="Hệ số nhân của thiên kiến. `score * (1 + weight · prior)`.",
    )
    parser.add_argument(
        "--clauses",
        help=(
            "JSON {query_id: [menh de, ...]} do nguoi hoac LLM tach san, "
            "theo TRINH TU THOI GIAN. Thieu query_id nao thi query do dung bo luat."
        ),
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW,
        help=(
            "Ban kinh thuong, tinh bang keyframe quanh vi tri DANTE da ghep. "
            "-1 = thuong CA VIDEO (ban da do duoc la te hon nen; giu de so)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Chi lay N cau KIS dau tien (thu nhanh duong day, khong phai phep do).",
    )
    parser.add_argument("--out", default="eval/results_btc_b1/scene-order.json")
    args = parser.parse_args()
    if args.window is not None and args.window < 0:
        args.window = None  # `-1` tren CLI = `None` trong `apply_prior`

    config = load_config(args.config)
    top_k = min(config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)

    queries = load_official_queryset(args.queryset)
    if is_heldout(queries):
        # Script này in HÀNH VI từng câu; heldout chỉ được đọc điểm tổng hợp.
        raise SystemExit(
            f"{args.queryset} chua cau HELD-OUT — script nay in hang tung cau."
        )
    queries = [q for q in queries if q.type == "KIS"]
    if not queries:
        raise SystemExit("khong co cau KIS nao trong queryset")
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} cau KIS")

    hand: dict[str, list[str]] = {}
    if args.clauses:
        raw = json.loads(Path(args.clauses).read_text(encoding="utf-8"))
        # Khoá `_...` là ghi chú trong chính tệp — JSON không có comment, và ghi
        # chú ở đây quan trọng: nó nói vì sao một câu CÓ từ nối lại cố ý vắng mặt.
        hand = {k: v for k, v in raw.items() if not k.startswith("_")}
        print(f"menh de tach san: {len(hand)} cau")

    # Nền lấy từ cache truy hồi — cùng con số `measure_rerank` đang đọc, nên hai
    # bảng so được với nhau.
    cache_key = evidence_key(
        args.config, args.queryset, config.retrieval.coarse_topk, args.en_source
    )
    cache_path = Path(args.evidence_cache) if args.evidence_cache else None
    evidences = load_evidence_cache(cache_path, queries, cache_key) if cache_path else None

    bundle = load_flat_bundle(args.flat_root)
    cascade = build_flat_cascade(config, args.flat_root, flat_bundle=bundle)
    if evidences is None:
        from tune_weights import collect_evidence  # noqa: PLC0415 — script canh script

        evidences = collect_evidence(
            cascade, queries, config.retrieval.coarse_topk, args.en_source
        )

    weights = config.retrieval.fusion_weights.model_dump()
    relational = RelationalMap(config.paths.relational_db)

    fusion = config.retrieval.fusion
    by_id = {e.query.query_id: e for e in evidences}

    #: Mệnh đề mà **không arm nào** trả về gì. Phải in ra to: một mệnh đề rỗng
    #: cho `prior` rỗng, `apply_prior` thành phép không-làm-gì, và bảng in
    #: `+0,000` — **giống hệt** một tầng chạy đủ mà vô ích. Đã mất một lượt chạy
    #: vì đúng chuyện này (`run_id` chứa `#` làm trượt pattern `evidence_id`,
    #: `ArmRegistry.run` nuốt `ValidationError`), nên chỗ phân biệt hai khả năng
    #: đó phải nằm trong chính bản báo cáo.
    empty_clauses: list[str] = []
    rows: list[dict] = []
    for query in queries:
        evidence = by_id.get(query.query_id)
        if evidence is None:
            continue
        base = fuse_and_group(
            evidence.per_event[0], _assignment(weights), cascade.group,
            fusion.normalizer, fusion.rrf_k, config.retrieval.coarse_topk,
        )
        base_wide, base_rank = _score(query, base, relational, top_k)

        texts = hand.get(query.query_id)
        clauses = clauses_from_texts(texts) if texts else split_clauses(query.text)
        row = {
            "query_id": query.query_id,
            "clauses": len(clauses),
            "source": "tay" if texts else "luat",
            "base": {"wide": base_wide, "rank": base_rank},
        }

        if len(clauses) < 2:
            row["after"] = row["base"]
            rows.append(row)
            continue

        per_clause = []
        for index, clause in enumerate(clauses):
            text_en = (
                cascade.resolve_english(clause.text, None)
                if args.en_source != "none" else None
            )
            # `-c{i}`, KHÔNG `#c{i}`. `run_id` đi thẳng vào `evidence_id`, và
            # `Candidate` bắt nó khớp `^[a-zA-Z0-9_]+:[a-zA-Z0-9_\-]+:\d+$` —
            # `#` trượt pattern, `ArmRegistry.run` nuốt `ValidationError` rồi
            # ghi log "arm ... returned nothing", nên MỌI arm trả rỗng và bảng
            # in ra `+0,000` y hệt một tầng chạy mà vô ích. Mất một lượt chạy vì
            # đúng chuyện này.
            arm_out = cascade.arms.run(
                clause.text, config.retrieval.coarse_topk,
                f"{query.query_id}-c{index}", query_text_en=text_en,
            )
            fused = fuse_and_group(
                arm_out, _assignment(weights), cascade.group,
                fusion.normalizer, fusion.rrf_k, config.retrieval.coarse_topk,
            )
            if not fused:
                empty_clauses.append(f"{query.query_id}-c{index}")
            per_clause.append(_to_hits(fused, relational))

        prior = video_prior(per_clause)
        after = apply_prior(
            base, prior,
            ordinal_of=lambda kid: relational.get_keyframe(kid).keyframe_index,
            weight=args.weight, window=args.window,
        )
        after_wide, after_rank = _score(query, after, relational, top_k)
        row["after"] = {"wide": after_wide, "rank": after_rank}
        row["videos_co_prior"] = len(prior)
        gt = prior.get(query.gt_video_id or "")
        row["prior_video_dung"] = round(gt.score, 3) if gt else 0.0
        rows.append(row)
        print(
            f"  {query.query_id:<16} {len(clauses)} menh de · "
            f"nen {base_wide:.2f}/{base_rank or '-'} -> "
            f"{after_wide:.2f}/{after_rank or '-'} · "
            f"prior video dung {row['prior_video_dung']}"
        )

    if empty_clauses:
        print(
            f"\n  !! {len(empty_clauses)} menh de KHONG arm nao tra ve gi: "
            f"{', '.join(empty_clauses[:6])}"
            f"{' …' if len(empty_clauses) > 6 else ''}"
        )
        print("     Bang duoi in +0,000 vi KHONG CHAY, khong phai vi vo ich.")

    touched = [r for r in rows if r["clauses"] >= 2]
    untouched = [r for r in rows if r["clauses"] < 2]

    print(f"\nCHAM: {len(touched)}/{len(rows)} cau tach duoc >=2 menh de")
    if touched:
        _report("cham", touched)
    print(f"\nKHONG CHAM: {len(untouched)} cau")
    changed = [
        r for r in untouched
        if r["after"]["wide"] != r["base"]["wide"] or r["after"]["rank"] != r["base"]["rank"]
    ]
    if changed:
        print(f"  ! {len(changed)} cau DOI dù không tách được — đó là LỖI, không phải kết quả")
        for r in changed[:5]:
            print(f"    {r['query_id']}: {r['base']} -> {r['after']}")
    else:
        print("  giu nguyen tung con so — dung nhu `apply_prior` hua")

    print(f"\nCA BO ({len(rows)} cau KIS)")
    _report("tong", rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"weight": args.weight, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nbao cao: {out}")
    relational.close()


def _assignment(weights: dict[str, float]) -> Assignment:
    """`Assignment` mang đúng bảng trọng số của config, không tắt arm nào.

    Arm trọng số 0,0 trong config vẫn **bật** ở đây. `Assignment.disabled()` chỉ
    tính `OFF`, và OFF khác 0,0: OFF vứt arm khỏi phép gộp, 0,0 giữ nó lại với
    phiếu bằng không. Bảng trọng số đem đi thi để `ocr` và `object` ở 0,0 chứ
    không OFF, nên nền đo ở đây phải y hệt.
    """
    return Assignment(settings=dict(weights))


def _report(label: str, rows: list[dict]) -> None:
    base_wide = statistics.fmean(r["base"]["wide"] for r in rows)
    after_wide = statistics.fmean(r["after"]["wide"] for r in rows)
    base_ranks = [r["base"]["rank"] for r in rows if r["base"]["rank"]]
    after_ranks = [r["after"]["rank"] for r in rows if r["after"]["rank"]]
    up = sum(
        1 for r in rows
        if r["after"]["rank"] and r["base"]["rank"] and r["after"]["rank"] < r["base"]["rank"]
    )
    down = sum(
        1 for r in rows
        if r["after"]["rank"] and r["base"]["rank"] and r["after"]["rank"] > r["base"]["rank"]
    )
    print(
        f"  {label:<6} RONG {base_wide:.3f} -> {after_wide:.3f} "
        f"(d {after_wide - base_wide:+.3f}) · "
        f"hangTB {statistics.fmean(base_ranks) if base_ranks else 0:.1f} -> "
        f"{statistics.fmean(after_ranks) if after_ranks else 0:.1f} · "
        f"len {up} · xuong {down}"
    )


if __name__ == "__main__":
    main()
