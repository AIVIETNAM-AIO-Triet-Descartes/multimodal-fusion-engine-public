"""Bốn tầng rerank đáng bao nhiêu điểm — đo từng tầng RIÊNG, trên cùng một đầu vào.

    python scripts/measure_rerank.py --queryset eval/queryset_dev.jsonl \
        --skip-types TRAKE --flat-root stores/flat \
        --evidence-cache .cache/tune/dev-kisqa.json --tiers ta,colpali,hf_cross

Vì sao đo riêng chứ không đo chuỗi
-----------------------------------
`apply_tiers` chạy các tầng nối tiếp, mỗi tầng ăn đầu ra của tầng trước — đó là
cách chúng chạy thật. Nhưng để **quyết định bật tầng nào**, chuỗi là phép đo sai:
tầng thứ hai chỉ được chấm trên phần tầng thứ nhất đã dọn, nên đóng góp của nó
lẫn với thứ tự đặt. Ở đây mỗi tầng nhận **cùng một** danh sách hợp nhất, và cột
`ΔRỘNG` đọc được độc lập.

Cột `chuỗi` ở cuối vẫn có mặt, vì đó là cấu hình sẽ đem đi thi.

Ba con số phải đọc cùng nhau
-----------------------------
* **ΔRỘNG** — Final Score đổi bao nhiêu. Bão hoà sớm: một đáp án đã nằm trong
  top 5 thì đẩy nó lên hạng 1 **không** đổi điểm.
* **hạng TB** — hạng đáp án đúng đầu tiên. Nó động khi Final Score đứng yên, nên
  đây là chỗ nhìn thấy một tầng đang làm việc trên bộ 42 câu.
* **giây/câu** — tầng đắt phải trả bằng điểm, không phải bằng lời hứa.

Ảnh: `KeyframeImages` kéo từ Modal Volume và cache ở `.cache/keyframes/`. Lượt
đầu trên 42 câu × `--depth` ảnh là hàng nghìn file — dùng `--prefetch-only` để
tách phần tải ra khỏi phần đo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from pathlib import Path

from mfe.core.cli import setup_console
from mfe.core.config import load_config
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY, evaluate_query
from mfe.eval.official import (
    QAQuery,
    TRAKEQuery,
    is_heldout,
    load_official_queryset,
    to_kis_submissions,
    to_qa_submissions,
)
from mfe.eval.tuning import (
    Assignment,
    evidence_key,
    fuse_and_group,
    load_evidence_cache,
    save_evidence_cache,
)
from mfe.mapping.relational import RelationalMap
from mfe.online.cascade import Cascade
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle
from mfe.online.keyframe_images import KeyframeImages
from mfe.online.rerank_tiers import DEFAULT_DEPTH, apply_tiers, build_tier

logger = logging.getLogger("measure_rerank")

def score_of(query, candidates, relational, top_k: int):
    """`(Final RỘNG, Final HẸP | None, hạng đáp án đúng đầu tiên | None)`."""
    if isinstance(query, QAQuery):
        submissions = list(to_qa_submissions(candidates, relational, query.answer(), limit=top_k))
    else:
        submissions = list(to_kis_submissions(candidates, relational, limit=top_k))
    wide = evaluate_query(submissions, query.ground_truth(), query_id=query.query_id)
    strict_gt = query.ground_truth_strict()
    strict = (
        evaluate_query(submissions, strict_gt, query_id=query.query_id)
        if strict_gt is not None
        else None
    )
    return wide.final_score, (strict.final_score if strict else None), wide.first_hit_rank


class Totals:
    """Cộng dồn một tầng qua cả bộ truy vấn."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.wide: list[float] = []
        self.strict: list[float] = []
        self.ranks: list[int] = []
        #: Một mục mỗi câu. Giữ cả danh sách chứ không cộng dồn: **câu đầu tiên
        #: gánh lượt nạp model** (đo được 44 giây cho `ta`), nên trung bình cộng
        #: nói dối về chi phí mỗi truy vấn còn trung vị thì không.
        self.per_query_s: list[float] = []
        self.scored = 0
        self.moved = 0
        self.ran = 0
        self.reasons: dict[str, int] = {}

    def add(self, wide, strict, rank) -> None:
        self.wide.append(wide)
        if strict is not None:
            self.strict.append(strict)
        if rank is not None:
            self.ranks.append(rank)

    def note(self, outcome) -> None:
        if outcome is None:
            return
        if outcome.ran:
            self.ran += 1
            self.scored += outcome.scored
            self.moved += len(outcome.moved)
            self.per_query_s.append(outcome.elapsed_s)
        else:
            self.reasons[outcome.reason or "?"] = self.reasons.get(outcome.reason or "?", 0) + 1

    @property
    def mean_wide(self) -> float:
        return sum(self.wide) / len(self.wide) if self.wide else 0.0

    @property
    def mean_strict(self) -> float | None:
        return sum(self.strict) / len(self.strict) if self.strict else None

    @property
    def mean_rank(self) -> float | None:
        return sum(self.ranks) / len(self.ranks) if self.ranks else None

    @property
    def median_s(self) -> float:
        return statistics.median(self.per_query_s) if self.per_query_s else 0.0

    @property
    def total_s(self) -> float:
        return sum(self.per_query_s)


class _EnglishForQuery:
    """Bản tiếng Anh cho `colpali` và `vlm` — kể cả khi truy hồi đọc từ cache.

    Chạy từ cache thì không có `Cascade`, mà hai tầng đó vẫn cần tiếng Anh và sẽ
    **ngồi ngoài** nếu không có. Bộ dịch dựng riêng ở đây, và nó đọc đúng
    `retrieval.translation.cache_path` — nên lượt đo không mua lại bản dịch mà
    `tune_weights.py` đã trả tiền.
    """

    def __init__(self, config, cascade, en_source: str) -> None:
        self.en_source = en_source
        self.cascade = cascade
        self._translator = None
        self._config = config

    def _lazy(self):
        if self._translator is None:
            from mfe.online.translate import build_translator  # noqa: PLC0415

            self._translator = build_translator(self._config)
        return self._translator

    def __call__(self, query, search_text: str) -> str | None:
        """`search_text` chứ không `query.text` — với QA hai thứ đó khác nhau."""
        if self.en_source == "none":
            return None
        if query.text_en:
            return query.text_en
        if self.en_source == "manual":
            return None
        if self.cascade is not None:
            return self.cascade.resolve_english(search_text, None)
        translator = self._lazy()
        if translator is None:
            return None
        translation = translator.translate(search_text, manual=None)
        return translation.text_en if translation is not None else None

    def save(self) -> None:
        if self._translator is not None:
            self._translator.save_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="measure_rerank", description=__doc__)
    parser.add_argument("--config", default="config/btc_b1.yaml")
    parser.add_argument("--queryset", default="eval/queryset_dev.jsonl")
    parser.add_argument("--flat-root", default="stores/flat")
    parser.add_argument("--evidence-cache", default=None)
    parser.add_argument("--en-source", choices=("manual", "auto", "none"), default="auto")
    parser.add_argument("--skip-types", default=None)
    parser.add_argument("--tiers", default="ta", help="ta,colpali,hf_cross,vlm")
    parser.add_argument(
        "--depth", type=int, default=None,
        help=f"Ghi de do sau cho MOI tang. Mac dinh tung tang: {DEFAULT_DEPTH}",
    )
    parser.add_argument("--budget-ms", type=int, default=600000,
                        help="Ngan sach tang `ta`. Mac dinh rong: day la phep DO.")
    parser.add_argument("--chain", action="store_true",
                        help="Do them cau hinh CHUOI: cac tang noi tiep, dung thu tu --tiers.")
    parser.add_argument("--prefetch-only", action="store_true",
                        help="Chi keo anh ve cache roi dung. Tach phan tai khoi phan do.")
    parser.add_argument("--limit", type=int, default=None, help="Chi lay N cau dau (thu nhanh).")
    parser.add_argument("--out", default="eval/results_btc_b1/rerank-tiers.json")
    return parser


def _load_env_file(path: str | Path = ".env") -> None:
    """Đưa `.env` vào môi trường, **không** đè lên biến đã đặt sẵn.

    Hai tầng đọc thẳng `os.environ`: `hf_cross` cần `HF_TOKEN`, `vlm` cần
    `GEMINI_API_KEY`. Thiếu bước này thì cả hai **báo đúng** ("chua co HF_TOKEN")
    và lượt đo vẫn chạy hết, in ra một bảng trông bình thường với hai dòng
    `+0,000` — nghĩa là "tầng không giúp gì" đọc giống hệt "tầng không chạy".
    Lượt đo 24-08 mất đúng một vòng vì chuyện này.
    """
    from mfe.offline.transport import load_dotenv  # noqa: PLC0415

    for key, value in load_dotenv(path).items():
        os.environ.setdefault(key, value)


def main() -> None:  # noqa: PLR0915 — một lượt đo tuyến tính, tách ra chỉ làm khó đọc
    setup_console()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    _load_env_file()

    config = load_config(args.config)
    queries = load_official_queryset(args.queryset)
    if is_heldout(queries):
        raise SystemExit(
            f"{args.queryset} chua cau HELD-OUT. Script nay in HANH VI tung cau, "
            "nen khong duoc chay tren heldout."
        )
    if args.skip_types:
        dropped = {t.strip().upper() for t in args.skip_types.split(",") if t.strip()}
        queries = [q for q in queries if q.type.upper() not in dropped]
    # TRAKE đi qua DANTE để ghép N sự kiện thành một dòng, nên "rerank rồi chấm"
    # là một phép đo khác hẳn. Bỏ ở đây thay vì chấm sai, và nói ra.
    trake = [q for q in queries if isinstance(q, TRAKEQuery)]
    if trake:
        print(f"bo {len(trake)} cau TRAKE — chung ghep qua DANTE, khong cung phep do")
        queries = [q for q in queries if not isinstance(q, TRAKEQuery)]
    if args.limit:
        queries = queries[: args.limit]

    top_k = min(config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)
    fusion = config.retrieval.fusion
    normalizer, rrf_k = fusion.resolved_normalizer(), fusion.rrf_k

    relational = RelationalMap(config.paths.relational_db)
    relational.init_schema()

    cache_key = evidence_key(
        args.config, args.queryset, config.retrieval.coarse_topk, args.en_source
    )
    cache_path = Path(args.evidence_cache) if args.evidence_cache else None
    evidences = load_evidence_cache(cache_path, queries, cache_key) if cache_path else None

    cascade = None
    if evidences is None:
        from tune_weights import collect_evidence  # noqa: PLC0415 — script canh script

        cascade = (
            build_flat_cascade(config, args.flat_root, flat_bundle=load_flat_bundle(args.flat_root))
            if args.flat_root
            else Cascade.from_config(config)
        )
        evidences = collect_evidence(
            cascade, queries, config.retrieval.coarse_topk, args.en_source
        )
        if cascade.translator is not None:
            cascade.translator.save_cache()
        if cache_path is not None:
            save_evidence_cache(cache_path, evidences, cache_key)

    grouper = (cascade if cascade is not None else Cascade(config=config)).group
    weights = config.retrieval.fusion_weights.model_dump()

    fused_by_query: dict[str, list] = {}
    for evidence in evidences:
        arms = sorted(evidence.arm_names)
        assignment = Assignment({name: float(weights.get(name, 1.0)) for name in arms})
        fused_by_query[evidence.query.query_id] = fuse_and_group(
            evidence.per_event[0], assignment, grouper, normalizer, rrf_k, top_k,
            run_id=evidence.query.query_id,
        )

    names = [t.strip() for t in args.tiers.split(",") if t.strip()]
    depths = {n: (args.depth or DEFAULT_DEPTH.get(n, 50)) for n in names}

    # Ảnh trước, đo sau. Tải xen giữa các phép đo sẽ tính thời gian mạng vào
    # `giây/câu` của tầng, và cột đó dùng để so tầng với nhau.
    images = KeyframeImages()
    pixel_depth = max(
        [depths[n] for n in names if n in {"ta", "colpali", "vlm"}] or [0]
    )
    if pixel_depth:
        wanted = [
            c.keyframe_id
            for candidates in fused_by_query.values()
            for c in candidates[:pixel_depth]
        ]
        unique = list(dict.fromkeys(wanted))
        started = time.perf_counter()
        ready = images.prefetch(unique)
        print(
            f"anh: {ready}/{len(unique)} san sang ({time.perf_counter() - started:.0f}s). "
            f"Thieu {len(unique) - ready} — ung vien khong co anh giu nguyen cho."
        )
    if args.prefetch_only:
        relational.close()
        return

    tiers = {name: build_tier(name, images, args.budget_ms) for name in names}
    for name, tier in tiers.items():
        ok, reason = tier.available()
        if not ok:
            print(f"  ! {name}: {reason}")

    totals = {"<hop nhat>": Totals("<hop nhat>")}
    for name in names:
        totals[name] = Totals(name)
    if args.chain:
        totals["<chuoi>"] = Totals("<chuoi>")

    # `hf_cross` chấm trên VĂN BẢN đi kèm ứng viên, nên nó cần bản gộp — kể cả
    # khi phần truy hồi đã đọc từ cache và không mở bản gộp nào.
    bundle = None
    if "hf_cross" in names:
        bundle = (
            cascade.flat_bundle
            if cascade is not None and cascade.flat_bundle is not None
            else load_flat_bundle(args.flat_root, models=set())
        )

    def texts_of(keyframe_id: str) -> str:
        if bundle is None:
            return ""
        parts = [bundle.texts[s].get(keyframe_id, "") for s in bundle.texts]
        return " ".join(p for p in parts if p)

    english = _EnglishForQuery(config, cascade, args.en_source)

    def search_text_of(query) -> str:
        """Đúng chuỗi tầng A đã tìm bằng — với QA đó là `retrieval_text`.

        KHÔNG phải `query.text`. Câu QA của BTC là một **câu hỏi**, còn SigLIP
        huấn luyện trên caption: đưa câu hỏi vào tầng rerank thì nó chấm sai đúng
        những câu tầng A đã làm đúng. Đo được lượt đầu (bản sai): `quang_qa_01`
        hạng 1 → 51, `quang_qa_03` hạng 2 → 38 — và tổng thể tầng `ta` trông như
        chỉ được +0,014.

        `QAQuery.retrieval_text` có sẵn cho đúng chuyện này, và
        `mfe.eval.official.score_one` cũng đưa nó xuống rerank theo cùng đường —
        nên đo bằng `query.text` là đo một hệ khác với hệ chạy thật.
        """
        if isinstance(query, QAQuery):
            return query.retrieval_text or query.text
        return query.text

    per_query: list[dict] = []
    for query in queries:
        fused = fused_by_query[query.query_id]
        base_wide, base_strict, base_rank = score_of(query, fused, relational, top_k)
        totals["<hop nhat>"].add(base_wide, base_strict, base_rank)
        row = {"query_id": query.query_id, "type": query.type,
               "base": {"wide": base_wide, "rank": base_rank}}

        search_text = search_text_of(query)
        query_en = english(query, search_text)

        for name in names:
            after, outcome = tiers[name].rerank(
                fused, query_vi=search_text, query_en=query_en,
                images=images, texts=texts_of, depth=depths[name],
            )
            wide, strict, rank = score_of(query, after, relational, top_k)
            totals[name].add(wide, strict, rank)
            totals[name].note(outcome)
            row[name] = {"wide": wide, "rank": rank, "ran": outcome.ran,
                         "reason": outcome.reason, "moved": len(outcome.moved)}

        if args.chain:
            after, outcomes = apply_tiers(
                fused, [tiers[n] for n in names], query_vi=search_text,
                query_en=query_en, images=images, texts=texts_of,
                depth=max(depths.values()),
            )
            wide, strict, rank = score_of(query, after, relational, top_k)
            totals["<chuoi>"].add(wide, strict, rank)
            for outcome in outcomes:
                totals["<chuoi>"].note(outcome)
            row["<chuoi>"] = {"wide": wide, "rank": rank}

        per_query.append(row)
        cells = " ".join(
            f"{n}:{row[n]['wide']:.2f}/{row[n]['rank'] or '-'}" for n in names
        )
        print(f"  {query.query_id:<16} nen:{base_wide:.2f}/{base_rank or '-':<4} {cells}")

    base = totals["<hop nhat>"]
    header = (
        f"\n{'tang':<12}{'RONG':>8}{'d':>8}{'HEP':>8}{'hangTB':>9}"
        f"{'trung':>7}{'chay':>7}{'doi cho':>9}{'s/cau*':>8}"
    )
    print(header)
    print("-" * len(header.strip("\n")))
    for label, total in totals.items():
        delta = "" if label == "<hop nhat>" else f"{total.mean_wide - base.mean_wide:+.3f}"
        strict = f"{total.mean_strict:.3f}" if total.mean_strict is not None else "-"
        rank = f"{total.mean_rank:.1f}" if total.mean_rank is not None else "-"
        print(
            f"{label:<12}{total.mean_wide:>8.3f}{delta:>8}{strict:>8}{rank:>9}"
            f"{len(total.ranks):>4}/{len(queries):<2}{total.ran:>7}"
            f"{total.moved:>9}{total.median_s:>8.1f}"
        )
    print(
        "\nRONG bao hoa: dap an da nam trong top 5 thi day len hang 1 KHONG doi diem.\n"
        "Doc `hangTB` cung luc — no dong khi RONG dung yen.\n"
        "* s/cau la TRUNG VI: cau dau tien ganh lan nap model, trung binh cong "
        "se noi doi ve chi phi moi truy van."
    )
    for label, total in totals.items():
        for reason, count in total.reasons.items():
            print(f"  ! {label}: {count}/{len(queries)} cau khong chay — {reason}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "queryset": args.queryset,
                "n_queries": len(queries),
                "tiers": names,
                "depths": depths,
                "weights": weights,
                "summary": {
                    label: {
                        "wide": total.mean_wide,
                        "strict": total.mean_strict,
                        "mean_rank": total.mean_rank,
                        "hits": len(total.ranks),
                        "ran": total.ran,
                        "moved": total.moved,
                        "median_s": total.median_s,
                        "total_s": total.total_s,
                        "reasons": total.reasons,
                    }
                    for label, total in totals.items()
                },
                "per_query": per_query,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nbao cao: {out}")
    english.save()
    relational.close()


if __name__ == "__main__":
    main()
