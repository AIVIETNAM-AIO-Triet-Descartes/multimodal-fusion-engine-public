"""Per-arm diagnostic: run each stage-A arm alone and report where it puts the answer.

    python scripts/measure_arms.py --config config/pilot.yaml --queryset eval/dev/pilot.jsonl

Step B1 of docs/triet/ke-hoach-merge-sota.md §3.4b. Fusion weights are only worth
tuning once you know which arms carry information at all; an arm that never ranks
the answer does not need a weight, it needs removing.

Reports the rank of the first candidate falling inside the WIDE ground-truth
window. "-" means the arm never found it within `coarse_topk`.

`--en-source` decides where the English for `query_language="en"` arms comes
from, and exists to answer one question directly: what does `siglip` cost when
the hand-written `text_en` is replaced by the machine (handoff §5.1)?

    manual  the `text_en` on each query — hand-written, does not scale to 115
    auto    ignore it, translate with the configured chain instead
    none    supply none at all, so English-only arms sit out — the floor

`--flat-root` chạy trên **kho phẳng** (`stores/flat/`) thay vì Elasticsearch +
Milvus. Đó là đường đang thật sự chạy — xem `mfe.online.flat_arms` — nên một
phép đo không có cờ này đo một hệ mà máy này không dựng nổi.
"""

from __future__ import annotations

import argparse

from mfe.core.config import load_config
from mfe.eval.official import load_official_queryset
from mfe.online.cascade import Cascade
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle


def _gt_ranges(query) -> tuple[str, list[tuple[int, int]]]:
    """(video_id, [wide ranges]) — TRAKE contributes one range per event."""
    if query.type == "TRAKE":
        return query.gt_video_id, list(query.gt_events)
    return query.gt_video_id, [query.gt_frame_range]


def _first_hit_rank(candidates, video_id: str, ranges: list[tuple[int, int]]) -> int | None:
    from mfe.contracts.ids import parse_keyframe_id

    for rank, cand in enumerate(candidates, start=1):
        parsed_video, frame_id = parse_keyframe_id(cand.keyframe_id)
        if parsed_video != video_id:
            continue
        if any(start <= frame_id <= end for start, end in ranges):
            return rank
    return None


def build_cascade(config, flat_root: str | None) -> Cascade:
    """Cascade trên kho phẳng khi có `--flat-root`, còn lại là đường có kho."""
    if not flat_root:
        return Cascade.from_config(config)
    bundle = load_flat_bundle(flat_root)
    print(f"kho phang: {bundle.report()}")
    return build_flat_cascade(config, flat_root, flat_bundle=bundle)


def main() -> None:
    parser = argparse.ArgumentParser(prog="measure_arms", description=__doc__)
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--queryset", default="eval/dev/pilot.jsonl")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--en-source",
        choices=("manual", "auto", "none"),
        default="manual",
        help="Where the English query comes from; see the module docstring.",
    )
    parser.add_argument(
        "--flat-root",
        default=None,
        help="Chay tren kho phang (vd stores/flat) thay vi Elasticsearch + Milvus.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    queries = load_official_queryset(args.queryset)
    cascade = build_cascade(config, args.flat_root)

    def search_text_of(query) -> str:
        """Đúng chuỗi tầng A tìm bằng — với QA đó là `retrieval_text`.

        Câu QA của BTC là một **câu hỏi**, còn `siglip` huấn luyện trên caption;
        `QAQuery.retrieval_text` có sẵn cho đúng chỗ này. Đo bằng `query.text` là
        đo một hệ mà `mfe.eval.official.score_one` không chạy.
        """
        from mfe.eval.official import QAQuery, TRAKEQuery

        if isinstance(query, QAQuery):
            return query.retrieval_text or query.text
        if isinstance(query, TRAKEQuery):
            # Một dòng cho cả chuỗi: bảng này đo arm, không đo phép ghép DANTE.
            return query.text
        return query.text

    def english_for(query, search_text: str) -> str | None:
        """The English form under the chosen policy.

        `auto` passes `manual=None` deliberately: the point of the run is to
        measure the machine's translation, so letting the hand-written form win
        would report the manual number under the automatic label. It also
        translates `search_text`, not `query.text` — dịch một chuỗi rồi tìm bằng
        chuỗi khác là để `siglip` đi tìm thứ sáu arm còn lại không tìm.
        """
        if args.en_source == "none":
            return None
        if args.en_source == "manual":
            return query.text_en
        return cascade.resolve_english(search_text, None)

    arms = list(cascade.arms)
    print(f"\n{len(arms)} arm, {len(queries)} truy van, do sau {args.top_k}\n")
    for arm in arms:
        if not arm.available:
            print(f"  ! {arm.name}: {arm.unavailable_reason}")

    usable = [a.name for a in arms if a.available]
    header = f"{'query':<17}" + "".join(f"{n[:11]:>12}" for n in usable)
    print("\n" + header)
    print("-" * len(header))

    totals = dict.fromkeys(usable, 0)
    found = dict.fromkeys(usable, 0)

    for query in queries:
        video_id, ranges = _gt_ranges(query)
        search_text = search_text_of(query)
        text_en = english_for(query, search_text)
        row = f"{query.query_id:<17}"
        for name in usable:
            results = cascade.arms.run_named(
                [name], search_text, args.top_k, "measure", query_text_en=text_en
            )
            rank = _first_hit_rank(results.get(name, []), video_id, ranges)
            row += f"{(rank if rank else '-'):>12}"
            if rank:
                found[name] += 1
                totals[name] += rank
        print(row)

    print("-" * len(header))
    print(f"{'so cau tim thay':<17}" + "".join(f"{found[n]:>12}" for n in usable))
    print(
        f"{'hang trung binh':<17}"
        + "".join(f"{(totals[n] / found[n] if found[n] else 0):>12.1f}" for n in usable)
    )
    print("\n(hang thap = tot; '-' = khong tim thay trong top-k)")
    print(f"text_en: --en-source {args.en_source}", end="")
    if cascade.translator is not None:
        print(f"  ·  {cascade.translator.usage or 'khong goi toi'}")
        cascade.translator.save_cache()
    else:
        print()


if __name__ == "__main__":
    main()
