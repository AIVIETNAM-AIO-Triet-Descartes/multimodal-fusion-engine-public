"""Đo ĐỘ TRỄ truy hồi: từ lúc có câu truy vấn tới lúc có danh sách xếp hạng.

    .venv/Scripts/python.exe scripts/measure_latency.py \
        --queryset eval/queryset_dev.jsonl --flat-root stores/flat --n 20

Vì sao có file này
-------------------
Con số duy nhất từng đo là **24,8 ms cho một phép nhân ma trận** trên 176.707
vector. Nhưng đó là *một arm*, không phải *một truy vấn*: một truy vấn thật còn
phải mã hoá câu hỏi bằng tháp text SigLIP, chạy 5 arm BM25, hợp nhất RRF, và
(với câu KIS ≥3 mệnh đề) chạy thêm `scene_order`. Lập lịch bàn nộp bằng 24,8 ms
là lập lịch cho một hệ không tồn tại.

⚠ Đo **sau khi đã nạp xong**, giống hệt máy chủ đang chạy. Chi phí NẠP là thứ
khác hẳn và đã ghi trong `query_flat.py`: BM25 0,6s · npz 4,2s · tháp text
SigLIP ~10s. Một tiến trình Streamlit trả cái đó **một lần**, rồi mọi truy vấn
sau ăn con số của file này.

⚠ Lượt ĐẦU luôn chậm hơn hẳn (lazy-load của torch, cache CPU nguội), nên bỏ
`--warmup` lượt rồi mới tính. Không bỏ thì trung vị bị kéo lệch bởi đúng một
mẫu, và đó là mẫu không đại diện cho truy vấn thứ hai trở đi.

⚠ KHÔNG đo `rerank_tiers` ở đây — `vlm` đo được **27,6 s/câu** và sẽ nuốt trọn
bảng. Nó là quyết định riêng (§4), đo riêng bằng `measure_rerank.py`.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time

sys.path.insert(0, "src")

from mfe.core.cli import setup_console  # noqa: E402
from mfe.eval.official import load_official_queryset  # noqa: E402
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle  # noqa: E402

logger = logging.getLogger(__name__)


def _search_text(query) -> str:
    """Đúng chuỗi tầng A tìm bằng — với QA đó là `retrieval_text`, không phải câu hỏi.

    Đo bằng `query.text` cho câu QA là đo một hệ mà `score_one` không chạy (§5).
    """
    from mfe.eval.official import QAQuery

    if isinstance(query, QAQuery):
        return query.retrieval_text or query.text
    return query.text


def main() -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="config/btc_b1.yaml")
    p.add_argument("--queryset", default="eval/queryset_dev.jsonl")
    p.add_argument("--flat-root", default="stores/flat")
    p.add_argument("--n", type=int, default=20, help="số câu đo")
    p.add_argument("--warmup", type=int, default=2, help="số lượt bỏ đi trước khi tính")
    p.add_argument("--top-k", type=int, default=100)
    args = p.parse_args()

    from mfe.core.config import load_config

    t0 = time.perf_counter()
    config = load_config(args.config)
    queries = load_official_queryset(args.queryset)
    bundle = load_flat_bundle(args.flat_root)
    cascade = build_flat_cascade(config, args.flat_root, flat_bundle=bundle)
    t_nap = time.perf_counter() - t0
    print(f"nạp xong {t_nap:.1f}s · {bundle.report()}")
    print(f"{len(queries)} câu trong bộ đề, đo {args.n} "
          f"(bỏ {args.warmup} lượt đầu)\n")

    chon = queries[: args.n + args.warmup]
    do: list[tuple[str, float]] = []
    dich: list[float] = []
    tim: list[float] = []
    for i, q in enumerate(chon):
        text = _search_text(q)
        t = time.perf_counter()
        try:
            # Tách DỊCH khỏi TÌM: `search` tự gọi `resolve_english` khi thiếu
            # bản tiếng Anh, và đó là một lượt suy luận seq2seq — trộn nó vào
            # con số "truy hồi" là giấu mất chi phí lớn nhất.
            en = cascade.resolve_english(text, None)
            t_dich = time.perf_counter() - t
            t2 = time.perf_counter()
            ra = cascade.search(text, top_k=args.top_k, run_id="do_do_tre",
                                query_text_en=en)
            t_tim = time.perf_counter() - t2
            n_ra = len(ra[0]) if isinstance(ra, tuple) else len(ra)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {getattr(q, 'query_id', i)}: {type(e).__name__}: {e}")
            continue
        dt = t_dich + t_tim
        nhan = "khởi động" if i < args.warmup else "       "
        print(f"  {nhan} {getattr(q, 'query_id', i):<24} {dt * 1000:7.0f} ms "
              f"= dịch {t_dich * 1000:6.0f} + tìm {t_tim * 1000:6.0f} "
              f"· {n_ra:3d} kết quả")
        if i >= args.warmup:
            do.append((getattr(q, "query_id", str(i)), dt))
            dich.append(t_dich)
            tim.append(t_tim)

    if not do:
        print("\nkhông câu nào chạy được")
        return 1

    ms = sorted(d * 1000 for _, d in do)
    print(f"\n{len(ms)} câu, SAU khi đã nạp:")
    print(f"  trung vị   {statistics.median(ms):7.0f} ms")
    print(f"  trung bình {statistics.fmean(ms):7.0f} ms")
    print(f"  nhanh nhất {ms[0]:7.0f} ms")
    print(f"  chậm nhất  {ms[-1]:7.0f} ms")
    if len(ms) >= 10:
        print(f"  p90        {ms[int(len(ms) * 0.9)]:7.0f} ms")
    if dich:
        md, mt = statistics.median(dich) * 1000, statistics.median(tim) * 1000
        print("\ntách hai phần (trung vị):")
        print(f"  DỊCH vi→en {md:7.0f} ms  ({md / (md + mt) * 100:.0f}%)")
        print(f"  TÌM        {mt:7.0f} ms  ({mt / (md + mt) * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
