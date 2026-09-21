"""`caption_qwen` nên ăn truy vấn tiếng VIỆT, tiếng ANH, hay CẢ HAI?

    .venv/Scripts/python.exe scripts/measure_caption_query_lang.py

Vì sao phép đo này tồn tại
---------------------------
`qwen_captions.json` là nửa `[CAPTION]`, **viết tiếng Anh** (quyết định 25-08).
Nhưng `TEXT_ARM_SPECS` trước 21-09 không có chỗ khai ngôn ngữ, nên arm này nhận
mặc định `"vi"` và đã tìm một chỉ mục tiếng Anh bằng truy vấn tiếng Việt suốt từ
đó — dù `_DESCRIPTIONS` của chính nó ghi "cần bản dịch truy vấn".

Ba cấu hình, cùng bộ đề, cùng kho, chỉ đổi CHUỖI TRUY VẤN:

    vi      truy vấn gốc            (hiện trạng trước khi sửa)
    en      bản dịch               (điều tài liệu vẫn tưởng đang chạy)
    vi+en   nối cả hai             (ý của Lead 21-09)

Cấu hình thứ ba không phải thoả hiệp cho vui. Đo được: **12,21%** token phân biệt
trong `qwen_captions.json` có dấu tiếng Việt — Qwen rò chữ trên màn hình vào nửa
caption bất chấp prompt. Những token đó hiếm nên idf rất cao (51,3% token tiếng
Việt trong caption V5 chỉ xuất hiện ở ĐÚNG 1 caption). Truy vấn tiếng Việt khớp
được chúng; bản dịch tiếng Anh thì không.

Dịch bằng cổng của Đạt (`mfe.online.trammay`) và **cache ra đĩa** — bản dịch
không đổi giữa các lượt, và trả tiền hai lần cho cùng một câu là lãng phí.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

from mfe.core.cli import setup_console
from mfe.core.config import load_config
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle
from mfe.online.trammay import TrammayClient, available, translate_backend

logger = logging.getLogger(__name__)

CACHE = Path(".cache/tune/dich-truy-van-trammay.json")


def nap_de(duong: Path, bo_dang: set[str]) -> list[dict]:
    ra = []
    for dong in duong.read_text(encoding="utf-8").splitlines():
        if not dong.strip():
            continue
        q = json.loads(dong)
        if q.get("type") in bo_dang:
            continue
        if q.get("gt_video_id"):
            ra.append(q)
    return ra


def hang_cua_dap_an(hits, q: dict) -> int | None:
    """Thứ hạng của khung đáp án ĐẦU TIÊN, hoặc None nếu không có trong danh sách.

    Dùng `gt_frame_range` (cửa sổ RỘNG) chứ không phải `_strict`: arm đơn lẻ chỉ
    cần đưa được đúng vùng lên, việc chốt khung là của tầng sau.
    """
    vid = q["gt_video_id"]
    lo, hi = q.get("gt_frame_range") or [0, 10**9]
    for i, c in enumerate(hits, start=1):
        v, _, f = c.keyframe_id.partition("#")
        if v == vid and lo <= int(f) <= hi:
            return i
    return None


def _nap_env(path: str = ".env") -> None:
    """Nạp `.env` trước khi hỏi `available()`.

    Không có bước này thì script báo "chưa có TRAMMAY_API_KEY" trong khi `.env`
    có sẵn — đúng lỗi đã làm `measure_rerank.py` mất trắng một lượt đo và làm
    `mfe.eval.official` xuất ra file nộp không hề rerank (CLAUDE.md §5).
    """
    import os  # noqa: PLC0415

    from mfe.offline.transport import load_dotenv  # noqa: PLC0415

    for key, value in load_dotenv(path).items():
        os.environ.setdefault(key, value)


def main() -> None:
    setup_console()
    _nap_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/btc_b1.yaml")
    p.add_argument("--queryset", default="eval/queryset_dev.jsonl")
    p.add_argument("--flat-root", default="stores/flat")
    p.add_argument("--arm", default="caption_qwen")
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--skip-types", default="TRAKE")
    p.add_argument("--limit", type=int, default=0, help="0 = tất cả")
    args = p.parse_args()

    ok, vi_sao = available()
    if not ok:
        logger.error("khong dich duoc: %s", vi_sao)
        sys.exit(1)

    de = nap_de(Path(args.queryset), {t for t in args.skip_types.split(",") if t})
    if args.limit:
        de = de[: args.limit]
    logger.info("%d cau de (bo %s)", len(de), args.skip_types)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str] = (
        json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    )
    backend = translate_backend(TrammayClient.from_env())
    moi = 0
    for q in de:
        t = q["text"]
        if t not in cache:
            cache[t] = backend.translate(t)
            moi += 1
            if moi % 10 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                logger.info("  da dich %d cau moi", moi)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    logger.info("dich xong: %d moi, %d trong cache", moi, len(cache))

    cfg = load_config(args.config)
    casc = build_flat_cascade(
        cfg, args.flat_root, flat_bundle=load_flat_bundle(args.flat_root)
    )

    # Khoá phải ASCII và không dấu: `run_id` đi vào `evidence_id`, mà mẫu của nó
    # là `^[a-zA-Z0-9_]+:[a-zA-Z0-9_\-]+:\d+$`. Một khoá `vi+en` làm mọi ứng viên
    # hỏng validation và `ArmRegistry.run` NUỐT arm hỏng im lặng — arm trả 0 kết
    # quả trông y hệt arm không tìm thấy gì. Đã mất một lượt đo vì `run_id="dò"`.
    cach = {
        "vi": lambda q: q["text"],
        "en": lambda q: cache[q["text"]],
        "vi_en": lambda q: f"{q['text']} {cache[q['text']]}",
    }
    hang: dict[str, list[int | None]] = {k: [] for k in cach}
    for q in de:
        for ten, lam in cach.items():
            chuoi = lam(q)
            # Đưa CÙNG một chuỗi vào cả hai tham số: phép đo này tự quyết định
            # arm thấy gì, nên nó không được phụ thuộc vào `query_language` mà
            # arm khai — đó đúng là thứ đang đem ra so.
            hits = (
                casc.arms.run_named(
                    [args.arm],
                    chuoi,
                    args.top_k,
                    run_id=f"lang_{ten}",
                    query_text_en=chuoi,
                )
                or {}
            ).get(args.arm) or []
            hang[ten].append(hang_cua_dap_an(hits, q))

    n = len(de)
    print(f"\n{'=' * 74}")
    print(f"arm `{args.arm}` — {n} câu, top {args.top_k}, chỉ đổi chuỗi truy vấn")
    print("=" * 74)
    print(f"  {'cách':<8}{'trúng':>8}{'R@1':>7}{'R@5':>7}{'R@20':>7}{'R@100':>8}{'hạng TB':>10}")
    print("  " + "-" * 62)
    for ten in cach:
        h = hang[ten]
        trung = [x for x in h if x]
        r = {k: sum(1 for x in trung if x <= k) / n for k in (1, 5, 20, 100)}
        tb = statistics.mean(trung) if trung else float("nan")
        print(f"  {ten:<8}{len(trung):>4}/{n:<3}{r[1]:>7.1%}{r[5]:>7.1%}"
              f"{r[20]:>7.1%}{r[100]:>8.1%}{tb:>10.1f}")

    print(f"\n{'=' * 74}\nSO CẶP — câu nào đổi hạng (bỏ câu cả hai cùng trượt)\n{'=' * 74}")
    for a, b in (("vi", "en"), ("vi", "vi_en"), ("en", "vi_en")):
        len_, xuong, hoa = 0, 0, 0
        for x, y in zip(hang[a], hang[b], strict=True):
            if x is None and y is None:
                continue
            if x is None:
                len_ += 1
            elif y is None:
                xuong += 1
            elif y < x:
                len_ += 1
            elif y > x:
                xuong += 1
            else:
                hoa += 1
        print(f"  {a:>6} -> {b:<7} lên {len_:>3} · xuống {xuong:>3} · hoà {hoa:>3}")


if __name__ == "__main__":
    main()
