"""Lát cắt N video: sinh caption mới rồi đo TRUY HỒI THẬT trên chính lát đó.

    # pha 1+2+3, chạy lại được sau khi đứt
    .venv/Scripts/python.exe scripts/slice_caption_ab.py --videos 15 --prompt v5_bao

Vì sao phải có lát cắt thay vì đo trên kho hiện tại
----------------------------------------------------
Phép đo `measure_caption_query_lang.py` chạy trên `qwen_captions.json`, nơi tiếng
Việt xuất hiện ở 12,21% token do Qwen **rò rỉ ngẫu nhiên**. Kho caption mới sẽ có
chữ tiếng Việt **được lọc theo ngữ cảnh** — bảng nguyên liệu, tiêu đề công thức,
giá tiền. Khác cả lượng lẫn chất, nên suy từ kho này sang kho kia là "sai nền".

Muốn trả lời "truy vấn nên là vi / en / cả hai trên kho MỚI" thì phải có một kho
mới **đủ lớn để xếp hạng bên trong**. 183 caption rời rạc không xếp hạng được gì.

Lát cắt này là kho thật thu nhỏ: mọi keyframe của N video, và khung đáp án phải
vượt lên giữa hàng trăm khung anh em **cùng video** — bài toán khó nhất, và đúng
bài toán thi.

Ba pha, mỗi pha ghi tăng dần
-----------------------------
1. TẢI ảnh còn thiếu từ Modal   -> `.cache/keyframes/`
2. SINH caption qua cổng Đạt    -> `out/caption.json`  (bỏ qua khung đã có)
3. ĐO vi / en / vi_en trên lát  -> in bảng

Đứt giữa chừng thì chạy lại lệnh cũ: pha 1 và 2 đều bỏ qua phần đã xong. Đây
không phải sự cẩn thận thừa — pha 2 mất ~50 phút cho 15 video, và mất trắng vì
máy ngủ là mất 50 phút.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mfe.contracts.ids import build_keyframe_id
from mfe.core.cli import setup_console
from mfe.offline.caption_prompt import PROMPT_VARIANTS
from mfe.online.keyframe_images import DEFAULT_MAPS, KeyframeImages
from mfe.online.trammay import TrammayClient, TrammayError, available, translate_backend

logger = logging.getLogger(__name__)

DICH_CACHE = Path(".cache/tune/dich-truy-van-trammay.json")


def _nap_env(path: str = ".env") -> None:
    """Nạp `.env` — thiếu bước này là `available()` báo thiếu khoá dù `.env` có (§5)."""
    import os  # noqa: PLC0415

    from mfe.offline.transport import load_dotenv  # noqa: PLC0415

    for key, value in load_dotenv(path).items():
        os.environ.setdefault(key, value)


def nap_de(duong: Path, bo_dang: set[str]) -> list[dict]:
    ra = []
    for dong in duong.read_text(encoding="utf-8").splitlines():
        if dong.strip():
            q = json.loads(dong)
            if q.get("type") not in bo_dang and q.get("gt_video_id"):
                ra.append(q)
    return ra


def chon_video(de: list[dict], n: int) -> list[str]:
    """N video phủ nhiều câu nhất. Phủ nhiều câu = mỗi ảnh sinh ra đắt xắt ra miếng."""
    return [v for v, _ in Counter(q["gt_video_id"] for q in de).most_common(n)]


def khung_cua_video(video_id: str, maps: Path = DEFAULT_MAPS) -> list[tuple[str, int]]:
    """`[(keyframe_id, n)]` của MỌI keyframe trong video — cả lát, không chỉ khung GT.

    Lấy đủ vì đây là pool xếp hạng: khung anh em chính là distractor khó nhất.
    """
    p = maps / f"{video_id}.csv"
    if not p.is_file():
        return []
    with p.open(encoding="utf-8") as fh:
        return [
            (build_keyframe_id(video_id, int(r["frame_idx"])), int(r["n"]))
            for r in csv.DictReader(fh)
        ]


def pha1_tai_anh(kfs: list[str], luong: int) -> dict[str, Path]:
    """Ảnh đã có thì dùng luôn; thiếu thì kéo từ Modal. Trả `{keyframe_id: path}`."""
    anh = KeyframeImages()
    co: dict[str, Path] = {}
    thieu: list[str] = []
    for k in kfs:
        p = anh.local_path(k)
        if p is not None and p.is_file() and p.stat().st_size > 0:
            co[k] = p
        else:
            thieu.append(k)
    logger.info("pha 1: %d/%d anh da co, can tai %d", len(co), len(kfs), len(thieu))
    if not thieu:
        return co

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=luong) as ex:
        for k, p in zip(thieu, ex.map(anh.get, thieu), strict=True):
            if p is not None:
                co[k] = p
    logger.info(
        "pha 1 xong: %d anh (%.0fs), %d khong lay duoc",
        len(co), time.time() - t0, len(kfs) - len(co),
    )
    return co


def pha2_sinh_caption(
    anh: dict[str, Path], prompt: str, max_tok: int, ra: Path, luong: int, client
) -> dict[str, str]:
    """Sinh caption, GHI TĂNG DẦN, bỏ qua khung đã có trong `ra`."""
    cu: dict[str, str] = json.loads(ra.read_text(encoding="utf-8")) if ra.is_file() else {}
    con = [k for k in anh if k not in cu]
    logger.info("pha 2: %d da co, con %d khung", len(cu), len(con))
    if not con:
        return cu

    def mot(k: str) -> tuple[str, str | None]:
        try:
            return k, client.chat(
                prompt, images=[anh[k]], max_tokens=max_tok, temperature=0.0
            )
        except TrammayError as e:
            logger.warning("  %s: %s", k, e)
            return k, None

    t0, xong, hong = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=luong) as ex:
        for k, c in ex.map(mot, con):
            if c is None:
                hong += 1
                continue
            cu[k] = c.replace("[FINAL_CAPTION]", "").strip()
            xong += 1
            # Ghi mỗi 50, không phải 100: lượt 21-09 chết sau ~100 ảnh và mọi
            # kết quả chưa tới mốc lưu đều mất trắng. Ghi một file 5.000 khoá
            # rẻ hơn nhiều so với gọi lại API cho 50 ảnh.
            if xong % 50 == 0:
                ra.write_text(json.dumps(cu, ensure_ascii=False), encoding="utf-8")
                tb = (time.time() - t0) / xong
                logger.info(
                    "  %d/%d · %.1fs/anh · con ~%.0f phut",
                    xong, len(con), tb, (len(con) - xong) * tb / 60,
                )
    ra.write_text(json.dumps(cu, ensure_ascii=False), encoding="utf-8")
    logger.info("pha 2 xong: %d caption (%d hong) trong %.0f phut",
                len(cu), hong, (time.time() - t0) / 60)
    return cu


def pha3_do(caps: dict[str, str], de: list[dict], top_k: int) -> None:
    """BM25 trên chính lát cắt, ba chuỗi truy vấn, cùng bộ đề."""
    import numpy as np  # noqa: PLC0415
    from rank_bm25 import BM25Okapi  # noqa: PLC0415

    from mfe.stores.inverted.index import _tokenize  # noqa: PLC0415

    keys = sorted(caps)
    logger.info("pha 3: dung BM25 tren %d caption", len(keys))
    bm25 = BM25Okapi([_tokenize(caps[k]) for k in keys])

    dich: dict[str, str] = (
        json.loads(DICH_CACHE.read_text(encoding="utf-8")) if DICH_CACHE.is_file() else {}
    )
    backend = translate_backend()
    for q in de:
        if q["text"] not in dich:
            dich[q["text"]] = backend.translate(q["text"])
    DICH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DICH_CACHE.write_text(json.dumps(dich, ensure_ascii=False), encoding="utf-8")

    cach = {
        "vi": lambda q: q["text"],
        "en": lambda q: dich[q["text"]],
        "vi_en": lambda q: f"{q['text']} {dich[q['text']]}",
    }
    hang: dict[str, list[int | None]] = {k: [] for k in cach}
    for q in de:
        vid = q["gt_video_id"]
        lo, hi = q.get("gt_frame_range") or [0, 10**9]
        for ten, lam in cach.items():
            toks = _tokenize(lam(q))
            diem = bm25.get_scores(toks) if toks else np.zeros(len(keys))
            thu_tu = np.argsort(diem)[::-1][:top_k]
            h = None
            for i, idx in enumerate(thu_tu, start=1):
                if diem[idx] <= 0:
                    break
                v, _, f = keys[idx].partition("#")
                if v == vid and lo <= int(f) <= hi:
                    h = i
                    break
            hang[ten].append(h)

    n = len(de)
    print(f"\n{'=' * 74}")
    print(f"LAT CAT — {len(keys):,} caption · {n} cau · BM25 trong chinh lat")
    print("=" * 74)
    print(f"  {'cach':<8}{'trung':>9}{'R@1':>8}{'R@5':>8}{'R@20':>8}{'hang TB':>10}")
    print("  " + "-" * 52)
    for ten in cach:
        t = [x for x in hang[ten] if x]
        r = {k: sum(1 for x in t if x <= k) / n for k in (1, 5, 20)}
        tb = statistics.mean(t) if t else float("nan")
        print(f"  {ten:<8}{len(t):>5}/{n:<3}{r[1]:>8.1%}{r[5]:>8.1%}{r[20]:>8.1%}{tb:>10.1f}")

    print(f"\n{'=' * 74}\nSO CAP\n{'=' * 74}")
    for a, b in (("vi", "en"), ("vi", "vi_en"), ("en", "vi_en")):
        len_ = xuong = hoa = 0
        for x, y in zip(hang[a], hang[b], strict=True):
            if x is None and y is None:
                continue
            if x is None or (y is not None and y < x):
                len_ += 1
            elif y is None or y > x:
                xuong += 1
            else:
                hoa += 1
        print(f"  {a:>6} -> {b:<7} len {len_:>3} · xuong {xuong:>3} · hoa {hoa:>3}")


def main() -> None:
    setup_console()
    _nap_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=int, default=15)
    p.add_argument("--prompt", default="v5_bao", choices=sorted(PROMPT_VARIANTS))
    p.add_argument("--queryset", default="eval/queryset_dev.jsonl")
    p.add_argument("--skip-types", default="TRAKE")
    p.add_argument("--out", default="scratchpad/lat_cat")
    # Hai nút thắt khác nhau, nên hai con số khác nhau:
    #   pha 1 nghẽn ở Modal + đĩa — nhiều luồng không nhanh hơn, chỉ dễ nghẽn hơn
    #   pha 2 nghẽn ở API — đo 20-09: 4 luồng 0,51 anh/s · 8 luồng 1,15 · 16 luồng
    #   1,84, và độ trễ mỗi ảnh GIỮ NGUYÊN ~5-6s.
    #
    # ⚠ 16 là TRẦN ĐÃ ĐO, đừng nâng mà không đo lại. Thử 30 luồng ngày 21-09:
    # cổng trả `EOF occurred in violation of protocol` + `RemoteDisconnected`
    # hàng loạt sau ~100 ảnh — đứt ở tầng TLS, tức bị chặn ở mức KẾT NỐI chứ
    # không phải HTTP. Cùng cổng, cùng khoá, 16 luồng thì 800/800 lượt sạch.
    # (Notebook của Bảo chạy 28 luồng, nhưng đó là Kaggle — mạng khác, IP khác.)
    p.add_argument("--luong-tai", type=int, default=8, help="luồng tải ảnh (pha 1)")
    p.add_argument("--luong-api", type=int, default=16, help="luồng gọi API (pha 2)")
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=0, help="0 = mac dinh cua bien the")
    p.add_argument("--chi-pha", type=int, default=0, help="1/2/3; 0 = chay ca ba")
    args = p.parse_args()

    ok, vi_sao = available()
    if not ok:
        logger.error("%s", vi_sao)
        sys.exit(1)

    de_all = nap_de(Path(args.queryset), {t for t in args.skip_types.split(",") if t})
    vids = chon_video(de_all, args.videos)
    de = [q for q in de_all if q["gt_video_id"] in vids]
    kfs = [k for v in vids for k, _ in khung_cua_video(v)]
    logger.info("%d video · %d keyframe · %d cau do duoc", len(vids), len(kfs), len(de))

    thu_muc = Path(args.out) / f"{args.videos}video_{args.prompt}"
    thu_muc.mkdir(parents=True, exist_ok=True)
    file_cap = thu_muc / "caption.json"
    (thu_muc / "video.json").write_text(json.dumps(vids, indent=2), encoding="utf-8")

    prompt, mac_dinh = PROMPT_VARIANTS[args.prompt]
    max_tok = args.max_tokens or mac_dinh

    anh: dict[str, Path] = {}
    if args.chi_pha in (0, 1, 2):
        anh = pha1_tai_anh(kfs, args.luong_tai)
    if args.chi_pha == 1:
        return

    caps: dict[str, str] = {}
    if args.chi_pha in (0, 2):
        caps = pha2_sinh_caption(
            anh, prompt, max_tok, file_cap, args.luong_api, TrammayClient.from_env()
        )
    if args.chi_pha == 2:
        return
    if not caps:
        caps = json.loads(file_cap.read_text(encoding="utf-8"))

    pha3_do(caps, de, args.top_k)


if __name__ == "__main__":
    main()
