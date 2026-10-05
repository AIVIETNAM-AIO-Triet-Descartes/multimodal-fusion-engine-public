"""Gộp batch 1 + batch 2 thành MỘT kho phẳng cho truy vấn.

    .venv/Scripts/python.exe scripts/gop_kho_hai_batch.py --out stores/flat_all

Sáu file ra, đúng sáu arm đang bật (`config/btc_b1.yaml: arms_disabled`):

    qwen_captions.json    caption_qwen      BM25 trên caption
    qwen_ocr.json         ocr_qwen25        BM25 trên chữ trong ảnh
    subtitles.json        subtitle_exact + subtitle_partial
    siglip_vectors.npz    siglip            (trọng số 2,0 — arm mạnh nhất)
    dense_vectors.npz     bge_m3
    subtitles_dense.npz   subtitle_bge_m3

**KHÔNG gộp `window_*`** (bản ghi cửa sổ / shot-level), cả JSON lẫn dense —
chưa arm nào đọc chúng. Đếm ở runtime: không tên nào chứa `window`/`shot` trong
`TEXT_ARM_SPECS` lẫn `DENSE_ARM_SPECS`, và `bundle.py` khai
`Artifact("window_dense", ..., None, ...)` với arm = None. Gộp vào bây giờ là
đổ dữ liệu vào một file không lối nào đọc tới.

⚠ CAPTION BATCH 1 LÀ BẢN GEMINI, KHÔNG PHẢI QWEN
-------------------------------------------------
`stores/flat` hiện có `qwen_captions.json` (Qwen2.5-VL) và `dense_vectors.npz`
nhúng từ chính nó. Script này **thay cả hai** bằng bản Gemini mới:

    text   /research/caption-gemini-trammay-v1/parts/captions/{video}.json   873
    dense  /bundle/aic_btc_b1_v1/parts/dense_vectors/{video}.npz             873

Hai nguồn này phải đi **cùng nhau**. `caption_qwen` (BM25) và `bge_m3` (dense)
là hai arm đọc **cùng một nội dung** qua hai cơ chế; trộn text Gemini với vector
Qwen thì hai arm mô tả hai thứ khác nhau dưới cùng một cái tên, hợp nhất cộng
điểm của hai bằng chứng không liên quan, và **không có gì báo**. Script kiểm
tập khoá của hai nguồn khớp nhau trước khi ghi.

⚠ Tên file ra vẫn là `qwen_captions.json` / `qwen_ocr.json` dù nội dung do
Gemini sinh. Đó là **chủ ý**: `SOURCE_FILE` trong `flat_arms.py` khoá theo tên
này, đổi tên là phải sửa hợp đồng ở hai nơi. Cái giá là một cái tên nói dối —
ghi rõ ở đây để người sau không đi truy "sao Qwen lại viết kiểu này".

Ba nguồn, ba nơi
-----------------
    B1 caption + dense   volume `keyframe-augmentation-research`
    B1 phần còn lại      `stores/flat` CỤC BỘ (đã gộp, đã dùng, khỏi gộp lại)
    B2 tất cả            volume `keyframes-batch-2-v2`

Bất biến giữ đúng
------------------
* **Gộp GIỮ BẢN ĐẦU** cho khoá trùng. B1 có 614 cặp keyframe trùng `frame_idx`
  (0,346%), B2 có 19 — nếu không gộp thì hai arm đánh địa chỉ khác nhau và hợp
  nhất cộng điểm của hai khung khác nhau.
* **Kiểm số chiều** từng part: siglip 1152, bge_m3 1024, subtitle_bge_m3 1024.
  Một part lệch chiều làm hỏng cả ma trận mà lỗi chỉ lộ ra ở tầng Milvus/numpy,
  cách chỗ sai vài lớp.
* **Chuẩn hoá lại L2**. Part đo được đã chuẩn hoá sẵn, nhưng một ma trận chưa
  chuẩn hoá vẫn cho ra thứ hạng trông hợp lý — chỉ sai.
* **Đếm lại trên file đã ghi**, không tin biến đếm trong vòng lặp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from mfe.core.cli import setup_console  # noqa: E402

VOL_B1 = "keyframe-augmentation-research"
VOL_B2 = "keyframes-batch-2-v2"

GOC_CAP_B1 = "/research/caption-gemini-trammay-v1"
GOC_BUNDLE_B1 = "/bundle/aic_btc_b1_v1"
GOC_B2 = "/bundle/aic_btc_b2_v1"

#: `tên file ra -> (kiểu, số chiều nếu là vector, nguồn B1, nguồn B2)`.
#:
#: Nguồn ghi theo ba dạng:
#:   ("local", <tên file trong stores/flat>)
#:   ("parts", <volume>, <gốc>, <tầng>)
#:   ("flat",  <volume>, <đường dẫn file phẳng>)
NGUON = {
    "qwen_captions.json": (
        "text", None,
        ("parts", VOL_B1, GOC_CAP_B1, "captions"),
        ("flat", VOL_B2, f"{GOC_B2}/data/vectors/gemini_captions.json"),
    ),
    "qwen_ocr.json": (
        "text", None,
        ("local", "qwen_ocr.json"),
        ("flat", VOL_B2, f"{GOC_B2}/data/vectors/gemini_ocr.json"),
    ),
    "subtitles.json": (
        "text", None,
        ("local", "subtitles.json"),
        ("parts", VOL_B2, GOC_B2, "subtitles"),
    ),
    "siglip_vectors.npz": (
        "vector", 1152,
        ("local", "siglip_vectors.npz"),
        ("parts", VOL_B2, GOC_B2, "siglip_vectors"),
    ),
    "dense_vectors.npz": (
        "vector", 1024,
        ("parts", VOL_B1, GOC_BUNDLE_B1, "dense_vectors"),
        ("parts", VOL_B2, GOC_B2, "dense_vectors"),
    ),
    "subtitles_dense.npz": (
        "vector", 1024,
        ("local", "subtitles_dense.npz"),
        ("parts", VOL_B2, GOC_B2, "subtitles_dense"),
    ),
}

_VOL: dict = {}


def vol(ten: str):
    from mfe.offline.transport import ModalVolumeTransport

    if ten not in _VOL:
        _VOL[ten] = ModalVolumeTransport(ten, create_if_missing=False)
    return _VOL[ten]


# ---------------------------------------------------------------------------
# Tải
# ---------------------------------------------------------------------------


def tai_parts(volume: str, goc: str, tang: str, cache: Path, workers: int) -> list[Path]:
    """Tải mọi part của một tầng về `cache`, bỏ qua file đã có.

    Có cache vì một lượt gộp đụng ~3.300 part và vài GB; đứt mạng giữa chừng
    phải chạy tiếp được chứ không phải tải lại từ đầu.
    """
    thu_muc = cache / volume / tang
    thu_muc.mkdir(parents=True, exist_ok=True)
    remote = sorted(p for p in vol(volume).listdir(f"{goc}/parts/{tang}")
                    if p.endswith((".json", ".npz")))
    can = [(r, thu_muc / r.rsplit("/", 1)[-1]) for r in remote]
    thieu = [(r, f) for r, f in can if not f.is_file()]
    if thieu:
        print(f"    tải {len(thieu)}/{len(can)} part ({tang}) ...", flush=True)
        t = time.perf_counter()

        def mot(rf):
            r, f = rf
            tam = f.with_suffix(f.suffix + ".tmp")
            vol(volume).get(r, tam)
            tam.replace(f)   # đổi tên nguyên tử: file dở không bị coi là xong

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(mot, thieu))
        print(f"    xong {time.perf_counter() - t:.0f}s", flush=True)
    return [f for _, f in can]


def tai_flat(volume: str, remote: str, cache: Path) -> Path:
    f = cache / volume / remote.rsplit("/", 1)[-1]
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.is_file():
        print(f"    tải {remote} ...", flush=True)
        tam = f.with_suffix(f.suffix + ".tmp")
        vol(volume).get(remote, tam)
        tam.replace(f)
    return f


def lay(nguon, b1_flat: Path, cache: Path, workers: int) -> list[Path]:
    """Danh sách file cục bộ của một nguồn, bất kể nó đến từ đâu."""
    kieu = nguon[0]
    if kieu == "local":
        f = b1_flat / "data" / "vectors" / nguon[1]
        if not f.is_file():
            raise SystemExit(f"không có {f} — kho batch 1 cục bộ thiếu file này")
        return [f]
    if kieu == "flat":
        return [tai_flat(nguon[1], nguon[2], cache)]
    return tai_parts(nguon[1], nguon[2], nguon[3], cache, workers)


# ---------------------------------------------------------------------------
# Gộp
# ---------------------------------------------------------------------------


def gop_text(files: list[Path]) -> tuple[dict[str, str], int]:
    """Gộp JSON `{keyframe_id: text}`. GIỮ BẢN ĐẦU khi trùng khoá."""
    ra: dict[str, str] = {}
    trung = 0
    for f in files:
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            if k in ra:
                trung += 1
                continue
            ra[k] = unicodedata.normalize("NFC", v) if isinstance(v, str) else v
    return ra, trung


def gop_vector(files: list[Path], dim: int) -> tuple[list[str], np.ndarray, int]:
    """Gộp npz `{ids, vectors}`. GIỮ BẢN ĐẦU, kiểm chiều, chuẩn hoá lại L2."""
    ids_all: list[str] = []
    khoi: list[np.ndarray] = []
    thay: set[str] = set()
    trung = 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            ids = [str(x) for x in z["ids"].tolist()]
            vec = np.asarray(z["vectors"], dtype=np.float32)
        if vec.ndim != 2 or vec.shape[1] != dim:
            raise SystemExit(f"{f.name}: chiều {vec.shape}, chờ đợi (n, {dim})")
        giu = []
        for i, k in enumerate(ids):
            if k in thay:
                trung += 1
                continue
            thay.add(k)
            giu.append(i)
            ids_all.append(k)
        if giu:
            khoi.append(vec[giu])
    if not khoi:
        return [], np.zeros((0, dim), dtype=np.float32), trung
    mat = np.concatenate(khoi, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids_all, mat / norms, trung


def nhom_video(khoa) -> Counter:
    return Counter(k.split("#")[0][0] for k in khoa)


# ---------------------------------------------------------------------------


def main() -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="stores/flat_all", help="gốc kho ra")
    p.add_argument("--b1-flat", default="stores/flat", help="kho phẳng batch 1 cục bộ")
    p.add_argument("--cache", default=".cache/gop", help="nơi giữ part đã tải")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--only", nargs="*", help="chỉ gộp các file này")
    args = p.parse_args()

    out = Path(args.out) / "data" / "vectors"
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    b1_flat = Path(args.b1_flat)

    muc = {k: v for k, v in NGUON.items() if not args.only or k in args.only}
    bao: list[str] = []
    khoa_caption: dict[str, set[str]] = {}

    for ten, (kieu, dim, ng_b1, ng_b2) in muc.items():
        print(f"\n{'=' * 70}\n{ten}")
        t0 = time.perf_counter()
        f1 = lay(ng_b1, b1_flat, cache, args.workers)
        f2 = lay(ng_b2, b1_flat, cache, args.workers)

        if kieu == "text":
            d1, t1 = gop_text(f1)
            d2, t2 = gop_text(f2)
            chung = set(d1) & set(d2)
            if chung:
                raise SystemExit(
                    f"{ten}: {len(chung)} khoá có ở CẢ hai batch, ví dụ "
                    f"{sorted(chung)[:3]} — id hai batch phải rời nhau"
                )
            gop = {**d1, **d2}
            (out / ten).write_text(
                json.dumps(gop, ensure_ascii=False), encoding="utf-8"
            )
            khoa = set(gop)
            print(f"  B1 {len(d1):>7,}  +  B2 {len(d2):>7,}  =  {len(gop):>7,} khoá"
                  f"   (trùng bỏ: {t1 + t2})")
        else:
            i1, v1, t1 = gop_vector(f1, dim)
            i2, v2, t2 = gop_vector(f2, dim)
            chung = set(i1) & set(i2)
            if chung:
                raise SystemExit(f"{ten}: {len(chung)} khoá có ở CẢ hai batch")
            ids = i1 + i2
            mat = np.concatenate([v1, v2], axis=0) if len(v2) else v1
            np.savez_compressed(
                out / ten, ids=np.asarray(ids, dtype=np.str_), vectors=mat
            )
            khoa = set(ids)
            print(f"  B1 {len(i1):>7,}  +  B2 {len(i2):>7,}  =  {mat.shape}"
                  f"   (trùng bỏ: {t1 + t2})")

        if ten in ("qwen_captions.json", "dense_vectors.npz"):
            khoa_caption[ten] = khoa
        nh = dict(sorted(nhom_video(khoa).items()))
        print(f"  theo nhóm video: {nh}")
        print(f"  {(out / ten).stat().st_size / 1e6:,.0f} MB · {time.perf_counter() - t0:.0f}s")
        bao.append(f"  {ten:<22} {len(khoa):>8,} khoá")

    # --- Phép kiểm QUAN TRỌNG NHẤT của script này ---------------------------
    #
    # `caption_qwen` (BM25 trên text) và `bge_m3` (dense) phải đọc CÙNG nội
    # dung. Nếu tập khoá lệch nhau thì một trong hai đang mô tả caption Qwen cũ
    # còn cái kia mô tả caption Gemini mới — hợp nhất sẽ cộng điểm của hai bằng
    # chứng không liên quan, và không có gì báo.
    if len(khoa_caption) == 2:
        a, b = khoa_caption["qwen_captions.json"], khoa_caption["dense_vectors.npz"]
        print(f"\n{'=' * 70}\nĐỐI CHIẾU text caption với vector caption")
        if a == b:
            print(f"  KHỚP — {len(a):,} khoá, cùng một nội dung qua hai arm")
        else:
            print(f"  ⚠ LỆCH: chỉ text {len(a - b):,} · chỉ vector {len(b - a):,}")
            print(f"    ví dụ chỉ text  : {sorted(a - b)[:3]}")
            print(f"    ví dụ chỉ vector: {sorted(b - a)[:3]}")
            print("    ⇒ `caption_qwen` và `bge_m3` đang đọc hai nội dung khác nhau.")

    print(f"\n{'=' * 70}\nĐẾM LẠI TRÊN FILE ĐÃ GHI ({out})")
    for f in sorted(out.iterdir()):
        if f.suffix == ".json":
            n = len(json.loads(f.read_text(encoding="utf-8")))
        else:
            with np.load(f, allow_pickle=False) as z:
                n = len(z["ids"])
        print(f"  {f.name:<22} {n:>8,} khoá   {f.stat().st_size / 1e6:>7,.0f} MB")
    print("\nDùng:  --flat-root", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
