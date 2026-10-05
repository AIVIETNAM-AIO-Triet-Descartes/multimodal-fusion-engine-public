"""Sinh thumbnail cho toàn bộ keyframe — phần cắt lưới 100 ảnh từ 38 s xuống.

    .venv/Scripts/python.exe scripts/build_thumbnails.py \
        --src /data/keyframes --dst /data/thumbs --workers 16

Vì sao có file này
-------------------
Đo 25-09-2026: một lưới 100 ảnh **nguội** qua Modal mất **38–41 giây**. Nhưng
100 ảnh chỉ nặng **~19 MB** (trung vị 179 KB/ảnh, mẫu 4.000 ảnh) — 19 MB trong
40 giây là ~0,5 MB/s, **không thể** là nghẽn băng thông. Nó nghẽn ở **100 lượt
khứ hồi**.

⚠ Vì vậy **thumbnail KHÔNG giúp gì trên Modal hay Object Storage** — ảnh nhỏ
hơn không làm số lượt khứ hồi ít đi. Nó chỉ có tác dụng khi ảnh đã nằm **trên
đĩa máy chủ**, lúc đó chỗ thắt chuyển sang băng thông và 19 MB → ~2 MB ăn
thẳng vào đúng chỗ. Cùng một biện pháp, hai kết luận ngược nhau tuỳ nền; đừng
chép kết luận từ nền này sang nền kia.

Cách dùng đúng: chạy MỘT LẦN trên VM sau khi chép ảnh lên đĩa, rồi cho nginx
phục vụ cây `--dst` cho lưới kết quả và cây `--src` cho ảnh phóng to.

Ba lựa chọn, đều có lý do
--------------------------
**Cây thư mục SONG SONG, không ghi đè.** Ảnh gốc vẫn cần cho lúc người vận hành
phóng to một khung để quyết định nộp hay không. Thu nhỏ tại chỗ là tiết kiệm
đĩa bằng cách vứt đi thứ đắt nhất.

**Bỏ qua file đã có.** 176.707 + 265.654 khung là một lượt chạy dài; đứt mạng
giữa chừng phải chạy tiếp được chứ không phải làm lại. `--force` cho lúc đổi
`--max-side`.

**Tiến trình, không luồng.** Pillow nhả GIL ở phần giải mã nhưng không nhả ở
phần dựng đối tượng, và đây là việc CPU thuần — trên VM 16 OCPU thì tiến trình
mới dùng hết được nhân.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "src")

from mfe.core.cli import setup_console  # noqa: E402

#: Cạnh dài của thumbnail. 320 px vừa một ô lưới trên màn hình 1080p ở mật độ
#: 5 cột, và đưa 179 KB xuống ~20 KB — tức 100 ảnh từ 19 MB còn ~2 MB.
MAX_SIDE = 320

#: 80 là chỗ Pillow còn giữ được chữ trên overlay đọc được ở cỡ thumbnail. Hạ
#: xuống 60 tiết kiệm thêm ~25% nhưng làm nhoè đúng thứ nhóm S cần nhìn (số
#: chặng, đồng hồ, tốc độ).
JPEG_QUALITY = 80

DUOI_ANH = {".jpg", ".jpeg", ".png", ".webp"}


def _mot_anh(viec: tuple[str, str, int, int, bool]) -> tuple[str, str | None]:
    """Thu nhỏ một ảnh. Trả `(đường dẫn, lý do bỏ qua hoặc None)`.

    Không ném: một ảnh hỏng trong 440.000 ảnh không được làm đổ cả lượt chạy
    sau nhiều giờ. Nhưng cũng không im lặng — lý do đi ngược về người gọi để
    đếm, vì "đừng tin log, hãy đếm dữ liệu".
    """
    src, dst, max_side, quality, force = viec
    src_p, dst_p = Path(src), Path(dst)
    if dst_p.exists() and not force:
        return src, "da co"
    try:
        from PIL import Image

        with Image.open(src_p) as anh:
            anh = anh.convert("RGB")
            anh.thumbnail((max_side, max_side))
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            tam = dst_p.with_suffix(dst_p.suffix + ".tmp")
            anh.save(tam, format="JPEG", quality=quality, optimize=True)
        # Đổi tên nguyên tử: một lượt chạy bị ngắt không được để lại file JPEG
        # cụt mà lần sau `dst.exists()` lại coi là đã xong.
        os.replace(tam, dst_p)
    except Exception as err:  # noqa: BLE001
        return src, f"{type(err).__name__}: {err}"
    return src, None


def main() -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, type=Path, help="gốc cây ảnh đầy đủ")
    p.add_argument("--dst", required=True, type=Path, help="gốc cây thumbnail")
    p.add_argument("--max-side", type=int, default=MAX_SIDE)
    p.add_argument("--quality", type=int, default=JPEG_QUALITY)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--force", action="store_true", help="ghi đè cả file đã có")
    p.add_argument("--limit", type=int, default=0, help="chỉ làm N ảnh đầu (để thử)")
    args = p.parse_args()

    if not args.src.is_dir():
        print(f"không có thư mục {args.src}")
        return 1

    print(f"quét {args.src} ...")
    nguon = [q for q in args.src.rglob("*") if q.suffix.lower() in DUOI_ANH]
    if args.limit:
        nguon = nguon[: args.limit]
    if not nguon:
        print("không tìm thấy ảnh nào")
        return 1
    print(f"{len(nguon):,} ảnh · {args.workers} tiến trình · cạnh dài {args.max_side}px")

    viec = [
        (
            str(q),
            # Giữ NGUYÊN cây con, nhưng đuôi luôn `.jpg`: một `.png` thu nhỏ vẫn
            # lưu JPEG, và nếu giữ đuôi cũ thì nginx trả sai `Content-Type`.
            str((args.dst / q.relative_to(args.src)).with_suffix(".jpg")),
            args.max_side, args.quality, args.force,
        )
        for q in nguon
    ]

    xong = bo_qua = 0
    loi: list[tuple[str, str]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tuong_lai = [pool.submit(_mot_anh, v) for v in viec]
        for i, f in enumerate(as_completed(tuong_lai), start=1):
            duong, ly_do = f.result()
            if ly_do is None:
                xong += 1
            elif ly_do == "da co":
                bo_qua += 1
            else:
                loi.append((duong, ly_do))
            if i % 5000 == 0 or i == len(viec):
                troi = time.perf_counter() - t0
                print(f"  {i:,}/{len(viec):,} · {i / troi:,.0f} ảnh/s · "
                      f"còn ~{(len(viec) - i) / max(i / troi, 1) / 60:.0f} phút")

    troi = time.perf_counter() - t0
    print(f"\nxong {xong:,} · bỏ qua (đã có) {bo_qua:,} · lỗi {len(loi):,} "
          f"· {troi / 60:.1f} phút")
    for duong, ly_do in loi[:20]:
        print(f"  ✗ {duong}: {ly_do}")
    if len(loi) > 20:
        print(f"  ... và {len(loi) - 20:,} lỗi nữa")

    # ĐẾM trên đĩa, không tin biến đếm trong vòng lặp — đúng §5 CLAUDE.md.
    #
    # Cộng KÍCH THƯỚC THẬT của từng file chứ không ngoại suy từ vài trăm ảnh
    # đầu: `rglob` trả theo thứ tự thư mục, nên "N ảnh đầu" là một video chứ
    # không phải một mẫu, và bản đầu của hàm này báo 38 KB/ảnh trong khi trung
    # vị thật của kho là 179 KB. `stat` trên 440.000 file mất vài giây.
    thumb = list(args.dst.rglob("*.jpg"))
    goc_b = sum(Path(v[0]).stat().st_size for v in viec)
    thu_b = sum(q.stat().st_size for q in thumb)
    print(f"\nđếm lại trên đĩa: {len(thumb):,} file trong {args.dst}")
    print(f"dung lượng: gốc {goc_b / 1e9:.2f} GB → thumb {thu_b / 1e9:.2f} GB "
          f"({goc_b / max(thu_b, 1):.1f}× nhỏ hơn)")
    if thumb:
        print(f"lưới 100 ảnh: {goc_b / len(viec) * 100 / 1e6:.1f} MB → "
              f"{thu_b / len(thumb) * 100 / 1e6:.1f} MB")
    return 1 if loi else 0


if __name__ == "__main__":
    sys.exit(main())
