r"""Gói / giải nén phần DỮ LIỆU mà git không mang theo — cho thành viên chạy giao diện.

    # Lead, một lần:
    python scripts/share_data.py pack --out mfe-data.zip

    # Thành viên, sau khi `git pull` và nhận zip:
    python scripts/share_data.py unpack mfe-data.zip

Vì sao có script này thay vì một dòng hướng dẫn
------------------------------------------------
`.gitignore` chặn `data/**` và `stores/**`, nên ba thư mục dưới đây phải chuyển
tay. Bảo người ta "giải nén vào gốc repo" thì **Windows Explorer làm sai theo mặc
định**: nút *Extract All* tự điền đích là `<thư mục hiện tại>\<tên zip>\`, nên
mọi thứ lệch đúng một tầng thư mục.

Và nó hỏng theo kiểu tệ nhất — im lặng. Giao diện vẫn mở, chỉ là truy hồi ra **0
kết quả** và mọi dòng ghi `không tra được frame_id`, trông y hệt lỗi cấu hình.
`unpack` giải nén đúng chỗ rồi **đếm lại**, nên sai là biết ngay lúc cài chứ
không phải lúc đang thi.

Vì sao pack và unpack nằm CÙNG một file
----------------------------------------
`REQUIRED` là bản kê duy nhất. Tách đôi thì một hôm thêm một nguồn dữ liệu vào
gói mà quên thêm vào phép kiểm — và phép kiểm sẽ báo "đủ" cho một gói thiếu.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: File kho phẳng. Tên cố định vì `mfe.online.flat_arms` tra theo tên.
#:
#: Danh sách này là **phép đếm**, không phải phép liệt kê cho vui: `pack` gom cả
#: thư mục bằng glob, nên một file mới vẫn vào zip dù không có tên ở đây — còn
#: `verify` thì chỉ so `min_files=len(FLAT_FILES)`. Quên cập nhật ở đây nghĩa là
#: một gói THIẾU bốn arm vẫn được báo "đủ", và người nhận chỉ biết khi mở giao
#: diện ra thấy `caption_qwen — không chạy được`.
FLAT_FILES = (
    "siglip_vectors.npz",
    "audio_features.npz",
    "objects_text.json",
    "ocr_text.json",
    "paddle_ocr.json",
    "subtitles.json",
    "viet_ocr.json",
    # Lượt 5 + 6 + 7 (28-08-2026). Bốn file này bật thêm bốn arm:
    "qwen_captions.json",    # caption_qwen  (BM25 trên caption Qwen2.5-VL)
    "qwen_ocr.json",         # ocr_qwen25    (BM25 trên OCR Qwen2.5-VL)
    "dense_vectors.npz",     # bge_m3        (nhúng CAPTION, 1024 chiều)
    "subtitles_dense.npz",   # subtitle_bge_m3 (nhúng PHỤ ĐỀ, 1024 chiều)
)


@dataclass(frozen=True)
class Need:
    """Một phần dữ liệu phải chuyển tay, kèm cách biết nó đủ."""

    path: str
    why: str
    #: Số file tối thiểu bên trong (0 = chính nó là một file).
    min_files: int = 0
    pattern: str = "*"


REQUIRED = (
    Need(
        "stores/flat/data/vectors",
        "kho phẳng — thiếu thì KHÔNG arm nào chạy, truy hồi ra 0 kết quả",
        min_files=len(FLAT_FILES),
    ),
    Need(
        "stores/btc_b1/relational/app.db",
        "bản đồ frame_idx — thiếu thì mọi dòng 'không tra được frame_id', "
        "KHÔNG xuất được file nộp nào",
    ),
    Need(
        "data/btc_b1/map-keyframes",
        "nối frame_idx -> số thứ tự keyframe (tên file ảnh) — thiếu thì LƯỚI "
        "KHÔNG HIỆN ẢNH NÀO dù token Modal đúng",
        min_files=800, pattern="*.csv",
    ),
)


def _files_under(base: Path, need: Need) -> list[Path]:
    if base.is_file():
        return [base]
    return sorted(p for p in base.rglob(need.pattern) if p.is_file())


def pack(out: Path) -> int:
    """Gói ba phần trên, đường dẫn TƯƠNG ĐỐI so với gốc repo."""
    missing = [n for n in REQUIRED if not (ROOT / n.path).exists()]
    if missing:
        for need in missing:
            print(f"  THIEU {need.path} — {need.why}")
        sys.exit("khong goi duoc: may nay cung thieu du lieu")

    total = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    # ZIP64 bật sẵn: riêng `siglip_vectors.npz` đã 757 MB.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for need in REQUIRED:
            base = ROOT / need.path
            files = _files_under(base, need)
            print(f"  {need.path}: {len(files)} file", flush=True)
            for path in files:
                archive.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
                total += path.stat().st_size
    print(
        f"\n{out}  ·  {out.stat().st_size / 2**20:.0f} MB nen "
        f"(goc {total / 2**20:.0f} MB)"
    )
    print("Gui kem cau lenh: python scripts/share_data.py unpack <ten file>.zip")
    return 0


def unpack(archive_path: Path) -> int:
    """Giải nén vào **gốc repo**, rồi đếm lại.

    Không hỏi người dùng đường dẫn đích: đó chính là chỗ Explorer làm sai.
    """
    if not archive_path.is_file():
        sys.exit(f"khong thay {archive_path}")
    print(f"giai nen {archive_path.name} vao {ROOT}")
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        odd = [n for n in names if n.startswith(("/", "..")) or ":" in n]
        if odd:
            # Đường dẫn tuyệt đối hay `..` trong zip là lối ghi đè ra ngoài repo.
            sys.exit(f"zip chua duong dan bat thuong, dung lai: {odd[:3]}")
        zf.extractall(ROOT)
    print(f"  {len(names)} file\n")
    return verify()


def verify() -> int:
    """Đếm dữ liệu, không đọc log. Trả 0 khi đủ."""
    print("kiem tra:")
    bad = 0
    for need in REQUIRED:
        base = ROOT / need.path
        if not base.exists():
            print(f"  ✗ THIEU  {need.path}\n           {need.why}")
            bad += 1
            continue
        files = _files_under(base, need)
        if base.is_file():
            print(f"  ✓ {need.path}  ({base.stat().st_size / 2**20:.0f} MB)")
        elif len(files) < need.min_files:
            print(
                f"  ✗ {need.path}: {len(files)} file, can it nhat {need.min_files}"
                f"\n           {need.why}"
            )
            bad += 1
        else:
            print(f"  ✓ {need.path}  ({len(files)} file)")

    flat = ROOT / "stores/flat/data/vectors"
    if flat.is_dir():
        for name in FLAT_FILES:
            if not (flat / name).is_file():
                print(f"  ✗ thieu {name} trong kho phang")
                bad += 1

    bad += _check_db()
    if bad:
        print(f"\n{bad} van de — xem tung dong o tren.")
        return 1
    print("\nDu. Chay:  streamlit run app/submit_lab.py")
    print("Nho chep .env.example thanh .env va dien MODAL_TOKEN_ID/SECRET.")
    return 0


def _check_db() -> int:
    """Mở thật `app.db` và đếm — một file đúng kích thước vẫn có thể rỗng ruột.

    176.707 là số sau khi gộp **614 cặp keyframe trùng `frame_idx`** (CLAUDE.md
    §2). Thấy 177.321 nghĩa là cầm bản CHƯA gộp, và hai arm sẽ đánh địa chỉ khác
    nhau mà không có gì báo.
    """
    db = ROOT / "stores/btc_b1/relational/app.db"
    if not db.is_file():
        return 0
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            videos = con.execute("select count(*) from videos").fetchone()[0]
            keyframes = con.execute("select count(*) from keyframes").fetchone()[0]
    except Exception as err:  # noqa: BLE001
        print(f"  ✗ app.db khong doc duoc: {err}")
        return 1
    print(f"    app.db: {videos} video · {keyframes} keyframe")
    if videos != 873 or keyframes != 176707:
        print(
            "  ! cho doi 873 video / 176707 keyframe. Lech nghia la ban do nay "
            "khong phai ban da gop 614 cap trung frame_idx (CLAUDE.md §2)."
        )
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="Lead goi du lieu thanh zip")
    p.add_argument("--out", default="mfe-data.zip")
    u = sub.add_parser("unpack", help="Thanh vien giai nen vao dung cho + kiem")
    u.add_argument("archive")
    sub.add_parser("verify", help="Chi kiem, khong giai nen")

    args = parser.parse_args()
    if args.cmd == "pack":
        sys.exit(pack(Path(args.out)))
    if args.cmd == "unpack":
        sys.exit(unpack(Path(args.archive)))
    sys.exit(verify())


if __name__ == "__main__":
    main()
