"""verify_clip_slices.py — Kiểm lát cắt CLIP CHỈ bằng dữ liệu năm nay.

    python scripts/verify_clip_slices.py

Vấn đề cần kiểm
---------------
`prepare_sample_2026.py` cắt vector cho từng video ra khỏi một mảng CLIP gộp
của cả batch, dựa trên GIẢ ĐỊNH: thứ tự toàn cục = thứ tự tên video sắp xếp,
và offset = tổng số keyframe của các video đứng trước. PDF không nói rõ quy tắc
sắp xếp, nên đây là suy đoán.

Nếu giả định sai, hậu quả **hoàn toàn im lặng**: mỗi video vẫn có đủ số vector,
shape vẫn đúng, không exception nào — chỉ là vector thuộc về video khác. Mọi
truy vấn vẫn trả kết quả, xếp hạng vẫn trông hợp lý, và không gì trong pipeline
báo động.

Vì sao không đối chiếu với bộ năm ngoái
---------------------------------------
Cách hiển nhiên là so với `data/sample/clip-features/` (bộ 2025 đã xác minh).
Nhưng bên đối chiếu khi đó là **dữ liệu năm khác**, nên cosine thấp không phân
biệt được hai nguyên nhân: offset sai, hay dữ liệu hai năm vốn khác nhau. Một
phép kiểm cho kết quả mơ hồ thì không dùng để ra quyết định được.

Script này khép kín trong bộ năm nay: lấy chính ảnh keyframe vừa tải từ link
BTC, encode lại bằng `clip-ViT-B-32` (checkpoint PDF §3 nêu đích danh), rồi so
với vector đã cắt. Cả hai vế cùng một nguồn, nên kết luận là dứt khoát.

Đọc kết quả
-----------
- **KHỚP**    : cosine trên đường chéo ~1.0 → lát cắt đúng video.
- **ĐỐI CHỨNG**: cosine với video KHÁC phải thấp hẳn. Nếu nó cũng cao thì phép
  kiểm này vô nghĩa (mọi vector đều giống nhau) — đừng kết luận gì.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

from mfe.core.cli import setup_console
from mfe.online.encoder import DEFAULT_CLIP_CHECKPOINT, encode_image

OUT = Path("data/sample_2026")

#: Ngưỡng chấp nhận trên đường chéo. Không đặt 0.99: vector BTC lưu float16 và
#: cách họ resize/crop ảnh không được công bố, nên khớp thật vẫn lệch chút.
#: Vẫn cách rất xa mức ~0.3-0.6 của một video khác.
MATCH_MIN = 0.90

#: Số ảnh lấy mẫu mỗi video. Ba là đủ: đầu / giữa / cuối bắt được cả lệch offset
#: cố định lẫn lệch tích luỹ.
SAMPLES_PER_VIDEO = 3


def keyframe_count(video_id: str) -> int | None:
    path = OUT / "map-keyframes" / f"{video_id}.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def main() -> int:
    setup_console(level=logging.WARNING)
    sliced_dir = OUT / "clip-features"
    files = sorted(sliced_dir.glob("*.npy"))
    if not files:
        print(f"Không thấy .npy nào trong {sliced_dir} — chạy prepare_sample_2026.py trước.")
        return 2

    print(f"Checkpoint ảnh: {DEFAULT_CLIP_CHECKPOINT}")
    print(f"Ngưỡng khớp   : {MATCH_MIN}\n")

    # --- 1) Kiểm cấu trúc trước: số vector phải bằng số dòng CSV -------------
    print("[1] Số vector so với số dòng map-keyframes")
    structural_ok = True
    vectors: dict[str, np.ndarray] = {}
    for path in files:
        video_id = path.stem
        arr = np.load(path)
        vectors[video_id] = arr
        expected = keyframe_count(video_id)
        mark = "OK"
        if expected is None:
            mark = "KHONG CO CSV"
        elif arr.shape[0] != expected:
            mark = f"LECH (CSV={expected})"
            structural_ok = False
        print(f"    {video_id:<12} {arr.shape}  {mark}")
    if not structural_ok:
        print("\n=> LECH SO LUONG. Offset chac chan sai, khong can chay tiep buoc 2.")
        return 1

    # --- 2) Kiểm nội dung: encode lại ảnh thật rồi so ------------------------
    print(f"\n[2] Encode lai {SAMPLES_PER_VIDEO} anh/video roi so voi vector da cat")
    diagonal: list[float] = []
    probes: list[tuple[str, int, np.ndarray]] = []

    for video_id, arr in vectors.items():
        n_rows = arr.shape[0]
        picks = sorted({1, (n_rows + 1) // 2, n_rows})
        picks = [n for n in picks if 1 <= n <= n_rows][:SAMPLES_PER_VIDEO]
        sims = []
        for n in picks:
            image_path = OUT / "keyframes" / video_id / f"{n:03d}.jpg"
            if not image_path.is_file():
                print(f"    {video_id:<12} n={n:<4} THIEU ANH {image_path.name}")
                continue
            ours = encode_image(str(image_path))
            sim = cosine(ours, arr[n - 1])
            sims.append(sim)
            diagonal.append(sim)
            probes.append((video_id, n, ours))
        if sims:
            flag = "OK" if min(sims) >= MATCH_MIN else "LECH"
            joined = " ".join(f"{s:.4f}" for s in sims)
            print(f"    {video_id:<12} {joined}   {flag}")

    if not diagonal:
        print("\n=> Khong encode duoc anh nao. Kiem tra thu muc keyframes/.")
        return 2

    # --- 3) Đối chứng: so ảnh của video này với vector của video KHÁC --------
    print("\n[3] Doi chung — anh video A so voi vector video B (phai THAP)")
    off_diagonal: list[float] = []
    video_ids = list(vectors)
    for video_id, n, ours in probes[: min(8, len(probes))]:
        other = next((v for v in video_ids if v != video_id), None)
        if other is None:
            continue
        other_arr = vectors[other]
        idx = min(n - 1, other_arr.shape[0] - 1)
        sim = cosine(ours, other_arr[idx])
        off_diagonal.append(sim)
        print(f"    {video_id} n={n}  vs  {other}  ->  {sim:.4f}")

    # --- Kết luận ------------------------------------------------------------
    lo, mean = min(diagonal), float(np.mean(diagonal))
    print("\n" + "-" * 56)
    print(f"  Duong cheo (dung video) : min={lo:.4f}  mean={mean:.4f}")
    if off_diagonal:
        hi_off = max(off_diagonal)
        print(f"  Doi chung  (khac video) : max={hi_off:.4f}")
    else:
        hi_off = None

    if lo < MATCH_MIN:
        print("\n=> LECH. Offset toan cuc dang SAI — vector thuoc ve video khac.")
        print("   KHONG upload bo nay len HF; phai tim manifest/offset that.")
        return 1
    if hi_off is not None and hi_off > lo:
        print("\n=> KHONG KET LUAN DUOC: doi chung cao hon duong cheo.")
        print("   Phep kiem khong phan biet duoc video — dung tin ket qua.")
        return 1

    print("\n=> KHOP. Lat cat CLIP thuoc dung video, chi dung du lieu nam nay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
