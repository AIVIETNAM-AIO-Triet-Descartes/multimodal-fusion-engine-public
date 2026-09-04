"""Gộp lô phẳng B0 (BTC) với delta B1 (keyframe Quang + Trọng trích thêm).

Vì sao là một kho RIÊNG chứ không ghi đè `stores/flat`
------------------------------------------------------
Phép so B0/B1 chỉ có nghĩa khi **cùng một câu hỏi chạy trên hai kho khác nhau
đúng một điều**: B1 có thêm 5.301 hàng. Ghi đè kho B0 là mất mất phía đối
chứng, và lúc đó không ai dựng lại được con số nền.

Vì sao gộp bằng "B0 thắng khi trùng khoá"
------------------------------------------
Đo được lúc nhận merge: **0** id B1 trùng id B0 (182.008 = 176.707 + 5.301), nên
luật này hiện không kích hoạt lần nào. Nó vẫn phải có mặt: `keyframe_id` mã hoá
`frame_idx`, nên một keyframe trích thêm rơi đúng `frame_idx` của một keyframe
BTC là **cùng một khung hình** — và khi ấy bản BTC là bản chuẩn. Im lặng ghi đè
theo chiều ngược lại là đổi nội dung một hàng mà cả hai phía phép đo đều tưởng
là giống nhau.

`objects_text.json` và `audio_features.npz` chỉ có ở B0 và được chép nguyên: các
keyframe B1 đơn giản là không có hàng trong hai nguồn đó, y hệt cách một khung
không có chữ thì không có hàng OCR. Arm `object` đang trọng số 0,0 và `clap`
đang tắt, nên chúng không tham gia phép so.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

#: File JSON `{keyframe_id: text}`.
TEXT_FILES = (
    "ocr_text.json",
    "paddle_ocr.json",
    "qwen_captions.json",
    "qwen_ocr.json",
    "subtitles.json",
    "viet_ocr.json",
)

#: File `.npz` với hai mảng `ids` (U15) và `vectors` (float32, N×D).
DENSE_FILES = (
    "siglip_vectors.npz",
    "dense_vectors.npz",
    "subtitles_dense.npz",
)

#: Chỉ có ở B0. Chép nguyên, không phải lỗi khi B1 không có.
B0_ONLY = ("objects_text.json", "audio_features.npz")


def merge_text(b0: Path, b1: Path, out: Path, name: str) -> str:
    base = json.loads((b0 / name).read_text(encoding="utf-8"))
    delta_path = b1 / name
    if not delta_path.is_file():
        (out / name).write_text(
            json.dumps(base, ensure_ascii=False), encoding="utf-8"
        )
        return f"  {name:<24}{len(base):>9,} + {'(khong co B1)':>13}  = {len(base):>9,}"
    delta = json.loads(delta_path.read_text(encoding="utf-8"))
    clash = len(set(base) & set(delta))
    merged = {**delta, **base}  # B0 thắng khi trùng khoá — xem docstring.
    (out / name).write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    return (
        f"  {name:<24}{len(base):>9,} + {len(delta):>9,}"
        f"  = {len(merged):>9,}   trung {clash}"
    )


def merge_dense(b0: Path, b1: Path, out: Path, name: str) -> str:
    with np.load(b0 / name, allow_pickle=False) as payload:
        base_ids = [str(x) for x in payload["ids"].tolist()]
        base_vec = np.asarray(payload["vectors"], dtype=np.float32)
    delta_path = b1 / name
    if not delta_path.is_file():
        np.savez(out / name, ids=np.array(base_ids), vectors=base_vec)
        return f"  {name:<24}{len(base_ids):>9,} + {'(khong co B1)':>13}"
    with np.load(delta_path, allow_pickle=False) as payload:
        delta_ids = [str(x) for x in payload["ids"].tolist()]
        delta_vec = np.asarray(payload["vectors"], dtype=np.float32)
    if delta_vec.shape[1] != base_vec.shape[1]:
        raise ValueError(
            f"{name}: B0 {base_vec.shape[1]} chieu, B1 {delta_vec.shape[1]} chieu. "
            "Hai nua khong cung khong gian thi khoang cach khong so duoc."
        )
    known = set(base_ids)
    keep = [i for i, k in enumerate(delta_ids) if k not in known]
    ids = base_ids + [delta_ids[i] for i in keep]
    vectors = np.concatenate([base_vec, delta_vec[keep]], axis=0)
    np.savez(out / name, ids=np.array(ids), vectors=vectors)
    return (
        f"  {name:<24}{len(base_ids):>9,} + {len(keep):>9,}"
        f"  = {len(ids):>9,}   bo trung {len(delta_ids) - len(keep)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0", type=Path, default=Path("stores/flat"))
    parser.add_argument("--b1", type=Path, default=Path("stores/flat_b1_delta"))
    parser.add_argument("--out", type=Path, default=Path("stores/flat_b1"))
    args = parser.parse_args()

    b0 = args.b0 / "data" / "vectors"
    b1 = args.b1 / "data" / "vectors"
    out = args.out / "data" / "vectors"
    out.mkdir(parents=True, exist_ok=True)

    print(f"B0 {b0}\nB1 {b1}\n-> {out}\n")
    for name in TEXT_FILES:
        print(merge_text(b0, b1, out, name))
    for name in DENSE_FILES:
        print(merge_dense(b0, b1, out, name))
    for name in B0_ONLY:
        source = b0 / name
        if source.is_file():
            shutil.copy2(source, out / name)
            print(f"  {name:<24}chep nguyen tu B0")


if __name__ == "__main__":
    main()
