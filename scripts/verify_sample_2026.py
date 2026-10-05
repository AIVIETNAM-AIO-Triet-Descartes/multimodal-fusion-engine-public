"""verify_sample_2026.py — Kiểm bộ mẫu đã đạt yêu cầu chưa (local + trên HF).

    python scripts/verify_sample_2026.py            # kiểm cả local và HF
    python scripts/verify_sample_2026.py --local    # chỉ kiểm local (không cần mạng)

Bốn thứ được kiểm, theo thứ tự từ rẻ tới đắt:

1. **Số lượng nhất quán** — với mỗi video, số dòng `map-keyframes/*.csv` phải
   bằng số ảnh keyframe, bằng số vector trong `.npy`, bằng số file objects. Đây
   là điều kiện `StoreWriter.verify_stores_agree()` sẽ đòi lúc ingest; lệch ở
   đây thì lệch luôn ở đó.
2. **Công thức frame_idx** — `floor(pts_time × fps)` phải khớp mọi dòng CSV.
3. **Đủ file phụ** — mỗi video một `.mp4` và một `media-info/*.json`.
4. **Local ↔ HF khớp từng file** — thiếu file nào, thừa file nào.

Không kiểm nội dung vector ở đây: việc đó là của `verify_clip_slices.py`, tốn
thời gian hơn nhiều vì phải nạp CLIP và encode lại ảnh.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path("data/sample_2026")
REPO_ID = "TrietTrinh/aic2026-sample"


def check_counts() -> tuple[bool, int, int]:
    """Đối chiếu số lượng giữa 4 kho. Trả về (ok, số video, tổng keyframe)."""
    print("[1] So luong 4 kho khop nhau")
    print(f"    {'video':<12} {'CSV':>5} {'anh':>5} {'npy':>5} {'obj':>5}")
    bad: list[str] = []
    total_kf = 0
    csv_files = sorted((OUT / "map-keyframes").glob("*.csv"))
    for path in csv_files:
        video_id = path.stem
        with path.open(encoding="utf-8", newline="") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
        kf_dir, obj_dir = OUT / "keyframes" / video_id, OUT / "objects" / video_id
        images = len(os.listdir(kf_dir)) if kf_dir.is_dir() else 0
        objects = len(os.listdir(obj_dir)) if obj_dir.is_dir() else 0
        npy_path = OUT / "clip-features" / f"{video_id}.npy"
        vectors = np.load(npy_path).shape[0] if npy_path.is_file() else 0
        ok = rows == images == vectors == objects
        if not ok:
            bad.append(video_id)
        total_kf += rows
        flag = "" if ok else "  <-- LECH"
        print(f"    {video_id:<12} {rows:>5} {images:>5} {vectors:>5} {objects:>5}{flag}")
    print(f"    => {len(csv_files) - len(bad)}/{len(csv_files)} video khop"
          + (f", lech: {bad}" if bad else ""))
    return not bad, len(csv_files), total_kf


def check_formula() -> bool:
    """floor(pts_time * fps) phải khớp cột frame_idx ở mọi dòng."""
    print("\n[2] Cong thuc floor(pts_time x fps) == frame_idx")
    wrong = total = 0
    for path in sorted((OUT / "map-keyframes").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                total += 1
                expected = math.floor(float(row["pts_time"]) * float(row["fps"]))
                if expected != int(row["frame_idx"]):
                    wrong += 1
    print(f"    sai {wrong}/{total:,} dong")
    return wrong == 0


def check_companions(n_videos: int) -> bool:
    """Mỗi video phải có đúng một .mp4 và một media-info JSON."""
    print("\n[3] File phu (mp4 + media-info)")
    mp4 = len(list((OUT / "videos").glob("*.mp4")))
    info = len(list((OUT / "media-info").glob("*.json")))
    print(f"    videos mp4 : {mp4}/{n_videos}")
    print(f"    media-info : {info}/{n_videos}")
    return mp4 == n_videos and info == n_videos


def check_remote() -> bool:
    """Đối chiếu từng file giữa local và repo HF."""
    print("\n[4] Local <-> HuggingFace")
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("    Khong doc duoc HF_TOKEN tu .env — bo qua buoc nay.")
        return False

    api = HfApi(token=token)
    info = api.dataset_info(REPO_ID)
    if not getattr(info, "private", False):
        print(f"    [CANH BAO] Repo {REPO_ID} dang PUBLIC — du lieu co ban quyen!")
        return False
    print(f"    repo private: OK ({REPO_ID})")

    remote = {
        p for p in api.list_repo_files(REPO_ID, repo_type="dataset")
        if p.startswith("data/sample_2026/")
    }
    local = {
        "data/sample_2026/" + str(p.relative_to(OUT)).replace("\\", "/")
        for p in OUT.rglob("*")
        if p.is_file()
    }
    missing, extra = local - remote, remote - local
    print(f"    local {len(local)} file | remote {len(remote)} file")
    print(f"    thieu tren remote: {len(missing)} | thua tren remote: {len(extra)}")
    for path in sorted(missing)[:5]:
        print(f"      thieu: {path}")
    for path in sorted(extra)[:5]:
        print(f"      thua : {path}")
    return not missing and not extra


def main() -> int:
    local_only = "--local" in sys.argv
    if not OUT.is_dir():
        print(f"Khong thay {OUT} — chay prepare_sample_2026.py truoc.")
        return 2

    counts_ok, n_videos, total_kf = check_counts()
    formula_ok = check_formula()
    companions_ok = check_companions(n_videos)
    remote_ok = True if local_only else check_remote()

    print("\n" + "=" * 60)
    results = [
        ("So luong 4 kho khop", counts_ok),
        ("Cong thuc frame_idx", formula_ok),
        ("File phu day du", companions_ok),
    ]
    if not local_only:
        results.append(("Local == HuggingFace", remote_ok))
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  {n_videos} video | {total_kf:,} keyframe")

    if all(ok for _, ok in results):
        print("\n=> DAT YEU CAU.")
        return 0
    print("\n=> CHUA DAT — xem cac muc FAIL o tren.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
