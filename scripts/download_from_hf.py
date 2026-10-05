"""
download_from_hf.py — Tải data/sample_2026/ từ HuggingFace về máy local.

    MFE_MEMBER=dat python scripts/download_from_hf.py      # 16 video CỦA BẠN
    python scripts/download_from_hf.py --all               # toàn bộ (Lead)
    python scripts/download_from_hf.py --list              # xem ai đã up gì

Cài trước:
    pip install huggingface_hub python-dotenv
Cần token có quyền ĐỌC repo private: đặt HF_TOKEN trong .env ở gốc repo.

Vì sao mặc định chỉ tải phần của mình
--------------------------------------
Repo chứa tối đa 80 video (5 người × 16) — khoảng 13 GB. Nhưng để viết ground
truth và chạy thử trên máy mình, bạn CHỈ cần 16 video của mình: query của bạn
chỉ được phép trỏ vào chúng (`mfe-merge-queryset` chặn nếu trỏ ra ngoài), và
kho vector/BM25 dựng từ 80 video sẽ khiến số đo local của bạn không so sánh
được với người khác.

Chỉ Lead cần `--all`, để chạy harness trên bộ gộp.

Danh sách video của bạn đọc từ `data/sample_2026/manifests/{member}.json` trên
repo — file do `prepare_sample_2026.py` sinh ra, mỗi người một file riêng.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

load_dotenv()  # đọc .env ở gốc repo, lấy HF_TOKEN

REPO_ID = "TrietTrinh/aic2026-sample"  # đổi khớp với upload_to_hf.py
LOCAL_DIR = "."  # tải về sao cho ra đúng ./data/sample_2026/... tại đây
PREFIX = "data/sample_2026"

#: Mỗi video trải trên 6 kho. Liệt kê tường minh thay vì dùng một pattern
#: `*{video_id}*`: `L21_V001` là tiền tố của `L21_V0010` nếu BTC từng đánh số
#: tới đó, và một pattern quá rộng sẽ kéo về video của người khác mà không báo.
_PER_VIDEO_PATTERNS = [
    f"{PREFIX}/keyframes/{{vid}}/*",
    f"{PREFIX}/objects/{{vid}}/*",
    f"{PREFIX}/map-keyframes/{{vid}}.csv",
    f"{PREFIX}/media-info/{{vid}}.json",
    f"{PREFIX}/clip-features/{{vid}}.npy",
    f"{PREFIX}/videos/{{vid}}.mp4",
]


def token() -> str | None:
    return os.environ.get("HF_TOKEN")


def list_manifests(api: HfApi) -> list[str]:
    """Tên các thành viên đã upload, đọc từ manifests/ trên repo."""
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(
        Path(f).stem
        for f in files
        if f.startswith(f"{PREFIX}/manifests/") and f.endswith(".json")
    )


def videos_of(member: str) -> list[str]:
    """Danh sách video của `member`, tải riêng file manifest về trước."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{PREFIX}/manifests/{member}.json",
        local_dir=LOCAL_DIR,
        token=token(),
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    videos = payload.get("videos") or []
    if not videos:
        raise SystemExit(f"manifest của '{member}' không có video nào — bộ này hỏng.")
    return videos


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tải bộ mẫu AIC 2026 từ HuggingFace (mặc định: chỉ phần của bạn)."
    )
    parser.add_argument(
        "--all", action="store_true", help="Tải toàn bộ 80 video (dành cho Lead)."
    )
    parser.add_argument(
        "--member", default=None, help="Ghi đè MFE_MEMBER (ví dụ --member dat)."
    )
    parser.add_argument(
        "--list", action="store_true", help="Chỉ liệt kê ai đã upload, không tải."
    )
    args = parser.parse_args()

    if not token():
        print("Không đọc được HF_TOKEN từ .env — cần token có quyền read.")
        return 2

    api = HfApi(token=token())
    members = list_manifests(api)

    if args.list:
        if not members:
            print(f"Chưa ai upload manifest lên {REPO_ID}.")
            return 0
        print(f"Đã upload ({len(members)}/5): {', '.join(members)}")
        for name in members:
            print(f"  {name:<8} {len(videos_of(name)):>3} video")
        return 0

    if args.all:
        print(f"Tải TOÀN BỘ {PREFIX}/ từ {REPO_ID} (có thể ~13 GB)...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{PREFIX}/**"],
            local_dir=LOCAL_DIR,
            token=token(),
        )
        print(f"\nXong. Dữ liệu ở ./{PREFIX}")
        return 0

    member = (args.member or os.environ.get("MFE_MEMBER", "")).strip().lower()
    if not member:
        print(
            "Chưa biết bạn là ai. Đặt MFE_MEMBER hoặc dùng --member.\n"
            f"  MFE_MEMBER=dat python {Path(sys.argv[0]).name}\n"
            f"Đã upload: {', '.join(members) if members else '(chưa ai)'}\n"
            "Lead cần toàn bộ thì dùng --all."
        )
        return 2
    if member not in members:
        print(
            f"Chưa thấy manifest của '{member}' trên repo.\n"
            f"Đã upload: {', '.join(members) if members else '(chưa ai)'}\n"
            "Chạy prepare_sample_2026.py + upload_to_hf.py trước, hoặc kiểm lại tên."
        )
        return 2

    videos = videos_of(member)
    patterns = [p.format(vid=vid) for vid in videos for p in _PER_VIDEO_PATTERNS]
    patterns.append(f"{PREFIX}/manifests/{member}.json")

    print(f"Tải {len(videos)} video của '{member}' từ {REPO_ID}:")
    print(f"  {', '.join(videos)}\n")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=LOCAL_DIR,
        token=token(),
    )
    print(f"\nXong. Dữ liệu ở ./{PREFIX}")
    print("Kiểm tra tiếp:  python scripts/verify_sample_2026.py --local")
    return 0


if __name__ == "__main__":
    sys.exit(main())
