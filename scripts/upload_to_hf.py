"""
upload_to_hf.py — Upload data/sample_2026/ lên HuggingFace Hub (dataset, PRIVATE).

Cài trước:
    pip install huggingface_hub python-dotenv
Token: đặt HF_TOKEN trong .env ở gốc repo (cần quyền "write").

Vì sao dùng upload_folder chứ không phải upload_large_folder:
    upload_large_folder() KHÔNG có tham số path_in_repo (đã kiểm chữ ký trên
    bản đang cài), nên muốn kết quả nằm ở data/sample_2026/ trên HF thì phải
    dựng một thư mục trung gian có đúng cấu trúc đó rồi trỏ vào — tức copy
    2,7 GB sang chỗ khác. Bản trước làm vậy bằng thư mục TẠM, và điều đó còn
    phá luôn khả năng resume: upload_large_folder ghi metadata resume vào
    chính folder_path, mà thư mục tạm thì bị xoá sau mỗi lần chạy.

    upload_folder() có path_in_repo, nên trỏ thẳng vào data/sample_2026 được:
    không copy, không tốn thêm đĩa. Nó vẫn resume được theo cách khác — file
    nào đã có trên repo với cùng hash thì bỏ qua, nên chạy lại sau khi đứt
    mạng là an toàn và rẻ.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()  # đọc .env ở gốc repo, lấy HF_TOKEN

REPO_ID = "TrietTrinh/aic2026-sample"  # đổi theo tài khoản/tổ chức thật
# CHỈ upload đúng thư mục này — không đụng data/sample, data/sample_01...
SAMPLE_DIR = Path("data/sample_2026")


def main() -> int:
    if not SAMPLE_DIR.exists():
        raise FileNotFoundError(f"Không thấy {SAMPLE_DIR} — chạy prepare_sample_2026.py trước.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Không đọc được HF_TOKEN từ .env — dừng.")
        return 2
    api = HfApi(token=token)

    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    # Kiểm lại quyền riêng tư THẬT, không tin vào tham số vừa truyền.
    # `exist_ok=True` KHÔNG chuyển một repo đang public thành private — nếu repo
    # đã tồn tại từ trước ở chế độ public thì lệnh trên im lặng bỏ qua. Dữ liệu
    # đây là video thời sự có bản quyền đài truyền hình, nên đẩy nhầm lên public
    # là chuyện không sửa được bằng cách xoá sau.
    info = api.dataset_info(REPO_ID)
    if not getattr(info, "private", False):
        print(f"[DỪNG] Repo {REPO_ID} đang ở chế độ PUBLIC.")
        print("       Dữ liệu có bản quyền — chuyển sang private rồi chạy lại:")
        print(f"       https://huggingface.co/datasets/{REPO_ID}/settings")
        return 1

    print(f"Repo (đã xác nhận private): https://huggingface.co/datasets/{REPO_ID}")

    n_files = sum(1 for _ in SAMPLE_DIR.rglob("*") if _.is_file())
    print(f"Đang upload {n_files} file từ {SAMPLE_DIR} -> data/sample_2026/ trên HF...")
    print("  (chạy lại được nếu đứt mạng — file đã có cùng hash sẽ bị bỏ qua)\n")

    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(SAMPLE_DIR),
        path_in_repo="data/sample_2026",
        commit_message="Add AIC 2026 batch-1 sample (16 videos)",
    )

    print("\nXong. Team cần được thêm quyền đọc repo (Settings > Collaborators trên HF)")
    print("hoặc dùng chung token có quyền read tới repo này để tải về (xem download_from_hf.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
