"""Máy này đã chạy được bàn nộp chưa — và nếu chưa thì THIẾU CÁI GÌ.

    python scripts/prepare_env.py              # kiểm, không tải gì
    python scripts/prepare_env.py --models     # + tải sẵn checkpoint (~7 GB)

Vì sao là một script chứ không phải một danh sách lệnh trong chat
-----------------------------------------------------------------
Mỗi thứ thiếu ở đây làm **một tầng biến mất im lặng**, không làm chương trình
đổ. Thiếu `google-genai` thì `vlm` ngồi ngoài và điểm vẫn in ra. Thiếu
`sentence-transformers` thì `bge_m3` vắng mặt. Thiếu `map-keyframes` thì lưới
không hiện ảnh nào dù token Modal đúng — trông y hệt lỗi mạng.

`QuestClient` từng bọc `from openai import OpenAI` trong `try/except ImportError:
pass`, và vì `openai` không có trong venv nên **hai trong ba nhà cung cấp không
tồn tại** mà không dòng log nào nói ra. Script này là hàm `available()` cho cả
cái máy: nó phải trả lời được "vì sao không".

Nó KHÔNG cài gì hộ. Nó in ra lệnh cần chạy, rồi người vận hành tự chạy — vì
`pip install` vào nhầm interpreter là một trong những cách hỏng khó thấy nhất
trên Windows.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INSTALL = 'pip install -e ".[online,browser,dev]"'


@dataclass(frozen=True)
class Dep:
    """Một gói, và **cái gì tắt đi** khi thiếu nó."""

    module: str
    cost: str


#: Gói -> tầng chết theo. Câu chữ ở cột hai quan trọng ngang cột một: một
#: `ModuleNotFoundError` thì ai cũng sửa được, còn "arm bge_m3 vắng mặt" thì
#: người ta đi tìm dữ liệu.
DEPS = (
    Dep("numpy", "MỌI arm — không có phép nhân ma trận nào chạy"),
    Dep("yaml", "không đọc được config, chương trình không khởi động"),
    Dep("pydantic", "contracts — không dựng được Candidate"),
    Dep("rank_bm25", "TẤT CẢ arm văn bản: ocr_*, subtitle_partial, object"),
    Dep("torch", "siglip (trọng số 2,0 — arm gánh cả hệ), clap, envit5"),
    Dep("transformers", "tháp text siglip + clap + envit5"),
    Dep("sentence_transformers", "bge_m3 và subtitle_bge_m3"),
    Dep("PIL", "nén ảnh gửi cho vlm và cho nút 'VLM đọc hộ'"),
    Dep("google.genai", "tầng rerank vlm, nút 'VLM đọc hộ', nút 'Quest viết lại'"),
    Dep("streamlit", "cả giao diện bàn nộp"),
)

#: Khoá môi trường: (tên, bắt buộc?, hậu quả khi thiếu).
#: KHÔNG in giá trị — chỉ in có/không.
ENV_KEYS = (
    ("GEMINI_API_KEY", True,
     "vlm + VLM đọc hộ + Quest đều ngồi ngoài. PHẢI là khoá RIÊNG từng người: "
     "trần free tier là 15 request/PHÚT cho cả project, dùng chung là chặn nhau"),
    ("MFE_GEMINI_VLM_MODEL", True,
     "mặc định rơi về gemini-3.6-flash = 20 request/NGÀY; với 49 đề nó chết ở "
     "câu 21 và 28 câu còn lại giữ thứ tự cũ mà không có gì đỏ lên"),
    ("MFE_GEMINI_QUEST_MODEL", False, "cùng trần quota, cho nút Quest"),
    ("MODAL_TOKEN_ID", False, "không tải được ảnh keyframe — lưới trống, "
     "truy hồi vẫn chạy bình thường"),
    ("MODAL_TOKEN_SECRET", False, "đi kèm MODAL_TOKEN_ID"),
)


def _have(module: str) -> bool:
    """`importlib.util.find_spec` chứ không `import`: nạp torch mất vài giây."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> int:
    want = (3, 11) <= sys.version_info[:2] < (3, 13)
    mark = "✓" if want else "✗"
    print(f"  {mark} python {sys.version_info.major}.{sys.version_info.minor} "
          f"({sys.executable})")
    if not want:
        print("      pyproject đòi >=3.11,<3.13")
    return 0 if want else 1


def check_deps() -> int:
    bad = 0
    for dep in DEPS:
        if _have(dep.module):
            print(f"  ✓ {dep.module}")
            continue
        bad += 1
        print(f"  ✗ {dep.module}\n      tắt theo: {dep.cost}")
    return bad


def check_env() -> int:
    from mfe.offline.transport import load_dotenv  # noqa: PLC0415

    path = ROOT / ".env"
    if not path.is_file():
        print(f"  ✗ KHÔNG CÓ {path}")
        print("      chép .env.example thành .env rồi điền khoá của RIÊNG BẠN")
        return 1

    for key, value in load_dotenv(path).items():
        os.environ.setdefault(key, value)

    bad = 0
    for key, required, cost in ENV_KEYS:
        if os.getenv(key):
            print(f"  ✓ {key}")
            continue
        if required:
            bad += 1
        print(f"  {'✗' if required else '·'} {key}\n      {cost}")
    return bad


def check_data() -> int:
    """Đếm dữ liệu, dùng lại `share_data.REQUIRED` — bản kê DUY NHẤT.

    Chép lại danh sách file vào đây là cách để một hôm gói zip thêm một nguồn
    mà phép kiểm vẫn báo "đủ".
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from share_data import verify  # noqa: PLC0415

    return verify()


def check_gemini_live() -> int:
    """Gọi THẬT một lượt. Liệt kê được ≠ gọi được.

    `gemini-2.5-flash` vẫn nằm trong `models.list()` của khoá nhóm nhưng
    `generateContent` trả 404. Một khoá "có mặt trong .env" nói lên rất ít.
    """
    if not os.getenv("GEMINI_API_KEY"):
        print("  · bỏ qua — chưa có GEMINI_API_KEY")
        return 0
    model = os.getenv("MFE_GEMINI_VLM_MODEL", "gemini-3.6-flash")
    try:
        from google import genai  # noqa: PLC0415

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        reply = client.models.generate_content(model=model, contents="tra loi: ok")
        print(f"  ✓ {model} trả lời được ({(reply.text or '').strip()[:30]!r})")
        return 0
    except Exception as err:  # noqa: BLE001 - day la phep kiem, khong phai duong chay
        print(f"  ✗ {model}: {type(err).__name__}: {str(err)[:200]}")
        return 1


#: Checkpoint tải một lần rồi nằm trong cache HuggingFace. Đây là phần CHẬM
#: nhất của lần chạy đầu, và nó KHÔNG phụ thuộc dữ liệu — nên đáng làm trong
#: lúc chờ kho phẳng.
MODELS = (
    ("google/siglip-so400m-patch14-384", "~3,5 GB", "arm siglip (trọng số 2,0)"),
    ("VietAI/envit5-translation", "~1,2 GB", "dịch truy vấn sang tiếng Anh"),
    ("BAAI/bge-m3", "~2,2 GB", "arm bge_m3 và subtitle_bge_m3"),
    ("laion/clap-htsat-unfused", "~0,6 GB", "arm clap (trọng số 0,5)"),
)


def pull_models() -> int:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    bad = 0
    for repo, size, why in MODELS:
        print(f"  … {repo}  {size}  — {why}", flush=True)
        try:
            snapshot_download(repo_id=repo)
            print(f"  ✓ {repo}")
        except Exception as err:  # noqa: BLE001
            bad += 1
            print(f"  ✗ {repo}: {type(err).__name__}: {str(err)[:160]}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true",
                        help="Tai san checkpoint (~7 GB). Lam trong luc cho.")
    parser.add_argument("--call-api", action="store_true",
                        help="Goi THAT mot luot Gemini de kiem khoa.")
    args = parser.parse_args()

    print("\n=== 1. Python ===")
    bad = check_python()

    print("\n=== 2. Gói ===")
    missing = check_deps()
    if missing:
        print(f"\n  -> thiếu {missing} gói. Chạy trong venv của repo:\n     {INSTALL}")
    bad += missing

    print("\n=== 3. .env ===")
    bad += check_env()

    if args.call_api:
        print("\n=== 3b. Gọi thật một lượt Gemini ===")
        bad += check_gemini_live()

    print("\n=== 4. Dữ liệu (git không mang theo) ===")
    bad += check_data()

    if args.models:
        print("\n=== 5. Tải checkpoint ===")
        bad += pull_models()
    else:
        print("\n=== 5. Checkpoint ===")
        print("  · chưa kiểm. Chạy lại với --models để tải sẵn (~7 GB) —")
        print("    đây là phần chậm nhất của lần chạy đầu và nó KHÔNG cần dữ liệu.")

    print()
    if bad:
        print(f"CHUA CHAY DUOC: {bad} muc can sua (xem dau ✗ o tren).")
        return 1
    print("SAN SANG. Mo giao dien:  streamlit run app/submit_lab.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
