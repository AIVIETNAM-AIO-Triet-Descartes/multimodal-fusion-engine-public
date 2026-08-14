"""
prepare_sample_2026.py — Trích mẫu nhiều video từ dữ liệu thật AIC 2026 (batch 1).

Cài trước:
    pip install remotezip huggingface_hub requests numpy pandas

Ý tưởng:
- Keyframes/Videos/Objects: tải CHỌN LỌC qua HTTP Range request (remotezip),
  không tải nguyên zip — quan trọng nhất với objects-aic25-b1.zip (ước tính
  ~900MB-1GB cho cả batch, trong khi ta chỉ cần vài chục video).
- map-keyframes/media-info: nhỏ, tải nguyên zip cho cả batch rồi lọc cục bộ.
- clip-features: PHẢI tải nguyên file (~94MB, chấp nhận được) vì PDF mô tả
  đây là MỘT mảng vector duy nhất cho cả batch — không cắt được từ xa nếu
  không biết trước offset của từng video.

CẢNH BÁO — các điểm cần tự xác minh trước khi tin kết quả (đã đánh dấu TODO
trong code): (1) prefix đường dẫn thật bên trong từng zip, (2) thứ tự toàn
cục thật của mảng CLIP feature. Script in ra dữ liệu thô để bạn đối chiếu
bằng mắt ở lần chạy đầu, đừng chạy toàn bộ rồi tin ngay.
"""
import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip

BASE_URL = "https://aic-data.ledo.io.vn"

# Số kết nối song song khi trích file nhỏ (keyframes/objects).
#
# Đây là việc CHỜ MẠNG chứ không phải tính toán: mỗi ảnh là một HTTP range
# request riêng, và phần lớn thời gian là round-trip chứ không phải truyền dữ
# liệu. Đo tuần tự: ~42 ảnh/phút — cả job mất ~4 giờ. Song song hoá ăn gần
# tuyến tính ở đây.
#
# Giữ ở mức vừa phải: server của BTC không hứa gì về rate limit, và bị chặn
# giữa chừng còn tốn thời gian hơn là chạy chậm. ĐO THẬT: 8 luồng chạy ngon qua
# L21/L23 rồi server bắt đầu từ chối range request giữa nhóm L25 (remotezip báo
# `RangeNotSupported`, nhưng thực chất là bị throttle — nó ném lỗi đó cho mọi
# response khác 206). 4 luồng vẫn nhanh gấp ~2 lần tuần tự mà nhẹ tay hơn.
MAX_WORKERS = 4

# Chọn rải rác nhiều nhóm để đa dạng (khác fps/độ dài) — đổi tùy ý.
GROUPS = ["L21", "L23", "L25", "L27"]
N_PER_GROUP = 4  # 4 nhóm x 4 video = 16 video (> 5 theo yêu cầu)

# Ai đang chạy script này. QUYẾT ĐỊNH VIDEO NÀO BẠN NHẬN — đừng bỏ qua.
#
# Không có nó thì mọi người cùng lấy `video_ids[:4]` của mỗi nhóm, tức là CẢ 5
# NGƯỜI NHẬN ĐÚNG 16 VIDEO GIỐNG HỆT NHAU. Bộ test "rải rác nhiều video" sẽ
# thành 16 video lặp lại 5 lần, và không ai phát hiện ra cho tới lúc gộp.
#
# Đặt qua biến môi trường trước khi chạy, ví dụ:  MFE_MEMBER=dat python scripts/...
MEMBERS = ["dat", "quang", "trong", "bao", "triet"]
MEMBER = os.environ.get("MFE_MEMBER", "").strip().lower()

OUT = Path("data/sample_2026")
SUBFOLDERS = ["clip-features", "keyframes", "map-keyframes", "media-info", "objects", "videos"]


def check_range_support(url: str) -> bool:
    """Kiểm tra server có hỗ trợ tải một phần (HTTP Range) không.

    Bắt buộc để remotezip hoạt động.
    """
    r = requests.get(url, headers={"Range": "bytes=0-1"}, timeout=20, stream=True)
    ok = r.status_code == 206
    r.close()
    return ok


def member_slice(video_ids: list[str], n: int) -> list[str]:
    """Cắt phần của MEMBER ra khỏi danh sách video đã sắp xếp.

    Người thứ k lấy `video_ids[k*n : (k+1)*n]`, nên hai người không bao giờ
    nhận chung một video. Cắt theo chỉ số cố định chứ không random: chạy lại
    phải ra đúng bộ cũ, nếu không thì ground truth đã viết trỏ vào video mình
    không còn giữ nữa.
    """
    if MEMBER not in MEMBERS:
        raise SystemExit(
            f"Chưa đặt MFE_MEMBER (hoặc đặt sai: {MEMBER!r}).\n"
            f"Hợp lệ: {', '.join(MEMBERS)}\n"
            f"Ví dụ:  MFE_MEMBER=dat python scripts/prepare_sample_2026.py\n"
            f"Không có nó thì cả 5 người tải về CÙNG một bộ 16 video."
        )
    k = MEMBERS.index(MEMBER)
    chunk = video_ids[k * n : (k + 1) * n]
    if len(chunk) < n:
        raise SystemExit(
            f"Nhóm này chỉ còn {len(chunk)} video cho '{MEMBER}' (cần {n}). "
            f"Tổng có {len(video_ids)} video, {len(MEMBERS)} người x {n} = "
            f"{len(MEMBERS) * n}. Giảm N_PER_GROUP hoặc thêm nhóm vào GROUPS."
        )
    return chunk


def pick_videos_in_group(group: str, n: int) -> list[str]:
    """Liệt kê video có trong Keyframes_{group}.zip qua central directory (không tải nội dung)."""
    url = f"{BASE_URL}/Keyframes_{group}.zip"
    with RemoteZip(url) as zf:
        names = zf.namelist()

    # Path dạng ".../{video_id}/{n}.jpg" — video_id là thư mục cha trực tiếp của ảnh.
    video_ids = sorted({n.split("/")[-2] for n in names if n.lower().endswith((".jpg", ".jpeg"))})
    if not video_ids:
        raise RuntimeError(
            f"Không suy ra được video_id từ path trong {url}. "
            f"5 entry đầu: {names[:5]}"
        )
    chunk = member_slice(video_ids, n)
    print(f"  [{group}] {len(video_ids)} video trong nhóm -> '{MEMBER}' nhận: {chunk}")
    return chunk


def entry_matches_video(entry: str, vid: str) -> bool:
    """
    Khớp một entry trong zip với video_id, xử lý cả 2 kiểu path đã xác nhận thật:
      - video_id là THƯ MỤC CHA:  keyframes/L21_V001/001.jpg
      - video_id là TÊN FILE:     video/L21_V001.mp4   (đã xác nhận từ Videos_L21_a.zip)
    """
    p = PurePosixPath(entry)
    if vid in p.parts[:-1]:
        return True
    if p.stem == vid:
        return True
    return False


def open_remote_zip(zip_url: str, attempts: int = 4) -> RemoteZip:
    """Mở RemoteZip, thử lại khi server từ chối range request.

    `RangeNotSupported` nghe như "server không hỗ trợ", nhưng remotezip ném nó
    cho MỌI response khác 206 — kể cả 429/503 lúc bị throttle. Đã gặp thật: mở
    được suốt L21/L23 rồi hỏng giữa L25. Lùi dần rồi thử lại là đủ.
    """
    delay = 3.0
    for attempt in range(attempts):
        try:
            return RemoteZip(zip_url)
        except Exception as err:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            print(
                f"    [retry {attempt + 1}/{attempts - 1}] mở {zip_url} lỗi "
                f"({type(err).__name__}), chờ {delay:.0f}s...",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _extract_chunk(zip_url: str, members: list[str], dest: Path, counter: dict, lock, total: int):
    """Trích một phần danh sách entry bằng KẾT NỐI RIÊNG của luồng này.

    `RemoteZip` giữ một HTTP session và con trỏ đọc dùng chung, nên không an
    toàn khi nhiều luồng cùng gọi `.extract()` trên một đối tượng. Mỗi luồng mở
    riêng một RemoteZip — tốn thêm một lần đọc central directory, đổi lại không
    có tranh chấp trạng thái.
    """
    done_here = 0
    with open_remote_zip(zip_url) as zf:
        for member in members:
            # Thử lại khi kết nối bị reset. Server BTC thỉnh thoảng đóng đột ngột
            # (ConnectionResetError 10054) — đo được 1 lỗi/1082 entry. Bỏ qua
            # một ảnh nghe như chuyện nhỏ, nhưng CSV vẫn có dòng cho nó, nên
            # ingest sẽ dừng ở đúng video đó: "map-keyframes lists n=104 nhưng
            # thiếu ảnh". Rẻ hơn nhiều nếu thử lại ngay tại đây.
            last_err = None
            for attempt in range(3):
                try:
                    zf.extract(member, path=dest)
                    last_err = None
                    break
                except Exception as err:  # noqa: BLE001 — một file hỏng không nên giết cả job
                    last_err = err
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
            if last_err is not None:
                with lock:
                    counter["failed"].append((member, f"{type(last_err).__name__}: {last_err}"))
                continue
            done_here += 1
            if done_here % 25 == 0:
                with lock:
                    counter["done"] += 25
                    print(f"    ...{counter['done']}/{total}", flush=True)
    with lock:
        counter["done"] += done_here % 25


def extract_selected(zip_url: str, video_ids: list[str], dest: Path, label: str):
    """Trích các entry thuộc về video_ids từ một zip từ xa — chỉ tải phần khớp.

    Danh sách entry đọc một lần (central directory), rồi chia đều cho
    MAX_WORKERS luồng, mỗi luồng một kết nối.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with open_remote_zip(zip_url) as zf:
        names = zf.namelist()
    matched = [n for n in names if any(entry_matches_video(n, vid) for vid in video_ids)]
    if not matched:
        print(f"  [CẢNH BÁO] Không khớp entry nào — kiểm tra lại quy ước path trong {zip_url}")
        return

    # Bỏ qua thứ đã có trên đĩa: server này hay ngắt giữa chừng (throttle,
    # connection reset), nên chạy lại là chuyện bình thường chứ không phải
    # ngoại lệ. Tải lại 1.4k ảnh chỉ để tới được chỗ hỏng là phí thời gian và
    # phí luôn hạn mức của server — đúng thứ đã làm nó chặn ta lần trước.
    wanted = [n for n in matched if not (dest / n).exists() or (dest / n).is_dir()]
    wanted = [n for n in wanted if not n.endswith("/")]
    skipped = len(matched) - len(wanted)
    print(
        f"  [{label}] khớp {len(matched)}/{len(names)} entry cho {len(video_ids)} video"
        + (f" — bỏ qua {skipped} đã có" if skipped else ""),
        flush=True,
    )
    if not wanted:
        print(f"  [{label}] đã đủ, không cần tải gì thêm.")
        return

    workers = min(MAX_WORKERS, max(1, len(wanted) // 10) or 1)
    chunks = [wanted[i::workers] for i in range(workers)]
    counter = {"done": 0, "failed": []}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_extract_chunk, zip_url, chunk, dest, counter, lock, len(wanted))
            for chunk in chunks
            if chunk
        ]
        for fut in as_completed(futures):
            fut.result()

    if counter["failed"]:
        print(f"  [{label}] LỖI {len(counter['failed'])} entry — 3 ví dụ đầu:")
        for member, err in counter["failed"][:3]:
            print(f"      {member}: {err}")


def fetch_zip_bytes(zip_name: str) -> bytes:
    """Tải nguyên một zip nhỏ về RAM, dùng lại được cho nhiều mục đích.

    Tách riêng khỏi bước lọc vì map-keyframes cần dùng HAI lần: một lần lấy CSV
    của video mục tiêu, một lần đếm keyframe của TOÀN BỘ video để tính offset
    CLIP. Tải hai lần là phí băng thông mà không được gì.
    """
    url = f"{BASE_URL}/{zip_name}"
    print(f"  Đang tải nguyên {zip_name} (nhỏ, không cần chọn lọc từ xa)...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


def filter_zip_bytes(content: bytes, zip_name: str, video_ids: list[str], dest: Path) -> list[str]:
    """Lọc các entry thuộc về video_ids ra `dest`. Trả về full namelist."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        print(f"  [debug] 5 entry đầu của {zip_name}: {names[:5]}")
        wanted = [n for n in names if any(entry_matches_video(n, vid) for vid in video_ids)]
        print(f"  [{zip_name}] khớp {len(wanted)}/{len(names)} entry")
        for member in wanted:
            zf.extract(member, path=dest)
    return names


def video_zip_urls(group: str) -> list[str]:
    """Các phần zip video thật sự tồn tại cho một nhóm.

    Không giả định hậu tố `_a`: một nhóm có thể bị chia thành `_a`, `_b`, ... và
    video ta chọn không nhất thiết nằm ở phần đầu. Dò bằng HEAD (rẻ) rồi trả về
    mọi phần tồn tại — bỏ sót một phần nghĩa là mất video mà không có lỗi nào.
    """
    found: list[str] = []
    for suffix in ("_a", "_b", "_c", "_d", ""):
        url = f"{BASE_URL}/Videos_{group}{suffix}.zip"
        try:
            if requests.head(url, timeout=20, allow_redirects=True).status_code == 200:
                found.append(url)
        except requests.RequestException:
            continue
    if not found:
        print(f"  [CẢNH BÁO] Không thấy zip video nào cho nhóm {group}")
    return found


def extract_clip_features(video_ids: list[str], dest: Path):
    """Trích .npy CLIP của từng video mục tiêu.

    ĐÃ KIỂM BẰNG DỮ LIỆU THẬT: `clip-features-32-aic25-b1.zip` chứa **một file
    .npy cho MỖI video** (`clip-features-32/L21_V001.npy`, ...), y hệt cấu trúc
    bộ 2025 — KHÔNG phải một mảng gộp cho cả batch như PDF §3 gợi ý.

    Bản trước của hàm này tin vào mô tả "một mảng duy nhất" rồi tự tính offset
    theo tổng số keyframe của các video đứng trước. Kết quả: nó nạp
    `L21_V001.npy` (307 vector) tưởng là mảng toàn cục, cắt được đúng video đầu
    nhờ offset 0, và trả về **mảng rỗng cho 15/16 video còn lại** — mà vẫn thoát
    exit 0. Không tính offset nữa thì không còn giả định nào để sai.
    """
    extract_selected(
        f"{BASE_URL}/clip-features-32-aic25-b1.zip", video_ids, dest, "clip-features"
    )


def flatten_redundant_nesting(dest: Path):
    """Bỏ lớp thư mục thừa do prefix bên trong zip tạo ra.

    Các zip của BTC gói sẵn theo dạng `keyframes/L21_V001/001.jpg`, nên giải nén
    vào `data/sample_2026/keyframes/` sẽ ra `.../keyframes/keyframes/L21_V001/`.
    Lồng hai lớp làm lệch với `config.paths` (vốn trỏ `.../keyframes/{video_id}/`)
    và lệch cả với bộ `data/sample/` đang dùng — ingest sẽ không thấy video nào
    mà cũng không báo lỗi gì.

    Chỉ làm phẳng khi thư mục con trùng đúng tên thư mục cha, để không đụng
    nhầm một video vô tình trùng tên.
    """
    children = [c for c in dest.iterdir()] if dest.is_dir() else []
    # Lớp thừa = dest chỉ chứa ĐÚNG MỘT thư mục con, không có gì khác.
    # Không so tên: prefix trong zip không nhất thiết trùng tên thư mục đích —
    # đã gặp thật, Videos_*.zip gói theo "video/" (số ít) trong khi ta ghi vào
    # "videos/". So tên sẽ trượt đúng trường hợp này.
    if len(children) != 1 or not children[0].is_dir():
        return
    nested = children[0]
    print(f"  [flatten] bỏ lớp lồng thừa: {nested} -> {dest}")
    for child in nested.iterdir():
        target = dest / child.name
        if target.exists():
            print(f"    [bỏ qua] đã tồn tại: {target}")
            continue
        child.rename(target)
    nested.rmdir()


def verify_clip_slices(sliced_dir: Path, kaggle_mirror_npy_dir: str | None):
    """
    Đối chiếu vector vừa cắt với bộ Kaggle mirror đã xác nhận trước đó
    (aresusayhi/ai-challenge-2025, clip-features-32/{video_id}.npy).
    Cosine similarity phải ~1.0 (chỉ lệch do làm tròn) — nếu không, thứ tự
    toàn cục giả định ở trên SAI, cần tìm manifest/offset thật thay vì đoán.
    """
    if not kaggle_mirror_npy_dir:
        print("  [Bỏ qua verify] Chưa trỏ kaggle_mirror_npy_dir —")
        print("  TỰ RỦI RO nếu dùng thẳng kết quả trên.")
        return
    kdir = Path(kaggle_mirror_npy_dir)
    for f in sorted(sliced_dir.glob("*.npy")):
        vid = f.stem
        ref_path = kdir / f"{vid}.npy"
        if not ref_path.exists():
            print(f"  [{vid}] không có bản đối chiếu trong Kaggle mirror, bỏ qua")
            continue
        got = np.load(f).astype(np.float32)
        ref = np.load(ref_path).astype(np.float32)
        if got.shape != ref.shape:
            print(f"  [{vid}] LỆCH SHAPE: {got.shape} vs {ref.shape} — offset tính sai chắc chắn")
            continue
        denom = np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-8
        sim = (got * ref).sum(axis=1) / denom
        print(f"  [{vid}] cosine similarity — mean={sim.mean():.4f}, min={sim.min():.4f}")


def main():
    for sub in SUBFOLDERS:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / ".gitkeep").touch()

    print("Kiểm tra server có hỗ trợ Range request (bắt buộc để tải chọn lọc)...")
    if not check_range_support(f"{BASE_URL}/Keyframes_{GROUPS[0]}.zip"):
        print("[DỪNG] Server KHÔNG hỗ trợ Range — remotezip sẽ không hoạt động.")
        print("       Dùng wget/aria2c tải nguyên từng zip nhóm thay vì script này.")
        return
    print("  OK.\n")

    all_video_ids: dict[str, list[str]] = {}
    for g in GROUPS:
        vids = pick_videos_in_group(g, N_PER_GROUP)
        all_video_ids[g] = vids
        print(f"Nhóm {g}: chọn {vids}")
    flat_video_ids = [v for vids in all_video_ids.values() for v in vids]
    print(f"\nTổng {len(flat_video_ids)} video được chọn.\n")

    # ---- 1) Keyframes + Videos: chọn lọc theo từng nhóm ----
    for g, vids in all_video_ids.items():
        print(f"[{g}] Trích keyframes (chọn lọc)...")
        extract_selected(f"{BASE_URL}/Keyframes_{g}.zip", vids, OUT / "keyframes", "keyframes")
        print(f"[{g}] Trích video gốc mp4 (chọn lọc)...")
        for vurl in video_zip_urls(g):
            extract_selected(vurl, vids, OUT / "videos", f"videos/{Path(vurl).name}")

    # ---- 2) Objects: zip nặng nhất cho cả batch -> BẮT BUỘC chọn lọc, không tải nguyên ----
    print("[objects] Trích chọn lọc (zip này nặng nhất, ước tính ~900MB-1GB toàn batch)...")
    extract_selected(f"{BASE_URL}/objects-aic25-b1.zip", flat_video_ids, OUT / "objects", "objects")

    # ---- 3) map-keyframes: tải nguyên (nhỏ), CẦN GIỮ LẠI keyframe_count của TOÀN BỘ video ----
    #     (không chỉ video đã chọn) để tính offset đúng cho bước cắt CLIP feature ở dưới.
    print("[map-keyframes] Tải nguyên + lọc cục bộ...")
    mapkf_bytes = fetch_zip_bytes("map-keyframes-aic25-b1.zip")
    filter_zip_bytes(mapkf_bytes, "map-keyframes", flat_video_ids, OUT / "map-keyframes")

    # Đếm keyframe cho TOÀN BỘ video trong batch (cần cho việc cắt CLIP đúng offset).
    # Dùng lại bytes đã tải ở trên — không tải lần hai.
    print("  Đếm keyframe toàn batch để tính offset CLIP...")
    keyframe_counts: dict[str, int] = {}
    all_video_ids_global_order: list[str] = []
    with zipfile.ZipFile(io.BytesIO(mapkf_bytes)) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".csv"):
                continue
            vid = Path(name).stem
            with zf.open(name) as f:
                df = pd.read_csv(f)
            keyframe_counts[vid] = len(df)
            all_video_ids_global_order.append(vid)
    print(f"  Đã đếm keyframe cho {len(all_video_ids_global_order)} video trong batch.")

    # ---- 4) media-info: tải nguyên (nhỏ), lọc cục bộ ----
    print("[media-info] Tải nguyên + lọc cục bộ...")
    filter_zip_bytes(
        fetch_zip_bytes("media-info-aic25-b1.zip"), "media-info", flat_video_ids, OUT / "media-info"
    )

    # ---- 5) clip-features: trích thẳng .npy của từng video ----
    print("[clip-features] Trích .npy theo từng video...")
    extract_clip_features(flat_video_ids, OUT / "clip-features")

    # ---- 6) Verify — BẮT BUỘC trước khi tin dùng phần clip-features ----
    # Phép kiểm PHỤ, chỉ để tham khảo: đối chiếu với bộ sample 2025 ở
    # data/sample/clip-features/ (chứa L21_V001, L23_V001 — trùng nhóm đang lấy).
    #
    # Không dùng nó để ra quyết định: bên đối chiếu là DỮ LIỆU NĂM KHÁC, nên
    # cosine thấp không phân biệt được "offset sai" với "hai năm vốn khác nhau".
    # Đề bài nói batch 1 năm nay chính là dữ liệu năm ngoái, nhưng đó là điều
    # cần kiểm chứ không phải điều được giả định khi đang đi kiểm thứ khác.
    #
    # Phép kiểm CHÍNH là scripts/verify_clip_slices.py — khép kín trong dữ liệu
    # năm nay (encode lại chính ảnh vừa tải, so với vector vừa cắt), nên kết
    # luận dứt khoát. BẮT BUỘC chạy nó trước khi upload.
    print("\n[verify-phụ] Đối chiếu tham khảo với bộ sample 2025...")
    verify_clip_slices(OUT / "clip-features", kaggle_mirror_npy_dir="data/sample/clip-features")
    print("\n[verify-chính] Chạy: python scripts/verify_clip_slices.py")
    print("  (khép kín trong dữ liệu năm nay — đừng upload trước khi nó báo KHỚP)")

    # ---- 6b) Làm phẳng lớp thư mục thừa do prefix trong zip ----
    print("\n[flatten] Chuẩn hoá cấu trúc thư mục cho khớp config.paths...")
    for sub in SUBFOLDERS:
        flatten_redundant_nesting(OUT / sub)

    # ---- 7) manifests/{member}.json ----
    #
    # Tên file có tên người, KHÔNG phải "manifest.json" dùng chung. Cả 5 người
    # upload vào cùng path `data/sample_2026/` trên HF, nên một file tên cố định
    # sẽ bị người upload sau GHI ĐÈ — và manifest lại chính là bản ghi duy nhất
    # cho biết video nào của ai. Mất nó là mất luôn khả năng tải đúng phần của
    # mình về, và mất luôn phép kiểm "viết query trên video người khác".
    manifest = {
        "source": f"{BASE_URL} (BTC AIC 2026, batch 1)",
        # eval/merge_queryset.py đọc field này để bắt lỗi "viết query trên video
        # của người khác" lúc gộp bộ test 5 người.
        "member": MEMBER,
        "groups": GROUPS,
        "n_per_group": N_PER_GROUP,
        "videos": flat_video_ids,
        "n_videos": len(flat_video_ids),
    }
    manifest_dir = OUT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / f"{MEMBER}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[manifest] {manifest_dir / f'{MEMBER}.json'} — {len(flat_video_ids)} video")

    print(f"\nHOÀN TẤT. Dữ liệu ở {OUT.resolve()}")
    print("QUAN TRỌNG: đọc kỹ log [verify] ở trên trước khi dùng phần clip-features/ —")
    print("nếu cosine similarity không gần 1.0, offset toàn cục đang tính SAI, đừng dùng vội.")


if __name__ == "__main__":
    main()
