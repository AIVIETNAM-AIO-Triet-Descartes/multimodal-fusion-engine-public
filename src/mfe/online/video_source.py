"""`video_id` -> chỗ phát được video gốc, cho việc tua tay trên bàn nộp.

Vì sao cần một mô-đun riêng cho một việc nhỏ thế này
-----------------------------------------------------
Vì nó **không** nhỏ ở chỗ ai cũng tưởng. Batch 1 là ~28 GB video và kho này cố ý
không tải chúng về (`config/btc_b1.yaml`: `videos:` khai để `Config` hợp lệ,
thư mục để rỗng). Nên "đường dẫn video" ở mỗi máy một khác: người có ổ cứng chép
sẵn, người trỏ vào một URL dùng chung, người chỉ có đúng vài video mẫu.

Bản của Bảo cắm cứng năm đường dẫn tuyệt đối trên máy anh ấy
(`D:/AIC/SOTA_V2_AIC/data/videos`, …). Chạy được đúng một máy, và khi không tìm
thấy thì chỉ hiện "Thiếu file" — người dùng không biết phải đặt cái gì ở đâu.

Hai chế độ, và chúng KHÔNG thay thế nhau
-----------------------------------------
* **URL** (`MFE_VIDEO_BASE_URL`) — trình duyệt tải thẳng, nên tua được mượt và
  không tốn đĩa. Đây là chế độ dùng được cho cả 873 video.
* **File trên đĩa** — Streamlit tự phục vụ qua `st.video`, nhưng khi ấy khung
  `<video>` nằm trong iframe của ta **không** đọc được đường dẫn `file://` (trình
  duyệt chặn), nên chế độ này mất phần đọc số khung trực tiếp.

`kind` nói rõ mình đang ở chế độ nào để phần giao diện chọn đúng trình phát, thay
vì đoán theo hình dạng chuỗi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: URL gốc, video nằm ở `{base}/{video_id}.mp4`. Có thể kèm query (SAS token của
#: Azure chẳng hạn): `https://x.blob.core.windows.net/videos?sv=…` — phần sau `?`
#: được nối lại SAU tên file, vì blob store đọc chữ ký trên đường dẫn đầy đủ.
ENV_BASE_URL = "MFE_VIDEO_BASE_URL"

#: Thư mục chứa `{video_id}.mp4`. Nhiều thư mục thì ngăn bằng `os.pathsep`.
ENV_DIR = "MFE_VIDEO_DIR"

#: Nơi tìm cuối cùng khi không đặt biến môi trường nào. Thư mục đầu là chỗ
#: `config/btc_b1.yaml` khai; thư mục sau là bộ video mẫu có sẵn trong repo.
FALLBACK_DIRS = (Path("data/btc_b1/videos"), Path("data/sample/videos"))


@dataclass(frozen=True)
class VideoSource:
    """Chỗ lấy được video, kèm **cách** lấy."""

    video_id: str
    #: `"url"` hoặc `"file"`.
    kind: str
    #: URL đầy đủ, hoặc đường dẫn tuyệt đối trên đĩa.
    location: str

    @property
    def is_url(self) -> bool:
        return self.kind == "url"


def _from_base_url(base: str, video_id: str) -> str:
    """`{base}/{video_id}.mp4`, giữ nguyên query nếu base có.

    Nối sai chỗ là hỏng im lặng với blob store: `…/videos?sig=abc/L21_V001.mp4`
    trả 404 hoặc 403 chứ không báo rằng URL bị ghép ngược.
    """
    base = base.strip()
    if "?" in base:
        root, query = base.split("?", 1)
        return f"{root.rstrip('/')}/{video_id}.mp4?{query}"
    return f"{base.rstrip('/')}/{video_id}.mp4"


def search_dirs() -> list[Path]:
    """Các thư mục sẽ dò, theo đúng thứ tự ưu tiên."""
    raw = os.environ.get(ENV_DIR, "").strip()
    dirs = [Path(p) for p in raw.split(os.pathsep) if p.strip()] if raw else []
    return dirs + list(FALLBACK_DIRS)


def resolve_local_video(video_id: str) -> VideoSource | None:
    """Find explicitly configured/local source files without testing a remote URL."""
    for folder in search_dirs():
        candidate = folder / f"{video_id}.mp4"
        if candidate.is_file():
            return VideoSource(video_id, "file", str(candidate.resolve()))
    return None


def resolve_video(video_id: str) -> VideoSource | None:
    """Chỗ phát được video này, hoặc `None` nếu máy này không có.

    URL đi trước vì nó là chế độ duy nhất phủ hết 873 video, và vì nó không đòi
    phải kiểm tồn tại — một lượt HEAD cho mỗi ô ứng viên là thứ giao diện không
    trả nổi.
    """
    base = os.environ.get(ENV_BASE_URL, "").strip()
    if base:
        return VideoSource(video_id, "url", _from_base_url(base, video_id))
    for folder in search_dirs():
        path = folder / f"{video_id}.mp4"
        if path.is_file():
            return VideoSource(video_id, "file", str(path.resolve()))
    return None


def why_missing(video_id: str) -> str:
    """Câu giải thích đưa thẳng lên giao diện khi `resolve_video` trả `None`.

    Nói ra **phải đặt gì ở đâu**. Một dòng "thiếu file" không nói điều đó, và
    người vận hành sẽ đi hỏi thay vì tự sửa được trong mười giây.
    """
    folders = ", ".join(f"`{d}`" for d in search_dirs())
    return (
        f"Không tìm thấy `{video_id}.mp4`. Đặt **`{ENV_BASE_URL}`** trong `.env` "
        f"(URL gốc chứa video, kèm SAS token nếu có) — đây là cách duy nhất phủ "
        f"hết 873 video. Hoặc chép file vào một trong: {folders}, hoặc trỏ "
        f"**`{ENV_DIR}`** tới thư mục chứa chúng."
    )
