"""Lấy ảnh keyframe về máy, có cache — cho rerank nhìn pixel và cho người xem.

Hai tầng rerank nhìn ảnh (`colpali`, `vlm`) và mọi giao diện đều cần đúng một
thứ: `keyframe_id -> đường dẫn ảnh`. Ảnh nằm trên Modal Volume, nên hàm này là
chỗ duy nhất biết cách đi lấy, và nó cache lại để lần sau khỏi đi.

Vì sao cần một bước dịch
-------------------------
`keyframe_id` mang **`frame_idx`** (toạ độ nộp bài), còn tên file ảnh là **`n`**
(số thứ tự trong video). Hai số khác nhau — `L21_V001#031124` là frame 31.124,
tình cờ là keyframe thứ 250 — và `map-keyframes` là chỗ duy nhất nối chúng.
Nhầm hai số này là cách dễ nhất để nhìn nhầm ảnh mà vẫn tin mình nhìn đúng.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path(".cache/keyframes")
DEFAULT_MAPS = Path("data/btc_b1/map-keyframes")
DEFAULT_BUNDLE = "aic_btc_b1_v1"
DEFAULT_VOLUME = "keyframes-btc"


@lru_cache(maxsize=1024)
def ordinal_for(video_id: str, frame_idx: int, maps: Path = DEFAULT_MAPS) -> int | None:
    """`n` của keyframe mang `frame_idx` này, hoặc None nếu CSV không có dòng nào.

    Trả None chứ không đoán: `round(frame_idx / fps)` sai ở đúng những chỗ khó,
    và một ảnh sai trông y hệt một ảnh đúng.
    """
    path = maps / f"{video_id}.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame_idx"]) == frame_idx:
                return int(row["n"])
    return None


class KeyframeImages:
    """`keyframe_id -> Path`, tải từ Modal Volume một lần rồi dùng bản cache.

    Transport được tạo **lười**: phần lớn lượt chạy chỉ đụng ảnh đã có sẵn trong
    cache, và dựng client Modal cho một lượt như thế là trả tiền cho một kết nối
    không dùng đến.
    """

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE,
        *,
        maps: Path | str = DEFAULT_MAPS,
        bundle: str = DEFAULT_BUNDLE,
        volume: str = DEFAULT_VOLUME,
        offline: bool = False,
        image_roots: dict[str, Path | str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.maps = Path(maps)
        self.bundle = bundle
        self.volume = volume
        #: True = chỉ đọc cache, không bao giờ ra mạng. Dùng khi chạy không có token.
        self.offline = offline
        # Optional local bundle roots.  The combined B1+B2 runtime keeps the
        # raw keyframes on /data, so serving an image must not first copy it into
        # the old three-digit cache layout.  The map still remains the only
        # source of truth for frame_idx -> ordinal.
        self.image_roots = {
            video: Path(root) for video, root in (image_roots or {}).items()
        }
        self._transport = None
        self._layout = None
        self.misses: list[str] = []

    def _remote(self):
        if self._transport is None:
            from mfe.offline.bundle import BundleLayout  # noqa: PLC0415
            from mfe.offline.transport import (  # noqa: PLC0415
                build_transport,
                ensure_modal_credentials,
            )

            ensure_modal_credentials(".env")
            self._layout = BundleLayout(self.bundle)
            self._transport = build_transport(
                "modal", volume=self.volume, local_root="bundle_local", version=2
            )
        return self._transport, self._layout

    def local_path(self, keyframe_id: str) -> Path | None:
        """Đường dẫn ảnh sẽ nằm ở đâu, chưa cần nó tồn tại."""
        video_id, _, frame = keyframe_id.partition("#")
        if not frame.isdigit():
            return None
        ordinal = ordinal_for(video_id, int(frame), self.maps)
        if ordinal is None:
            return None
        direct_root = self.image_roots.get(video_id)
        if direct_root is not None:
            for filename in (f"{ordinal:03d}.jpg", f"{ordinal:04d}.jpg"):
                candidate = direct_root / video_id / filename
                if candidate.is_file():
                    return candidate
        width = 4 if ordinal >= 1000 else 3
        return self.cache_dir / video_id / f"{ordinal:0{width}d}.jpg"

    def get(self, keyframe_id: str) -> Path | None:
        """Ảnh của một keyframe. Tải nếu chưa có; None nếu không lấy được."""
        target = self.local_path(keyframe_id)
        if target is None:
            self.misses.append(keyframe_id)
            return None
        if target.is_file() and target.stat().st_size > 0:
            return target
        if self.offline:
            return None

        video_id = keyframe_id.split("#")[0]
        transport, layout = self._remote()
        target.parent.mkdir(parents=True, exist_ok=True)
        remote = f"{layout.keyframes_dir(video_id)}/{target.name}"
        try:
            transport.get(remote, target)
        except Exception as err:  # noqa: BLE001 - anh thieu khong duoc lam do ca luot
            logger.warning("%s: khong tai duoc anh (%s)", keyframe_id, err)
            self.misses.append(keyframe_id)
            target.unlink(missing_ok=True)
            return None
        return target if target.is_file() and target.stat().st_size > 0 else None

    def cached(self, keyframe_id: str) -> Path | None:
        """Ảnh **đã có trên đĩa**, hoặc `None`. KHÔNG bao giờ ra mạng.

        Dùng sau `prefetch`. `get` tải khi thiếu, và khi nó nằm trong một vòng
        lặp dựng giao diện thì mỗi ảnh thiếu là một lượt tải **tuần tự** chèn
        vào giữa việc vẽ trang — đúng thứ `prefetch` vừa bỏ công song song hoá
        để tránh. Ở đây thiếu ảnh thì ô đó trống, và đó là lựa chọn đúng: người
        vận hành cần trang hiện ra ngay, không cần đủ 100 ảnh.
        """
        target = self.local_path(keyframe_id)
        if target is None or not target.is_file() or target.stat().st_size == 0:
            return None
        return target

    def is_allowed_local_path(self, path: Path) -> bool:
        """Return whether ``path`` belongs to one of the configured image roots."""
        try:
            resolved = path.resolve()
            roots = [
                self.cache_dir.resolve(),
                *(root.resolve() for root in self.image_roots.values()),
            ]
            return any(resolved == root or root in resolved.parents for root in roots)
        except OSError:
            return False

    def prefetch(self, keyframe_ids: Iterable[str], workers: int = 8) -> int:
        """Tải song song. Trả về số ảnh sẵn sàng.

        Rerank nhìn ảnh cần *tất cả* ảnh trước khi bắt đầu, và tải tuần tự 50 ảnh
        qua Modal mất hàng chục giây — gần bằng chính phép rerank.
        """
        wanted = list(keyframe_ids)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self.get, wanted))
        return sum(1 for r in results if r is not None)

    def __call__(self, keyframe_id: str) -> Path | None:
        return self.get(keyframe_id)
