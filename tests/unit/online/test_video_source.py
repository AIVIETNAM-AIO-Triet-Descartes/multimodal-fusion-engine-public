"""`video_id` -> chỗ phát video, và lý do khi không có."""

from __future__ import annotations

import os

import pytest

from mfe.online import video_source
from mfe.online.video_source import (
    ENV_BASE_URL,
    ENV_DIR,
    resolve_video,
    search_dirs,
    why_missing,
)


@pytest.fixture(autouse=True)
def _sach_moi_truong(monkeypatch: pytest.MonkeyPatch) -> None:
    """Không để biến môi trường của máy chạy test quyết định kết quả."""
    monkeypatch.delenv(ENV_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_DIR, raising=False)
    monkeypatch.setattr(video_source, "FALLBACK_DIRS", ())


class TestURL:
    def test_noi_duoi_ten_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_BASE_URL, "https://kho.example/videos")
        source = resolve_video("L21_V001")
        assert source is not None
        assert source.is_url
        assert source.location == "https://kho.example/videos/L21_V001.mp4"

    def test_giu_query_SAU_ten_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SAS token phải nằm sau `.mp4`, không phải giữa đường dẫn.

        Ghép ngược ra `…/videos?sig=abc/L21_V001.mp4` — blob store trả 403 chứ
        không nói rằng URL bị dựng sai, nên lỗi này chỉ lộ ra ở trình duyệt.
        """
        monkeypatch.setenv(ENV_BASE_URL, "https://kho.example/videos?sv=2024&sig=abc")
        source = resolve_video("L21_V001")
        assert source is not None
        assert source.location == (
            "https://kho.example/videos/L21_V001.mp4?sv=2024&sig=abc"
        )

    def test_bo_dau_gach_thua(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_BASE_URL, "https://kho.example/videos/")
        source = resolve_video("L22_V014")
        assert source is not None
        assert source.location == "https://kho.example/videos/L22_V014.mp4"

    def test_url_KHONG_kiem_ton_tai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Đặt URL là tin URL: một lượt HEAD cho mỗi ô ứng viên là quá đắt."""
        monkeypatch.setenv(ENV_BASE_URL, "https://kho.example/videos")
        assert resolve_video("KHONG_TON_TAI") is not None


class TestFile:
    def test_tim_thay_trong_thu_muc_khai_bang_env(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "L23_V001.mp4").write_bytes(b"x")
        monkeypatch.setenv(ENV_DIR, str(tmp_path))
        source = resolve_video("L23_V001")
        assert source is not None
        assert not source.is_url
        assert source.location.endswith("L23_V001.mp4")

    def test_nhieu_thu_muc_ngan_bang_pathsep(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mot, hai = tmp_path / "mot", tmp_path / "hai"
        mot.mkdir()
        hai.mkdir()
        (hai / "L25_V052.mp4").write_bytes(b"x")
        monkeypatch.setenv(ENV_DIR, f"{mot}{os.pathsep}{hai}")
        source = resolve_video("L25_V052")
        assert source is not None
        assert source.location.endswith("L25_V052.mp4")

    def test_khong_co_thi_tra_None(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_DIR, "/khong/co/thu/muc/nay")
        assert resolve_video("L21_V001") is None

    def test_URL_duoc_uu_tien_hon_file(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "L21_V001.mp4").write_bytes(b"x")
        monkeypatch.setenv(ENV_DIR, str(tmp_path))
        monkeypatch.setenv(ENV_BASE_URL, "https://kho.example/v")
        source = resolve_video("L21_V001")
        assert source is not None and source.is_url


class TestLyDoThieu:
    def test_noi_ro_phai_dat_gi_o_dau(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Một dòng "thiếu file" không sửa được gì; câu này phải sửa được."""
        monkeypatch.setenv(ENV_DIR, "/kho/cua/toi")
        why = why_missing("L21_V001")
        assert "L21_V001.mp4" in why
        assert ENV_BASE_URL in why
        assert ENV_DIR in why

    def test_liet_ke_dung_cac_thu_muc_se_do(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_DIR, "/kho/cua/toi")
        assert search_dirs()[0].as_posix().endswith("kho/cua/toi")
