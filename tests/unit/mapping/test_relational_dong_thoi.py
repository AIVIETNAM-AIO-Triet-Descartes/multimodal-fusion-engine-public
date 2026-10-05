"""`RelationalMap` phải chịu được NHIỀU LUỒNG cùng đọc — bàn nộp chung kết.

Vì sao có file này: bản cũ dùng **một** connection dùng chung với
`check_same_thread=False`, và docstring tự biện minh *"an toàn vì mọi lối dùng
đều tuần tự — Streamlit chạy một lần vẽ tại một thời điểm cho mỗi phiên"*.

Câu đó đúng cho MỘT phiên. `@st.cache_resource` chia một đối tượng cho MỌI
phiên, còn Streamlit chạy mỗi phiên trong một luồng riêng — nên 5 người cùng
truy vấn là 5 luồng dùng chung một connection. Chung kết cần đúng 5 người.

Test này chạy 5 luồng thật, vì một giả định về đồng thời chỉ kiểm được bằng
cách vi phạm nó.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.relational import RelationalMap

SO_LUONG = 5          # đúng số người ngồi bàn nộp
SO_LUOT = 40


@pytest.fixture
def db_file(tmp_path):
    """DB trên ĐĨA, không phải `:memory:` — xem `_la_bo_nho`."""
    db = RelationalMap(str(tmp_path / "map.db"))
    db.init_schema()
    db.upsert_video(VideoRow(
        video_id="L21_V001", source_path="fake.mp4", fps=25.0,
        duration_ms=10_000, total_frames=250, width=1280, height=720))
    db.upsert_keyframes([
        KeyframeRow(keyframe_id=f"L21_V001#{i:06d}", video_id="L21_V001",
                    keyframe_index=i, frame_idx=i, pts_time=i / 25.0,
                    timestamp_ms=int(i / 25.0 * 1000), shot_id="shot_1")
        for i in range(1, 51)
    ])
    yield db
    db.close()


class TestNhieuLuongCungDoc:
    def test_nam_luong_doc_khong_nem(self, db_file):
        """Bản cũ ném `ProgrammingError` hoặc trả sai ở đây."""
        loi: list[BaseException] = []

        def doc(i: int):
            try:
                kid = f"L21_V001#{(i % 50) + 1:06d}"
                row = db_file.get_keyframe(kid)   # NÉM nếu không thấy
                assert row.keyframe_id == kid
            except BaseException as e:      # noqa: BLE001
                loi.append(e)

        with ThreadPoolExecutor(max_workers=SO_LUONG) as ex:
            list(ex.map(doc, range(SO_LUOT * SO_LUONG)))
        assert not loi, f"{len(loi)} lỗi, ví dụ: {loi[0]!r}"

    def test_moi_luong_MOT_connection_rieng(self, db_file):
        """Mỗi luồng phải thấy một object connection khác nhau."""
        thay: dict[int, int] = {}
        khoa = threading.Lock()

        def lay(_i: int):
            conn = db_file._conn
            with khoa:
                thay[threading.get_ident()] = id(conn)

        with ThreadPoolExecutor(max_workers=SO_LUONG) as ex:
            list(ex.map(lay, range(SO_LUONG * 4)))
        # Mỗi luồng một id connection riêng; không luồng nào dùng chung.
        assert len(set(thay.values())) == len(thay), thay


class TestBoNhoTrongVanChungConnection:
    """`:memory:` KHÔNG chia sẻ được giữa các connection.

    Chia theo luồng ở đây sẽ cho luồng thứ hai một DB **rỗng** — không có cả
    bảng, vì `init_schema` chạy ở luồng chính. Mọi truy vấn ném
    `OperationalError: no such table`, tức ồn ào ở một chỗ không liên quan gì
    tới nguyên nhân. Nên bộ nhớ trong phải giữ MỘT connection dùng chung.
    """

    def test_memory_dung_chung_mot_connection(self):
        db = RelationalMap(":memory:")
        try:
            db.init_schema()
            ra: list[int] = []
            with ThreadPoolExecutor(max_workers=3) as ex:
                list(ex.map(lambda _: ra.append(id(db._conn)), range(6)))
            assert len(set(ra)) == 1, "`:memory:` phải dùng chung một connection"
        finally:
            db.close()

    def test_memory_luong_khac_van_thay_du_lieu(self):
        """Phép thử thật sự: luồng khác có đọc được thứ luồng chính đã ghi không.

        `get_video` NÉM khi không thấy, nên luồng thấy DB rỗng sẽ làm test đỏ.
        """
        db = RelationalMap(":memory:")
        try:
            db.init_schema()
            db.upsert_video(VideoRow(
                video_id="L21_V001", source_path="f.mp4", fps=25.0,
                duration_ms=1000, total_frames=25, width=640, height=480))
            ra: list[str] = []
            loi: list[BaseException] = []

            def doc(_i):
                try:
                    ra.append(db.get_video("L21_V001").video_id)
                except BaseException as e:   # noqa: BLE001
                    loi.append(e)

            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(doc, range(4)))
            assert not loi, f"luồng khác thấy DB rỗng: {loi[0]!r}"
            assert ra == ["L21_V001"] * 4
        finally:
            db.close()


class TestDong:
    def test_close_dong_connection_cua_MOI_luong(self, tmp_path):
        """Windows: connection còn mở là còn khoá file."""
        db = RelationalMap(str(tmp_path / "m.db"))
        db.init_schema()
        with ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(lambda _: db._conn.execute("SELECT 1").fetchone(), range(6)))
        assert len(db._da_mo) >= 2
        db.close()
        assert db._da_mo == []
        for conn in []:
            conn.close()

    def test_close_hai_lan_khong_nem(self, tmp_path):
        db = RelationalMap(str(tmp_path / "m.db"))
        db.init_schema()
        db.close()
        db.close()

    def test_dung_lai_sau_close_mo_connection_moi(self, tmp_path):
        """`close()` không được làm đối tượng chết vĩnh viễn."""
        db = RelationalMap(str(tmp_path / "m.db"))
        db.init_schema()
        db.close()
        assert db._conn.execute("SELECT 1").fetchone()[0] == 1
        db.close()


class TestBusyTimeout:
    def test_co_dat_busy_timeout(self, tmp_path):
        """Mặc định sqlite3 là 0 — hỏng NGAY khi gặp khoá ghi."""
        db = RelationalMap(str(tmp_path / "m.db"))
        try:
            db.init_schema()
            (giay,) = db._conn.execute("PRAGMA busy_timeout;").fetchone()
            assert giay == RelationalMap.BUSY_TIMEOUT_MS
        finally:
            db.close()

    def test_foreign_keys_van_bat_tren_connection_moi(self, tmp_path):
        """Mỗi connection mới phải tự bật lại PRAGMA — nó KHÔNG kế thừa."""
        db = RelationalMap(str(tmp_path / "m.db"))
        try:
            db.init_schema()
            ra: list[int] = []
            with ThreadPoolExecutor(max_workers=3) as ex:
                list(ex.map(
                    lambda _: ra.append(
                        db._conn.execute("PRAGMA foreign_keys;").fetchone()[0]),
                    range(6)))
            assert all(v == 1 for v in ra), ra
        finally:
            db.close()


def test_sqlite3_van_la_backend(tmp_path):
    """Chốt kiểu, để đổi backend là test này đỏ chứ không trôi im lặng."""
    db = RelationalMap(str(tmp_path / "m.db"))
    try:
        assert isinstance(db._conn, sqlite3.Connection)
    finally:
        db.close()
