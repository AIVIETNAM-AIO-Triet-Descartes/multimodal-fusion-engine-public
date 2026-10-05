# mapping/backends/

**Owner:** Nhật Bảo. **Spec:** `docs/05-validator-mapping.md` §1.

## Trạng thái Sprint 1

`RelationalMap` (`../relational.py`) **inline sqlite3 trực tiếp** — chưa dùng lớp backend riêng. Thư mục này để **dành chỗ** cho lúc thêm Postgres.

docs/05 §1: SQLite "đủ cho sprint, đổi sang Postgres khi scale".

## Thêm Postgres sau

1. Trích phần connect/execute của `relational.py` thành interface `SqliteBackend`, đặt ở đây.
2. Thêm `postgres.py` cùng interface.
3. Thêm key chọn backend vào `config/*.yaml`; `relational.py` dispatch theo config.

Giữ `schema.sql` trung tính đủ để chuyển gần như cơ học — tránh syntax riêng SQLite trừ khi có lý do.

## Bẫy SQLite (đã xử trong `relational.py`)

`PRAGMA foreign_keys = ON` **mỗi connection**. SQLite mặc định **tắt** FK. Không bật thì `REFERENCES` trong `schema.sql` chỉ là trang trí, và một keyframe trỏ vào video không tồn tại vẫn insert được — đúng thứ P0 này sinh ra để chặn. `relational.py.__init__` đã bật.
