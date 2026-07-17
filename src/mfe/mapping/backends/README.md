# mapping/backends/

**Owner:** TV4. **Spec:** `docs/05-validator-mapping.md` §1.

## Vì sao tách backend từ ngày 1

docs/05 §1 nói SQLite "đủ cho sprint, đổi sang Postgres khi scale". Tách sẵn nghĩa là lúc đổi, diff nằm gọn trong thư mục này — `relational.py`, `validator.py`, `submit.py` không đụng tới.

| File | Trạng thái |
|---|---|
| `sqlite.py` | Sprint 1 — hiện thực duy nhất |
| `postgres.py` | Chưa có. Sprint scale-out thêm vào đây |

## Thêm Postgres sau

1. Thêm `postgres.py` cùng interface với `sqlite.py`.
2. Thêm key chọn backend vào `config/*.yaml`.
3. `relational.py` dispatch theo config.

Giữ `schema.sql` trung tính đủ để chuyển gần như cơ học — tránh syntax riêng SQLite trừ khi có lý do.

## Bẫy SQLite

`PRAGMA foreign_keys = ON` **mỗi connection**. SQLite mặc định **tắt** FK. Không bật thì `REFERENCES` trong `schema.sql` chỉ là trang trí, và một keyframe trỏ vào video không tồn tại vẫn insert được — đúng thứ P0 này sinh ra để chặn.
