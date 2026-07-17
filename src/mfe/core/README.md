# core/ — infra dùng chung

**Owner:** Lead. **Spec:** `docs/01-lead-coordination-infra.md` §3.

## Viết gì vào đây

Thứ cả 4 workstream đều dùng và không thuộc về ai: load config, đọc/ghi bytes, ghi log có cấu trúc.

| Module | Vai trò |
|---|---|
| `config.py` | YAML → object đã validate. Thiếu key = lỗi lúc load, không phải lúc dùng |
| `storage.py` | `read_bytes` / `write_bytes` — seam local ↔ Modal (docs/01 §3.2) |
| `backends/local.py` | Sprint 1: hiện thực duy nhất |
| `backends/modal.py` | Stub. Fail to, không âm thầm fallback về local |
| `logging.py` | Event stream → Lead tính TTCA (docs/04 §5) |

## Đừng viết gì vào đây

- **Logic của workstream nào.** Nếu nó chỉ TV2 dùng, nó thuộc `stores/`.
- **Contract/schema.** Đó là `contracts/`. `core/` *dùng* contract, không định nghĩa.

## Luật quan trọng nhất của package này

**Không ai ngoài `core/backends/` được gọi `open()` hay `Path.write_bytes()` lên đường dẫn store.** Nếu TV1 ghi keyframe thẳng bằng `Path.write_bytes`, thì lúc chuyển Modal phải sửa code TV1 — đúng thứ mà lớp trừu tượng này sinh ra để chặn (docs/01 §3.2: "đổi từ local sang Modal chỉ bằng đổi config, không sửa logic").

## Thêm backend mới

Thêm file vào `backends/`, đăng ký trong `get_backend()`. Không caller nào phải đổi.
