# infra/

**Owner:** Vỹ Triết. **Spec:** `docs/01-lead-coordination-infra.md` §3.

| Thư mục | Trạng thái |
|---|---|
| `modal/` | Stub. Sprint sau |

## Sprint 1 không cần gì ở đây

docs/00 §1 xếp "Full dataset trên Modal" và "Scale-out, GPU orchestration thật" **ngoài sprint**. Chạy local, một máy, một lệnh.

## Phân biệt với `src/mfe/core/backends/`

| Chỗ | Gì |
|---|---|
| `src/mfe/core/backends/modal.py` | **code** — hiện thực `read_bytes`/`write_bytes` trên volume |
| `infra/modal/` | **deploy** — app definition, image, mount, GPU config |

Code biết cách đọc/ghi. Infra biết chạy nó ở đâu. Đừng trộn.

## Nếu sau này cần local infra (docker-compose cho Postgres...)

Thêm `infra/local/`. Cây không vỡ.
