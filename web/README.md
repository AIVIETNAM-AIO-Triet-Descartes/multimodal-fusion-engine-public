# HTML web runtime

Đây là bản nguồn của website HTML đang được triển khai tại `/home/ubuntu/web`.
Website dùng `config/btc_b12.yaml` và kho runtime trên `/data`, giữ Streamlit
ở port 8502 và phục vụ HTML ở port 8501.

Các file `accounts.json`, `CREDENTIALS.txt` và `.session_secret` là dữ liệu
vận hành, không đưa vào Git. Tạo tài khoản cố định trên máy chạy web bằng:

```bash
cd /home/ubuntu/web
.venv/bin/python create_accounts.py
```

Khởi động sau khi đã có runtime bundle:

```bash
cd /home/ubuntu/web
./start_html.sh
```

Sau khi đăng nhập, mục **Cấu hình các nhánh** cho phép bật/tắt arm và đổi
trọng số theo từng job. Mỗi lane có nút QUEST riêng: KIS hiện bản dịch/mở rộng,
QA hiện search hint/OCR/logic question, còn TRAKE hiện các mệnh đề để sửa và
sắp xếp trước khi tìm. Các lời gọi QUEST và QA VLM dùng `TRAMMAY_API_KEY` qua
`llm-gateway.invalid`.

Để bật player video, đặt `MFE_VIDEO_BASE_URL` trong `.env` thành URL container
Azure có SAS, ví dụ `https://...blob.core.windows.net/video?sp=...&sig=...`.
Backend sẽ nối `{video_id}.mp4` trước phần query SAS, yêu cầu đăng nhập ở
`/video/{video_id}`, rồi redirect sang Blob để trình duyệt hỗ trợ Range/seek.
Không ghi SAS vào Git hoặc HTML.

Khi cập nhật từ Git, đồng bộ các file nguồn trong thư mục này vào
`/home/ubuntu/web`, giữ lại các file credentials và session của máy chủ.
