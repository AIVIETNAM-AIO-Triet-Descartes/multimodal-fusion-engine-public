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

Khi cập nhật từ Git, đồng bộ các file nguồn trong thư mục này vào
`/home/ubuntu/web`, giữ lại các file credentials và session của máy chủ.
