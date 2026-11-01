# HTML web runtime

Đây là bản nguồn của website HTML đang được triển khai tại `web/`.
Website dùng `config/btc_b12.yaml` và kho runtime trên `/data`, giữ Streamlit
ở port 8502 và phục vụ HTML ở port 8501.

Các file `accounts.json`, `CREDENTIALS.txt` và `.session_secret` là dữ liệu
vận hành, không đưa vào Git. Tạo tài khoản cố định trên máy chạy web bằng:

```bash
cd web
.venv/bin/python create_accounts.py
```

Khởi động sau khi đã có runtime bundle:

```bash
cd web
./start_html.sh
```

Sau khi đăng nhập, mục **Cấu hình các nhánh** cho phép bật/tắt arm và đổi
trọng số theo từng job. Mỗi lane có nút QUEST riêng: KIS hiện bản dịch/mở rộng,
QA hiện search hint/OCR/logic question, còn TRAKE hiện các mệnh đề để sửa và
sắp xếp trước khi tìm. Các lời gọi QUEST và QA VLM dùng `TRAMMAY_API_KEY` qua
`MFE_TRAMMAY_BASE_URL`.

QA hybrid gom năm frame liên tục quanh ứng viên đầu từ relational map, sau đó
thêm các ứng viên rải cho đủ tối đa 15 ảnh. Caption dài bị giới hạn 200 ký tự
trong prompt; OCR và phụ đề được ưu tiên. Công tắc **Gemini caption rerank** ở
mỗi lane mặc định bật để ưu tiên độ chính xác; có thể tắt khi cần chạy nhanh.
Khi bật, caption/OCR/phụ đề của top ứng viên được gửi qua Trammay rồi ghi lại
trạng thái rerank trong kết quả.

Runtime giữ cache kết quả truy hồi trong RAM (mặc định 128 truy vấn), cùng cache
phản hồi QUEST và thứ tự rerank Trammay. Hai nhánh dịch và mở rộng của KIS được
gọi song song. Mặc định vẫn là `ag/gemini-3.8-flash-high`; trong phần tuỳ chọn
của từng lane có thể chọn `high` (chất lượng cao), `medium` (cân bằng) hoặc
`low` (nhanh). Lựa chọn được áp dụng cho QUEST, Gemini caption rerank và VLM
đáp án QA, đồng thời được đưa vào cache key để các mức không dùng nhầm kết quả
của nhau.

Tab **Đề xuất** lưu ghi chú dùng chung cho mọi tài khoản đang đăng nhập. Nút
**Nộp DRES** chỉ xuất hiện trên kết quả đã chạy; backend đọc `user_dres` /
`pass_dres`, đồng thời chấp nhận tên biến cũ `user_dress` / `pass_dress`. Session
DRES không được gửi xuống trình duyệt. Có thể đặt `MFE_DRES_EVALUATION_ID` để
chọn evaluation mặc định, hoặc đọc danh sách evaluation ACTIVE bằng nút làm
mới cạnh hộp chọn. Ghi chú được lưu ở `web/suggestions.json` và
được giới hạn 200 mục gần nhất.

Khu vực **Nộp KIS / QA / TRAKE thủ công** cho phép nhập trực tiếp mã video,
thời điểm hoặc frame, câu trả lời QA và danh sách frame TRAKE mà không cần chạy
QUEST hay truy hồi trước. Các ô KIS, QA và TRAKE trong từng lane cũng nhận nhập
tay; QUEST chỉ điền giá trị gợi ý để người dùng sửa lại.

TRAKE cũng tự tách ở backend khi dán nguyên văn các dòng `E1: ...`, `E2: ...`.
Câu mô tả đứng trước các dòng được dùng làm ngữ cảnh chung cho từng event;
thứ tự dòng được giữ nguyên khi gửi sang DANTE.

Sau khi TRAKE có kết quả, khu vực **Sửa riêng từng event TRAKE** cho phép sửa
một mệnh đề rồi chạy lại cả chuỗi với các event còn lại được giữ nguyên. Nếu
mệnh đề đúng nhưng frame được chọn sai, mở video ngay tại event đó và bấm
**Chốt frame đang xem**; frame đã chốt sẽ được dùng trong payload DRES của chuỗi.

Với TRAKE, nút đề xuất ở đầu mỗi chuỗi lưu toàn bộ các event; các nút trong
từng event vẫn cho phép đánh dấu riêng một frame để kiểm tra.

Khi mở video từ một kết quả, có thể tua tới đúng thời điểm rồi bấm **Chốt
frame đang xem**. Frame được giữ theo đúng lane và kết quả/event đang mở, hiển
thị lại bằng dấu `📌`, được dùng khi tạo đề xuất và được gửi cùng payload DRES
nếu người vận hành nộp kết quả đó.

Bấm trực tiếp vào ảnh kết quả để mở ảnh phóng to; có thể đóng bằng nút **Đóng**,
bấm ra nền tối hoặc nhấn `Esc`.

Sau khi DRES nhận một đáp án, backend ghi fingerprint của **đúng payload đáp án**
vào `web/dres_submissions.json` (mode 600). Vì vậy một đáp án khác
cho cùng truy vấn vẫn có thể được nộp lại để sửa lượt sai, còn payload giống hệt
vẫn bị chặn ở DRES hoặc ở web. Verdict `CORRECT/WRONG/PENDING` hiện ngay sau khi
gửi.

Để bật player video, đặt `MFE_VIDEO_BASE_URL` trong `.env` thành URL container
Azure có SAS, ví dụ `https://...blob.core.windows.net/video?sp=...&sig=...`.
Backend sẽ nối `{video_id}.mp4` trước phần query SAS, yêu cầu đăng nhập ở
`/video/{video_id}`, rồi redirect sang Blob để trình duyệt hỗ trợ Range/seek.
Không ghi SAS vào Git hoặc HTML.

Khi cập nhật từ Git, đồng bộ các file nguồn trong thư mục này vào
`web/`, giữ lại các file credentials và session của máy chủ.
