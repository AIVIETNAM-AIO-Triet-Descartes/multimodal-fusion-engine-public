"""Prompt sinh caption + OCR cho Qwen2.5-VL. **DỮ LIỆU, không phải mã.**

Tách khỏi `stages.py` vì hai lý do, và lý do thứ hai mới là lý do thật:

1. `stages.py` đã hơn 2.000 dòng, và prompt là thứ người ta sửa nhiều nhất.
2. **Xuống dòng cho vừa 100 cột là đổi đúng chuỗi ký tự model nhận được.** Một
   prompt bị sửa vì lý do định dạng là thứ không ai nghĩ tới khi đi tìm nguyên
   nhân kết quả đổi. Nên file này được `pyproject.toml` tha `E501` — cùng cách
   `mfe/online/quest.py` đã được tha, vì cùng một lý do.

Hai biến thể cùng tồn tại, và cả hai đều được dùng:

| biến thể | dùng khi | tiếng | max_tokens |
|---|---|---|---|
| `bao` | lượt **probe** đối chiếu với bản Bảo đã chạy | Anh | 256 |
| `aic` | lượt chạy **thật** | Việt | 384 |

Giữ `bao` không phải để dự phòng: nó là phép kiểm **đường ống** duy nhất ta có.
`notebooks/reference/dense_captions_qwen.json` (855 keyframe) chỉ đối chiếu được
với chính prompt đã sinh ra nó. Đổi prompt rồi so với nó là so hai thứ khác nhau,
và khi lệch sẽ không biết lỗi nằm ở địa chỉ khoá, ở động cơ vLLM, hay ở prompt.
"""

from __future__ import annotations

#: Prompt cua Bao, CHEP NGUYEN VAN tu `notebooks/reference/caption.ipynb`.
#:
#: GIU LAI du bien the `aic` moi la cai dem di chay that. Ly do: ban chay cua Bao
#: (`notebooks/reference/dense_captions_qwen.json`, 855 keyframe cua L21_V001..V003)
#: la thu DUY NHAT doi chieu duoc, va no chi doi chieu duoc voi CHINH prompt da
#: sinh ra no. Chay bien the nay o luot probe la phep kiem **duong ong**: dia chi
#: khoa, dinh dang, dong co vLLM. Doi prompt roi so voi no la so hai thu khac nhau.
_QWEN25_PROMPT_BAO = (
    "Task 1: Describe all visual details, actions, and objects in this image in English.\n"
    "Task 2: Transcribe ALL visible text in this image exactly as written "
    "(including Vietnamese if any). If no text, write 'None'.\n\n"
    "Format your output EXACTLY like this:\n"
    "[CAPTION] (your detailed description here)\n"
    "[TEXT] (your extracted text here)"
)

#: Phan CHI DAN cua bien the `aic`. Doc nguoc tu `diem-yeu-pipeline-theo-49-de.md`
#: §3, va moi dieu khoan o day tra loi mot con so dem duoc tren 49 de that:
#:
#:   mau gan vao TUNG vat  63%  ·  dem so luong  51%  ·  vi tri  45%
#:   chu trong khung hinh  27%  ·  goc may       16%
#:
#: Ba khac biet lon so voi prompt cua Bao, va ca ba deu co gia phai tra:
#:
#: 1. **TIENG VIET, khong phai tieng Anh.** Arm `caption_qwen` la BM25 tren truy
#:    van tieng Viet. Caption tieng Anh va truy van tieng Viet **khong co mot
#:    token nao chung** — dung bai hoc `labels_vi` (§3 cua tai lieu diem yeu).
#:    Gia: tieng Viet ton nhieu token hon tieng Anh trong tokenizer cua Qwen, nen
#:    `max_tokens` phai nang len (xem `PROMPT_VARIANTS`).
#: 2. **Cam mo dau bang "Hinh anh cho thay".** Khong phai chuyen tham my: moi cau
#:    tra loi deu mo dau giong nhau thi cum do thanh stopword tren toan chi muc —
#:    dung co che da giet arm `object` (`mot` 91,0% tai lieu, idf 0,09).
#: 3. **Bo duong thoat "None".** Prompt cua Bao cho san mot tu de bo cuoc, va mot
#:    model se dung no. `stage_n5_ocr_qwen` da tra gia cho bai hoc nay: cho san
#:    duong thoat thi 305/307 anh tra ve "None" trong khi easyocr tim thay chu
#:    tren ca 307. Khong co chu thi de TRONG.
_QWEN25_INSTRUCTION_AIC = """Bạn đang mô tả MỘT khung hình cắt từ video truyền hình Việt Nam.

Trả về ĐÚNG hai khối, đúng thứ tự này, mỗi khối mở đầu bằng nhãn của nó:

[CAPTION] mô tả bằng TIẾNG VIỆT, 2-4 câu, CHỈ những gì nhìn thấy.
[TEXT] mọi chữ đọc được trong khung hình, chép NGUYÊN VĂN.

LUẬT CHO [CAPTION] — bốn thứ BẮT BUỘC nêu khi khung hình có:
1. MÀU gắn vào TỪNG vật, không phải màu chung của cảnh.
   ĐÚNG: "tay đua áo vàng quần đen chạy trước tay đua áo xanh dương quần đỏ"
   SAI:  "các tay đua mặc trang phục nhiều màu"
2. SỐ LƯỢNG khi đếm được: "ba người", "hai chiếc xe máy", "năm cái ghế nhựa".
   Đếm được bao nhiêu ghi bấy nhiêu. Đông quá không đếm nổi thì ghi "nhiều".
3. VỊ TRÍ tương đối: bên trái / ở giữa / bên phải / phía trước / phía sau.
4. GÓC MÁY khi rõ: cận cảnh, toàn cảnh, nhìn từ trên cao, quay từ dưới lên, quay chậm.

CẤM trong [CAPTION]:
- Đoán thứ không nhìn thấy: bối cảnh, địa danh, tên người, nghề nghiệp, cảm xúc, mục đích.
  "một bản tin về lũ lụt" là ĐOÁN. "ô tô lội trong nước nâu ngập nửa bánh" là NHÌN THẤY.
- Mở đầu bằng "Hình ảnh cho thấy", "Bức ảnh này", "Trong hình" — vào thẳng nội dung.
- Viết tiếng Anh.

LUẬT CHO [TEXT]:
- Chép NGUYÊN VĂN: giữ nguyên dấu tiếng Việt, chữ HOA, số, dấu câu.
- MỖI vùng chữ rời nhau trên màn hình là MỘT DÒNG riêng, ngăn bằng ký tự xuống dòng.
  Logo đài, đồng hồ, tiêu đề, dòng chữ chạy dưới chân màn hình là các vùng KHÁC NHAU.
- Không dịch, không diễn giải, không thêm chú thích của bạn.
- Không có chữ nào thì để TRỐNG sau [TEXT]. Đừng viết "None", "không có", hay gạch ngang."""

#: Few-shot cho bien the `aic`. **CHI VAN BAN, khong kem anh mau** — mot anh mau
#: ton hang tram token thi giac moi cai, va o quy mo 177.321 keyframe thi do la
#: hang chuc gio GPU chi de lap lai cung mot vi du.
#:
#: Vi du day dinh dang, khong day noi dung: model van nhin anh THAT de mo ta. Ba
#: vi du, moi cai gai mot thu khac nhau:
#:
#:   1. co nhieu vung chu roi nhau  -> moi vung MOT DONG
#:   2. dem + mau gan vat + trai/giua/phai, va KHONG co chu -> [TEXT] de TRONG
#:   3. slide co chu dai            -> chep nguyen van, giu chu HOA va dau
#:
#: Vi du 2 la vi du quan trong nhat: no la cho duy nhat model thay rang "khong co
#: chu" duoc the hien bang mot khoang TRONG chu khong phai bang mot tu.
_QWEN25_FEWSHOT_AIC = """Dưới đây là ba ví dụ về ĐỊNH DẠNG đầu ra đúng (cho các khung hình khác).
Học cách trình bày, đừng chép nội dung.

### Ví dụ 1
[CAPTION] Hai người dẫn chương trình đứng sau bàn dẫn cong màu trắng. Người bên trái là nam, đeo kính, mặc áo sơ mi xanh navy đậm với cà vạt sáng màu; người bên phải là nữ, tóc dài, mặc áo xanh dương nhạt. Cả hai cầm một tập tài liệu bìa đỏ. Phông nền là toàn cảnh thành phố lúc hoàng hôn màu cam vàng.
[TEXT] HTV9 HD
06:31:14
TIN CHÍNH
Đợt mưa lũ vừa qua khiến 3 người thiệt mạng

### Ví dụ 2
[CAPTION] Năm người ngồi trên ghế nhựa thấp quanh một đống măng rừng lột dở trên sân bê tông. Ngoài cùng bên trái là người phụ nữ mặc áo sơ mi kẻ caro đỏ trắng và quần vàng; ở giữa có một người mặc áo thun xanh lá; ngoài cùng bên phải là người đàn ông mặc áo hoa văn rằn ri. Phía sau là dãy thùng phuy nhựa màu xanh dương và một rổ nhựa màu vàng.
[TEXT]

### Ví dụ 3
[CAPTION] Slide bài giảng nền trắng, cận cảnh. Ba ô bấm bo tròn màu xanh dương xếp dọc ở nửa trái, mỗi ô kèm một khung trắng rỗng bên phải. Góc dưới bên trái là ô hình một nam giảng viên mặc áo sơ mi trắng ngồi trước laptop.
[TEXT] THANH NIÊN
HOA KÌ
NHẬT BẢN
TÂY ÂU
CHƯƠNG TRÌNH ÔN THI THPT MÔN LỊCH SỬ

Bây giờ làm đúng như vậy cho khung hình được cung cấp."""

_QWEN25_PROMPT_AIC = f"{_QWEN25_INSTRUCTION_AIC}\n\n{_QWEN25_FEWSHOT_AIC}"

#: `ten bien the -> (prompt, max_tokens mac dinh)`.
#:
#: `max_tokens` khac nhau la mot phep do chu khong phai so thich: caption tieng
#: Viet ton nhieu token hon caption tieng Anh cung noi dung, va 256 cua Bao da
#: cat cut mot phan ngay tren ban chay cua chinh ban ay ("...and the corr").
#: 384 la gia tri khoi diem cho bien the `aic`; o do throughput cua notebook 08
#: DO lai anh huong that cua no truoc khi ap len 177.321 anh.
PROMPT_VARIANTS: dict[str, tuple[str, int]] = {
    "bao": (_QWEN25_PROMPT_BAO, 256),
    "aic": (_QWEN25_PROMPT_AIC, 384),
}
