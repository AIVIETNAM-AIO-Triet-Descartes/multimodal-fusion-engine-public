# Private Kaggle ZIP cache cho bundle BTC

Mục tiêu của cache là tránh tải lặp lại các ZIP lớn từ `aic-data.ledo.io.vn`. Dữ liệu đầu ra,
phân công, model và Modal Volume không thay đổi.

## 1. File của `trong`

Giữ nguyên tên và nội dung của bảy ZIP dưới đây; không giải nén rồi nén lại:

| File | Kích thước chuẩn (byte) |
|---|---:|
| `Keyframes_L21.zip` | 1.447.092.929 |
| `Keyframes_L23.zip` | 506.221.125 |
| `Keyframes_L24.zip` | 1.726.819.558 |
| `Keyframes_L25.zip` | 6.092.515.522 |
| `Keyframes_L26_b.zip` | 2.418.048.525 |
| `Keyframes_L29.zip` | 2.509.637.849 |
| `Keyframes_L30.zip` | 1.411.485.866 |

Tổng cộng 16.111.821.374 byte, xấp xỉ 15 GiB. Với notebook 02, nên cache thêm hai file dùng
chung:

- `objects-aic25-b1.zip`
- `map-keyframes-aic25-b1.zip`

Notebook 03 dùng lại bảy ZIP keyframe và `map-keyframes-aic25-b1.zip`. Notebook 04 lấy video bằng
HTTP Range nên không sử dụng các ZIP keyframe này.

## 2. Tạo Dataset trên Kaggle

1. Vào Kaggle, chọn **Datasets → New Dataset**.
2. Upload các ZIP nguyên bản. Có thể chia thành nhiều Dataset nếu giao diện không nhận cả bộ một
   lần; không được để hai Dataset đã attach chứa cùng một tên ZIP.
3. Đặt Visibility là **Private** trước khi tạo.
4. Không đưa token, `.env`, source repo hoặc dữ liệu giải nén vào Dataset này.

Chỉ upload lại dữ liệu BTC khi quy định sử dụng dữ liệu của cuộc thi và Lead cho phép lưu bản sao
private trên Kaggle.

## 3. Attach và kiểm tra trước khi chạy

Trong Kaggle Notebook, chọn **Add Input**, tìm Dataset private vừa tạo và attach. Chạy cell kiểm tra
sau trước khi chạy notebook chính:

```python
from pathlib import Path

required = {
    "Keyframes_L21.zip": 1_447_092_929,
    "Keyframes_L23.zip": 506_221_125,
    "Keyframes_L24.zip": 1_726_819_558,
    "Keyframes_L25.zip": 6_092_515_522,
    "Keyframes_L26_b.zip": 2_418_048_525,
    "Keyframes_L29.zip": 2_509_637_849,
    "Keyframes_L30.zip": 1_411_485_866,
}

root = Path("/kaggle/input")
for name, expected in required.items():
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"{name}: cần đúng một file, tìm thấy {matches}")
    actual = matches[0].stat().st_size
    if actual != expected:
        raise RuntimeError(f"{name}: {actual:,} byte, chờ đợi {expected:,}")
    print(f"OK  {name:<24} {actual / 2**30:5.2f} GiB  {matches[0]}")
```

Nếu cả bảy dòng đều `OK`, cache phần keyframe của `trong` đã sẵn sàng.

## 4. Chạy notebook

1. Dùng notebook trong `notebooks/btc/`, không dùng bản cũ nằm ngoài repo.
2. Đặt `MEMBER = "trong"` ở cell đầu.
3. Giữ nguyên `VOLUME = "keyframes-btc"`, `VOL_VERSION = 2` và các file shard.
4. Attach cùng Dataset cache cho notebook 02 và 03.
5. Trong log, mỗi ZIP cache phải hiện dòng `cache Kaggle: /kaggle/input/...`.
6. Nếu log hiện `tải <tên ZIP>`, file đó không có trong cache và notebook đang fallback về website.

ZIP trong `/kaggle/input` là read-only. Notebook chỉ đọc trực tiếp và không xóa chúng. ZIP tải
fallback vào `/kaggle/working/zips` vẫn được xóa sau khi xử lý để giải phóng đĩa.
