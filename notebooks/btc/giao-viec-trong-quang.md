# Giao việc: nạp bundle BTC lên Modal Volume — Trọng & Quang

**Người làm:** Trọng, Quang. Chỉ hai người. Việc chia dữ liệu (slice) đã cố định trong
`config/zips_keyframes_btc.yaml` cho đúng hai tên này — **đừng chia lại, đừng thêm người**, lý do ở §4.3.

**Đích:** Modal Volume `keyframes-btc` (v2), bundle `aic_btc_b1_v1`.

**Ngoài phạm vi:** sinh caption (`n6_recap`, `n6_bge_m3`) — Lead hoãn. Trích keyframe bằng DAKE —
việc khác, volume khác (`notebooks/dake/`).

---

## 1. Phân công

Đổi `MEMBER` ở ô đầu mỗi notebook thành `"trong"` hoặc `"quang"`. Không cần chỉnh gì khác.

| | zip | video | ảnh | video gốc |
|---|---|---|---|---|
| **trong** | L21, L23, L24, L25, L26_b, L29, L30 | 404 | 88.664 | 38,94 GB |
| **quang** | L22, L26_a, L26_c, L26_d, L26_e, L27, L28 | 469 | 88.657 | 38,34 GB |

Cân theo **cả hai trục** vì hai nhóm tầng có chi phí khác nhau: N4/N5 chạy theo **từng ảnh**
(88.664 vs 88.657), N2/N3 chạy theo **thời lượng** (38,94 vs 38,34 GB video gốc).

> ⚠️ **Trước khi chạy, kiểm tra trạng thái thật trên Volume.** Bảng phân công vừa đổi từ 5 người
> sang 2, nên có thể một phần L25 đã được `triet` nhập trước đó dưới tên khác. Notebook sẽ tự nhập
> lại (dữ liệu giống hệt, chỉ tốn băng thông) và ghi lại author cho đúng. Nếu thấy lạ thì báo Lead
> trước khi chạy tiếp, đừng tự sửa config.

---

## 2. Ba lượt, chạy theo thứ tự

Cùng volume, cùng bundle, cùng phân công. Mỗi notebook **bấm Run All**; đứt phiên thì bấm lại, nó
bỏ qua việc đã xong.

| | Notebook | Tầng | Requirements |
|---|---|---|---|
| 1 | `02-keyframes-btc-import.ipynb` | `n0_import_btc` | `../requirements-kaggle-nomodel.txt` |
| 2 | `03-btc-siglip-ocr.ipynb` | `n4_siglip`, `n5_ocr_easyocr`, `n5_ocr_qwen` | `../requirements-kaggle-btc-images.txt` |
| 3 | `04-btc-subtitles-audio.ipynb` | `n2_subtitles`, `n3_audio` | `../requirements-kaggle-btc-video.txt` |

Lượt 1 phải xong **trước** lượt 2 và 3 cho từng video (notebook tự bỏ qua video chưa nhập). Lượt 2
và 3 độc lập nhau — chạy song song hai tab được.

**Kaggle Secrets cần có:** `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `GITHUB_TOKEN`.
Accelerator: lượt 1 **bất kỳ**; lượt 2 và 3 **GPU T4**.

### Model dùng cho từng feature

**Không được đổi hay bớt model nào.** Nếu một model không nạp được, không chạy được, hay chậm hơn
dự tính — **báo Lead**, đừng tự thay bằng model khác. Đổi model là đổi phép đo, và bundle sẽ thành
hỗn hợp hai phép đo mà cosine vẫn trông hợp lý nên không ai phát hiện.

| Artifact | Model | Ghi chú |
|---|---|---|
| `siglip_vectors` | `google/siglip-so400m-patch14-384` | **chỉ tháp ảnh**, 1152 chiều. Tháp text nằm ở tầng online |
| `ocr_text` | easyocr `["vi", "en"]`, GPU | nguồn OCR thứ nhất |
| `sviptr_ocr` | `Qwen/Qwen2-VL-7B-Instruct`, 4-bit | nguồn OCR thứ hai. Tên `sviptr` là **di sản**, không phải tên model |
| `subtitles` + `subtitle_segments` | `faster-whisper large-v3` | `language="vi"`, `vad_filter=True`, `beam_size=5` |
| `audio_features` | `laion/clap-htsat-unfused` | 512 chiều, cửa sổ ±5s quanh keyframe, sr 48 kHz |
| `objects/` | **không chạy model** | BTC phát hành sẵn nhãn; lượt 1 nhập nguyên văn 5 trường |
| `dense_captions`, `dense_vectors` | — | **KHÔNG sinh ở lượt này** |

`n4_objects` không có trong kế hoạch: BTC đã cấp nhãn và lượt 1 nạp chúng. `mfe.offline.ingest` đọc
thẳng `objects/{video_id}/{n:03d}.json` nên không cần thêm bước nào. (Chạy detector của ta lên chính
bộ keyframe này là một phép đo **khác** — §3.6 của kế hoạch, không phải việc của task này.)

---

## 3. Hai artifact sẽ trống, và đó là dự kiến

Bỏ caption nghĩa là bundle không có `dense_captions.json` và `dense_vectors.npz`. Hệ quả:

- `merge_bundle.py` **không** coi đó là lỗi — nó gộp những gì có.
- Tầng online sẽ thiếu arm `caption` và arm dense-vector. Khi so hiệu năng với bộ DAKE, **phải so
  trên cùng tập arm**, nếu không con số không nói gì.

Không cần làm gì thêm. Chỉ đừng "sinh tạm" caption bằng model khác để bù chỗ trống.

---

## 4. Hợp đồng phải tuân thủ

Notebook đã làm đúng những điều dưới đây; ghi ra để khi đọc log hay debug thì biết cái gì là đúng.
**Ba mục 4.1–4.3 nếu sai thì không ném lỗi, chỉ sai** — đó là lý do chúng được liệt kê.

### 4.1 Khoá — `keyframe_id`, không phải số thứ tự

```
keyframe_id = {video_id}#{frame_idx:06d}        ví dụ  L21_V001#037716
```

`frame_idx` là **frame của video gốc**, và nó được **copy** từ cột `frame_idx` trong
`map-keyframes/{video_id}.csv` của BTC — không bao giờ tính lại từ `pts_time × fps`.

Số thứ tự keyframe `n` (1, 2, 3…) là thuộc tính của **lần trích**, không phải của video. Dùng `n`
làm khoá thì hai bộ keyframe khác nhau cho ra cùng một khoá trỏ vào hai frame khác nhau — và không
có gì báo lỗi.

`n` vẫn dùng ở đúng một chỗ: **tên file** ảnh và nhãn object (`{n:03d}.jpg`, `{n:03d}.json`), vì hai
thứ đó đánh địa chỉ theo thứ tự trong cùng một lần trích.

### 4.2 Bố cục trên Volume

```
/bundle/aic_btc_b1_v1/
├── keyframes/{video_id}/{n:03d}.jpg
├── objects/{video_id}/{n:03d}.json          5 trường, nguyên văn của BTC
├── map-keyframes/{video_id}.csv             cột: n, pts_time, fps, frame_idx
├── subtitles_segments/{video_id}.json       segment thô của Whisper
└── parts/
    ├── {artifact}/{video_id}.json           text:   {keyframe_id: text}, đã chuẩn hoá NFC
    ├── {artifact}/{video_id}.npz            vector: HAI mảng `ids` + `vectors`
    └── manifest/{video_id}.json             artifact nào đã có, model gì, tham số gì

/_status/{member}/{stage}/{video_id}.json    marker "đã xong"
```

**Vector là `.npz` với hai mảng, không phải `.npy` chứa dict.** `ids` là mảng chuỗi thật
(`dtype=np.str_`), `vectors` là `float32`. Cụ thể để `np.load` **không** cần `allow_pickle=True` —
nạp file pickle là thực thi code, mà artifact này được truyền giữa nhiều máy.

**Marker là điểm chốt, không phải file dữ liệu.** Một thư mục ảnh đẩy dở dang **vẫn tồn tại** trên
Volume, nên "có file" không có nghĩa là "đã xong". Notebook luôn đẩy xong rồi mới cắm mốc.

### 4.3 Một video có đúng MỘT author

`parts/manifest/{video_id}.json` ghi một trường `author`, và `merge_bundle.py` **từ chối cả bundle**
nếu author khác người được giao trong bảng phân công.

Vì thế: **đừng chạy phần của người khác dưới tên mình**, và đừng chia lại bảng phân công. Nếu cần
đổi (một người không kịp, thêm người), Lead sửa `config/zips_keyframes_btc.yaml` — sửa ở đó thì cả
ba lượt tự khớp, vì cả ba đọc cùng file đó.

---

## 5. Sau khi ingest xong — gộp thế nào

Ai xong phần mình thì **tự kiểm trước**, rồi báo Lead. Chạy trên máy cá nhân (cần `.env` có
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`), không chạy trên Kaggle:

```bash
python scripts/verify_bundle.py --shards config/ingest_shards_batch1_btc_images.yaml --volume keyframes-btc --member trong
```

Đổi `--shards` sang `..._btc.yaml` / `..._btc_video.yaml` để kiểm từng lượt, và `--member` thành tên
mình. Thêm `--check-hashes` cho lượt kiểm cuối: nó tải lại từng mảnh để đối chiếu sha256 với marker,
chậm, nhưng là **phép kiểm duy nhất bắt được file bị ghi đè dang dở**.

Khi **cả hai** người xong, **Lead** gộp:

```bash
python scripts/merge_bundle.py --shards config/ingest_shards_batch1_btc_images.yaml --volume keyframes-btc --emit sharded --upload
```

Chạy một lần cho mỗi file `--shards` (ba lượt = ba lần). `--emit sharded` là bắt buộc ở quy mô này:
`single` gộp thành một file phẳng mỗi artifact, hợp cho bộ mẫu 16 video, không hợp cho 873.

`merge_bundle.py` **từ chối** chứ không cảnh báo, ở bốn trường hợp:

1. hai người cùng ghi một `keyframe_id`;
2. author không khớp bảng phân công (§4.3);
3. khoá sai định dạng, hoặc `keyframe_id` không có trong `map-keyframes`;
4. hai shard lệch phiên bản ở nhóm gói **ảnh hưởng số** (`torch`, `transformers`,
   `faster-whisper`, `easyocr`, `numpy`…).

Điều 4 là lý do cả hai người phải cài từ `requirements-kaggle.lock.txt` nếu file đó đã có trong
repo. Nếu chưa có: **người chạy trước** commit bản `.lock` mà notebook sinh ra, người sau cài theo.

---

## 6. Băng thông — chỗ tốn thời gian nhất, và cách xử lý

`aic-data.ledo.io.vn` bóp băng thông khi nhiều người cùng kéo. Đã gặp: tụt từ ~20 MB/s xuống
**~6 KB/s**, ETA nhảy lên 5 ngày.

- **Hai người hẹn nhau chạy lệch giờ.** Nối tiếp nhanh hơn song song khi nghẽn ở phía server.
- Tốc độ **đều** ở mức vài KB/s = bị bóp, không phải đứt → **restart session** để đổi IP Kaggle
  rồi Run All lại.
- `wget` đã có `--read-timeout=60` nên kết nối chết tự nối lại, và mỗi gói được **kiểm kích thước**
  sau khi tải. Chậm chỉ tốn thời gian, không hỏng dữ liệu.
- Lượt 2 tải lại `Keyframes_*.zip` (13,7–15,0 GB mỗi người). Lượt 3 lấy **từng video** bằng HTTP
  Range nên không tải gói nào — đĩa cao nhất ~325 MB.

---

## 7. Khi nào báo Lead thay vì tự xử lý

- Model nào không nạp được, hoặc muốn đổi/bớt model → **báo, không tự quyết**.
- `verify_bundle.py` hay `merge_bundle.py` từ chối → gửi nguyên output, đừng sửa config cho nó hết
  báo lỗi.
- Notebook ném lỗi ở một video và đi tiếp: bình thường, cuối phiên nó in danh sách. Nhưng nếu
  **cùng một lỗi** ở nhiều video thì dừng và báo.
- Số lượng ảnh hoặc nhãn object lệch nhau: lượt 1 tự từ chối video đó. Lệch một hai video thì ghi
  lại; lệch cả gói thì báo.
- Muốn đổi bảng phân công vì bất kỳ lý do gì → Lead sửa, không sửa tại chỗ (§4.3).

Chi tiết kỹ thuật và lý do từng quyết định:
[`docs/triet/ke-hoach-offline-ingest.md`](../../docs/triet/ke-hoach-offline-ingest.md) §7.1.
