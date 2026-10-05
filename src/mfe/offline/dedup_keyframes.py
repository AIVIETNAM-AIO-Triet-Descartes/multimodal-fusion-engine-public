"""Lọc keyframe lân cận gần trùng nhau — hai chỗ đặt được, hai tín hiệu khác nhau.

    notebooks/dake/hieu-chinh-tham-so-dake.md §3 "Hệ quả cho kế hoạch lọc trùng"

Bài toán, viết lại cho gọn: DAKE **được phép** chọn hai frame cách nhau rất gần,
và ở cấu hình cũ (`min_gap`=5 frame = 0,20s) thì **36% khoảng cách keyframe dưới
nửa giây**. Đó không phải nội dung tình cờ giống nhau — đó là tham số sinh ra
trùng lặp. P1 đưa `min_gap` lên 1,0s và tỉ lệ đó xuống 1–3%, nên việc này **không
còn gấp**; nhưng nó vẫn nên được trả lời, và đây là câu trả lời.

Hai chỗ đặt, và chúng KHÔNG thay thế nhau
------------------------------------------
**(a) `signature_filter` — chữ ký rẻ, chạy trong N1.** N1 đã giải mã *mọi* frame
(92% chi phí của tầng), nên tính thêm một chữ ký perceptual trong cùng lượt gần
như miễn phí. Không thêm model, chỉ là số học: DCT 8×8 trên ảnh xám 32×32, so
bằng khoảng cách Hamming. Đây là cách **duy nhất** khiến "lọc ngay từ đầu" khả
thi — tín hiệu OCR và embedding là *đầu ra* của N4/N5, nên chúng không thể tồn
tại trước khi keyframe được chọn.

**(b) `embedding_filter` — embedding thật, chèn giữa `n4_siglip` và `n5_ocr_*`.**
Dùng chính vector SigLIP đã tính, so bằng cosine. Đắt hơn (SigLIP vẫn phải chạy
trên toàn bộ) nhưng cứu được tầng đắt nhất phía sau. Nó thấy được thứ pHash
không thấy: hai khung *nhìn* khác nhau về pixel mà cùng một nội dung ngữ nghĩa.

Vì sao cả hai đều so với KHUNG ĐÃ GIỮ, không phải khung liền trước
-------------------------------------------------------------------
So với khung liền trước rồi bỏ nó đi sẽ trôi: A≈B, B≈C nhưng A có thể rất khác C,
và lọc theo cặp liền kề sẽ giữ lại A rồi bỏ cả B lẫn C. Cả hai hàm ở đây so ứng
viên với **khung gần nhất còn được giữ**, nên chuỗi trôi chậm vẫn được lấy mẫu.

Vì sao không có ngưỡng mặc định "đúng"
---------------------------------------
`PHASH_MAX_DISTANCE` và `EMBED_MAX_COSINE` dưới đây là **điểm khởi đầu để quét**,
không phải hằng số đã hiệu chỉnh. §3 của tài liệu nói rõ vì sao chỗ này phải bảo
thủ hơn ρ: quét lại ρ chỉ tốn lượt 2 (~8% chi phí) vì `frame_sizes` đã cache, còn
embedding của frame **đã bỏ** thì không cache được — quét lại ngưỡng lọc là chạy
lại cả tầng. Đo trước, chốt sau.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: Cạnh ảnh xám trước khi DCT. 32 là quy ước của pHash: đủ để DCT có ý nghĩa,
#: đủ nhỏ để phép biến đổi không đáng kể so với chi phí giải mã đã bỏ ra.
PHASH_SIDE = 32

#: Số hệ số DCT tần thấp giữ lại mỗi chiều -> chữ ký 8×8 = 64 bit.
PHASH_KEEP = 8

#: Khoảng cách Hamming TỐI ĐA để coi hai khung là trùng. 64 bit, nên 5 là ~8%.
#: Điểm khởi đầu để quét, không phải giá trị đã hiệu chỉnh — xem docstring.
PHASH_MAX_DISTANCE = 5

#: Cosine TỐI THIỂU để coi hai khung là trùng về nội dung. SigLIP đẩy các khung
#: cùng cảnh lên rất cao, nên ngưỡng phải sát 1,0 mới chỉ bắt trùng thật.
EMBED_MAX_COSINE = 0.97


def phash(frame: np.ndarray) -> int:
    """Chữ ký perceptual 64 bit của một khung BGR (hoặc ảnh xám).

    Không dùng OpenCV: N1 đã import cv2 rồi, nhưng hàm này cũng chạy trong test
    và trong notebook nơi ảnh có thể đến từ PIL. numpy là đủ, và giữ nó thuần
    numpy nghĩa là chữ ký **không đổi** theo phiên bản OpenCV — một chữ ký đổi
    giá trị giữa hai lần chạy sẽ làm mọi so sánh trước đó vô nghĩa mà không báo.

    So sánh với mắt người thì pHash bắt được "cùng một cảnh, khác vài pixel" và
    bỏ qua khác biệt về độ sáng tổng thể — đúng thứ cần khi hai keyframe cách
    nhau 0,2 giây.
    """
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim == 3:
        # BGR -> xám theo trọng số luminance của Rec. 601 (thứ tự kênh của cv2).
        image = image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299
    small = _resize_box(image, PHASH_SIDE)
    coefficients = _dct2(small)[:PHASH_KEEP, :PHASH_KEEP]
    # Bỏ hệ số DC (0,0): nó mã hoá độ sáng trung bình, và hai khung chỉ khác
    # nhau về độ sáng vẫn là cùng một cảnh.
    rest = coefficients.flatten()[1:]
    bits = rest > np.median(rest)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    """Số bit khác nhau giữa hai chữ ký."""
    return bin(left ^ right).count("1")


@dataclass
class FilterReport:
    """Giữ lại gì, bỏ đi gì, và bỏ vì giống khung nào.

    `dropped` mang cả khung bị bỏ **và** khung nó bị coi là trùng, vì một báo cáo
    chỉ nói "đã bỏ 37 khung" thì không kiểm lại được: người vận hành cần mở đúng
    hai ảnh đó ra nhìn mới biết ngưỡng đặt đúng hay sai.
    """

    kept: list[int] = field(default_factory=list)
    #: (chỉ số bị bỏ, chỉ số được giữ mà nó trùng, khoảng cách/độ tương đồng)
    dropped: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    @property
    def ratio_dropped(self) -> float:
        total = self.n_kept + self.n_dropped
        return self.n_dropped / total if total else 0.0

    def summary(self) -> str:
        return (
            f"giu {self.n_kept}, bo {self.n_dropped} "
            f"({self.ratio_dropped:.1%} so voi dau vao)"
        )


def signature_filter(
    signatures: Sequence[int],
    *,
    max_distance: int = PHASH_MAX_DISTANCE,
) -> FilterReport:
    """(a) Lọc bằng chữ ký perceptual. `signatures[i]` là pHash của keyframe thứ i.

    Trả về `FilterReport` với chỉ số **theo thứ tự đầu vào**, không phải
    `frame_idx` — người gọi giữ ánh xạ đó, và hàm này không cần biết.

    >>> report = signature_filter([0b1010, 0b1010, 0b1111_0000], max_distance=1)
    >>> report.kept
    [0, 2]
    >>> report.dropped
    [(1, 0, 0.0)]
    """
    report = FilterReport()
    last_kept: int | None = None
    for index, signature in enumerate(signatures):
        if last_kept is None:
            report.kept.append(index)
            last_kept = index
            continue
        distance = hamming(signature, signatures[last_kept])
        if distance <= max_distance:
            report.dropped.append((index, last_kept, float(distance)))
        else:
            report.kept.append(index)
            last_kept = index
    return report


def embedding_filter(
    vectors: np.ndarray,
    *,
    max_cosine: float = EMBED_MAX_COSINE,
) -> FilterReport:
    """(b) Lọc bằng embedding SigLIP. `vectors` là (N, D), theo thứ tự thời gian.

    Vector được chuẩn hoá lại tại chỗ: `n4_siglip` ghi ra vector chưa chắc có
    norm 1 (float16 -> float32 làm norm lệch vài 1e-4), và cosine tính trên
    vector chưa chuẩn hoá thì không phải cosine.

    >>> import numpy as np
    >>> v = np.array([[1.0, 0.0], [1.0, 0.001], [0.0, 1.0]])
    >>> embedding_filter(v, max_cosine=0.99).kept
    [0, 2]
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"can ma tran (N, D), nhan duoc shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    report = FilterReport()
    last_kept: int | None = None
    for index in range(len(matrix)):
        if last_kept is None:
            report.kept.append(index)
            last_kept = index
            continue
        cosine = float(matrix[index] @ matrix[last_kept])
        if cosine >= max_cosine:
            report.dropped.append((index, last_kept, cosine))
        else:
            report.kept.append(index)
            last_kept = index
    return report


def sweep_signature(
    signatures: Sequence[int], distances: Sequence[int]
) -> list[tuple[int, FilterReport]]:
    """Quét nhiều ngưỡng Hamming trong một lượt. Rẻ: chữ ký đã có sẵn.

    Đây là cách ngưỡng NÊN được chốt — nhìn đường cong bỏ-bao-nhiêu theo ngưỡng,
    chứ không lấy một con số từ một bài báo dùng dataset khác.
    """
    return [(d, signature_filter(signatures, max_distance=d)) for d in distances]


def sweep_embedding(
    vectors: np.ndarray, cosines: Sequence[float]
) -> list[tuple[float, FilterReport]]:
    """Quét nhiều ngưỡng cosine trong một lượt."""
    return [(c, embedding_filter(vectors, max_cosine=c)) for c in cosines]


# ---------------------------------------------------------------------------
# Số học nội bộ. Tách ra để `phash` đọc được, và để test được từng mảnh.
# ---------------------------------------------------------------------------


def _resize_box(image: np.ndarray, side: int) -> np.ndarray:
    """Thu nhỏ về `side`×`side` bằng trung bình theo ô (box filter).

    Không lấy mẫu điểm: lấy mẫu điểm trên ảnh có nhiễu nén cho ra chữ ký nhảy
    lung tung giữa hai khung gần như giống hệt — đúng trường hợp hàm này tồn tại
    để nhận ra.
    """
    height, width = image.shape
    rows = np.linspace(0, height, side + 1).astype(int)
    cols = np.linspace(0, width, side + 1).astype(int)
    out = np.empty((side, side), dtype=np.float64)
    for r in range(side):
        r0, r1 = rows[r], max(rows[r] + 1, rows[r + 1])
        for c in range(side):
            c0, c1 = cols[c], max(cols[c] + 1, cols[c + 1])
            out[r, c] = image[r0:r1, c0:c1].mean()
    return out


def _dct2(block: np.ndarray) -> np.ndarray:
    """DCT-II hai chiều, chuẩn hoá trực giao. Thuần numpy, không cần scipy."""
    return _dct1(_dct1(block.T).T)


def _dct1(block: np.ndarray) -> np.ndarray:
    n = block.shape[0]
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    scale = np.full((n, 1), np.sqrt(2.0 / n))
    scale[0] = np.sqrt(1.0 / n)
    return scale * (basis @ block)
