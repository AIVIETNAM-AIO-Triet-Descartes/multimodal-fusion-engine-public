"""Arm đọc thẳng bản gộp trong bộ nhớ — không Elasticsearch, không Milvus.

    docs/triet/ke-hoach-offline-ingest.md §6.3 (hình dạng bản gộp)

Vì sao có file này
-------------------
Mười trong mười hai arm đọc Elasticsearch (8 arm văn bản) và Milvus (2 arm
dense). Cả hai là **máy chủ**, không phải thư viện: compose gắn chúng vào
`127.0.0.1`, và Kaggle thì không chạy Docker được. Nên muốn truy vấn trên Kaggle
— hoặc trên bất kỳ máy nào không dựng nổi hai kho đó — thì phải có một đường
khác.

Đường đó rẻ hơn nhiều người tưởng ở quy mô batch 1:

* **Dense.** 176.707 vector × 1152 chiều float32 là 815 MB. Vector đã chuẩn hoá
  L2 từ lúc trích, nên tìm kiếm cosine **là một phép nhân ma trận** —
  `matrix @ query`. Trên numpy mất ~0,2 giây; Milvus nhanh hơn nhờ chỉ mục xấp
  xỉ, nhưng ở đây ta muốn **chính xác**, và 0,2 giây thì không ai chờ.
* **Văn bản.** `rank_bm25` đã là phụ thuộc sẵn có, và nó chính là thứ
  `InvertedIndex` đang dùng.

Hợp đồng giữ nguyên tuyệt đối
-------------------------------
Cùng tên arm, cùng `Candidate`, cùng `query_language`. `Cascade`, `fusion`,
`dedup`, `DANTE` và cả bảng arm trong `run_submission.py` **không biết** chúng
đang chạy trên kho nào. Đó là điều kiện để so được kết quả hai đường với nhau:
đổi tên arm hay đổi hình dạng điểm là mất luôn phép so.

Ba chỗ dễ sai, đều im lặng
----------------------------
**Chuẩn hoá văn bản.** Dùng đúng `_normalize_text`/`_tokenize` của
`mfe.stores.inverted.index`. Hai bộ tách từ khác nhau cho hai bảng xếp hạng khác
nhau trên cùng dữ liệu, và không có gì báo — đúng cảnh báo trong docstring của
`InvertedIndex.add`.

**`subtitle_exact` không phải BM25.** Bên Elasticsearch nó là `match_phrase` —
khớp nguyên văn cụm. Thay bằng BM25 thì nó thành bản sao của `subtitle_partial`,
và fusion sẽ đếm cùng một bằng chứng hai lần.

**Khoá là `keyframe_id` của bản gộp.** File phẳng đã dùng `{video}#{frame:06d}`,
nên không có bước tra ordinal nào ở đây — đúng lý do bản gộp tồn tại.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from mfe.contracts.ids import parse_keyframe_id
from mfe.contracts.tools import Candidate
from mfe.online.arms import Arm
from mfe.online.dense_arms import DENSE_ARM_SPECS, encode_query
from mfe.online.text_arms import TEXT_ARM_SPECS
from mfe.online.tools.evidence import create_evidence_id
from mfe.stores.inverted.index import _normalize_text, _tokenize

logger = logging.getLogger(__name__)

#: `source` của arm văn bản -> tên file phẳng trong `data/vectors/`.
#: Cùng bảng với `mfe.offline.adapters.sota_payloads.SOURCE_FILES`; nhập lại ở
#: đây thì hai bảng sẽ lệch, nên nó được đối chiếu trong `FlatBundle.load`.
SOURCE_FILE = {
    "subtitle": "subtitles.json",
    "ocr": "ocr_text.json",
    "ocr_sviptr": "sviptr_ocr.json",
    "ocr_paddle": "paddle_ocr.json",
    "ocr_vietocr": "viet_ocr.json",
    "object": "objects_text.json",
    "caption": "dense_captions.json",
}

#: Tên arm dense -> tên file phẳng.
DENSE_FILE = {
    "siglip": "siglip_vectors.npz",
    "bge_m3": "dense_vectors.npz",
    "clap": "audio_features.npz",
}


class FlatBundle:
    """Bản gộp đã nạp vào RAM: văn bản đã tách từ, vector đã xếp thành ma trận."""

    def __init__(self) -> None:
        self.texts: dict[str, dict[str, str]] = {}
        self.bm25: dict[str, object] = {}
        self.text_keys: dict[str, list[str]] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.vector_keys: dict[str, list[str]] = {}
        self.missing: dict[str, str] = {}

    @classmethod
    def load(cls, root: str | Path, *, sources=None, models=None) -> FlatBundle:
        """Nạp `{root}/data/vectors/` — hoặc chính `{root}` nếu file nằm thẳng đó.

        Thiếu một file **không phải lỗi**: `paddle_ocr` chưa chạy xong thì arm đó
        đơn giản là không khả dụng, kèm lý do. Cái không được phép là im lặng.
        """
        from mfe.offline.adapters.sota_payloads import SOURCE_FILES

        drift = {k: v for k, v in SOURCE_FILE.items() if SOURCE_FILES.get(k) != v}
        if drift:
            raise ValueError(
                f"SOURCE_FILE lech voi sota_payloads.SOURCE_FILES: {drift}. "
                "Hai bang phai cung ten file, neu khong mot arm se doc nham nguon."
            )

        root = Path(root)
        base = root / "data" / "vectors"
        if not base.is_dir():
            base = root
        self = cls()

        for source, filename in SOURCE_FILE.items():
            if sources is not None and source not in sources:
                continue
            path = base / filename
            if not path.is_file():
                self.missing[source] = f"khong co {filename} trong {base}"
                continue
            started = time.perf_counter()
            payload = json.loads(path.read_text(encoding="utf-8"))
            keys = [k for k, v in payload.items() if str(v).strip()]
            self.texts[source] = {k: payload[k] for k in keys}
            self.text_keys[source] = keys
            logger.info(
                "%s: %d/%d dong co chu (%.1fs)",
                source, len(keys), len(payload), time.perf_counter() - started,
            )

        for model, filename in DENSE_FILE.items():
            if models is not None and model not in models:
                continue
            path = base / filename
            if not path.is_file():
                self.missing[model] = f"khong co {filename} trong {base}"
                continue
            started = time.perf_counter()
            with np.load(path, allow_pickle=False) as payload:
                ids = [str(x) for x in payload["ids"].tolist()]
                matrix = np.asarray(payload["vectors"], dtype=np.float32)
            # Chuẩn hoá lại cho chắc. Bản gộp ĐÃ chuẩn hoá, nhưng một ma trận
            # chưa chuẩn hoá vẫn cho ra thứ hạng trông hợp lý — chỉ sai. Chi phí
            # là một lần chia, một lần duy nhất lúc nạp.
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors[model] = matrix / norms
            self.vector_keys[model] = ids
            logger.info(
                "%s: %s (%.1fs)", model, matrix.shape, time.perf_counter() - started
            )
        return self

    def build_bm25(self, source: str):
        """Dựng BM25 cho một nguồn, chỉ khi thật sự cần.

        Lười có chủ ý: dựng đủ 7 nguồn tốn cả phút và phần lớn lượt chạy chỉ
        đụng vài arm.
        """
        if source in self.bm25:
            return self.bm25[source]
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        keys = self.text_keys[source]
        corpus = [_tokenize(self.texts[source][k]) for k in keys]
        started = time.perf_counter()
        self.bm25[source] = BM25Okapi(corpus) if corpus else None
        logger.info(
            "bm25 %s: %d tai lieu (%.1fs)", source, len(corpus),
            time.perf_counter() - started,
        )
        return self.bm25[source]

    def report(self) -> str:
        text = ", ".join(f"{s}:{len(k)}" for s, k in sorted(self.text_keys.items()))
        dense = ", ".join(f"{m}:{v.shape[0]}" for m, v in sorted(self.vectors.items()))
        return f"van ban [{text}] · dense [{dense}] · thieu {sorted(self.missing)}"


def _candidates(pairs, arm_name: str, run_id: str) -> list[Candidate]:
    """`[(keyframe_id, score)]` -> `Candidate`, đúng hình dạng của arm có kho."""
    out: list[Candidate] = []
    for rank, (keyframe_id, score) in enumerate(pairs, start=1):
        video_id, _ = parse_keyframe_id(keyframe_id)
        out.append(
            Candidate(
                keyframe_id=keyframe_id,
                video_id=video_id,
                score=float(score),
                evidence_id=create_evidence_id(arm_name, run_id, rank),
            )
        )
    return out


def build_flat_text_arms(bundle: FlatBundle, weights=None, enabled=None) -> list[Arm]:
    """Một `Arm` mỗi mục `TEXT_ARM_SPECS`, đọc từ `bundle`."""
    weights = weights or {}
    enabled = enabled or {}
    arms: list[Arm] = []
    for name, (source, is_phrase) in TEXT_ARM_SPECS.items():
        if source not in bundle.texts:
            arms.append(
                Arm(
                    name=name,
                    search=None,
                    weight=float(weights.get(name, 1.0)),
                    enabled=bool(enabled.get(name, True)),
                    unavailable_reason=bundle.missing.get(
                        source, f"khong co nguon {source} trong ban gop"
                    ),
                )
            )
            continue

        def search(q, k, run, _source=source, _name=name, _phrase=is_phrase):
            if _phrase:
                # `match_phrase` của Elasticsearch, làm bằng tay: khớp nguyên văn
                # cụm sau khi chuẩn hoá. KHÔNG được thay bằng BM25 — làm vậy thì
                # arm này thành bản sao của `subtitle_partial`, và fusion đếm
                # cùng một bằng chứng hai lần.
                needle = _normalize_text(q)
                if not needle:
                    return []
                hits = [
                    (key, 1.0)
                    for key in bundle.text_keys[_source]
                    if needle in _normalize_text(bundle.texts[_source][key])
                ]
                return _candidates(hits[:k], _name, run)

            model = bundle.build_bm25(_source)
            if model is None:
                return []
            tokens = _tokenize(q)
            if not tokens:
                return []
            scores = model.get_scores(tokens)
            keys = bundle.text_keys[_source]
            top = np.argsort(scores)[::-1][:k]
            return _candidates(
                [(keys[i], scores[i]) for i in top if scores[i] > 0], _name, run
            )

        arms.append(
            Arm(
                name=name,
                search=search,
                weight=float(weights.get(name, 1.0)),
                enabled=bool(enabled.get(name, True)),
                description=f"{'khop cum' if is_phrase else 'BM25'} tren `{source}` (ban gop)",
            )
        )
    return arms


def build_flat_dense_arms(bundle: FlatBundle, weights=None, enabled=None) -> list[Arm]:
    """Một `Arm` mỗi mục `DENSE_ARM_SPECS`, cosine bằng một phép nhân ma trận."""
    weights = weights or {}
    enabled = enabled or {}
    arms: list[Arm] = []
    for name, (_collection, dim, checkpoint, language, description) in DENSE_ARM_SPECS.items():
        if name not in bundle.vectors:
            arms.append(
                Arm(
                    name=name,
                    search=None,
                    weight=float(weights.get(name, 1.0)),
                    enabled=bool(enabled.get(name, True)),
                    query_language=language,
                    unavailable_reason=bundle.missing.get(
                        name, f"khong co vector {name} trong ban gop"
                    ),
                    description=description,
                )
            )
            continue

        matrix = bundle.vectors[name]
        if matrix.shape[1] != dim:
            arms.append(
                Arm(
                    name=name, search=None,
                    weight=float(weights.get(name, 1.0)),
                    query_language=language,
                    unavailable_reason=(
                        f"ban gop co {matrix.shape[1]} chieu, arm cho doi {dim}"
                    ),
                    description=description,
                )
            )
            continue

        def search(q, k, run, _name=name, _ckpt=checkpoint, _dim=dim):
            vector = encode_query(q, _ckpt, _dim)
            scores = bundle.vectors[_name] @ vector
            keys = bundle.vector_keys[_name]
            top = np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)):]
            top = top[np.argsort(scores[top])[::-1]]
            return _candidates([(keys[i], scores[i]) for i in top], _name, run)

        arms.append(
            Arm(
                name=name,
                search=search,
                weight=float(weights.get(name, 1.0)),
                enabled=bool(enabled.get(name, True)),
                query_language=language,
                description=f"{description} (ban gop, cosine day du)",
            )
        )
    return arms
