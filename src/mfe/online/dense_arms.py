"""Stage-A dense arms backed by Milvus collections.

One arm per embedding model, because Milvus holds one dimension per collection
and because the models answer different questions:

    bge_m3   BAAI/bge-m3 over the latest English dense captions.
             **Cross-lingual** — a Vietnamese query matches an English caption
             without any
             translation step, which is why this arm carries `query_language`
             "any" and why it was worth doing before SigLIP.
    subtitle_bge_m3
             The same model over the **subtitles**. Same checkpoint, different
             source: one embeds what the frame shows, the other what the frame
             says.
    siglip   google/siglip-so400m-patch14-384 over the keyframe images. Its text
             tower is English-only, so the arm declares "en" and stays dark
             unless the query supplies an English form.

The query encoder is loaded lazily and cached per checkpoint: `bge-m3` is ~2.2 GB
and must not be paid for by a run that never enables the arm.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

import numpy as np

from mfe.contracts.ids import build_keyframe_id
from mfe.contracts.tools import Candidate
from mfe.online.arms import Arm
from mfe.online.tools.evidence import create_evidence_id

logger = logging.getLogger(__name__)

#: arm name -> (collection, dimension, checkpoint, query_language, description)
DENSE_ARM_SPECS: dict[str, tuple[str, int, str, str, str]] = {
    "bge_m3": (
        "aic_keyframe_bge_m3_v1",
        1024,
        "BAAI/bge-m3",
        "any",
        "Ngữ nghĩa caption qua BGE-M3 — đa ngữ, không cần dịch truy vấn",
    ),
    # Cùng model, KHÁC nguồn: `bge_m3` nhúng caption Gemini mới nhất (thứ NHÌN
    # thấy trong khung), còn arm này nhúng **phụ đề** (thứ NGHE thấy). Nên nó
    # không phải bản sao — hai arm trả lời hai câu hỏi khác nhau, và câu QA của BTC hay hỏi
    # về lời người dẫn chương trình nói hơn là về vật trong khung.
    #
    # Nó cũng KHÁC `subtitle_partial`/`subtitle_exact` trên cùng `subtitles.json`:
    # hai arm kia khớp CHỮ (BM25 và nguyên văn), arm này khớp NGHĨA. Truy vấn
    # "người dẫn nói về ô nhiễm không khí" không trùng token nào với "chất lượng
    # môi trường đang xấu đi", nhưng cosine thì gần.
    "subtitle_bge_m3": (
        "aic_keyframe_subtitle_bge_m3_v1",
        1024,
        "BAAI/bge-m3",
        "any",
        "Ngữ nghĩa PHỤ ĐỀ qua BGE-M3 — khớp nghĩa, không khớp chữ như hai arm "
        "subtitle kia",
    ),
    "siglip": (
        "aic_keyframe_siglip_v1",
        1152,
        "google/siglip-so400m-patch14-384",
        "en",
        "Ảnh qua SigLIP SO400M — tháp text CHỈ hiểu tiếng Anh",
    ),
    # CLAP nghe ÂM THANH, không nhìn ảnh. Nó chỉ trả lời được truy vấn mô tả
    # tiếng động ("tiếng trống", "tiếng còi xe"), và trên một bộ đề toàn mô tả
    # thị giác nó sẽ đóng góp gần bằng không — đó không phải lỗi của arm.
    #
    # Vector vẫn đánh địa chỉ theo `keyframe_id` như mọi arm khác: cửa sổ âm
    # thanh được gán về keyframe gần nhất lúc offline, nên hợp nhất không cần
    # biết đây là tín hiệu khác loại.
    "clap": (
        "aic_keyframe_clap_v1",
        512,
        "laion/clap-htsat-unfused",
        "en",
        "Âm thanh qua CLAP — tháp text tiếng Anh, chỉ hợp truy vấn về tiếng động",
    ),
    "window_bge_m3": (
        "aic_window_bge_m3_v1",
        1024,
        "BAAI/bge-m3",
        "any",
        "Ngữ nghĩa bản ghi cửa sổ 15 giây qua BGE-M3",
    ),
}


#: SigLIP nạp tháp text với ĐỘ DÀI CỐ ĐỊNH 64 lúc huấn luyện, nên truy vấn phải
#: được đệm tới đúng 64 token. Đây không phải chi tiết trang trí — xem
#: `_encode_siglip`.
_SIGLIP_MAX_LENGTH = 64


#: Nạp model phải VÀO TỪNG LƯỢT, vì `ArmRegistry.run` chạy các arm song song.
#:
#: ⚠ Đây là lỗi ĐÃ XẢY RA, không phải đề phòng lý thuyết. Lượt đầu của ablation
#: 25-09 in ra:
#:
#:     arm siglip failed (cannot import name 'AutoModel' from 'transformers')
#:     arm clap   failed (cannot import name 'ClapModel' from 'transformers')
#:     quang_kis_01: 9 arm(s) answered      <- các câu sau đều 11
#:
#: `transformers` nạp lười và nạp lâu; hai luồng cùng `from transformers import`
#: lúc gói mới đi được nửa đường thì luồng thứ hai thấy một module CHƯA có
#: thuộc tính đó. `lru_cache` không cứu được: nó chỉ nhớ KẾT QUẢ, còn hai luồng
#: cùng trượt cache vẫn cùng chạy thân hàm.
#:
#: Hậu quả không phải là lỗi ồn ào mà là **mất arm im lặng** — `arms.run` bắt
#: mọi `Exception` để một arm hỏng không giết cả truy vấn, nên câu đầu tiên chạy
#: thiếu `siglip` (trọng số 2,0, arm mạnh nhất: bỏ nó là −0,093 Final) mà chỉ có
#: một dòng WARNING.
_KHOA_NAP = threading.Lock()


@lru_cache(maxsize=4)
def _load_sentence_transformer(checkpoint: str) -> Any:
    with _KHOA_NAP:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading query encoder %s (first call downloads weights)", checkpoint)
        return SentenceTransformer(checkpoint)


@lru_cache(maxsize=2)
def _load_siglip(checkpoint: str) -> Any:
    with _KHOA_NAP:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading SigLIP text tower %s", checkpoint)
        return (AutoTokenizer.from_pretrained(checkpoint),
                AutoModel.from_pretrained(checkpoint).eval())


@lru_cache(maxsize=2)
def _load_clap(checkpoint: str) -> Any:
    with _KHOA_NAP:
        from transformers import ClapModel, ClapProcessor

        logger.info("Loading CLAP text tower %s", checkpoint)
        return (ClapProcessor.from_pretrained(checkpoint),
                ClapModel.from_pretrained(checkpoint).eval())


def _pooled(out) -> np.ndarray:
    pooled = getattr(out, "pooler_output", None)
    return (pooled if pooled is not None else out).numpy()


def _encode_siglip(text: str, checkpoint: str) -> np.ndarray:
    """Tháp text SigLIP qua `transformers`, đệm tới `max_length`.

    KHÔNG dùng `SentenceTransformer` ở đây. Đo trên chính kho phẳng
    (176.707×1152): hai đường cho hai vector truy vấn **khác nhau** —
    cosine 0,658 với *"a landslide burying a mountain pass road"* và 0,798 với
    *"a cyclist in a yellow jersey raising both arms after the finish line"*,
    trùng nhau 3/10 và 7/10 ở top-10. Nguyên nhân: SigLIP huấn luyện tháp text
    ở độ dài cố định 64, còn `SentenceTransformer` đệm động theo lô.

    Cả hai đều trả về thứ hạng trông hợp lý, nên sai ở đây không nổ ra chỗ nào
    — nó chỉ làm mọi phép đo `siglip` đo một hệ khác với hệ mà
    `scripts/query_flat.py` và `app/rerank_lab.py` chạy.
    """
    import torch

    tokenizer, model = _load_siglip(checkpoint)
    batch = tokenizer(
        [text],
        padding="max_length",
        max_length=_SIGLIP_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return _pooled(model.get_text_features(**batch))[0]


def _encode_clap(text: str, checkpoint: str) -> np.ndarray:
    """Tháp text CLAP — cùng checkpoint `n3_audio` dùng cho phía âm thanh."""
    import torch

    processor, model = _load_clap(checkpoint)
    batch = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        return _pooled(model.get_text_features(**batch))[0]


#: Bao nhiêu vector truy vấn giữ lại. 64 là thừa cho một phiên: mỗi truy vấn
#: sinh tối đa 3 mục (một cho mỗi tháp text), và người vận hành không gõ lại 20
#: câu trong một lượt.
_CACHE_VECTOR = 64


@lru_cache(maxsize=_CACHE_VECTOR)
def _encode_cached(text: str, checkpoint: str, expected_dim: int) -> np.ndarray:
    """Thân của `encode_query`, nhớ theo `(text, checkpoint, dim)`.

    Vì sao cần: `bge_m3` và `subtitle_bge_m3` dùng **cùng checkpoint**
    `BAAI/bge-m3` trên **cùng chuỗi truy vấn** — hai arm, hai nguồn khác nhau,
    nhưng một vector. `lru_cache` trên `_load_sentence_transformer` chỉ nhớ
    **model**, nên model không nạp lại mà forward pass vẫn chạy hai lần. Đo
    25-09: **75 ms** mỗi lượt BGE trên CPU, tức 75 ms trả thừa mỗi truy vấn.

    Khoá phải gồm **cả checkpoint và số chiều**, không chỉ text: ba tháp text
    khác nhau cùng đi qua hàm này và cho ba vector khác nhau cho cùng một câu.

    Mảng trả về đặt **read-only**: `lru_cache` trả về CHÍNH đối tượng đó cho
    mọi lượt trúng cache, nên một người gọi lỡ tay sửa tại chỗ sẽ làm hỏng mọi
    truy vấn sau mà không có gì báo.
    """
    vector = _encode_raw(text, checkpoint, expected_dim)
    vector.setflags(write=False)
    return vector


def encode_query(text: str, checkpoint: str, expected_dim: int) -> np.ndarray:
    """Embed a query, L2-normalized, with the dimension checked. Có nhớ cache."""
    return _encode_cached(text, checkpoint, expected_dim)


def preload_query_encoders(arm_names: list[str] | None = None) -> None:
    """Load query towers before the first user query.

    Streamlit keeps the cascade in ``st.cache_resource`` for the process, so
    preloading here moves model startup cost out of the first search. The
    warm-up vector is cached too, but it is only one entry in the bounded query
    cache; subsequent user text is encoded normally.
    """
    names = list(DENSE_ARM_SPECS) if arm_names is None else arm_names
    warmup_text = "warm up query encoder"
    for name in names:
        try:
            _, dimension, checkpoint, _, _ = DENSE_ARM_SPECS[name]
        except KeyError:
            continue
        encode_query(warmup_text, checkpoint, dimension)


def _encode_raw(text: str, checkpoint: str, expected_dim: int) -> np.ndarray:
    """Mã hoá thật, không qua cache.

    Tháp text được chọn theo **họ checkpoint**, không phải theo một thư viện
    chung: SigLIP và CLAP không phải model câu, và nạp chúng qua
    `SentenceTransformer` cho ra vector khác (`_encode_siglip`). `bge-m3` thì
    đúng là model câu và đi đường `SentenceTransformer`.

    The check is not ceremony: a checkpoint that quietly returns a different
    width produces a Milvus error about the wire format, several layers away
    from the line that chose the wrong model.
    """
    if checkpoint.startswith("google/siglip"):
        vector = _encode_siglip(text, checkpoint)
    elif "clap" in checkpoint.lower():
        vector = _encode_clap(text, checkpoint)
    else:
        vector = _load_sentence_transformer(checkpoint).encode(
            text, normalize_embeddings=True
        )
    vector = np.asarray(vector, dtype=np.float32).ravel()
    if vector.shape[0] != expected_dim:
        raise ValueError(
            f"{checkpoint} produced a {vector.shape[0]}-dim query vector, "
            f"but its collection holds {expected_dim}-dim ones"
        )
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _to_candidates(hits, arm_name: str, run_id: str) -> list[Candidate]:
    return [
        Candidate(
            keyframe_id=build_keyframe_id(hit.video_id, hit.frame_id),
            video_id=hit.video_id,
            score=float(hit.score),
            evidence_id=create_evidence_id(arm_name, run_id, rank),
        )
        for rank, hit in enumerate(hits, start=1)
    ]


def build_dense_arms(
    store_factory,
    weights: dict[str, float] | None = None,
    enabled: dict[str, bool] | None = None,
) -> list[Arm]:
    """Build one `Arm` per entry of `DENSE_ARM_SPECS`.

    `store_factory(collection, dimension)` returns a ready `MilvusVectorStore`
    or raises. Raising is fine and expected — the collection may not be ingested
    yet — and becomes an unavailable arm carrying the reason.
    """
    weights = weights or {}
    enabled = enabled or {}
    arms: list[Arm] = []

    for arm_name, (collection, dim, checkpoint, language, description) in DENSE_ARM_SPECS.items():
        try:
            store = store_factory(collection, dim)
        except Exception as err:  # noqa: BLE001 — "not ingested yet" is normal
            arms.append(
                Arm(
                    name=arm_name,
                    weight=float(weights.get(arm_name, 1.0)),
                    unavailable_reason=f"collection {collection}: {err}",
                    description=description,
                    query_language=language,
                )
            )
            continue

        def search(query, top_k, run_id, _store=store, _cp=checkpoint, _d=dim, _n=arm_name):
            embedding = encode_query(query, _cp, _d)
            hits = _store.search(embedding.tolist(), top_k=min(top_k, 500))
            return _to_candidates(hits, _n, run_id)

        arms.append(
            Arm(
                name=arm_name,
                search=search,
                weight=float(weights.get(arm_name, 1.0)),
                enabled=bool(enabled.get(arm_name, True)),
                description=description,
                query_language=language,
            )
        )

    return arms
