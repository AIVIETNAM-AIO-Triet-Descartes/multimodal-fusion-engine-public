"""Stage-A dense arms backed by Milvus collections.

One arm per embedding model, because Milvus holds one dimension per collection
and because the models answer different questions:

    bge_m3   BAAI/bge-m3 over Bảo's English dense captions. **Cross-lingual** —
             a Vietnamese query matches an English caption without any
             translation step, which is why this arm carries `query_language`
             "any" and why it was worth doing before SigLIP.
    siglip   google/siglip-so400m-patch14-384 over the keyframe images. Its text
             tower is English-only, so the arm declares "en" and stays dark
             unless the query supplies an English form.

The query encoder is loaded lazily and cached per checkpoint: `bge-m3` is ~2.2 GB
and must not be paid for by a run that never enables the arm.
"""

from __future__ import annotations

import logging
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
    "siglip": (
        "aic_keyframe_siglip_v1",
        1152,
        "google/siglip-so400m-patch14-384",
        "en",
        "Ảnh qua SigLIP SO400M — tháp text CHỈ hiểu tiếng Anh",
    ),
}


@lru_cache(maxsize=4)
def _load_encoder(checkpoint: str) -> Any:
    from sentence_transformers import SentenceTransformer

    logger.info("Loading query encoder %s (first call downloads weights)", checkpoint)
    return SentenceTransformer(checkpoint)


def encode_query(text: str, checkpoint: str, expected_dim: int) -> np.ndarray:
    """Embed a query, L2-normalized, with the dimension checked.

    The check is not ceremony: a checkpoint that quietly returns a different
    width produces a Milvus error about the wire format, several layers away
    from the line that chose the wrong model.
    """
    vector = _load_encoder(checkpoint).encode(text, normalize_embeddings=True)
    vector = np.asarray(vector, dtype=np.float32).ravel()
    if vector.shape[0] != expected_dim:
        raise ValueError(
            f"{checkpoint} produced a {vector.shape[0]}-dim query vector, "
            f"but its collection holds {expected_dim}-dim ones"
        )
    return vector


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
