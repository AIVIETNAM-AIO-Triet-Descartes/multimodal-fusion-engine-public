"""Query text -> CLIP vector. The missing half of stage A.

BTC ships CLIP features for every keyframe but not the model, so until now a
text query had no way into the vector space and stage A ran on BM25 alone.

The checkpoint is not a guess. The official PDF (§3, "Thông tin dữ liệu vòng sơ
tuyển – Đợt 1") states the features come from **clip-ViT-B-32** — which is the
HuggingFace identifier `sentence-transformers/clip-ViT-B-32`, not a loose
reference to the architecture. That model wraps both towers in one object and
encodes images and text into the same 512-dim space through one `.encode()`.

Matching checkpoints is not a nicety. A text tower from one CLIP against an
image tower from another produces scores that are confidently, unfalsifiably
wrong: every query returns something, ranked plausibly, and none of it means
anything. `verify_against_reference()` exists so that claim gets tested rather
than assumed — see `scripts/verify_clip_checkpoint.py`.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from mfe.contracts.errors import ContractError, ErrorCode

logger = logging.getLogger(__name__)

#: Image tower — the checkpoint named in the official PDF §3. This one must not
#: change: it is what BTC used to build the shipped features, and our index is
#: those features.
DEFAULT_CLIP_CHECKPOINT = "clip-ViT-B-32"

#: Text tower — the multilingual distillation of the same model.
#:
#: `clip-ViT-B-32` is English-only, and measured on this dataset that is not a
#: degradation but a total failure: three Vietnamese queries of completely
#: different meaning ("xe may tren duong", "dam dong nguoi", "o to mau trang")
#: returned the *same* three keyframes, because the tokenizer reduces Vietnamese
#: to near-identical unknown subwords. The results were query-independent.
#:
#: This variant is distilled to land in the **same image space**, so it drops in
#: with no re-index, no dimension change and no new collection — and it answers
#: Vietnamese correctly ("nguoi dan chuong trinh trong truong quay" -> keyframes
#: labelled Television / Human face).
#:
#: The competition asks its queries in Vietnamese (official PDF §1.1), so this
#: is the tower that matters.
DEFAULT_TEXT_CHECKPOINT = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

#: Dimension the checkpoint emits; must equal `retrieval.embed_dim` in config.
CLIP_EMBED_DIM = 512

#: Cosine similarity between our re-encoding of a keyframe and the vector BTC
#: shipped for it, above which we accept that the checkpoints match. Not 0.99:
#: the shipped features are float16 and BTC's image preprocessing (resize
#: filter, crop) is not specified, so a genuine match still lands slightly
#: below 1.0. Well clear of the ~0.3-0.6 a *different* checkpoint would give.
CHECKPOINT_MATCH_THRESHOLD = 0.90


@lru_cache(maxsize=2)
def _load_model(checkpoint: str) -> Any:
    """Load and cache the SentenceTransformer. First call downloads weights."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as err:
        raise ContractError(
            ErrorCode.STORE_UNAVAILABLE,
            "text search needs sentence-transformers: pip install -e '.[online]'",
        ) from err

    logger.info("Loading CLIP checkpoint %s (first call downloads weights)", checkpoint)
    return SentenceTransformer(checkpoint)


def encode_query_text(
    text: str, checkpoint: str = DEFAULT_TEXT_CHECKPOINT
) -> np.ndarray:
    """Encode a query into the same space as the shipped keyframe features.

    Returns an L2-normalized float32 vector of length 512. Normalized because
    the Vector Index is `IndexFlatIP` — inner product only equals cosine when
    both sides have unit norm, and an unnormalized query does not fail, it just
    ranks wrongly.
    """
    if not text or not text.strip():
        raise ContractError(ErrorCode.INVALID_ARGUMENT, "query text must not be empty")

    vector = _load_model(checkpoint).encode(text, normalize_embeddings=True)
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (CLIP_EMBED_DIM,):
        raise ContractError(
            ErrorCode.SCHEMA_MISMATCH,
            f"{checkpoint} produced shape {vector.shape}; expected ({CLIP_EMBED_DIM},)",
        )
    return vector


def encode_image(image_path: str, checkpoint: str = DEFAULT_CLIP_CHECKPOINT) -> np.ndarray:
    """Encode a keyframe image with the same checkpoint.

    Used by `search_image` and by the checkpoint verification below.
    """
    try:
        from PIL import Image
    except ImportError as err:
        raise ContractError(
            ErrorCode.STORE_UNAVAILABLE,
            "image encoding needs Pillow: pip install -e '.[online]'",
        ) from err

    with Image.open(image_path) as image:
        vector = _load_model(checkpoint).encode(
            image.convert("RGB"), normalize_embeddings=True
        )
    return np.asarray(vector, dtype=np.float32)


def verify_against_reference(
    image_path: str,
    reference_embedding: np.ndarray,
    checkpoint: str = DEFAULT_CLIP_CHECKPOINT,
) -> tuple[bool, float]:
    """Re-encode a keyframe and compare with the vector BTC shipped for it.

    Returns ``(matches, cosine_similarity)``.

    Both vectors are unit-norm, so their dot product is the cosine. Reading the
    number matters more than the boolean:

        > 0.95   same checkpoint, same preprocessing — no further doubt
        0.90-0.95 same checkpoint, our resize/crop differs slightly — fine
        0.60-0.90 suspicious; likely a related but different CLIP variant
        < 0.60   different model. Do not ship: text queries would score
                 against an image space that is not the one they live in.
    """
    ours = encode_image(image_path, checkpoint=checkpoint)
    reference = np.asarray(reference_embedding, dtype=np.float32)
    if ours.shape != reference.shape:
        raise ContractError(
            ErrorCode.SCHEMA_MISMATCH,
            f"shape mismatch: ours {ours.shape} vs reference {reference.shape}",
        )

    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, "reference embedding is the zero vector"
        )
    similarity = float(np.dot(ours, reference / reference_norm))
    return similarity >= CHECKPOINT_MATCH_THRESHOLD, similarity
