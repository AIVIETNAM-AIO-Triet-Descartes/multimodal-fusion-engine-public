"""CLIP embedding — REQUIRED module. docs/02 §3, feeds the Vector Index.

The semantic backbone: it generalizes best across the data types 2026 might
bring (docs/02 §3). docs/context/buoi-1.md §4 is blunt that it is a good
baseline, not the answer — it is weak on small text, numbers, proper nouns,
relations and negation, which is what the Inverted Index is for.
"""

from __future__ import annotations

from typing import Any

from mfe.offline.features.base import FeatureModule


class ClipModule(FeatureModule):
    """Produces the L2-normalized vector stored in VectorRecord.embedding.

    Owner: TV1. TODO(TV1): implement.

    - Normalize L2 before returning. docs/00 §4.1 requires it, and
      `VectorRecord` rejects a vector whose norm is off — so an unnormalized
      return fails loudly at the boundary rather than quietly ruining cosine
      scores.
    - The vector length must equal `retrieval.embed_dim` in config (512 for
      ViT-B/32, 768 for larger). A mismatch is caught only when Faiss refuses
      the add, several thousand keyframes in.
    - The *same* checkpoint must encode queries at search time (TV3). A text
      tower from one model against an image tower from another produces
      confident nonsense.
    """

    config_key = "clip"

    def is_enabled(self, models_config: Any) -> bool:
        raise NotImplementedError("TODO(TV1)")

    def load(self, model_name: str) -> None:
        raise NotImplementedError("TODO(TV1): load CLIP checkpoint")

    def extract(self, image_bytes: bytes) -> list[float]:
        raise NotImplementedError("TODO(TV1): image -> L2-normalized embedding")
