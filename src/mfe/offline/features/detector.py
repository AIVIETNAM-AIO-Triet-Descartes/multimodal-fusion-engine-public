"""Object detection — REQUIRED module. docs/02 §3, feeds the Inverted Index.

Useful across both data types the 2026 set might contain: news footage and
static cameras (docs/02 §3). Enables queries like "có xe máy và người"
(docs/03 §2.1) that CLIP handles only fuzzily.

docs/context/buoi-1.md slide 45 notes BTC previously shipped detections from
Faster R-CNN + InceptionResNetV2 on Open Images V4. If they ship them again,
the cheapest correct move is to ingest theirs rather than re-run our own — but
only after checking their labels join to our keyframe_ids.
"""

from __future__ import annotations

from typing import Any

from mfe.offline.features.base import FeatureModule


class DetectorModule(FeatureModule):
    """Produces InvertedRecord.object_labels.

    Owner: Quốc Đạt. TODO(Quốc Đạt): implement.

    - Return normalized label strings; Công Quang indexes them as terms (docs/03 §2.2).
      Agree the vocabulary with Công Quang first — if Quốc Đạt emits "handbag" and a query
      expands to "purse", the term never matches.
    - The confidence threshold belongs in config, not here.
    - Sprint 1 stores labels only, not boxes. Boxes become necessary for the
      spatial tools in docs/context/buoi-3.md §4 (crop/zoom); leaving room for
      them later means not baking "labels only" into the store schema.
    """

    config_key = "detector"

    def is_enabled(self, models_config: Any) -> bool:
        raise NotImplementedError("TODO(Quốc Đạt)")

    def load(self, model_name: str) -> None:
        raise NotImplementedError("TODO(Quốc Đạt): load detector checkpoint")

    def extract(self, image_bytes: bytes) -> list[str]:
        raise NotImplementedError("TODO(Quốc Đạt): image -> object labels")
