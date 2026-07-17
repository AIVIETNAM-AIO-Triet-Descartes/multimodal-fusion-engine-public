"""OCR — OPTIONAL module. docs/02 §3, feeds the Inverted Index.

Set `models.ocr: null` to disable. docs/02 §6 requires the pipeline to run
unchanged with it off.

Why optional rather than always-on: "Cực giá trị với thời sự (chữ/số), vô dụng
với camera tĩnh không chữ" (docs/02 §3). This is the place where CLIP most
reliably fails — proper nouns, numbers, signage (docs/context/buoi-1.md §8).
"""

from __future__ import annotations

from typing import Any

from mfe.offline.features.base import FeatureModule


class OcrModule(FeatureModule):
    """Produces InvertedRecord.ocr_text. Empty string when disabled.

    Owner: TV1. TODO(TV1): implement.

    - Must handle Vietnamese diacritics. The docs' own example is "BẢN TIN 18H"
      (docs/03 §2.1) — an engine that strips diacritics turns "Thời sự" into
      something no query matches.
    - Agree normalization with TV2: their term index example lowercases ("18h",
      docs/03 §2.2). If TV1 stores raw case and TV2 lowercases at query time
      only, exact match breaks.
    - This is the slowest optional module. Measure it into throughput
      (docs/02 §5) so its cost at full-dataset scale is known before committing.
    """

    config_key = "ocr"

    def is_enabled(self, models_config: Any) -> bool:
        raise NotImplementedError("TODO(TV1): return False when models.ocr is null")

    def load(self, model_name: str) -> None:
        raise NotImplementedError("TODO(TV1): load OCR engine")

    def extract(self, image_bytes: bytes) -> str:
        raise NotImplementedError("TODO(TV1): image -> text")
