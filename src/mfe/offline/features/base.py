"""Feature module interface + registry — docs/02 §3.

Each signal is a plug-in that config turns on or off. docs/02 §3: "Vì chưa biết
tính chất data 2026, mỗi tín hiệu là một module bật/tắt được qua config. Đừng
nhúng cứng."

Concretely: OCR is gold on news footage (burnt-in text, tickers, dates) and
worthless on a static camera pointed at an intersection. docs/context/buoi-3.md
§9 calls the general principle "action space gọn" — a useless module is not
free, it adds noise.

Adding a modality later (ASR is already named in docs/02 §3 as "để sau") means
adding one file here and one config key. Nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FeatureModule(ABC):
    """One extractable signal for one keyframe image."""

    #: Config key under `models:` that enables this module. Null value = off.
    config_key: str

    @abstractmethod
    def is_enabled(self, models_config: Any) -> bool:
        """Whether config asks for this module. A null model name means off."""

    @abstractmethod
    def load(self, model_name: str) -> None:
        """Load weights. Called once, not per keyframe."""

    @abstractmethod
    def extract(self, image_bytes: bytes) -> Any:
        """Extract this module's signal from one keyframe image.

        Return type is module-specific: an embedding for CLIP, labels for the
        detector, a string for OCR. `writers.py` routes each to its store.
        """


_REGISTRY: dict[str, type[FeatureModule]] = {}


def register(name: str, module_cls: type[FeatureModule]) -> None:
    """Register a feature module under a config key.

    Owner: TV1. TODO(TV1): implement.
    """
    raise NotImplementedError("TODO(TV1): register module class under name")


def build_enabled_modules(models_config: Any) -> list[FeatureModule]:
    """Instantiate every module config switches on.

    Owner: TV1. TODO(TV1): implement.

    Must enforce docs/02 §3's required/optional split: CLIP and the detector are
    required, so a null there is a config error, not a silent skip. OCR and
    metadata are genuinely optional and a null is fine. docs/02 §6 requires the
    pipeline to still run with OCR off.
    """
    raise NotImplementedError("TODO(TV1): read config -> instantiate enabled modules")
