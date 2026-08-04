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

from mfe.contracts.errors import ContractError, ErrorCode


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

#: docs/02 §3's required/optional split. A null under `models:` means "off" for
#: an optional module and "misconfigured" for a required one — silently skipping
#: CLIP would produce an empty Vector Index and a system that returns nothing.
REQUIRED_MODULES = ("clip", "detector")


def register(name: str, module_cls: type[FeatureModule]) -> None:
    """Register a feature module under a config key. Owner: Quốc Đạt.

    Re-registering the same name is rejected rather than overwritten: two
    modules claiming `clip` is a mistake, and the one that wins would depend on
    import order.
    """
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not module_cls:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"feature module {name!r} is already registered to {existing.__name__}",
        )
    _REGISTRY[name] = module_cls


def build_enabled_modules(
    models_config: Any, features_config: Any = None
) -> list[FeatureModule]:
    """Instantiate every module config switches on. Owner: Quốc Đạt.

    Two sources of features (docs/02 §3, `config.features.source`):

    - ``provided`` — Sprint 1. BTC ships `clip-features/*.npy` and
      `objects/*.json`, so no checkpoint is loaded and no module runs. Returns
      `[]`, and `mfe.offline.provided` supplies the same signals by reading
      files. A non-null `models.*` here is a contradiction, not a preference.
    - ``model`` — later sprint. Enforces the required/optional split above:
      CLIP and the detector are required, so a null there is a config error;
      OCR and metadata are genuinely optional and a null just switches them off
      (docs/02 §6 requires the pipeline to still run with OCR off).
    """
    source = getattr(features_config, "source", "provided") if features_config else "provided"

    if source == "provided":
        configured = [
            name
            for name in ("clip", "detector", "ocr", "metadata")
            if getattr(models_config, name, None)
        ]
        if configured:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"features.source is 'provided' but models.{configured} are set. "
                "Set features.source to 'model' to run checkpoints, or clear those keys.",
            )
        return []

    modules: list[FeatureModule] = []
    for name in REQUIRED_MODULES:
        if not getattr(models_config, name, None):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"models.{name} is required when features.source is 'model' "
                "(docs/02 §3); a null there would silently produce an empty store",
            )

    for name, module_cls in _REGISTRY.items():
        model_name = getattr(models_config, name, None)
        module = module_cls()
        if not module.is_enabled(models_config):
            continue
        module.load(str(model_name))
        modules.append(module)
    return modules
