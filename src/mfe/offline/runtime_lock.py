"""Build an exact Modal image lock from the already-validated Kaggle runtime."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from mfe.offline.bundle import NUMERIC_PACKAGES

# Paddle itself is deliberately installed afterwards from the exact cu126 wheel.
# Installing either distribution here can silently put the CPU and GPU packages
# on top of the same ``paddle/`` directory — the failure Notebook 06 R7 fixed.
_PADDLE_DISTRIBUTIONS = {
    canonicalize_name("paddlepaddle"),
    canonicalize_name("paddlepaddle-gpu"),
}

_ROOTS = (
    *NUMERIC_PACKAGES,
    "paddleocr",
    "vietocr",
    "torchvision",
    "pillow",
    "pyyaml",
    "astor",
    "opt-einsum",
    "nvidia-nccl-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cuda-runtime-cu12",
)


def resolved_requirement_lock() -> list[str]:
    """Return exact installed pins plus their active dependency closure.

    The function is called only after Notebook 06 has passed its Paddle→Torch→
    Paddle GPU probe.  Modal therefore copies a measured environment instead of
    resolving the notebook's broad version ranges a second time.
    """

    excluded = {*_PADDLE_DISTRIBUTIONS, canonicalize_name("modal")}
    queue = list(_ROOTS)
    seen: set[str] = set()
    pins: dict[str, tuple[str, str]] = {}

    while queue:
        requested = queue.pop()
        canonical = canonicalize_name(requested)
        if canonical in seen or canonical in excluded:
            continue
        seen.add(canonical)
        try:
            dist = distribution(requested)
        except PackageNotFoundError:
            # Absence of optional numeric packages is part of freeze_environment;
            # a clean Modal image also leaves them absent.
            continue

        name = dist.metadata.get("Name") or requested
        pins[canonical] = (str(name), dist.version)
        for raw_requirement in dist.requires or ():
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in seen and dependency not in excluded:
                queue.append(requirement.name)

    return [
        f"{name}=={version}"
        for _canonical, (name, version) in sorted(pins.items())
    ]


def write_modal_runtime_lock(path: str | Path) -> Path:
    """Write the lock consumed by ``infra/modal/btc_ocr2.py``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sinh tu Notebook 06 SAU khi GPU probe thanh cong; khong sua tay.",
        "# Paddle GPU duoc cai rieng bang exact cu126 wheel trong Modal image.",
        *resolved_requirement_lock(),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
