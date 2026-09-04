"""Repository-local launcher for ``mfe-keyframe-augmentation``."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.offline.keyframe_augmentation.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
