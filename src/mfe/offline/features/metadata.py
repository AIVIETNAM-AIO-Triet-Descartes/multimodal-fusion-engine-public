"""Source metadata text — OPTIONAL module. docs/02 §3, feeds the Inverted Index.

Title / description / tags from the source (YouTube metadata was provided in
past years — docs/context/buoi-1.md slide 48). Optional because it depends
entirely on provenance: a YouTube clip has it, a raw camera feed does not
(docs/02 §3).

Note this is per-*video* data being denormalized onto every keyframe of that
video (docs/03 §2.1 shows `metadata_text` on the keyframe row). That is
intentional — the Inverted Index is a flat document view keyed by keyframe_id —
but it means the same string repeats across every row of a video, and it cannot
discriminate *within* a video.
"""

from __future__ import annotations

from typing import Any

from mfe.offline.features.base import FeatureModule


class MetadataModule(FeatureModule):
    """Produces InvertedRecord.metadata_text. Empty string when unavailable.

    Owner: Quốc Đạt. TODO(Quốc Đạt): implement.

    Unlike the other modules this reads a sidecar file, not the image bytes —
    so `extract` ignores its argument and the module is really a per-video
    lookup. If that shape fights the FeatureModule interface, say so rather than
    contorting: the interface serves the modules, not the reverse.
    """

    config_key = "metadata"

    def is_enabled(self, models_config: Any) -> bool:
        raise NotImplementedError("TODO(Quốc Đạt): return False when models.metadata is null")

    def load(self, model_name: str) -> None:
        raise NotImplementedError("TODO(Quốc Đạt): locate the metadata sidecar")

    def extract(self, image_bytes: bytes) -> str:
        raise NotImplementedError("TODO(Quốc Đạt): per-video metadata -> text")
