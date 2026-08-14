"""Readers for the artifacts BTC ships with the dataset — docs/02 §3.

Sprint 1 does not run a single checkpoint. The Kaggle "AI Challenge 2025"
release already contains, per video:

    keyframes/{video_id}/{n:03d}.jpg    the keyframe images themselves
    map-keyframes/{video_id}.csv        n, pts_time, fps, frame_idx  <- SOURCE OF TRUTH
    clip-features/{video_id}.npy        (n_keyframes, D) float16 CLIP embeddings
    objects/{video_id}/{n:03d}.json     object detection output
    media-info/{video_id}.json          title / keywords / author / length

So `offline/features/*.py` (which loads models) stays untouched and this module
supplies the same signals by reading files. `config.features.source` selects
between the two; see `mfe.offline.features.base.build_enabled_modules`.

Everything here is deliberately dependency-light — numpy for the .npy, stdlib
for the rest, including a small JPEG header parser so that reading an image's
dimensions does not drag Pillow or OpenCV into the validator's install.
"""

from __future__ import annotations

import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfe.contracts.errors import ContractError, ErrorCode

#: Tolerance on an embedding's L2 norm, matching contracts.records._L2_TOLERANCE.
#: float16 -> float32 leaves norms a few 1e-4 off 1.0, which is why we
#: renormalize on read rather than trusting the shipped vectors.
_L2_TOLERANCE = 1e-3

#: JPEG frame markers that carry image dimensions. Everything else is skipped.
_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


@dataclass(frozen=True)
class MapRow:
    """One row of `map-keyframes/{video_id}.csv`.

    These four numbers are copied into the Relational Map verbatim. Recomputing
    any of them is the bug docs/adr/0002 exists to prevent.
    """

    n: int
    pts_time: float
    fps: float
    frame_idx: int


def read_jpeg_size(data: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` from JPEG bytes, without an image library.

    Walks the marker segments to the Start-Of-Frame header, which is the only
    place the dimensions live. Raises rather than guessing: a wrong width here
    would be written into AssetRecord and silently mis-scale the browser grid.
    """
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise ContractError(ErrorCode.INVALID_ARGUMENT, "not a JPEG (missing SOI marker)")

    offset = 2
    end = len(data)
    while offset < end:
        # Segments are 0xFF <marker>; fill bytes of 0xFF are legal padding.
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < end and data[offset] == 0xFF:
            offset += 1
        if offset >= end:
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue  # standalone markers, no length field
        if offset + 2 > end:
            break
        (segment_length,) = struct.unpack(">H", data[offset : offset + 2])
        if marker in _JPEG_SOF_MARKERS:
            if offset + 7 > end:
                break
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return int(width), int(height)
        offset += segment_length

    raise ContractError(ErrorCode.INVALID_ARGUMENT, "JPEG has no Start-Of-Frame header")


class ProvidedDataset:
    """Typed access to the shipped artifacts for one dataset root.

    Owner: Quốc Đạt. Every path comes from `config.paths` — nothing is hardcoded
    (docs/00 §7).
    """

    def __init__(
        self,
        keyframes_dir: str,
        map_keyframes_dir: str,
        clip_features_dir: str,
        objects_dir: str,
        media_info_dir: str,
        videos_dir: str,
    ) -> None:
        self.keyframes_dir = Path(keyframes_dir)
        self.map_keyframes_dir = Path(map_keyframes_dir)
        self.clip_features_dir = Path(clip_features_dir)
        self.objects_dir = Path(objects_dir)
        self.media_info_dir = Path(media_info_dir)
        self.videos_dir = Path(videos_dir)

    def video_ids(self) -> list[str]:
        """Every video with a map-keyframes CSV, sorted.

        The CSV is the entry point on purpose: a video without one has no
        coordinates, so it cannot produce a submittable keyframe and indexing it
        would only create candidates the validator must later reject.
        """
        if not self.map_keyframes_dir.is_dir():
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE,
                f"map-keyframes directory not found: {self.map_keyframes_dir}. "
                "Extract the dataset into data/sample/ — see data/README.md.",
            )
        return sorted(path.stem for path in self.map_keyframes_dir.glob("*.csv"))

    def read_map_keyframes(self, video_id: str) -> list[MapRow]:
        """Parse `map-keyframes/{video_id}.csv` into ordered rows.

        Validates that `n` runs 1..N without gaps. A gap would silently shift
        every embedding's row index relative to its keyframe — the exact
        off-by-one docs/03 §4 requires the sidecar to be 100% free of.
        """
        source = self.map_keyframes_dir / f"{video_id}.csv"
        if not source.is_file():
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE, f"missing map-keyframes CSV: {source}"
            )

        rows: list[MapRow] = []
        with source.open(encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(csv.DictReader(handle), start=2):
                try:
                    rows.append(
                        MapRow(
                            n=int(raw["n"]),
                            pts_time=float(raw["pts_time"]),
                            fps=float(raw["fps"]),
                            frame_idx=int(raw["frame_idx"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as err:
                    raise ContractError(
                        ErrorCode.INVALID_ARGUMENT,
                        f"{source}:{line_number} is not a valid map-keyframes row "
                        f"(need n, pts_time, fps, frame_idx): {err}",
                    ) from err

        if not rows:
            raise ContractError(ErrorCode.INVALID_ARGUMENT, f"{source} has no rows")

        expected = list(range(1, len(rows) + 1))
        if [row.n for row in rows] != expected:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"{source}: column n must run 1..{len(rows)} with no gaps or "
                "reordering; embeddings are matched to keyframes by row order",
            )
        return rows

    def read_clip_features(self, video_id: str, embed_dim: int) -> np.ndarray:
        """Load `clip-features/{video_id}.npy` as L2-normalized float32.

        Renormalized here rather than trusted: the file is float16, so norms
        arrive a few 1e-4 off 1.0 and `VectorRecord` rejects anything past
        1e-3 (docs/00 §4.1). Rejecting an entire ingest over a storage format's
        precision would be the wrong call; silently indexing unnormalized
        vectors would be worse, because inner product would stop being cosine.
        """
        source = self.clip_features_dir / f"{video_id}.npy"
        if not source.is_file():
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE, f"missing CLIP features: {source}"
            )

        features = np.load(source).astype(np.float32, copy=False)
        if features.ndim != 2:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"{source}: expected a 2-D (n_keyframes, D) array, got shape {features.shape}",
            )
        if features.shape[1] != embed_dim:
            raise ContractError(
                ErrorCode.SCHEMA_MISMATCH,
                f"{source}: embedding dimension {features.shape[1]} does not match "
                f"retrieval.embed_dim={embed_dim} in config",
            )
        if not np.isfinite(features).all():
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT, f"{source} contains NaN or infinity"
            )

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"{source} contains a zero vector, which has no direction to search on",
            )
        return features / norms

    def read_media_info(self, video_id: str) -> dict:
        """Load `media-info/{video_id}.json`, or `{}` when absent.

        Absence is tolerated: metadata is an optional signal (docs/02 §3), and a
        video without it should still be searchable by CLIP and by objects.
        """
        source = self.media_info_dir / f"{video_id}.json"
        if not source.is_file():
            return {}
        return json.loads(source.read_text(encoding="utf-8"))

    def read_object_labels(
        self, video_id: str, n: int, score_min: float, translate: bool = False
    ) -> list[str]:
        """Object entity names for keyframe `n`, above `score_min`, deduplicated.

        With `translate=True`, Vietnamese synonyms are appended (labels_vi.py).
        BTC names entities in English and asks in Vietnamese, so untranslated
        labels give the text arm nothing a query can match at keyframe level.

        The shipped files carry ~100 boxes per keyframe, the long tail of which
        is noise below 0.05. Indexing all of them would bury the real labels:
        BM25 weighs a term by how rare it is, and a label attached to every
        keyframe in the corpus stops discriminating between them.

        Returns entity names ("Lantern"), not the machine-readable class ids
        ("/m/01jfsr") — an operator types the former.
        """
        source = self.objects_dir / video_id / f"{n:03d}.json"
        if not source.is_file():
            return []

        payload = json.loads(source.read_text(encoding="utf-8"))
        entities = payload.get("detection_class_entities") or []
        scores = payload.get("detection_scores") or []

        labels: list[str] = []
        seen: set[str] = set()
        for entity, score in zip(entities, scores, strict=False):
            # Scores ship as strings, e.g. "0.79673874".
            if float(score) < score_min:
                continue
            name = str(entity).strip()
            key = name.casefold()
            if name and key not in seen:
                seen.add(key)
                labels.append(name)

        if translate:
            from mfe.offline.labels_vi import translate_labels

            return translate_labels(labels)
        return labels

    def keyframe_image_path(self, video_id: str, n: int) -> Path:
        """Path to `keyframes/{video_id}/{n:03d}.jpg`.

        The `{n:03d}` padding matches `contracts.ids.KEYFRAME_INDEX_DIGITS`, so
        the image filename and the keyframe_id agree by construction.
        """
        return self.keyframes_dir / video_id / f"{n:03d}.jpg"

    def video_path(self, video_id: str) -> Path:
        """Path to the source `.mp4`. Stored on VideoRow so the browser and the
        validator can reopen the original for verification (docs/05 §1.1)."""
        return self.videos_dir / f"{video_id}.mp4"
