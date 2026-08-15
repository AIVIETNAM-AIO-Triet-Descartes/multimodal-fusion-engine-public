"""Translate SOTA_AIC (Nhật Bảo) keyframe ids into ours.

    Bảo:  L21_V001_001     suffix = `n`, the keyframe's ORDINAL
    ours: L21_V001#000090  suffix = `frame_idx`, the SOURCE-VIDEO FRAME

The two suffixes look alike and mean different things. Ordinal 1 of L21_V001 is
frame 0; ordinal 250 is frame 31124. So this is **a lookup, not a string edit** —
the only authority is `map-keyframes/{video_id}.csv`, the same file the Relational
Map is built from.

An earlier draft of the merge plan assumed `_001` and `#001` were interchangeable.
They were, briefly, back when our key was also the ordinal. Moving our key to
`frame_idx` (docs/triet/ke-hoach-merge-sota.md §3.1) removed that coincidence,
which is a small gain worth naming: forgetting the lookup now yields an id that
does not exist, instead of an id that exists and points at the wrong frame.

On duplicate frames: 614 CSV row pairs in batch 1 share a `frame_idx` (two images
extracted less than one frame apart). Ingest collapses them, keeping the lower
ordinal. That needs no special case here — both ordinals map to the same
`frame_idx`, hence to the same `keyframe_id`, which is the correct answer: they
denote one frame. `to_sota_id` returns the surviving ordinal.
"""

from __future__ import annotations

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id, validate_video_id
from mfe.offline.provided import ProvidedDataset


def parse_sota_id(sota_id: str) -> tuple[str, int]:
    """Split ``L21_V001_001`` into ``("L21_V001", 1)``.

    Splits from the right: `video_id` contains underscores too, so a left split
    would read the video group as the ordinal.

    >>> parse_sota_id("L21_V001_001")
    ('L21_V001', 1)
    """
    video_id, separator, ordinal = sota_id.rpartition("_")
    if not separator or not ordinal.isdigit():
        raise ContractError(
            ErrorCode.INVALID_ID,
            f"malformed SOTA keyframe id {sota_id!r}; expected {{video_id}}_{{n:03d}}",
        )
    validate_video_id(video_id)
    return video_id, int(ordinal)


def build_sota_id(video_id: str, ordinal: int) -> str:
    """Inverse of :func:`parse_sota_id`. Bảo pads to 3 digits."""
    validate_video_id(video_id)
    if ordinal < 0:
        raise ContractError(ErrorCode.INVALID_ID, f"ordinal must be >= 0, got {ordinal}")
    return f"{video_id}_{ordinal:03d}"


class SotaIdMap:
    """Bidirectional ordinal <-> frame translation, backed by map-keyframes.

    Built once per run and reused. Reading 16 CSVs is cheap; doing it per lookup
    is not.
    """

    def __init__(self, dataset: ProvidedDataset, video_ids: list[str]) -> None:
        self._ordinal_to_frame: dict[tuple[str, int], int] = {}
        self._frame_to_ordinal: dict[tuple[str, int], int] = {}

        for video_id in video_ids:
            for row in dataset.read_map_keyframes(video_id):
                self._ordinal_to_frame[(video_id, row.n)] = row.frame_idx
                # Lowest ordinal wins, matching `collapse_duplicate_frames`. The
                # two must agree: if this kept the highest and ingest kept the
                # lowest, `to_sota_id` would name an ordinal that has no row in
                # the Relational Map.
                key = (video_id, row.frame_idx)
                if key not in self._frame_to_ordinal or row.n < self._frame_to_ordinal[key]:
                    self._frame_to_ordinal[key] = row.n

    def __len__(self) -> int:
        return len(self._ordinal_to_frame)

    @property
    def video_ids(self) -> set[str]:
        return {video_id for video_id, _ in self._ordinal_to_frame}

    def to_keyframe_id(self, sota_id: str) -> str:
        """``L21_V001_001`` -> ``L21_V001#000000``.

        Raises rather than guessing. Bảo's `p0_adapter.get_frame_idx` falls back
        to parsing the filename and finally to returning ``"0"`` — a submittable,
        plausible, wrong answer. That fallback is the reason this raises.
        """
        video_id, ordinal = parse_sota_id(sota_id)
        frame_idx = self._ordinal_to_frame.get((video_id, ordinal))
        if frame_idx is None:
            raise ContractError(
                ErrorCode.KEYFRAME_NOT_FOUND,
                f"{sota_id}: map-keyframes/{video_id}.csv has no row n={ordinal}. "
                "Not guessing a frame — a wrong frame scores 0 and looks valid.",
            )
        return build_keyframe_id(video_id, frame_idx)

    def to_sota_id(self, keyframe_id: str) -> str:
        """``L21_V001#000000`` -> ``L21_V001_001``. Only for calling back into Bảo's code."""
        video_id, frame_idx = parse_keyframe_id(keyframe_id)
        ordinal = self._frame_to_ordinal.get((video_id, frame_idx))
        if ordinal is None:
            raise ContractError(
                ErrorCode.KEYFRAME_NOT_FOUND,
                f"{keyframe_id}: map-keyframes/{video_id}.csv has no row with "
                f"frame_idx={frame_idx}",
            )
        return build_sota_id(video_id, ordinal)

    def translate_mapping(self, data: dict[str, object]) -> tuple[dict[str, object], list[str]]:
        """Re-key one of Bảo's JSON payloads. Returns ``(translated, unknown_ids)``.

        Unknown ids are *reported*, not raised: Bảo's exports may cover videos we
        did not ingest, and dropping those is correct. Silence is not — the caller
        logs the count so "the OCR arm returned nothing" can never be a mystery.
        """
        translated: dict[str, object] = {}
        unknown: list[str] = []
        for sota_id, value in data.items():
            try:
                translated[self.to_keyframe_id(sota_id)] = value
            except ContractError:
                unknown.append(sota_id)
        return translated, unknown
