"""Every adapter's output keys must satisfy the validator's schema rule.

The two live in different files owned by different concerns — `adapters/*.py`
builds the record, `validator._check_schema` decides whether it is well-formed —
and nothing but this test connects them. Renaming a key on one side alone blocks
every valid submission with SCHEMA_MISMATCH, which reads at a glance like a data
problem rather than a rename that was only half done.

That is not hypothetical: `by_frame` emitted `frame_idx` (our internal name for
the column) while BTC calls the field `frame_id`, and adopting BTC's name meant
touching both sides.
"""

from __future__ import annotations

import pytest

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters import base as adapter_base
from mfe.mapping.validator import Validator


@pytest.fixture
def keyframe() -> KeyframeRow:
    return KeyframeRow(
        keyframe_id="L23_V001#000375",
        video_id="L23_V001",
        keyframe_index=8,
        frame_idx=375,
        pts_time=15.0,
        timestamp_ms=15000,
    )


@pytest.fixture
def video() -> VideoRow:
    return VideoRow(
        video_id="L23_V001",
        source_path="data/sample_2026/videos/L23_V001.mp4",
        fps=25.0,
        duration_ms=174800,
        total_frames=4371,
        width=1280,
        height=720,
    )


def _required_keys(schema_format: str) -> tuple[str, ...]:
    """The keys `Validator._check_schema` demands, read off the validator."""
    validator = Validator.__new__(Validator)  # no Relational Map needed
    # Probe with an empty record: the failure message names every missing key.
    failure = validator._check_schema({}, schema_format)
    if failure is None:
        return ()
    return tuple(part.strip() for part in failure.message.split(":", 1)[1].split(","))


@pytest.mark.parametrize(
    "schema_format", ["by_frame", "by_timestamp", "by_frame_range", "raw_keyframe"]
)
def test_adapter_output_satisfies_the_validator(
    schema_format: str, keyframe: KeyframeRow, video: VideoRow
) -> None:
    record = adapter_base.get_adapter(schema_format).build(keyframe, video)

    validator = Validator.__new__(Validator)
    assert validator._check_schema(record, schema_format) is None, (
        f"{schema_format}: adapter emits {sorted(record)} but the validator "
        f"requires {sorted(_required_keys(schema_format))}"
    )


def test_by_frame_uses_btc_field_name_and_the_source_frame(
    keyframe: KeyframeRow, video: VideoRow
) -> None:
    """Two things at once, because getting either wrong is a silent zero.

    The key is `frame_id` (BTC's name, PDF §2.1.1), and the value is the real
    source-video frame 375 — not the keyframe ordinal 8 that names the image
    file. Submitting the ordinal produces a perfectly well-formed record that
    points at the wrong moment.
    """
    record = adapter_base.get_adapter("by_frame").build(keyframe, video)

    assert record == {"video_id": "L23_V001", "frame_id": 375}
    assert record["frame_id"] != keyframe.keyframe_index


def test_every_registered_format_is_covered_by_this_file() -> None:
    """A new adapter must not slip in without a schema-agreement check."""
    covered = {"by_frame", "by_timestamp", "by_frame_range", "raw_keyframe"}
    assert set(adapter_base._REGISTRY) == covered, (
        "a submit format was added or removed; add it to the parametrize list above"
    )
