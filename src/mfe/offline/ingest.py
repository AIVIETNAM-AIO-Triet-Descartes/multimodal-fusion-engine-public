"""Ingestion entrypoint — docs/02 §2.

    python -m mfe.offline.ingest --config config/local.yaml
    mfe-ingest --config config/local.yaml

Pipeline (docs/02 §2), as it runs in Sprint 1:

    data/sample/
      -> [1] probe metadata (fps, duration, resolution) from map-keyframes + media-info
      -> [2] register provided keyframes  -> Asset Store + Relational Map rows
      -> [3] read the provided features per keyframe:
             CLIP embedding   clip-features/{video_id}.npy -> Vector Index
             object detection objects/{video_id}/{n}.json  -> Inverted Index
             metadata text    media-info/{video_id}.json   -> Inverted Index
             OCR              not shipped; off in Sprint 1 (docs/02 §3)
      -> [4] write the 4 stores per the record format in docs/00 §4

Step [3] loads no checkpoint: `config.features.source` is `provided`, and BTC
ships the features already computed. Swapping to `model` is a config change, and
the modules it would run are in `features/*.py`.

Success here is not accuracy — it is reproducibility. docs/02 §1: "chĩa vào thư
mục bất kỳ, chạy một lệnh, ra đủ 4 kho, không crash".
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.records import InvertedRecord, VectorRecord
from mfe.core.cli import setup_console
from mfe.core.config import Config, load_config
from mfe.core.storage import get_backend
from mfe.mapping.relational import RelationalMap
from mfe.offline.keyframes import assign_shot_ids, extract_keyframes
from mfe.offline.probe import probe_from_provided
from mfe.offline.provided import ProvidedDataset, read_jpeg_size
from mfe.offline.throughput import ThroughputRecorder
from mfe.offline.writers import StoreWriter
from mfe.stores.asset.manifest import AssetStore
from mfe.stores.inverted.index import InvertedIndex
from mfe.stores.vector.index import VectorIndex

logger = logging.getLogger(__name__)

#: float32 bytes per embedding component, for the storage projection of docs/02 §5.
_VECTOR_BYTES_PER_COMPONENT = 4


def _warn_untranslated_labels(config: Config, dataset: ProvidedDataset) -> None:
    """Report detector classes with no Vietnamese entry in `labels_vi`.

    An unmapped label is not an error — it degrades to English-only, which is
    what every label did before the table existed. But it is invisible: the
    keyframe still indexes, the run still succeeds, and the only symptom is a
    query that never matches. So the gap is counted and named here, where the
    person who just changed the dataset is watching.
    """
    if not config.features.translate_labels:
        return

    from mfe.offline.labels_vi import missing_labels

    seen: set[str] = set()
    for video_id in dataset.video_ids():
        for row in dataset.read_map_keyframes(video_id):
            seen.update(
                dataset.read_object_labels(
                    video_id, row.n, config.features.object_score_min, translate=False
                )
            )

    missing = missing_labels(seen)
    if missing:
        shown = ", ".join(missing[:15])
        logger.warning(
            "%d/%d detector class(es) have no Vietnamese synonym and will only "
            "match an English query: %s%s. Add them to offline/labels_vi.py.",
            len(missing),
            len(seen),
            shown,
            "…" if len(missing) > 15 else "",
        )
    else:
        logger.info("labels_vi covers all %d detector class(es) in this dataset.", len(seen))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfe-ingest",
        description="Turn a directory of videos into the 4 stores (docs/02).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Dataset root holding keyframes/, map-keyframes/, clip-features/, "
            "objects/, media-info/, videos/. Defaults to the parent of "
            "paths.keyframes in the config."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/local.yaml",
        help="Config file. Paths, models and module on/off switches all come from here.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N videos. For the small-sample vertical slice (docs/00 §8 risk 3).",
    )
    parser.add_argument(
        "--videos",
        default=None,
        help=(
            "Comma-separated video_ids to ingest, e.g. L21_V001,L21_V002. "
            "Unlike --limit this names them, so a run stays reproducible when the "
            "dataset gains videos. Unknown ids are an error, not a silent skip."
        ),
    )
    return parser


def _open_dataset(config: Config, input_dir: str | None) -> ProvidedDataset:
    """Build a ProvidedDataset from config, optionally re-rooted at `--input`.

    `--input` exists because docs/02 §1 asks for "chĩa vào thư mục bất kỳ". It
    overrides only the input paths; output stores stay where config says, so
    pointing at a second dataset never silently redirects the stores too.
    """
    if input_dir is None:
        return ProvidedDataset(
            keyframes_dir=config.paths.keyframes,
            map_keyframes_dir=config.paths.map_keyframes,
            clip_features_dir=config.paths.clip_features,
            objects_dir=config.paths.objects,
            media_info_dir=config.paths.media_info,
            videos_dir=config.paths.videos,
        )

    root = Path(config.resolve_path(input_dir))
    return ProvidedDataset(
        keyframes_dir=str(root / "keyframes"),
        map_keyframes_dir=str(root / "map-keyframes"),
        clip_features_dir=str(root / "clip-features"),
        objects_dir=str(root / "objects"),
        media_info_dir=str(root / "media-info"),
        videos_dir=str(root / "videos"),
    )


def _metadata_text(media_info: dict, fields: list[str]) -> str:
    """Flatten the configured media-info fields into one searchable string.

    Kept to short fields (title, keywords, author) rather than the full
    description: a description is hundreds of words of channel boilerplate
    repeated across every keyframe of the video, which in BM25 terms is a term
    that appears everywhere and therefore discriminates nothing.
    """
    parts: list[str] = []
    for field in fields:
        value = media_info.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def run(
    config_path: str,
    input_dir: str | None = None,
    limit: int | None = None,
    videos: list[str] | None = None,
) -> dict:
    """Run the full offline pipeline. Owner: Quốc Đạt.

    Returns the throughput summary, so a caller (the eval harness) can report
    without re-reading the file.

    Two things that are easy to get wrong and expensive to fix:

    - `keyframe_id` comes from `mfe.contracts.ids.build_keyframe_id`. docs/02 §4
      calls this "điểm dễ sai nhất — sai một chỗ là cả pipeline lệch".
    - fps is read per video from the map-keyframes CSV, never assumed
      (docs/00 §7).
    """
    config = load_config(config_path)
    if config.features.source != "provided":
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"features.source={config.features.source!r} needs the model path "
            "(offline/features/*.py), which Sprint 1 does not implement. "
            "Set features.source: provided.",
        )

    dataset = _open_dataset(config, input_dir)
    backend = get_backend(
        config.storage.backend, root=config.repo_root, volume=config.storage.volume
    )
    recorder = ThroughputRecorder()

    video_ids = dataset.video_ids()
    if videos is not None:
        # Name the videos rather than taking a prefix: `--limit 3` happens to
        # select the pilot set today only because the list is sorted, which is a
        # coincidence that breaks the day the dataset grows.
        available = set(video_ids)
        missing = [v for v in videos if v not in available]
        if missing:
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE,
                f"--videos names {missing} but the dataset has no map-keyframes CSV for them; "
                f"available: {', '.join(sorted(available))}",
            )
        video_ids = [v for v in video_ids if v in set(videos)]
    if limit is not None:
        video_ids = video_ids[:limit]
    if not video_ids:
        raise ContractError(
            ErrorCode.STORE_UNAVAILABLE,
            "no videos found — extract the dataset into data/sample/ (see data/README.md)",
        )
    logger.info("Ingesting %d video(s): %s", len(video_ids), ", ".join(video_ids))

    Path(config.paths.relational_db).parent.mkdir(parents=True, exist_ok=True)
    relational = RelationalMap(config.paths.relational_db)
    try:
        relational.init_schema()
        writer = StoreWriter(
            relational=relational,
            vector=VectorIndex(
                store_path=config.paths.vector_store,
                embed_dim=config.retrieval.embed_dim,
            ),
            inverted=InvertedIndex(store_path=config.paths.inverted_store),
            asset=AssetStore(store_path=config.paths.asset_store, backend=backend),
        )

        # internal_id is a global running position across every video, because
        # Faiss numbers rows in one flat index — restarting it per video would
        # make the sidecar translate every hit after the first video wrongly.
        next_internal_id = 0

        for video_id in video_ids:
            recorder.start_stage("read_provided")
            map_rows = dataset.read_map_keyframes(video_id)
            media_info = dataset.read_media_info(video_id)
            first_image = dataset.keyframe_image_path(video_id, map_rows[0].n)
            frame_size = read_jpeg_size(backend.read_bytes(str(first_image)))
            recorder.end_stage("read_provided")

            video = probe_from_provided(
                video_id=video_id,
                source_path=str(dataset.video_path(video_id)),
                map_rows=map_rows,
                media_info=media_info,
                frame_size=frame_size,
            )
            writer.write_videos([video])

            recorder.start_stage("keyframes")
            pairs = list(
                extract_keyframes(
                    video=video,
                    map_rows=map_rows,
                    keyframes_dir=config.paths.keyframes,
                    backend=backend,
                )
            )
            keyframe_rows = assign_shot_ids([row for row, _ in pairs])
            asset_records = [record for _, record in pairs]
            recorder.end_stage("keyframes")

            recorder.start_stage("features")
            embeddings = dataset.read_clip_features(video_id, config.retrieval.embed_dim)
            if len(embeddings) != len(map_rows):
                raise ContractError(
                    ErrorCode.SCHEMA_MISMATCH,
                    f"{video_id}: clip-features has {len(embeddings)} rows but "
                    f"map-keyframes has {len(map_rows)}. They are matched by row "
                    "order, so a mismatch means every embedding after the gap "
                    "belongs to the wrong keyframe.",
                )

            vector_records = [
                VectorRecord(
                    internal_id=next_internal_id + offset,
                    keyframe_id=row.keyframe_id,
                    video_id=video_id,
                    embedding=embeddings[offset].tolist(),
                )
                for offset, row in enumerate(keyframe_rows)
            ]
            next_internal_id += len(vector_records)

            metadata_text = _metadata_text(media_info, config.features.metadata_fields)
            inverted_records = [
                InvertedRecord(
                    keyframe_id=row.keyframe_id,
                    video_id=video_id,
                    object_labels=dataset.read_object_labels(
                        video_id,
                        row.keyframe_index,
                        config.features.object_score_min,
                        translate=config.features.translate_labels,
                    ),
                    ocr_text="",  # not shipped with the dataset; module off (docs/02 §3)
                    metadata_text=metadata_text,
                )
                for row in keyframe_rows
            ]
            recorder.end_stage("features")

            recorder.start_stage("write")
            writer.write_keyframes(keyframe_rows)
            writer.write_assets(asset_records)
            writer.write_vectors(vector_records)
            writer.write_inverted(inverted_records)
            recorder.end_stage("write")

            recorder.record_video(
                duration_ms=video.duration_ms, keyframe_count=len(keyframe_rows)
            )
            recorder.record_bytes(
                vector_bytes=len(vector_records)
                * config.retrieval.embed_dim
                * _VECTOR_BYTES_PER_COMPONENT,
                asset_bytes=sum(record.byte_size for record in asset_records),
            )
            logger.info("%s: %d keyframes indexed", video_id, len(keyframe_rows))

        _warn_untranslated_labels(config, dataset)

        writer.verify_stores_agree()

        problems = relational.verify_mapping()
        if problems:
            raise ContractError(
                ErrorCode.SCHEMA_MISMATCH,
                f"relational map failed self-verification ({len(problems)} problem(s)); "
                f"docs/00 §1 requires 100% on the sample. First: {problems[0]}",
            )

        writer.save()
    finally:
        relational.close()

    recorder.write(config.paths.throughput)
    summary = recorder.summary()
    logger.info(
        "Done: %d video(s), %d keyframes, %.1fs wall clock",
        summary["sample"]["videos"],
        summary["sample"]["keyframes"],
        summary["sample"]["wall_clock_s"],
    )
    return summary


def main() -> None:
    setup_console()
    args = build_parser().parse_args()
    selected = [v.strip() for v in args.videos.split(",") if v.strip()] if args.videos else None
    run(config_path=args.config, input_dir=args.input, limit=args.limit, videos=selected)


if __name__ == "__main__":
    main()
