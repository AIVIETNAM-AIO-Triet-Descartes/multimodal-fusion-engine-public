"""Load SOTA_AIC's text exports into the production Elasticsearch index.

    python scripts/ingest_sota_text.py --sota-root path/to/SOTA_AIC \
        --config config/pilot.yaml --videos L21_V001,L21_V002,L21_V003

Reads Bảo's `data/vectors/*.json`, translates every keyframe id through
`map-keyframes` (`SotaIdMap`), and writes one Elasticsearch document per
`(video_id, frame_id, source)`. Milvus is untouched: these payloads are text.

Idempotent — the document id is derived from the logical key, so a re-run
updates in place rather than duplicating.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mfe.core.config import load_config
from mfe.offline.adapters.sota_ids import SotaIdMap
from mfe.offline.adapters.sota_payloads import SOURCE_FILES, load_all
from mfe.offline.provided import ProvidedDataset
from mfe.stores.production.config import ElasticsearchConfig
from mfe.stores.production.elasticsearch import ElasticsearchTextStore

logger = logging.getLogger("ingest_sota_text")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest_sota_text", description=__doc__)
    parser.add_argument("--sota-root", required=True, help="SOTA_AIC checkout root")
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument(
        "--videos", required=True, help="Comma-separated video_ids present in the export"
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCE_FILES),
        help=f"Comma-separated subset of {sorted(SOURCE_FILES)}",
    )
    parser.add_argument("--index", default=None, help="Override MFE_ELASTICSEARCH_INDEX")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and translate, report counts, write nothing.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    config = load_config(args.config)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    dataset = ProvidedDataset(
        keyframes_dir=config.paths.keyframes,
        map_keyframes_dir=config.paths.map_keyframes,
        clip_features_dir=config.paths.clip_features,
        objects_dir=config.paths.objects,
        media_info_dir=config.paths.media_info,
        videos_dir=config.paths.videos,
    )
    id_map = SotaIdMap(dataset, videos)
    logger.info("id map: %d keyframe(s) across %s", len(id_map), ", ".join(videos))

    vectors_dir = Path(args.sota_root) / "data" / "vectors"
    records, reports = load_all(vectors_dir, id_map, sources=sources)

    print("\n--- doc tu SOTA_AIC ---")
    for report in reports:
        print(f"  {report.summary()}")
    print(f"  TONG: {len(records)} ban ghi")

    missing = [r.source for r in reports if not r.present]
    if missing:
        # Not fatal: four of the five files have no producer in Bảo's repo
        # (ke-hoach-merge-sota.md §1b). Loud, so a thin index is never a surprise.
        print(f"\n  CANH BAO: thieu file cho source: {', '.join(missing)}")

    if not records:
        print("\nKhong co ban ghi nao — dung.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: khong ghi gi.")
        return

    es_config = ElasticsearchConfig.from_env()
    if args.index:
        es_config = es_config.model_copy(update={"index": args.index})
    store = ElasticsearchTextStore(es_config)

    health = store.health()
    if not health.connection_ok:
        print(f"\nElasticsearch khong ket noi duoc: {health.message}")
        sys.exit(1)

    store.ensure_schema()
    report = store.ingest(records)

    print("\n--- ghi vao Elasticsearch ---")
    print(f"  index      : {es_config.index}")
    print(f"  analyzer   : {store.analyzer_name} (mode={es_config.analyzer_mode})")
    print(f"  attempted  : {report.attempted}")
    print(f"  inserted   : {report.inserted}")
    print(f"  updated    : {report.updated}")
    print(f"  failed     : {report.failed}")
    for error in report.errors[:5]:
        print(f"    ! {error.logical_key} {error.message[:160]}")
    print(f"  tong trong index: {store.count()}")

    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
