"""Audit mapping/embedding contracts without modifying dataset files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mfe.stores.production.audit import audit_indexing_core, audit_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--indexing-core",
        action="store_true",
        help="audit mapping/vector/keyframe/object trees even when videos/media-info are absent",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="omit the per-video detail list from JSON output",
    )
    parser.add_argument(
        "--only-anomalies",
        action="store_true",
        help="retain only per-video entries that contain reported anomalies",
    )
    parser.add_argument(
        "--deep-object-json",
        action="store_true",
        help="parse every object JSON file; slower but detects corrupt artifacts",
    )
    args = parser.parse_args()
    audit = (
        audit_indexing_core(args.root, validate_object_json=args.deep_object_json)
        if args.indexing_core
        else audit_sample(args.root)
    )
    if args.only_anomalies and "videos" in audit:
        audit["videos"] = [video for video in audit["videos"] if video.get("anomalies")]
    if args.summary:
        audit.pop("videos", None)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
