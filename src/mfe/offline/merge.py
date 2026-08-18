"""Fold five members' per-video parts into one loadable bundle.

    docs/triet/ke-hoach-offline-ingest.md §6.3 (gộp trước hay nạp dần), §6.4 (điều kiện từ chối)

**Why merge at all, when the stores would accept the parts one at a time.** They
would: Milvus upserts by `keyframe_id`, Elasticsearch by `(keyframe_id, source)`,
so loading five shards in sequence lands the same rows as loading one merged file
— *provided the keys are disjoint and the models identical*. That proviso is the
whole argument. Loaded one at a time, a duplicate key is an upsert and the later
shard silently wins; five manifests are never in the same room, so no model
mismatch can ever be noticed. Merging first is what gives those two checks a
place to stand. It is the same class of failure as the ordinal-id trap in
`mfe.contracts.ids`: nothing raises, the data is just wrong.

So the merge always runs, and only its *output shape* follows scale: one flat
file at 16-video scale, sharded parts at batch-1 scale (§6.3). The checks are
identical either way, deliberately — a check that gets traded away for size is
one that will be traded away exactly when the data gets big enough to need it.

Refusals rather than warnings, throughout. A warning scrolls past; the damage
here is an index that looks fine and scores zero.
"""

from __future__ import annotations

import csv
import json
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from mfe.contracts.ids import parse_keyframe_id
from mfe.offline.bundle import (
    ARTIFACTS,
    NUMERIC_PACKAGES,
    OTHER_PACKAGES,
    BundleLayout,
    ShardPlan,
    utc_now,
)

if TYPE_CHECKING:  # pragma: no cover
    from mfe.offline.transport import Transport

logger = logging.getLogger(__name__)

Emit = Literal["single", "sharded"]

#: Rows above which `--emit single` is only a warning away from being a bad idea.
#: 100k x 1152 x 4B is ~0.5 GB per artifact, and concatenation peaks at twice
#: that. Batch 1 lands near 308k (§5.5b), which is why the sharded mode exists.
SINGLE_EMIT_SOFT_LIMIT = 100_000


@dataclass(frozen=True)
class Refusal:
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}"


@dataclass
class ArtifactMerge:
    name: str
    parts: int = 0
    rows: int = 0
    dropped: int = 0
    written: list[str] = field(default_factory=list)


@dataclass
class MergeReport:
    bundle: str
    videos: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    artifacts: dict[str, ArtifactMerge] = field(default_factory=dict)
    refusals: list[Refusal] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fingerprint_check: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.refusals

    def refuse(self, kind: str, detail: str) -> None:
        self.refusals.append(Refusal(kind, detail))

    def to_dict(self) -> dict:
        return {
            "bundle": self.bundle,
            "generated_at": utc_now(),
            "videos": self.videos,
            "authors": self.authors,
            "ok": self.ok,
            "fingerprint_check": self.fingerprint_check,
            "artifacts": {
                name: {
                    "parts": a.parts,
                    "rows": a.rows,
                    "dropped": a.dropped,
                    "written": a.written,
                }
                for name, a in self.artifacts.items()
            },
            "refusals": [{"kind": r.kind, "detail": r.detail} for r in self.refusals],
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Reading the bundle back off the transport
# ---------------------------------------------------------------------------


def _fetch(transport: Transport, remote: str, workdir: Path) -> Path:
    local = workdir / remote.lstrip("/").replace("/", "__")
    transport.get(remote, local)
    return local


def read_keyframe_set(
    transport: Transport, layout: BundleLayout, videos: list[str], workdir: Path
) -> tuple[set[str], dict[str, int]]:
    """Every `keyframe_id` the bundle's own `map-keyframes` declares.

    This is the set every payload key is checked against — the explicit form of
    the filter that keying by `keyframe_id` removed (`SotaIdMap.has_keyframe_id`
    carries the full argument). Duplicated `frame_idx` rows collapse here exactly
    as ingest collapses them, so the two agree on what exists.
    """
    known: set[str] = set()
    per_video: dict[str, int] = {}
    for video_id in videos:
        local = _fetch(transport, layout.map_keyframes(video_id), workdir)
        with local.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        before = len(known)
        for row in rows:
            known.add(f"{video_id}#{int(row['frame_idx']):06d}")
        per_video[video_id] = len(known) - before
    return known, per_video


def _list_parts(transport: Transport, layout: BundleLayout) -> dict[str, dict[str, str]]:
    """`{artifact: {video_id: remote_path}}` for everything actually uploaded."""
    found: dict[str, dict[str, str]] = {}
    for remote in transport.listdir(f"{layout.root}/parts"):
        pieces = remote.strip("/").split("/")
        if len(pieces) < 2:
            continue
        artifact, filename = pieces[-2], pieces[-1]
        if artifact not in ARTIFACTS:
            continue
        video_id = filename.rsplit(".", 1)[0]
        found.setdefault(artifact, {})[video_id] = remote
    return found


def _read_part_manifests(
    transport: Transport, layout: BundleLayout, videos: list[str], workdir: Path
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for video_id in videos:
        try:
            local = _fetch(transport, layout.part_manifest(video_id), workdir)
        except Exception:  # noqa: BLE001 - absence is reported by the caller
            continue
        out[video_id] = json.loads(local.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def check_fingerprints(manifests: dict[str, dict], report: MergeReport) -> None:
    """Every shard must claim the same model, params and numeric-package versions.

    This check is the entire reason `parts/manifest/` exists. Sharding by video
    buys parallelism and pays for it with five machines running one model; if two
    of them resolved different `transformers` versions, the bundle is a mixture of
    two measurements and cosine stays plausible throughout, so nothing downstream
    can tell. Comparing five manifests is the only moment that is visible.
    """
    for artifact in ARTIFACTS:
        seen: dict[tuple[str, str], list[str]] = {}
        for video_id, manifest in manifests.items():
            body = manifest.get("artifacts", {}).get(artifact)
            if not body:
                continue
            key = (
                str(body.get("model", "")),
                json.dumps(body.get("params", {}), sort_keys=True, ensure_ascii=False),
            )
            seen.setdefault(key, []).append(video_id)
        if len(seen) > 1:
            detail = "; ".join(
                f"{model} {params} <- {', '.join(sorted(videos))}"
                for (model, params), videos in sorted(seen.items())
            )
            report.refuse("model_lech", f"{artifact}: {detail}")
        elif seen:
            report.fingerprint_check[artifact] = next(iter(seen))[0]

    for package in NUMERIC_PACKAGES:
        # Absence is a value, not a skip. A package one shard loaded and another
        # never had is a divergence of exactly the kind this check exists for —
        # treating "missing" as agreement would wave through the case where one
        # member's environment is simply not the same stack.
        versions: dict[str, list[str]] = {}
        for video_id, manifest in manifests.items():
            value = manifest.get("env", {}).get(package) or "(khong cai)"
            versions.setdefault(value, []).append(video_id)
        if len(versions) > 1:
            detail = "; ".join(
                f"{v} <- {', '.join(sorted(vids))}" for v, vids in sorted(versions.items())
            )
            report.refuse("env_lech", f"{package}: {detail}")

    for package in OTHER_PACKAGES:
        seen_versions = {
            manifest.get("env", {}).get(package) or "(khong cai)"
            for manifest in manifests.values()
        }
        if len(seen_versions) > 1:
            report.warnings.append(
                f"{package} khac nhau giua cac manh ({', '.join(sorted(seen_versions))}) "
                "— khong anh huong so, chi ghi lai"
            )


def check_ownership(manifests: dict[str, dict], plan: ShardPlan, report: MergeReport) -> None:
    """A shard may only carry videos its author was assigned.

    Within one Modal workspace there is no write boundary between members
    (§6.5), so this cannot *prevent* anything — it detects. That is still worth
    having: it turns the convention in `config/ingest_shards.yaml` into something
    a machine checks, and a video written by the wrong person is exactly the shape
    a stray loop leaves behind.
    """
    for video_id, manifest in manifests.items():
        author = manifest.get("author")
        expected = plan.owner_of(video_id)
        if expected is None:
            report.refuse("video_la", f"{video_id} khong co trong bang phan cong")
        elif author != expected:
            report.refuse(
                "sai_tac_gia",
                f"{video_id} do {author!r} ghi nhung duoc giao cho {expected!r}",
            )


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _merge_text(
    transport: Transport,
    parts: dict[str, str],
    known: set[str],
    workdir: Path,
    report: MergeReport,
    artifact: str,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    owner: dict[str, str] = {}
    dropped = 0
    for video_id, remote in sorted(parts.items()):
        local = _fetch(transport, remote, workdir)
        payload = json.loads(local.read_text(encoding="utf-8"))
        for key, text in payload.items():
            if key in merged:
                report.refuse(
                    "khoa_trung",
                    f"{artifact}: {key} co trong ca {owner[key]} va {video_id}",
                )
                continue
            if key not in known:
                dropped += 1
                continue
            merged[key] = unicodedata.normalize("NFC", text)
            owner[key] = video_id
    report.artifacts[artifact].dropped += dropped
    if dropped:
        report.refuse(
            "khoa_ngoai_bo_keyframe",
            f"{artifact}: {dropped} khoa khong co trong map-keyframes cua bundle",
        )
    return merged


def _merge_vectors(
    transport: Transport,
    parts: dict[str, str],
    known: set[str],
    workdir: Path,
    report: MergeReport,
    artifact: str,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_ids: list[str] = []
    blocks: list[np.ndarray] = []
    seen: dict[str, str] = {}
    dropped = 0
    for video_id, remote in sorted(parts.items()):
        local = _fetch(transport, remote, workdir)
        payload = np.load(local, allow_pickle=False)
        ids = [str(x) for x in payload["ids"].tolist()]
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != dim:
            report.refuse(
                "so_chieu_lech",
                f"{artifact}/{video_id}: {vectors.shape}, cho doi (n, {dim})",
            )
            continue
        keep_rows: list[int] = []
        for index, key in enumerate(ids):
            if key in seen:
                report.refuse(
                    "khoa_trung", f"{artifact}: {key} co trong ca {seen[key]} va {video_id}"
                )
                continue
            if key not in known:
                dropped += 1
                continue
            seen[key] = video_id
            all_ids.append(key)
            keep_rows.append(index)
        blocks.append(vectors[keep_rows])

    report.artifacts[artifact].dropped += dropped
    if dropped:
        report.refuse(
            "khoa_ngoai_bo_keyframe",
            f"{artifact}: {dropped} khoa khong co trong map-keyframes cua bundle",
        )
    stacked = (
        np.concatenate(blocks).astype(np.float32)
        if blocks
        else np.zeros((0, dim), dtype=np.float32)
    )
    if stacked.size and not np.isfinite(stacked).all():
        report.refuse("vector_khong_huu_han", f"{artifact}: con NaN/inf sau khi gop")
    # Fixed-width unicode, never object: keeps the merged file loadable with
    # allow_pickle=False, same reason as in bundle.write_vector_part.
    return np.asarray(all_ids, dtype=np.str_), stacked


def merge_bundle(
    transport: Transport,
    plan: ShardPlan,
    workdir: str | Path,
    *,
    emit: Emit = "single",
    layout: BundleLayout | None = None,
) -> MergeReport:
    """Read every part, check it, and write the flat files. One writer only."""
    layout = layout or BundleLayout(plan.bundle)
    workdir = Path(workdir)
    (workdir / "in").mkdir(parents=True, exist_ok=True)
    staging = workdir / "in"
    outdir = workdir / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    report = MergeReport(bundle=plan.bundle, videos=plan.videos)
    known, per_video = read_keyframe_set(transport, layout, plan.videos, staging)

    manifests = _read_part_manifests(transport, layout, plan.videos, staging)
    missing_manifests = sorted(set(plan.videos) - set(manifests))
    if missing_manifests:
        report.refuse(
            "thieu_manifest",
            f"khong co parts/manifest cho: {', '.join(missing_manifests)}",
        )
    report.authors = sorted({m.get("author", "?") for m in manifests.values()})
    check_ownership(manifests, plan, report)
    check_fingerprints(manifests, report)

    found = _list_parts(transport, layout)
    wanted = [name for name, spec in ARTIFACTS.items() if spec.stage in plan.stages]

    for artifact in wanted:
        spec = ARTIFACTS[artifact]
        parts = found.get(artifact, {})
        report.artifacts[artifact] = ArtifactMerge(name=artifact, parts=len(parts))

        absent = sorted(set(plan.videos) - set(parts))
        if absent:
            report.refuse(
                "thieu_manh",
                f"{artifact}: thieu {len(absent)} video ({', '.join(absent[:5])}"
                f"{', …' if len(absent) > 5 else ''})",
            )
        extra = sorted(set(parts) - set(plan.videos))
        if extra:
            report.refuse("manh_la", f"{artifact}: co video ngoai phan cong: {', '.join(extra)}")

        if spec.kind == "text":
            merged = _merge_text(transport, parts, known, staging, report, artifact)
            report.artifacts[artifact].rows = len(merged)
            report.artifacts[artifact].written = _write_text(
                merged, outdir, spec.flat_name, emit, report
            )
        else:
            ids, vectors = _merge_vectors(
                transport, parts, known, staging, report, artifact, spec.dim or 0
            )
            report.artifacts[artifact].rows = len(ids)
            report.artifacts[artifact].written = _write_vectors(
                ids, vectors, outdir, spec.flat_name, emit, report
            )

    manifest = {
        "bundle": plan.bundle,
        "key_format": "keyframe_id",
        "keyframe_set": {"kind": plan.keyframe_set},
        "videos": plan.videos,
        "authors": report.authors,
        "counts": {"videos": len(plan.videos), "keyframes": len(known)},
        "keyframes_per_video": per_video,
        "artifacts": {
            name: {"rows": a.rows, "parts": a.parts, "dropped": a.dropped}
            for name, a in report.artifacts.items()
        },
        "fingerprint_check": report.fingerprint_check,
        "emit": emit,
        "generated_at": utc_now(),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _write_text(
    merged: dict[str, str], outdir: Path, flat_name: str, emit: Emit, report: MergeReport
) -> list[str]:
    target = outdir / "data" / "vectors"
    target.mkdir(parents=True, exist_ok=True)
    if emit == "single":
        if len(merged) > SINGLE_EMIT_SOFT_LIMIT:
            report.warnings.append(
                f"{flat_name}: {len(merged)} dong voi --emit single; can nhac sharded"
            )
        path = target / flat_name
        path.write_text(
            json.dumps(merged, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
        )
        return [str(path)]

    stem = flat_name.rsplit(".", 1)[0]
    shard_dir = target / stem
    shard_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, dict[str, str]] = {}
    for key, text in merged.items():
        by_video.setdefault(parse_keyframe_id(key)[0], {})[key] = text
    written = []
    for video_id, payload in sorted(by_video.items()):
        path = shard_dir / f"part-{video_id}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
        )
        written.append(str(path))
    (shard_dir / "index.json").write_text(
        json.dumps({"parts": [Path(p).name for p in written]}, indent=1), encoding="utf-8"
    )
    return written


def _write_vectors(
    ids: np.ndarray,
    vectors: np.ndarray,
    outdir: Path,
    flat_name: str,
    emit: Emit,
    report: MergeReport,
) -> list[str]:
    target = outdir / "data" / "vectors"
    target.mkdir(parents=True, exist_ok=True)
    if emit == "single":
        if len(ids) > SINGLE_EMIT_SOFT_LIMIT:
            report.warnings.append(
                f"{flat_name}: {len(ids)} dong voi --emit single; can nhac sharded"
            )
        path = target / flat_name
        np.savez_compressed(path, ids=ids, vectors=vectors)
        return [str(path)]

    stem = flat_name.rsplit(".", 1)[0]
    shard_dir = target / stem
    shard_dir.mkdir(parents=True, exist_ok=True)
    order: dict[str, list[int]] = {}
    for index, key in enumerate(ids.tolist()):
        order.setdefault(parse_keyframe_id(str(key))[0], []).append(index)
    written = []
    for video_id, rows in sorted(order.items()):
        path = shard_dir / f"part-{video_id}.npz"
        np.savez_compressed(path, ids=ids[rows], vectors=vectors[rows])
        written.append(str(path))
    (shard_dir / "index.json").write_text(
        json.dumps({"parts": [Path(p).name for p in written]}, indent=1), encoding="utf-8"
    )
    return written
