"""Bundle layout, per-video parts, and the done-marker protocol.

    docs/triet/ke-hoach-offline-ingest.md §1 (hợp đồng đầu ra), §5 (chống mất dữ liệu)

Six people write into one bundle from six machines, across sessions that get cut
off, and the result has to be a single artifact somebody can hand to
`ingest_sota_text.py`. Three rules make that work, and this module is where they
live:

1. **Two kinds of file, and only one of them can collide.** Anything whose path
   contains a `video_id` — images, `map-keyframes`, `objects`, and every
   `parts/` file — is disjoint by construction once video assignment is disjoint.
   Only the eight flat files under `data/vectors/` are shared, and those are
   written by exactly one person at merge time (`scripts/merge_bundle.py`). So
   the concurrent-write question never has to be answered; it is designed away.

2. **Per-video parts, not one growing flat file.** A session that dies on video 12
   loses video 12. Flattening happens once, at the end, by one writer.

3. **The marker is the commit point, never the data file.** A half-uploaded file
   still exists, so "does the file exist" cannot mean "is it done". A marker is
   written only after the bytes are up *and* their hash has been read back, which
   makes the answer to "can I skip this?" trustworthy enough to build restart on.

Nothing here imports torch, transformers or OpenCV — the notebook imports this
module and supplies the models. That keeps every rule above testable on a laptop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id

if TYPE_CHECKING:  # pragma: no cover
    from mfe.offline.transport import Transport

logger = logging.getLogger(__name__)

ArtifactKind = Literal["text", "vector"]


@dataclass(frozen=True)
class Artifact:
    """One of the eight payloads that end up flat in `data/vectors/`."""

    name: str
    kind: ArtifactKind
    #: Filename once flattened. For text artifacts this **must** equal the entry
    #: in `sota_payloads.SOURCE_FILES` — the assertion below keeps the two
    #: registries from drifting, which would otherwise show up as an
    #: Elasticsearch tier that is quietly missing.
    flat_name: str
    #: Elasticsearch `source` tag; `None` for vectors.
    source: str | None
    #: Vector width; `None` for text.
    dim: int | None
    stage: str


ARTIFACTS: dict[str, Artifact] = {
    a.name: a
    for a in (
        Artifact("subtitles", "text", "subtitles.json", "subtitle", None, "n2_subtitles"),
        Artifact("ocr_text", "text", "ocr_text.json", "ocr", None, "n5_ocr_easyocr"),
        Artifact("sviptr_ocr", "text", "sviptr_ocr.json", "ocr_sviptr", None, "n5_ocr_qwen"),
        Artifact("objects_text", "text", "objects_text.json", "object", None, "n4_objects"),
        Artifact("dense_captions", "text", "dense_captions.json", "caption", None, "n6_recap"),
        Artifact("siglip_vectors", "vector", "siglip_vectors.npz", None, 1152, "n4_siglip"),
        Artifact("dense_vectors", "vector", "dense_vectors.npz", None, 1024, "n6_bge_m3"),
        Artifact("audio_features", "vector", "audio_features.npz", None, 512, "n3_audio"),
    )
}

#: Stages that produce per-video directories rather than a `parts/` payload.
DIRECTORY_STAGES = ("n1_keyframes", "n0_import_btc")

#: Packages whose version changes the numbers. `merge_bundle.py` refuses a bundle
#: whose shards disagree here; everything else only warns. Five people running
#: five `transformers` versions produce a bundle that is a mixture of two
#: measurements, and cosine stays plausible throughout — so nobody notices.
NUMERIC_PACKAGES = (
    "torch",
    "transformers",
    "tensorflow",
    "tensorflow-hub",
    "faster-whisper",
    "sentence-transformers",
    "easyocr",
    "numpy",
)

#: Recorded for provenance, ignored when comparing shards.
OTHER_PACKAGES = ("modal", "opencv-python", "pillow", "librosa", "google-genai")

_SHA_CHUNK = 1 << 20


def _assert_registries_agree() -> None:
    from mfe.offline.adapters.sota_payloads import SOURCE_FILES

    for artifact in ARTIFACTS.values():
        if artifact.kind != "text":
            continue
        expected = SOURCE_FILES.get(artifact.source or "")
        if expected != artifact.flat_name:
            raise AssertionError(
                f"ARTIFACTS['{artifact.name}'] writes {artifact.flat_name} but "
                f"sota_payloads.SOURCE_FILES['{artifact.source}'] reads {expected}. "
                "A bundle written against one and read against the other loses a "
                "whole tier without raising anything."
            )


_assert_registries_agree()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SHA_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleLayout:
    """Every path in §1.1, in one place.

    Paths are built here and nowhere else, because the failure mode for getting
    one wrong is silence: `ingest_sota_text.py` treats a missing file as a valid
    state (four of the five sources had no producer for most of the project), so
    a payload written one directory off produces an index quietly short a tier
    rather than an error.
    """

    bundle: str

    # -- remote (on the Volume) --------------------------------------------
    @property
    def root(self) -> str:
        return f"/bundle/{self.bundle}"

    def keyframes_dir(self, video_id: str) -> str:
        return f"{self.root}/keyframes/{video_id}"

    def map_keyframes(self, video_id: str) -> str:
        return f"{self.root}/map-keyframes/{video_id}.csv"

    def objects_dir(self, video_id: str) -> str:
        return f"{self.root}/objects/{video_id}"

    def media_info(self, video_id: str) -> str:
        return f"{self.root}/media-info/{video_id}.json"

    def frame_sizes(self, video_id: str) -> str:
        return f"{self.root}/frame_sizes/{video_id}.npy"

    def subtitle_segments(self, video_id: str) -> str:
        return f"{self.root}/subtitles_segments/{video_id}.json"

    def recap_memory(self, video_id: str) -> str:
        return f"{self.root}/recap_memory/{video_id}.json"

    def part(self, artifact: str, video_id: str) -> str:
        suffix = "npz" if ARTIFACTS[artifact].kind == "vector" else "json"
        return f"{self.root}/parts/{artifact}/{video_id}.{suffix}"

    def part_manifest(self, video_id: str) -> str:
        return f"{self.root}/parts/manifest/{video_id}.json"

    def flat(self, artifact: str) -> str:
        return f"{self.root}/data/vectors/{ARTIFACTS[artifact].flat_name}"

    @property
    def manifest(self) -> str:
        return f"{self.root}/manifest.json"

    def marker(self, member: str, stage: str, video_id: str) -> str:
        return f"/_status/{member}/{stage}/{video_id}.json"

    def marker_dir(self, member: str) -> str:
        return f"/_status/{member}"

    def calibration(self, member: str) -> str:
        return f"/_status/{member}/_calib/report.json"


# ---------------------------------------------------------------------------
# Who runs what
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardPlan:
    """`config/ingest_shards.yaml`, validated.

    Validation is not ceremony: the disjointness check below is the single
    assumption every later simplification rests on. Two members sharing a video
    means two writers on one path, duplicate keys at merge, and — if it survived
    to Milvus — an upsert that silently keeps whichever shard was loaded last.
    """

    bundle: str
    keyframe_set: str
    calibration_video: str
    members: dict[str, list[str]]
    stages: list[str]
    #: `{member: [zip name, …]}` when work is split by archive. Empty otherwise.
    #: BTC ships videos in 14 grouped zips, so a member downloads archives, not
    #: files — see `zips_for`.
    zips: dict[str, list[str]] = field(default_factory=dict)
    #: `{zip name: url}` for whatever this plan covers.
    zip_urls: dict[str, str] = field(default_factory=dict)
    #: `{zip name: [video_id, …]}`, measured from each archive's central directory.
    #: Read rather than guessed from the name: `Videos_L26_a.zip` … `_e.zip` all
    #: start with `L26`, so deriving the contents from the filename would give
    #: every one of the five the whole group.
    zip_videos: dict[str, list[str]] = field(default_factory=dict)

    #: Exact size of each archive, measured by `make_zip_inventory.py`. The only
    #: thing that tells a finished download from a plausible-looking one: `wget
    #: -c` resuming against a server that ignores `Range` appends a second copy
    #: of the body instead of continuing, and the file then looks complete.
    zip_bytes: dict[str, int] = field(default_factory=dict)

    def zips_for(self, member: str) -> list[tuple[str, str]]:
        """`[(zip name, url), …]` this member must download. Empty if not zip-split."""
        return [(name, self.zip_urls[name]) for name in self.zips.get(member, [])]

    def videos_in(self, zip_name: str) -> list[str]:
        return list(self.zip_videos.get(zip_name, []))

    @property
    def videos(self) -> list[str]:
        return sorted(v for videos in self.members.values() for v in videos)

    def videos_for(self, member: str) -> list[str]:
        if member not in self.members:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"khong co thanh vien {member!r} trong bang phan cong; "
                f"co: {', '.join(sorted(self.members))}",
            )
        return list(self.members[member])

    def owner_of(self, video_id: str) -> str | None:
        for member, videos in self.members.items():
            if video_id in videos:
                return member
        return None


def read_video_list(path: str | Path) -> list[str]:
    """One video id per line; blanks and `#` comments ignored. Sorted on return.

    Batch 1 is 873 videos (`docs/triet/dataset_report.md` §A.1), which is why the
    ids live in a file instead of in the yaml: hand-listing 873 ids across five
    members is not a plan, it is a source of typos, and a typo here means a video
    silently belongs to nobody.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return sorted(
        line.strip() for line in lines if line.strip() and not line.startswith("#")
    )


def round_robin(video_ids: list[str], members: list[str]) -> dict[str, list[str]]:
    """Deal sorted videos to members one at a time. Disjoint by construction.

    Round robin over the *sorted* list rather than contiguous blocks, because ids
    are grouped by programme (`L21_*` news, `L25_*` lectures, …) and the groups
    differ in typical length. Contiguous blocks would hand one member every
    lecture; dealing spreads each programme across all five.

    It balances by **count, not duration** — durations are not known until the
    videos are fetched. Good enough to start, and the per-member elapsed time in
    the markers is what tells us whether it needs revisiting.

    >>> round_robin(["a", "b", "c"], ["x", "y"])
    {'x': ['a', 'c'], 'y': ['b']}
    """
    out: dict[str, list[str]] = {name: [] for name in members}
    for index, video_id in enumerate(sorted(video_ids)):
        out[members[index % len(members)]].append(video_id)
    return out


def load_shard_plan(
    path: str | Path = "config/ingest_shards.yaml",
    video_list: str | Path | None = None,
) -> ShardPlan:
    """Read the assignment, in either of its two forms.

    `explicit` spells out each member's videos — right for the 16-video sample,
    where the pilot videos must land on a specific person. `round_robin` deals a
    video-id list, which is the only workable shape at batch-1 scale.

    `video_list` overrides the path named in the yaml. The notebook uses it to
    point at a copy fetched from the Volume, which matters more than it sounds:
    **all five members must deal the same list**. Generated independently, a
    listing taken before and after a video is added shifts every assignment after
    the insertion point, and two members silently swap work already done. One
    list, written once, read by everyone.
    """
    import yaml  # noqa: PLC0415 - only the notebook and the merge script need it

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mode = raw.get("assign", "explicit")

    if mode == "zips":
        # BTC publishes 14 grouped archives, so the unit of work is an archive,
        # not a video (§5.6). Splitting by video while downloading by archive
        # would make every member pull nearly all 77 GB — exactly what splitting
        # the work exists to avoid.
        catalogue = Path(raw["zip_inventory"])
        if not catalogue.is_absolute():
            catalogue = path.parent.parent / catalogue
        book = yaml.safe_load(catalogue.read_text(encoding="utf-8"))
        host = book["host"].rstrip("/")
        zips = {member: list(names) for member, names in book["assignment"].items()}
        members = {
            member: sorted(v for name in names for v in book["zips"][name]["videos"])
            for member, names in zips.items()
        }
        plan_extra = {
            "zips": zips,
            "zip_urls": {name: f"{host}/{name}" for name in book["zips"]},
            "zip_videos": {name: list(body["videos"]) for name, body in book["zips"].items()},
            "zip_bytes": {name: int(body["bytes"]) for name, body in book["zips"].items()},
        }
    elif mode == "round_robin":
        names = list(raw["members"])
        listing = Path(video_list) if video_list else Path(raw["video_list"])
        if not listing.is_absolute():
            listing = path.parent.parent / listing
        members = round_robin(read_video_list(listing), names)
        plan_extra = {}
    elif mode == "explicit":
        members = {name: list(body["videos"]) for name, body in raw["members"].items()}
        plan_extra = {}
    else:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"assign phai la 'explicit', 'round_robin' hoac 'zips', khong phai {mode!r}",
        )

    seen: dict[str, str] = {}
    for member, videos in members.items():
        for video_id in videos:
            if video_id in seen:
                raise ContractError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"{video_id} duoc giao cho ca {seen[video_id]} va {member}. "
                    "Phan cong phai ROI NHAU — do la gia thiet moi buoc gop dua vao.",
                )
            seen[video_id] = member

    plan = ShardPlan(
        bundle=raw["bundle"],
        keyframe_set=raw["keyframe_set"],
        calibration_video=raw["calibration_video"],
        members=members,
        stages=list(raw["stages"]),
        **plan_extra,
    )
    unknown = set(plan.stages) - set(DIRECTORY_STAGES) - {a.stage for a in ARTIFACTS.values()}
    if unknown:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"stages lạ trong {path}: {sorted(unknown)}. "
            f"Biet: {sorted(set(DIRECTORY_STAGES) | {a.stage for a in ARTIFACTS.values()})}",
        )
    return plan


# ---------------------------------------------------------------------------
# Done markers
# ---------------------------------------------------------------------------


@dataclass
class StageStatus:
    """What this member has already finished, read **once** per session.

    Once, because the alternative is an existence check per (video, stage) — 9
    stages x 4 videos is 36 network round trips to answer a question one listing
    answers. On a 9-hour session budget that is not free, and it runs before any
    useful work starts.
    """

    member: str
    done: set[tuple[str, str]] = field(default_factory=set)

    @classmethod
    def load(cls, transport: Transport, layout: BundleLayout, member: str) -> StageStatus:
        status = cls(member=member)
        for remote in transport.listdir(layout.marker_dir(member)):
            parts = remote.strip("/").split("/")
            if len(parts) >= 2 and parts[-1].endswith(".json"):
                status.done.add((parts[-2], parts[-1][: -len(".json")]))
        logger.info("%s: da xong %d viec", member, len(status.done))
        return status

    def is_done(self, stage: str, video_id: str) -> bool:
        return (stage, video_id) in self.done

    def pending(self, stages: list[str], videos: list[str]) -> list[tuple[str, str]]:
        return [(s, v) for s in stages for v in videos if not self.is_done(s, v)]


# ---------------------------------------------------------------------------
# Validation — runs before anything is uploaded
# ---------------------------------------------------------------------------


def check_text_payload(mapping: dict[str, str], video_id: str) -> dict[str, str]:
    """NFC-normalise and validate a `{keyframe_id: text}` part.

    NFC at the source, not only at the ingest boundary: `StorageRecord` also
    normalises, but then only the index is clean while the file on the Volume
    keeps whatever the model emitted. Measured earlier in the project — NFD text
    scores zero on every text arm and raises nothing at all — so a file five
    people will copy around is worth normalising where it is written.
    """
    out: dict[str, str] = {}
    for key, text in mapping.items():
        got_video, frame_id = parse_keyframe_id(key)  # raises on a malformed key
        if got_video != video_id:
            raise ContractError(
                ErrorCode.INVALID_ID,
                f"manh cua {video_id} chua khoa cua {got_video}: {key}. "
                "Manh phai chi chua mot video — do la thu lam cho phep gop "
                "khong can hoa giai gi.",
            )
        if not isinstance(text, str):
            raise ContractError(ErrorCode.INVALID_ARGUMENT, f"{key}: text khong phai chuoi")
        cleaned = unicodedata.normalize("NFC", text).strip()
        if cleaned:
            out[build_keyframe_id(got_video, frame_id)] = cleaned
    return out


@dataclass
class VectorCheck:
    kept_ids: list[str]
    kept: np.ndarray
    dropped: int
    renormalised: int


def check_vectors(
    ids: list[str], vectors: np.ndarray, video_id: str, dim: int
) -> VectorCheck:
    """Drop what `StorageRecord.from_values` would reject, and count it.

    Dropping here rather than at ingest is the difference between a number in the
    manifest and a crash three days later on a different machine. `dropped` being
    non-zero is not automatically a problem — a silent video really does produce
    zero-norm audio embeddings — but it has to be *visible*, which is the same
    reason `LoadReport` counts empties.
    """
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{video_id}: vectors {vectors.shape} khong khop {len(ids)} id",
        )
    if vectors.shape[1] != dim:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{video_id}: so chieu {vectors.shape[1]}, cho doi {dim}",
        )

    vectors = np.asarray(vectors, dtype=np.float32)
    finite = np.isfinite(vectors).all(axis=1)
    norms = np.linalg.norm(vectors, axis=1)
    keep = finite & (norms > 0)

    kept_ids: list[str] = []
    for index, key in enumerate(ids):
        got_video, frame_id = parse_keyframe_id(key)
        if got_video != video_id:
            raise ContractError(
                ErrorCode.INVALID_ID, f"manh cua {video_id} chua khoa cua {got_video}: {key}"
            )
        if keep[index]:
            kept_ids.append(build_keyframe_id(got_video, frame_id))

    kept = vectors[keep]
    kept_norms = norms[keep]
    renormalised = int(np.count_nonzero(np.abs(kept_norms - 1.0) > 1e-3))
    if kept.size:
        kept = (kept / kept_norms[:, None]).astype(np.float32)
    return VectorCheck(
        kept_ids=kept_ids,
        kept=kept,
        dropped=int(len(ids) - len(kept_ids)),
        renormalised=renormalised,
    )


# ---------------------------------------------------------------------------
# The write-then-mark protocol
# ---------------------------------------------------------------------------


class BundleWriter:
    """Writes one member's work into the shared bundle, resumably.

    Every public method follows §5.2 in order: compute to a local temp file,
    validate it, upload, read the hash back, and only then write the marker. The
    ordering is the whole design — reverse the last two steps and a session that
    dies mid-upload leaves a marker claiming work that is not there, which is
    worse than no marker at all because restart will believe it.
    """

    def __init__(
        self,
        transport: Transport,
        layout: BundleLayout,
        member: str,
        workdir: str | Path,
        *,
        verify_readback: bool = True,
    ) -> None:
        self.transport = transport
        self.layout = layout
        self.member = member
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.verify_readback = verify_readback

    # -- low level ---------------------------------------------------------

    def _upload(self, local: Path, remote: str) -> str:
        digest = sha256_file(local)
        self.transport.put(local, remote)
        if self.verify_readback:
            back = self.transport.sha256(remote)
            if back is not None and back != digest:
                raise RuntimeError(
                    f"{remote}: hash doc lai khong khop ({back[:12]} != {digest[:12]}). "
                    "File len khong tron ven — chay lai viec nay."
                )
        return digest

    def mark_done(self, stage: str, video_id: str, **facts: Any) -> dict:
        """Write the marker. This, and only this, means the work is finished."""
        marker = {
            "video_id": video_id,
            "stage": stage,
            "author": self.member,
            "bundle": self.layout.bundle,
            "finished_at": utc_now(),
            **facts,
        }
        local = self.workdir / f"_marker_{stage}_{video_id}.json"
        local.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        self.transport.put(local, self.layout.marker(self.member, stage, video_id))
        local.unlink(missing_ok=True)
        return marker

    # -- artifacts ---------------------------------------------------------

    def write_text_part(
        self,
        artifact: str,
        video_id: str,
        mapping: dict[str, str],
        *,
        model: str,
        params: dict | None = None,
        elapsed_s: float | None = None,
    ) -> dict:
        spec = ARTIFACTS[artifact]
        cleaned = check_text_payload(mapping, video_id)
        local = self.workdir / f"{artifact}_{video_id}.json"
        local.write_text(
            json.dumps(cleaned, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
        )
        remote = self.layout.part(artifact, video_id)
        digest = self._upload(local, remote)
        marker = self.mark_done(
            spec.stage,
            video_id,
            artifact=artifact,
            rows=len(cleaned),
            empty=len(mapping) - len(cleaned),
            sha256=digest,
            bytes=local.stat().st_size,
            model=model,
            params=params or {},
            elapsed_s=elapsed_s,
        )
        local.unlink(missing_ok=True)
        return marker

    def write_vector_part(
        self,
        artifact: str,
        video_id: str,
        ids: list[str],
        vectors: np.ndarray,
        *,
        model: str,
        params: dict | None = None,
        elapsed_s: float | None = None,
    ) -> dict:
        spec = ARTIFACTS[artifact]
        if spec.dim is None:  # pragma: no cover - registry guarantees this
            raise ContractError(ErrorCode.INVALID_ARGUMENT, f"{artifact} khong phai vector")
        checked = check_vectors(ids, vectors, video_id, spec.dim)
        local = self.workdir / f"{artifact}_{video_id}.npz"
        # `dtype=object` here would make our own .npz require allow_pickle=True
        # on load — the exact hazard §1.5 rejects Bảo's pickled dict for, and it
        # would be worse coming from us because five machines swap these files.
        # A fixed-width unicode array loads with allow_pickle=False.
        np.savez_compressed(
            local, ids=np.asarray(checked.kept_ids, dtype=np.str_), vectors=checked.kept
        )
        remote = self.layout.part(artifact, video_id)
        digest = self._upload(local, remote)
        marker = self.mark_done(
            spec.stage,
            video_id,
            artifact=artifact,
            rows=len(checked.kept_ids),
            dropped=checked.dropped,
            renormalised=checked.renormalised,
            dim=spec.dim,
            sha256=digest,
            bytes=local.stat().st_size,
            model=model,
            params=params or {},
            elapsed_s=elapsed_s,
        )
        local.unlink(missing_ok=True)
        return marker

    def write_file(self, local: str | Path, remote: str) -> str:
        """A single side file (map-keyframes, segments, memory). No marker."""
        return self._upload(Path(local), remote)

    def write_directory(self, local_dir: str | Path, remote_dir: str) -> tuple[int, int]:
        """Upload a whole per-video directory. Returns ``(files, bytes)``.

        Hashes are not read back file by file here — for a keyframe directory
        that is thousands of extra round trips. The count-and-bytes check is what
        the marker records; a full hash sweep is `verify_bundle.py`'s job, run
        once by one person instead of by five people continuously.
        """
        local_dir = Path(local_dir)
        files = sorted(p for p in local_dir.rglob("*") if p.is_file())
        pairs = [
            (path, f"{remote_dir}/{path.relative_to(local_dir).as_posix()}") for path in files
        ]
        total = sum(path.stat().st_size for path in files)

        # One batch for the whole directory when the transport can do it. Modal's
        # `put` opens a `batch_upload` context per call, so uploading a keyframe
        # directory file by file would be ~350 separate commits per video — and
        # ~62,000 for one member's share of batch 1, against a service that asks
        # for no more than ~5 concurrent commits. Batching turns that into one.
        put_many = getattr(self.transport, "put_many", None)
        if put_many is not None:
            put_many(pairs)
        else:
            for path, remote in pairs:
                self.transport.put(path, remote)
        return len(files), total

    # -- manifest ----------------------------------------------------------

    def write_part_manifest(
        self, video_id: str, markers: list[dict], keyframes: int, *, merge: bool = False
    ) -> dict:
        """The per-video manifest. `merge=True` keeps artifacts already recorded.

        Needed as soon as **one bundle is filled by more than one pass**, which
        is exactly the shape of the BTC bundle: `02` imports the keyframes, `03`
        adds SigLIP and the two OCR sources, `04` adds subtitles and CLAP. All
        three write `part-manifest/{video}.json`, and without merging each pass
        *erases* the artifacts of the ones before it.

        That erasure is unrecoverable in practice. The markers stay on the
        Volume, so `StageStatus` correctly reports the work as done and no
        re-run will ever regenerate the manifest entry — the bundle ends up
        holding data its own manifest does not mention.

        Off by default because a single-pass bundle writes every artifact in one
        call, and there the read-back is a wasted round trip per video.
        """
        artifacts: dict[str, dict] = {}
        if merge:
            local = self.workdir / f"prev_manifest_{video_id}.json"
            try:
                self.transport.get(self.layout.part_manifest(video_id), local)
                artifacts = json.loads(local.read_text(encoding="utf-8")).get("artifacts", {})
            except Exception:  # noqa: BLE001 - chua co manifest la truong hop binh thuong
                artifacts = {}
            finally:
                local.unlink(missing_ok=True)

        artifacts.update(
            {
                m["artifact"]: {k: v for k, v in m.items() if k not in ("video_id", "stage")}
                for m in markers
                if "artifact" in m
            }
        )
        manifest = {
            "video_id": video_id,
            "author": self.member,
            "bundle": self.layout.bundle,
            "key_format": "keyframe_id",
            "counts": {"keyframes": keyframes},
            "artifacts": artifacts,
            "env": freeze_environment(),
            "generated_at": utc_now(),
        }
        local = self.workdir / f"manifest_{video_id}.json"
        local.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.transport.put(local, self.layout.part_manifest(video_id))
        local.unlink(missing_ok=True)
        return manifest


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def freeze_environment() -> dict[str, str]:
    """Installed versions of the packages that matter, `{}` entries omitted.

    Recorded per shard so `merge_bundle.py` can compare five of them. Reading it
    off the running interpreter rather than off a requirements file is the point:
    the requirements file says what was asked for, this says what actually loaded.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out: dict[str, str] = {}
    for name in (*NUMERIC_PACKAGES, *OTHER_PACKAGES):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def write_lockfile(path: str | Path = "notebooks/requirements-kaggle.lock.txt") -> Path:
    """Freeze what this session resolved, for the other four members to install.

    Step 1 of the two-step pin in `notebooks/requirements-kaggle.txt`: the first
    run records reality, everyone after installs that. Guessing pins ahead of the
    first run would look verified without being.
    """
    env = freeze_environment()
    path = Path(path)
    lines = [
        "# SINH TU DONG boi mfe.offline.bundle.write_lockfile — dung sua tay.",
        f"# Ghi lai boi lượt chạy dau tien, {utc_now()}.",
        "# Bon thanh vien con lai cai file NAY, khong cai requirements-kaggle.txt.",
        "",
    ]
    lines += [f"{name}=={value}" for name, value in sorted(env.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class Stopwatch:
    """`with Stopwatch() as t: ...` then `t.seconds`. Recorded in every marker."""

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.seconds = round(time.perf_counter() - self._start, 2)
