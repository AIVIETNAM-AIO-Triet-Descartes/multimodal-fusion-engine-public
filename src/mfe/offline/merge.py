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
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from mfe.contracts.ids import parse_keyframe_id
from mfe.offline.bundle import (
    ARTIFACTS,
    NUMERIC_PACKAGES,
    OTHER_PACKAGES,
    STAGE_PACKAGES,
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
    #: `{stage: {package: agreed version}}` — what every shard that ran the stage
    #: had installed. Written only where they agreed, so it is the provenance the
    #: bundle can actually stand behind, not a sample of one shard.
    env_check: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Lech ma Lead da xem va quyet dinh giu, kem ly do. Di vao manifest cua
    #: bundle gop de bundle mang theo loi canh bao cua chinh no.
    accepted_divergences: list[dict] = field(default_factory=list)

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
            "env_check": self.env_check,
            "accepted_divergences": self.accepted_divergences,
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


def _local_for(remote: str, workdir: Path) -> Path:
    return workdir / remote.lstrip("/").replace("/", "__")


def _fetch(transport: Transport, remote: str, workdir: Path) -> Path:
    """Tải một tệp, hoặc dùng lại tệp `_prefetch` đã kéo về.

    Tệp **0 byte không được tính là đã có**: một lượt tải đứt giữa chừng để lại
    đúng hình dạng đó, và coi nó là bộ nhớ đệm sẽ biến một cú đứt mạng thành
    một mảnh rỗng đi thẳng vào bản gộp.
    """
    local = _local_for(remote, workdir)
    if local.is_file() and local.stat().st_size > 0:
        return local
    transport.get(remote, local)
    return local


def _prefetch(
    transport: Transport,
    remotes: list[str],
    workdir: Path,
    *,
    label: str,
    workers: int = 4,
    tries: int = 3,
    reuse: bool = False,
) -> None:
    """Kéo trước cả danh sách, song song và có thử lại. Thiếu một tệp là DỪNG.

    Vì sao cần: `merge_bundle.py` tải 873 manifest rồi 873 mảnh mỗi artifact,
    **tuần tự và không timeout**. Ngày 21-08-2026 lượt gộp đứng im 25 phút ở
    `L22_V008`, để lại một tệp 0 byte — Modal bận, `transport.get` không ném lỗi
    mà cũng không trả về. Vòng lặp cũ bắt `Exception` nên một cú đứt thì bỏ qua
    được, còn một cú TREO thì không: không có gì để bắt.

    **Không tự đặt luật cho "thiếu file".** Nó chỉ lo song song và thử lại; ai
    gọi nó thì giữ nguyên chính sách của mình — thiếu `map-keyframes` là
    `FileNotFoundError`, thiếu manifest là refusal `thieu_manifest`. Cho hàm này
    ném khi thiếu là đổi hai luật đó cùng lúc, và cả hai đều có test khoá.

    Cái nó kéo về được thì vòng tuần tự phía sau đọc từ đĩa; cái nó không kéo
    được thì để vòng đó xử theo đúng luật cũ. Một file vắng mặt thật sẽ ném
    ngay, không treo — chỉ có Volume bận mới treo, và chỗ đó đã thử lại 3 lần.
    """
    if not remotes:
        return
    # `reuse` là một LỰA CHỌN CÓ Ý THỨC, không phải mặc định.
    #
    # Mặc định tải lại tất cả vì `workdir` sống qua nhiều lượt: một bản cũ sẽ
    # che mất thay đổi trên Volume, kể cả việc tệp đã bị XOÁ — đó là cách một
    # phép kiểm "thiếu manifest" báo đủ trong khi manifest không còn ở đó.
    #
    # Nhưng ở quy mô batch 1 (1.746 mảnh cho một config), tải lại sau một lần
    # chạy hỏng là mười phút. Nên có đường tắt, và nó phải được gõ ra.
    todo = [
        r for r in remotes
        if not (reuse and _local_for(r, workdir).is_file()
                and _local_for(r, workdir).stat().st_size > 0)
    ]
    if not todo:
        print(f"  dung lai {len(remotes)} {label} da tai (--reuse-downloads)")
        return
    print(f"  tai {len(todo)}/{len(remotes)} {label} ({workers} luong) …")

    def grab(remote: str) -> tuple[str, str | None]:
        local = _local_for(remote, workdir)
        # XOA truoc khi tai. Khong bo qua tep da co, va khong giu tep cu:
        # `workdir` song qua nhieu luot chay, nen mot ban cu se che mat thay doi
        # tren Volume — ke ca viec no da bi XOA. Do la cach mot phep kiem
        # "thieu manifest" bao la du du trong khi manifest khong con o do nua.
        local.unlink(missing_ok=True)
        last = ""
        for attempt in range(tries):
            try:
                transport.get(remote, local)
                if local.is_file() and local.stat().st_size > 0:
                    return remote, None
                last = "tep 0 byte"
            except Exception as err:  # noqa: BLE001
                last = f"{type(err).__name__}: {err}"
            local.unlink(missing_ok=True)
            if attempt + 1 < tries:
                time.sleep(2 ** attempt)
        return remote, last

    failed: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for remote, err in pool.map(grab, todo):
            done += 1
            if err:
                failed[remote] = err
            if done % 200 == 0 or done == len(todo):
                print(f"    … {done}/{len(todo)}")
    if failed:
        sample = "; ".join(f"{Path(r).name} ({e})" for r, e in list(failed.items())[:5])
        logger.warning(
            "khong tai truoc duoc %d/%d %s sau %d lan thu — de vong tuan tu xu "
            "theo luat cua no: %s", len(failed), len(remotes), label, tries, sample,
        )
        print(f"  (canh bao) {len(failed)}/{len(remotes)} {label} chua tai duoc: {sample}")


def read_keyframe_set(
    transport: Transport, layout: BundleLayout, videos: list[str], workdir: Path,
    workers: int = 4,
    reuse: bool = False,
) -> tuple[set[str], dict[str, int]]:
    """Every `keyframe_id` the bundle's own `map-keyframes` declares.

    This is the set every payload key is checked against — the explicit form of
    the filter that keying by `keyframe_id` removed (`SotaIdMap.has_keyframe_id`
    carries the full argument). Duplicated `frame_idx` rows collapse here exactly
    as ingest collapses them, so the two agree on what exists.
    """
    known: set[str] = set()
    per_video: dict[str, int] = {}
    _prefetch(transport, [layout.map_keyframes(v) for v in videos], workdir,
              label="map-keyframes", workers=workers, reuse=reuse)
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
    transport: Transport, layout: BundleLayout, videos: list[str], workdir: Path,
    workers: int = 4,
    reuse: bool = False,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    _prefetch(transport, [layout.part_manifest(v) for v in videos], workdir,
              label="manifest", workers=workers, reuse=reuse)
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


def check_fingerprints(
    manifests: dict[str, dict],
    report: MergeReport,
    accepted: dict[str, dict] | None = None,
) -> None:
    """Every shard must claim the same model and params, and — per stage — the
    same package versions.

    This check is the entire reason `parts/manifest/` exists. Sharding by video
    buys parallelism and pays for it with five machines running one model; if two
    of them resolved different `transformers` versions, the bundle is a mixture of
    two measurements and cosine stays plausible throughout, so nothing downstream
    can tell. Comparing five manifests is the only moment that is visible.

    The environment half lives in `check_stage_environments`, which compares per
    stage rather than per manifest — see there for why the finer grain is what
    makes the check survive a bundle filled by four notebooks.
    """
    accepted = accepted or {}
    for artifact in ARTIFACTS:
        bodies = {
            video_id: body
            for video_id, manifest in manifests.items()
            if (body := manifest.get("artifacts", {}).get(artifact))
        }
        if not bodies:
            continue

        _compare(
            artifact,
            "model",
            {v: str(b.get("model", "")) for v, b in bodies.items()},
            report,
            accepted,
        )
        # Từng khoá `params` một, không phải cả dict một cục.
        #
        # So cả dict thì một lệch ở `beam_size` và một lệch ở `window` ra cùng
        # một dòng, và **chấp nhận một cái là chấp nhận cả hai** — thứ Lead ký
        # duyệt không còn là thứ được bỏ qua. So theo khoá thì mỗi quyết định
        # đúng bằng phạm vi của nó, và một khoá mới lệch về sau vẫn chặn.
        keys = sorted({k for b in bodies.values() for k in (b.get("params") or {})})
        for key in keys:
            _compare(
                artifact,
                f"params.{key}",
                {
                    v: json.dumps(
                        (b.get("params") or {}).get(key, "(khong khai)"),
                        sort_keys=True, ensure_ascii=False,
                    )
                    for v, b in bodies.items()
                },
                report,
                accepted,
            )
        if artifact not in report.fingerprint_check:
            models = {str(b.get("model", "")) for b in bodies.values()}
            if len(models) == 1:
                report.fingerprint_check[artifact] = next(iter(models))

    check_stage_environments(manifests, report)

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


def _compare(
    artifact: str,
    field_name: str,
    values: dict[str, str],
    report: MergeReport,
    accepted: dict[str, dict],
) -> None:
    """Refuse if shards disagree on one field — unless the Lead accepted it.

    An accepted divergence is **not** a silenced check. It still runs, still
    reports the exact values and which videos carry them, and the reason the
    Lead wrote is carried into the merged bundle's own manifest. The only thing
    acceptance changes is whether it stops the merge.
    """
    groups: dict[str, list[str]] = {}
    for video_id, value in values.items():
        groups.setdefault(value, []).append(video_id)
    if len(groups) < 2:
        return

    key = f"{artifact}.{field_name}"
    detail = "; ".join(
        f"{value} <- {len(videos)} video ({', '.join(sorted(videos)[:3])}"
        f"{', …' if len(videos) > 3 else ''})"
        for value, videos in sorted(groups.items())
    )
    decision = accepted.get(key)
    if decision is None:
        report.refuse("model_lech", f"{key}: {detail}")
        return
    report.accepted_divergences.append(
        {
            "key": key,
            "values": {value: sorted(videos) for value, videos in sorted(groups.items())},
            "detail": detail,
            **{k: decision[k] for k in ("reason", "decided_by", "decided_on")},
        }
    )
    report.warnings.append(
        f"LECH DA CHAP NHAN {key}: {detail} — {decision['reason']} "
        f"({decision['decided_by']}, {decision['decided_on']})"
    )


def stage_env(manifest: dict, stage: str) -> dict[str, str] | None:
    """Environment recorded for `stage`, or `None` if this manifest never recorded it.

    `None` is the third answer, and keeping it distinct from `{}` is the point of
    this function. A manifest written before per-stage env existed carries one
    top-level `env` belonging to whichever pass wrote last; reading that as this
    stage's environment is a guess, and a guess is exactly what produced the
    spurious refusals per-stage env was added to remove.
    """
    body = (manifest.get("stages") or {}).get(stage)
    if isinstance(body, dict) and body.get("env"):
        return body["env"]
    return None


def check_stage_environments(manifests: dict[str, dict], report: MergeReport) -> None:
    """Per stage, every shard that ran it must have run it in the same versions.

    Compared **per stage** rather than per manifest because one bundle is filled
    by several passes (`02` imports, `03` adds SigLIP and OCR, `04` adds
    subtitles and CLAP). A whole-manifest comparison asks "did these two videos
    finish in the same environment", which on a multi-pass bundle is answered
    "no" almost always and means almost nothing — the difference is usually
    between two notebooks, not between two measurements of the same thing.

    Which packages matter is declared per stage in `bundle.STAGE_PACKAGES`, so a
    refusal always names a version that could have moved a number this stage
    wrote.

    Stages a manifest never recorded are reported as **unknown**, not as
    agreement and not as a refusal. Nothing can be proved about them either way,
    and pretending otherwise in either direction is worse than saying so.
    """
    stages = sorted({s for m in manifests.values() for s in (m.get("stages") or {})})
    for stage in stages:
        packages = STAGE_PACKAGES.get(stage)
        if packages is None:
            report.warnings.append(
                f"tang {stage} chua khai bao trong STAGE_PACKAGES — so TAT CA goi "
                f"NUMERIC cho no. Khai bao lai de phep kiem noi dung chinh xac."
            )
            packages = NUMERIC_PACKAGES

        recorded = {v: env for v, m in manifests.items() if (env := stage_env(m, stage))}
        for package in packages:
            # Absence is a value, not a skip. A package one shard loaded and
            # another never had is a divergence of exactly the kind this check
            # exists for — treating "missing" as agreement would wave through the
            # case where one member's environment is simply not the same stack.
            versions: dict[str, list[str]] = {}
            for video_id, env in recorded.items():
                versions.setdefault(env.get(package) or "(khong cai)", []).append(video_id)
            if len(versions) > 1:
                detail = "; ".join(
                    f"{v} <- {', '.join(sorted(vids))}" for v, vids in sorted(versions.items())
                )
                report.refuse("env_lech", f"{stage}/{package}: {detail}")
            elif versions:
                report.env_check.setdefault(stage, {})[package] = next(iter(versions))

        # Manifests that hold an artifact of this stage but recorded no
        # environment for it. Countable because the artifact proves the stage
        # ran; a directory stage leaves no artifact, so it can only be counted in
        # the whole-manifest summary below.
        produced = {a for a, spec in ARTIFACTS.items() if spec.stage == stage}
        blind = sorted(
            video_id
            for video_id, manifest in manifests.items()
            if video_id not in recorded and (set(manifest.get("artifacts") or {}) & produced)
        )
        if blind:
            report.warnings.append(
                f"{stage}: {len(blind)}/{len(manifests)} manh co artifact nhung KHONG "
                f"ghi env cua tang nay — khong ket luan duoc gi ve moi truong cua chung "
                f"(vd {', '.join(blind[:5])})"
            )

    legacy = sorted(v for v, m in manifests.items() if not (m.get("stages") or {}))
    if legacy:
        report.warnings.append(
            f"{len(legacy)}/{len(manifests)} manifest khong co `stages` (ghi truoc "
            "20-08-2026): chi co MOT `env` cua luot ghi cuoi, khong biet tang nao chay "
            "trong moi truong nao. Chay lai mot tang bat ky se ghi lai `stages` cho "
            "video do; den luc do moi truong cua chung la CHUA KIEM, khong phai da dat."
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
    workers: int = 4,
    reuse: bool = False,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    owner: dict[str, str] = {}
    dropped = 0
    _prefetch(transport, [r for _, r in sorted(parts.items())], workdir,
              label=artifact, workers=workers, reuse=reuse)
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
    workers: int = 4,
    reuse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    all_ids: list[str] = []
    blocks: list[np.ndarray] = []
    seen: dict[str, str] = {}
    dropped = 0
    collapsed = 0
    _prefetch(transport, [r for _, r in sorted(parts.items())], workdir,
              label=artifact, workers=workers)
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
                if seen[key] == video_id:
                    # Trùng TRONG CÙNG MỘT VIDEO — không phải hai người đụng nhau.
                    #
                    # `map-keyframes` của BTC có 614 cặp dòng khác `n` mà cùng
                    # `frame_idx`, và khoá là `frame_idx` (`mfe.contracts.ids`).
                    # Tầng sinh vector chạy theo TỪNG DÒNG CSV nên nó xuất hai
                    # vector cho một `keyframe_id`. Đây là tính chất đã biết của
                    # dữ liệu BTC, không phải lỗi của mảnh.
                    #
                    # Giữ dòng ĐẦU = giữ `n` nhỏ nhất, đúng luật mà
                    # `SotaIdMap._frame_to_ordinal` và
                    # `scripts/build_btc_relational.py` dùng. Ba chỗ phải cùng
                    # luật: lệch nhau thì một `keyframe_id` do kho trả về sẽ trỏ
                    # tới hàng không tồn tại trong Relational Map.
                    collapsed += 1
                    continue
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
    if collapsed:
        # Cảnh báo, không từ chối — nhưng CÓ ĐẾM. Con số này phải khớp số cặp
        # trùng `frame_idx` mà `build_btc_relational.py` gộp (614 cho batch 1);
        # lệch nghĩa là hai chỗ đang gộp theo hai luật khác nhau.
        report.warnings.append(
            f"{artifact}: gop {collapsed} dong trung keyframe_id TRONG cung mot "
            "video (map-keyframes cua BTC co dong khac `n` cung `frame_idx`); "
            "giu dong dau, cung luat voi Relational Map"
        )
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
    workers: int = 4,
    reuse: bool = False,
) -> MergeReport:
    """Read every part, check it, and write the flat files. One writer only."""
    layout = layout or BundleLayout(plan.bundle)
    workdir = Path(workdir)
    (workdir / "in").mkdir(parents=True, exist_ok=True)
    staging = workdir / "in"
    outdir = workdir / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    report = MergeReport(bundle=plan.bundle, videos=plan.videos)
    known, per_video = read_keyframe_set(transport, layout, plan.videos, staging, workers, reuse)

    manifests = _read_part_manifests(transport, layout, plan.videos, staging, workers, reuse)
    missing_manifests = sorted(set(plan.videos) - set(manifests))
    if missing_manifests:
        report.refuse(
            "thieu_manifest",
            f"khong co parts/manifest cho: {', '.join(missing_manifests)}",
        )
    report.authors = sorted({m.get("author", "?") for m in manifests.values()})
    check_ownership(manifests, plan, report)
    check_fingerprints(manifests, report, plan.accepted_divergences)

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
            merged = _merge_text(transport, parts, known, staging, report, artifact, workers, reuse)
            report.artifacts[artifact].rows = len(merged)
            report.artifacts[artifact].written = _write_text(
                merged, outdir, spec.flat_name, emit, report
            )
        else:
            ids, vectors = _merge_vectors(
                transport, parts, known, staging, report, artifact, spec.dim or 0, workers, reuse
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
        # Which versions produced which stage, kept beside the model fingerprints
        # for the same reason: the merged bundle outlives the sessions that filled
        # it, and `parts/manifest/` is not shipped with it.
        "env_check": report.env_check,
        # BANG CHUNG DI THEO BUNDLE. `parts/manifest/` khong duoc ship cung ban
        # gop, nen neu cho nay khong ghi thi sau khi gop xong khong con gi tren
        # dia noi rang mot nua `subtitles` duoc giai ma kieu khac.
        "accepted_divergences": report.accepted_divergences,
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
