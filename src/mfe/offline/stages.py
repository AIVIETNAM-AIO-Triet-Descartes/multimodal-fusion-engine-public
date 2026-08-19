"""The nine extraction stages, one function each — §2, §4.5, §4.6.

The notebook is a loop over these; nothing here lives in a notebook cell. That is
the §9 rule ("notebook `import` chứ không chép") and it buys three things: a stage
can be fixed without re-pasting a cell into five people's notebooks, the
non-model logic is reviewable in the repo, and every stage records its own model
and parameters into the marker the same way.

**Every model import is lazy and local to its stage.** Loading torch, TensorFlow
and OpenCV at module import would make `merge_bundle.py` — which needs none of
them — impossible to run on a laptop, and it is exactly the "load everything in
one process for six hours" shape that §4.2 blames for VRAM leaks. Each stage
loads, runs, and returns; the notebook frees between stages.

The signature is uniform on purpose: `(ctx) -> marker`. A stage that cannot say
what it produced does not get to claim it finished.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mfe.contracts.ids import build_keyframe_id, frame_idx_from_pts
from mfe.offline.bundle import BundleLayout, BundleWriter, Stopwatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model cache — nap mot lan, dung cho nhieu video
# ---------------------------------------------------------------------------

#: Loaded models, keyed by checkpoint. **Empty by default and never filled
#: implicitly** — a stage only reaches in here, and only the caller decides when
#: to let go (`unload_models`).
#:
#: Why this exists. The signature is `(ctx) -> marker`, one video per call, and
#: loading inside the call was correct while the loop was video-major over nine
#: stages: §4.2 blames "everything resident in one process" for the VRAM leaks,
#: and freeing between stages is the fix. But on the BTC bundle the loop is
#: **stage-major** — one model, every video — and there re-loading per video is
#: measured at 5-16 hours per member of pure loading, doing nothing.
#:
#: So both shapes are needed, and which one applies is the notebook's business,
#: not the stage's. A video-major notebook calls `unload_models()` between
#: stages and gets exactly the old behaviour; a stage-major one calls it between
#: *stages* and keeps the model across videos.
_MODELS: dict[str, object] = {}


def cached_model(key: str, build):
    """The model for `key`, built once. `build` is only called on a miss."""
    if key not in _MODELS:
        logger.info("nap model %s", key)
        _MODELS[key] = build()
    return _MODELS[key]


def unload_models() -> None:
    """Drop every cached model and free the VRAM.

    Call this when the *stage* changes, never merely when the video changes —
    that distinction is the whole point of the cache.
    """
    if _MODELS:
        logger.info("tha %d model", len(_MODELS))
    _MODELS.clear()
    free_gpu()


def _siglip_feature_tensor(output: Any) -> Any:
    """Return the pooled SigLIP image tensor across Transformers versions.

    Older Transformers releases returned the tensor directly from
    ``get_image_features``. Newer releases return ``BaseModelOutputWithPooling``
    and put the projected image embedding in ``pooler_output``. Keeping this
    compatibility at the stage boundary prevents dependency drift from turning
    every video into the same per-video failure.
    """
    pooled = getattr(output, "pooler_output", None)
    features = pooled if pooled is not None else output
    if not callable(getattr(features, "float", None)):
        raise TypeError(
            "SigLIP get_image_features() returned neither a tensor nor a "
            "non-empty pooler_output"
        )
    return features


@dataclass(frozen=True)
class Keyframe:
    """One row of `map-keyframes`, plus where its image sits locally."""

    n: int
    pts_time: float
    fps: float
    frame_idx: int
    image: Path

    @property
    def keyframe_id_for(self) -> str:  # pragma: no cover - trivial
        raise NotImplementedError  # video_id lives on the context, see `ids()`


@dataclass
class StageContext:
    """Everything a stage needs, and nothing about how it is scheduled."""

    video_id: str
    video_path: Path
    keyframes: list[Keyframe]
    writer: BundleWriter
    layout: BundleLayout
    workdir: Path
    params: dict = field(default_factory=dict)

    def ids(self) -> list[str]:
        return [build_keyframe_id(self.video_id, kf.frame_idx) for kf in self.keyframes]

    def images(self) -> list[Path]:
        return [kf.image for kf in self.keyframes]

    def audio_path(self) -> Path:
        """16 kHz mono wav, extracted once and shared by N2 and N3."""
        wav = self.workdir / f"{self.video_id}.wav"
        if not wav.is_file():
            import subprocess  # noqa: PLC0415

            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-i", str(self.video_path),
                 "-vn", "-ac", "1", "-ar", "48000", str(wav)],
                check=True, capture_output=True,
            )
        return wav


def read_keyframes(map_csv: Path, keyframes_dir: Path) -> list[Keyframe]:
    """Load `map-keyframes/{video}.csv` and locate each image.

    `frame_idx` is **copied**, never recomputed — copying beats recomputing, and
    `mfe.contracts.ids` is emphatic about which of the two is the contract. The
    consistency check below is therefore a real check rather than a tautology:
    it compares BTC's number against the formula, and it is the thing that
    catches an fps read off the wrong video.
    """
    rows = list(csv.DictReader(map_csv.open(encoding="utf-8")))
    out: list[Keyframe] = []
    for row in rows:
        n = int(row["n"])
        pts_time, fps = float(row["pts_time"]), float(row["fps"])
        frame_idx = int(row["frame_idx"])
        derived = frame_idx_from_pts(pts_time, fps)
        if derived != frame_idx:
            raise ValueError(
                f"{map_csv.name} n={n}: frame_idx={frame_idx} nhung "
                f"floor({pts_time}*{fps})={derived}. Sai fps hoac sai file."
            )
        out.append(
            Keyframe(n, pts_time, fps, frame_idx, keyframes_dir / f"{n:03d}.jpg")
        )
    if [kf.n for kf in out] != list(range(1, len(out) + 1)):
        raise ValueError(f"{map_csv.name}: cot n phai chay 1..N lien tuc, khong ho")
    return out


# ---------------------------------------------------------------------------
# N1 — DAKE
# ---------------------------------------------------------------------------


def stage_n1_keyframes(ctx: StageContext) -> dict:
    """Decode, measure JPEG size, pick keyframes, write images + mapping.

    Only runs on the DAKE pass. On the BTC pass the keyframes are given, and
    re-deriving them would mix two changes into one measurement (§7).

    **Two walks through the video, and that is inherent.** Every frame's size has
    to be known before any frame can be chosen, so nothing can be written on the
    first pass. What is *not* inherent is paying full decode twice: the second
    walk uses `grab()` and only reconstructs the ~2% of frames that were selected.

    The size curve is cached to the Volume, so a re-run — or a later sweep over
    `rho`, which is the whole reason the cache exists (§3.4) — costs a download
    instead of a decode.
    """
    import cv2  # noqa: PLC0415

    from mfe.offline.dake import (
        MAX_GAP_SECONDS,
        RHO,
        cuda_decode_available,
        decode_frames,
        decode_selected,
        jpeg_sizes,
        pts_for_frame,
        select_keyframes,
        video_fps,
    )

    rho = ctx.params.get("rho", RHO)
    max_gap = ctx.params.get("max_gap_seconds", MAX_GAP_SECONDS)
    rank_by = ctx.params.get("rank_by", "signed")
    backend = ctx.params.get("decode_backend", "auto")
    if backend == "auto":
        backend = "nvdec" if cuda_decode_available() else "cpu"
    fps, fps_raw = video_fps(str(ctx.video_path))

    cache = ctx.workdir / f"frame_sizes_{ctx.video_id}.npy"
    reused = False
    if not cache.is_file():
        # A previous session may have measured this video before dying. Downloading
        # a few hundred KB beats re-decoding twenty thousand frames.
        try:
            ctx.writer.transport.get(ctx.layout.frame_sizes(ctx.video_id), cache)
            reused = True
        except Exception:  # noqa: BLE001 - not measured yet, the normal first case
            cache.unlink(missing_ok=True)

    with Stopwatch() as timer:
        if cache.is_file():
            sizes = np.load(cache).tolist()
        else:
            sizes = list(
                jpeg_sizes(decode_frames(str(ctx.video_path), backend=backend))
            )
            np.save(cache, np.asarray(sizes, dtype=np.int64))
        picked = select_keyframes(
            sizes, fps=fps, rho=rho, max_gap_seconds=max_gap, rank_by=rank_by
        )

        out_dir = ctx.workdir / "keyframes" / ctx.video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        order_of = {frame: order for order, frame in enumerate(picked, start=1)}
        rows: list[tuple] = [("n", "pts_time", "fps", "frame_idx")]
        written = 0
        for index, frame in decode_selected(str(ctx.video_path), picked):
            order = order_of[index]
            cv2.imwrite(str(out_dir / f"{order:03d}.jpg"), frame)
            written += 1
            # NOT `round(index / fps, 4)`: that breaks the contract
            # `frame_idx == floor(pts_time * fps)` on 34% of rows at 30fps and 50%
            # at 29.97, and `read_keyframes` then rejects the mapping we just spent
            # a full decode producing. See `dake.pts_for_frame`.
            rows.append((order, pts_for_frame(index, fps), fps, index))

    if written != len(picked):
        # The mapping claims N rows and `clip-features` is an (N, D) array indexed
        # by row order, so a short write is not a partial result — it is a bundle
        # whose files disagree with each other.
        raise RuntimeError(
            f"{ctx.video_id}: chon {len(picked)} keyframe nhung chi ghi duoc "
            f"{written} anh. Bo giai ma dung som — khong cam moc."
        )

    map_csv = ctx.workdir / f"map_{ctx.video_id}.csv"
    with map_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    ctx.writer.write_file(map_csv, ctx.layout.map_keyframes(ctx.video_id))
    ctx.writer.write_file(cache, ctx.layout.frame_sizes(ctx.video_id))
    files, total_bytes = ctx.writer.write_directory(
        out_dir, ctx.layout.keyframes_dir(ctx.video_id)
    )
    gaps = np.diff(picked) if len(picked) > 1 else np.array([0])
    return ctx.writer.mark_done(
        "n1_keyframes",
        ctx.video_id,
        keyframes=len(picked),
        source_frames=len(sizes),
        ratio=round(len(picked) / max(1, len(sizes)), 5),
        max_gap_s=round(float(gaps.max()) / fps, 3),
        median_gap_s=round(float(np.median(gaps)) / fps, 3),
        fps=fps,
        fps_raw=round(fps_raw, 6),
        # Which decoder actually ran. Recorded because "the session had a GPU"
        # and "the GPU decoded anything" are different claims, and only the
        # second one explains a timing.
        decode_backend=backend,
        sizes_reused=reused,
        images=files,
        bytes=total_bytes,
        model="DAKE (khong model)",
        params={
            "rho": rho,
            "lookahead": 3,
            "min_gap": 5,
            "max_gap_seconds": max_gap,
            "rank_by": rank_by,
        },
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N0 — import BTC's own keyframes and objects
# ---------------------------------------------------------------------------


def stage_n0_import_btc(ctx: StageContext) -> dict:
    """Copy one video's shipped keyframes, objects and mapping onto the Volume.

    No model, no decode — a transfer. It exists because BTC publishes its own
    keyframe set alongside the videos, so the comparison bundle for §3.7 costs
    only bandwidth. Building it was going to be the expensive half of "is DAKE
    worth it"; it turns out to be the cheap half.

    Reuses the marker protocol rather than being a one-off script, so this gets
    the same resume behaviour, the same author check and the same
    `verify_bundle.py` as every extraction stage. A transfer that dies half way
    through 873 videos needs restart just as much as a decode does.

    `ctx.params` must carry the three local directories a caller has already
    unpacked from the archives:

        keyframes_dir  keyframes/{video_id}/{n:03d}.jpg
        objects_dir    objects/{video_id}/{n:03d}.json
        map_csv        map-keyframes/{video_id}.csv
    """
    keyframes_dir = Path(ctx.params["keyframes_dir"])
    objects_dir = Path(ctx.params.get("objects_dir", "")) if ctx.params.get("objects_dir") else None
    map_csv = Path(ctx.params["map_csv"])

    with Stopwatch() as timer:
        # The mapping is BTC's, so the floor(pts*fps) check here is a real check
        # against their numbers rather than a tautology — it is the one that
        # catches an fps read off the wrong video (see `read_keyframes`).
        keyframes = read_keyframes(map_csv, keyframes_dir)

        missing = [kf.n for kf in keyframes if not kf.image.is_file()]
        if missing:
            raise RuntimeError(
                f"{ctx.video_id}: thieu {len(missing)} anh keyframe "
                f"(vd n={missing[:5]}). Giai nen chua xong — khong cam moc."
            )

        ctx.writer.write_file(map_csv, ctx.layout.map_keyframes(ctx.video_id))
        images, image_bytes = ctx.writer.write_directory(
            keyframes_dir, ctx.layout.keyframes_dir(ctx.video_id)
        )

        objects = 0
        if objects_dir is not None and objects_dir.is_dir():
            objects, _ = ctx.writer.write_directory(
                objects_dir, ctx.layout.objects_dir(ctx.video_id)
            )
            if objects != images:
                # `read_object_labels` addresses by the same ordinal as the image,
                # so a count mismatch means some keyframe silently has no labels —
                # and BM25 is sensitive to term counts, which would skew the very
                # A/B this bundle exists to enable.
                raise RuntimeError(
                    f"{ctx.video_id}: {images} anh nhung {objects} file objects. "
                    "Hai thu nay danh dia chi theo cung mot so thu tu `n`."
                )

    return ctx.writer.mark_done(
        "n0_import_btc",
        ctx.video_id,
        keyframes=len(keyframes),
        images=images,
        objects=objects,
        bytes=image_bytes,
        fps=keyframes[0].fps if keyframes else None,
        model="BTC cap san (khong trich)",
        params={"source": "aic-data.ledo.io.vn"},
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N2 — subtitles
# ---------------------------------------------------------------------------


def stage_n2_subtitles(ctx: StageContext) -> dict:
    """faster-whisper large-v3, then segments assigned to keyframe windows.

    Two outputs and both are needed: the raw segments are the real data and the
    only thing that lets `subtitles.json` be regenerated when the keyframe set
    changes; the flat map is what Elasticsearch indexes.

    The window is `[pts_time(n), pts_time(n+1))` rather than a fixed +/-3s,
    because keyframe spacing varies (BTC: median 84 frames, worst 210), so a
    fixed window both misses and double-counts depending where you are. Bảo's
    notebook uses +/-3s; which is better is unmeasured (§4.5), and `window` here
    is the switch that lets K2 measure it instead of arguing about it.
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415

    beam = ctx.params.get("beam_size", 5)
    window = ctx.params.get("window", "next_keyframe")

    with Stopwatch() as timer:
        model = cached_model(
            "faster-whisper large-v3",
            lambda: WhisperModel("large-v3", device="cuda", compute_type="float16"),
        )
        segments, info = model.transcribe(
            str(ctx.audio_path()),
            language="vi",
            beam_size=beam,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        spans = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments
        ]

    segments_path = ctx.workdir / f"segments_{ctx.video_id}.json"
    segments_path.write_text(
        json.dumps({"language": info.language, "segments": spans}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    ctx.writer.write_file(segments_path, ctx.layout.subtitle_segments(ctx.video_id))

    mapping: dict[str, str] = {}
    for index, keyframe in enumerate(ctx.keyframes):
        start = keyframe.pts_time
        if window == "fixed3":
            low, high = start - 3.0, start + 3.0
        else:
            nxt = ctx.keyframes[index + 1].pts_time if index + 1 < len(ctx.keyframes) else 1e12
            low, high = start, nxt
        text = " ".join(s["text"] for s in spans if s["end"] > low and s["start"] < high)
        if text.strip():
            mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = text

    empty_ratio = 1 - len(mapping) / max(1, len(ctx.keyframes))
    if empty_ratio > 0.30 and spans:
        # §4.5 stop condition. Raised, not warned: a video that transcribed but
        # landed on almost no keyframes usually means the window or the audio
        # track is wrong, and finishing the run would bury that in a manifest.
        raise RuntimeError(
            f"{ctx.video_id}: {empty_ratio:.0%} keyframe khong co subtitle du "
            f"Whisper ra {len(spans)} doan. Nghi sai cua so hoac sai kenh audio."
        )

    return ctx.writer.write_text_part(
        "subtitles",
        ctx.video_id,
        mapping,
        model="faster-whisper large-v3",
        params={"beam_size": beam, "language": "vi", "vad_filter": True, "window": window},
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N3 — CLAP audio
# ---------------------------------------------------------------------------


def stage_n3_audio(ctx: StageContext) -> dict:
    """CLAP over a window around each keyframe.

    Note what this tier is: Bảo ships it disabled, the CLAP arm does not exist in
    the online layer yet, and the `audio_features.json` in the pilot bundle is not
    reproducible from any timestamp formula (§4.5). So this is generated to be
    *measurable later*, not to be used now — which is also why the shard plan puts
    it last, where losing it to a session timeout costs least.
    """
    import librosa  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import ClapModel, ClapProcessor  # noqa: PLC0415

    checkpoint = "laion/clap-htsat-unfused"
    half_window = ctx.params.get("half_window_s", 5.0)
    sample_rate = 48_000

    with Stopwatch() as timer:
        audio, _ = librosa.load(str(ctx.audio_path()), sr=sample_rate, mono=True)
        processor, model = cached_model(
            checkpoint,
            lambda: (
                ClapProcessor.from_pretrained(checkpoint),
                ClapModel.from_pretrained(checkpoint).eval().cuda(),
            ),
        )

        vectors: list[np.ndarray] = []
        for keyframe in ctx.keyframes:
            centre = int(keyframe.pts_time * sample_rate)
            span = int(half_window * sample_rate)
            clip = audio[max(0, centre - span) : centre + span]
            if len(clip) < 2 * span:
                clip = np.pad(clip, (0, 2 * span - len(clip)))
            # `audio=` first: transformers 5.x raises ValueError on the old
            # `audios=` name, and a `except TypeError` guard misses it — which is
            # how Bảo's notebook reported success on 3/3 failed videos.
            inputs = processor(audio=clip, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                out = model.get_audio_features(**{k: v.cuda() for k, v in inputs.items()})
            # `get_audio_features` returns a pooled output object, not a tensor.
            tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
            vectors.append(tensor.squeeze(0).float().cpu().numpy())

    return ctx.writer.write_vector_part(
        "audio_features",
        ctx.video_id,
        ctx.ids(),
        np.stack(vectors),
        model=checkpoint,
        params={"half_window_s": half_window, "sample_rate": sample_rate},
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N4 — objects and SigLIP
# ---------------------------------------------------------------------------


def stage_n4_objects(ctx: StageContext) -> dict:
    """Faster R-CNN / Inception-ResNet-v2, OpenImages V4. Five fields, verbatim.

    The vocabulary is not negotiable: `labels_vi.py` translates 544 OpenImages
    labels and nothing else, and swapping detectors would change the label set at
    the same time as the keyframe set, leaving the A/B unreadable (§3.5).
    """
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf  # noqa: PLC0415
    import tensorflow_hub as hub  # noqa: PLC0415

    handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    threshold = ctx.params.get("score_min", 0.3)

    with Stopwatch() as timer:
        detector = cached_model(handle, lambda: hub.load(handle).signatures["default"])
        out_dir = ctx.workdir / "objects" / ctx.video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, str] = {}

        for keyframe in ctx.keyframes:
            image = tf.image.decode_jpeg(tf.io.read_file(str(keyframe.image)), channels=3)
            result = detector(tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...])
            payload = {
                # Scores as strings, matching BTC's own files byte for byte —
                # `read_object_labels` calls float() anyway, and keeping the shape
                # identical is what makes the §3.6 comparison possible at all.
                "detection_scores": [str(x) for x in result["detection_scores"].numpy()],
                "detection_class_names": [
                    x.decode() for x in result["detection_class_names"].numpy()
                ],
                "detection_class_entities": [
                    x.decode() for x in result["detection_class_entities"].numpy()
                ],
                "detection_boxes": result["detection_boxes"].numpy().tolist(),
                "detection_class_labels": [
                    str(x) for x in result["detection_class_labels"].numpy()
                ],
            }
            (out_dir / f"{keyframe.n:03d}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            labels = [
                entity
                for entity, score in zip(
                    payload["detection_class_entities"], payload["detection_scores"], strict=True
                )
                if float(score) >= threshold
            ]
            if labels:
                mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = " ".join(labels)

    ctx.writer.write_directory(out_dir, ctx.layout.objects_dir(ctx.video_id))
    return ctx.writer.write_text_part(
        "objects_text",
        ctx.video_id,
        mapping,
        model=handle,
        # Must equal `features.object_score_min` in the config, or the BM25 branch
        # and the Elasticsearch `object` arm read two different label sets.
        params={"score_min": threshold},
        elapsed_s=timer.seconds,
    )


def stage_n4_siglip(ctx: StageContext) -> dict:
    """SigLIP image tower only. The text tower lives in the online layer."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

    checkpoint = "google/siglip-so400m-patch14-384"
    batch_size = ctx.params.get("batch_size", 16)

    with Stopwatch() as timer:
        processor, model = cached_model(
            checkpoint,
            lambda: (
                AutoProcessor.from_pretrained(checkpoint),
                AutoModel.from_pretrained(checkpoint).eval().cuda(),
            ),
        )
        vectors: list[np.ndarray] = []
        images = ctx.images()
        for start in range(0, len(images), batch_size):
            batch = [Image.open(p).convert("RGB") for p in images[start : start + batch_size]]
            inputs = processor(images=batch, return_tensors="pt").to("cuda")
            with torch.no_grad():
                features = _siglip_feature_tensor(model.get_image_features(**inputs))
            vectors.append(features.float().cpu().numpy())

    return ctx.writer.write_vector_part(
        "siglip_vectors",
        ctx.video_id,
        ctx.ids(),
        np.concatenate(vectors),
        model=checkpoint,
        params={"batch_size": batch_size, "tower": "image"},
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N5 — the two OCR sources
# ---------------------------------------------------------------------------


def stage_n5_ocr_easyocr(ctx: StageContext) -> dict:
    import easyocr  # noqa: PLC0415

    with Stopwatch() as timer:
        reader = cached_model("easyocr vi+en", lambda: easyocr.Reader(["vi", "en"], gpu=True))
        mapping: dict[str, str] = {}
        for keyframe in ctx.keyframes:
            text = " ".join(reader.readtext(str(keyframe.image), detail=0))
            if text.strip():
                mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = text

    return ctx.writer.write_text_part(
        "ocr_text",
        ctx.video_id,
        mapping,
        model="easyocr",
        params={"lang": ["vi", "en"]},
        elapsed_s=timer.seconds,
    )


#: Hai buoc, va buoc 1 la ly do buoc 2 hoat dong.
#:
#: Ban dau prompt hoi thang "chep chu, khong co thi tra loi None". Do duoc tren
#: `L21_V001`: model tra "None" cho **305/307** anh, trong khi easyocr tim thay
#: chu tren **ca 307** anh cua chinh bo do. Cung anh, mot model thay chu khap
#: noi, model kia khong thay gi — nen loi nam o prompt, khong o anh.
#:
#: Hai thay doi, moi cai chua mot nguyen nhan:
#:
#: 1. **Bo duong thoat "None".** Cho san mot token de bo cuoc thi mot model 7B
#:    luong tu 4-bit se dung no. Khong co chu thi de trong sau `CHU:`, va cho
#:    tang nay dem so dong trong — im lang tu nhien, khong phai mot tu ma phep
#:    hau xu ly phai doan.
#: 2. **Buoc dinh vi truoc buoc chep.** Prompt cua Bao chay duoc tren dung bo
#:    anh nay (41 khoa co chu that, 153 khoa watermark) vi no bat model MO TA
#:    anh truoc. Ta khong muon caption o day (§2.2 — caption va ocr_sviptr phai
#:    la hai lan sinh doc lap), nen buoc 1 chi hoi CHU NAM O DAU. Duoc phan chu
#:    y vao anh ma khong sinh ra caption.
_QWEN_PROMPT = """Ảnh này là một khung hình từ bản tin truyền hình Việt Nam.
BƯỚC 1 — liệt kê các vùng có chữ: logo kênh, đồng hồ, thanh tin chạy, tiêu đề, \
phụ đề, biển hiệu, chữ trên vật thể. Nếu không có vùng nào, ghi: khong
BƯỚC 2 — chép lại CHÍNH XÁC toàn bộ chữ đã thấy, giữ nguyên dấu tiếng Việt. \
Không dịch, không giải thích, không mô tả hình ảnh.
Trả về đúng hai dòng:
VUNG: <các vùng>
CHU: <toàn bộ chữ>"""

#: Tu ma model dung khi bo cuoc, du prompt khong con moi no. Bat de DEM, khong
#: de tin: `"BARBER SHOP\nNone"` la cau tra loi CO chu bi dan them mot tu rac —
#: phep loc cu chi so sanh voi CA chuoi nen no di thang vao index BM25.
_BAIL_WORDS = {"none", "khong", "không", "no text", "n/a", "-", "empty"}

#: Anh moi lan `generate()`. Do duoc: mot anh mot lan la **10,8 giay/anh** tren
#: T4, tuc ~532 gio GPU cho 177.321 keyframe cua batch 1. Chan boi VRAM chu khong
#: boi tinh toan, nen gia tri lam viec phai DO — tang tu dong ha lo khi OOM va
#: ghi gia tri cuoi vao marker.
OCR_BATCH = 8


def parse_qwen_ocr(text: str) -> tuple[str, bool]:
    """`(chu doc duoc, co bo cuoc)` tu cau tra loi tho cua Qwen2-VL.

    Doc phan sau `CHU:` khi co, con khong thi lay ca cau tra loi — model khong
    luon giu dinh dang, va mot cau tra loi khong dinh dang van co the chua chu
    that. Sau do go cac tu bo cuoc **o tung dong**, vi cho ma phep loc cu that
    bai la khi model tra ca chu that VA tu bo cuoc trong cung cau tra loi.
    """
    cleaned = (text or "").strip()
    for marker in ("CHU:", "CHỮ:", "TEXT:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[1]
            break
    else:
        # Khong co nhan: bo dong VUNG neu no lot vao, giu phan con lai.
        cleaned = "\n".join(
            line
            for line in cleaned.splitlines()
            if not line.strip().upper().startswith(("VUNG:", "VÙNG:", "REGION:"))
        )

    kept: list[str] = []
    bailed = False
    for raw in cleaned.splitlines():
        line = raw.strip().strip("*").strip()
        if not line:
            continue
        if line.lower().strip(" .:\"'") in _BAIL_WORDS:
            bailed = True
            continue
        kept.append(line)
    out = "\n".join(kept).strip()
    return out, bailed or not out


def stage_n5_ocr_qwen(ctx: StageContext) -> dict:
    """Qwen2-VL, 4-bit — the model that actually produced `sviptr_ocr.json`.

    Two things this deliberately does *not* copy from Bảo's notebook. It asks for
    text only, instead of a two-task `[CAPTION]`/`[TEXT]` prompt whose output is
    split by string search — that prompt silently dumps the whole response into
    the caption when the markers are missing, with no counter. And the caption
    comes from N6 instead, so `caption` and `ocr_sviptr` stop being two fields cut
    from one generation, which is the most likely explanation for `ocr_sviptr`
    ranking 1.0 alone yet contributing negatively in ablation (§2.2).

    That makes them two genuinely separate signals — but they are still the same
    *model*, so it is not yet the independent second OCR source §2.2 asks the Lead
    to decide on.

    Chay theo LO, va do la mot sua loi hieu nang chu khong phai doi model. Do
    duoc tren `L21_V001`: mot anh mot lan `generate()` cho **10,8 giay/anh**, tuc
    **~532 gio GPU cho batch 1**. Trong cung repo nay `build_qwen_generate()` da
    gop lo cho ReCap; tang nay thi khong, khong vi ly do nao.
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoProcessor, BitsAndBytesConfig  # noqa: PLC0415
    from transformers import Qwen2VLForConditionalGeneration as Qwen2VL  # noqa: PLC0415

    checkpoint = "Qwen/Qwen2-VL-7B-Instruct"
    batch_size = int(ctx.params.get("batch_size", OCR_BATCH))
    max_new = int(ctx.params.get("max_new_tokens", 256))

    def load():
        processor = AutoProcessor.from_pretrained(checkpoint)
        # Sinh theo lo thi PHAI dem trai. Dem phai day token dem vao GIUA prompt
        # va cho model viet tiep, nen no sinh tu sau mot chuoi dem — output rac
        # ma khong ném loi. `build_qwen_generate()` cung dat dong nay.
        processor.tokenizer.padding_side = "left"
        model = Qwen2VL.from_pretrained(
            checkpoint,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto",
        ).eval()
        return processor, model

    with Stopwatch() as timer:
        processor, model = cached_model(checkpoint, load)

        def ask(keyframes: list[Keyframe]) -> list[str]:
            """Mot lan `generate()` cho ca lo. Tra ve dung `len(keyframes)` cau."""
            messages = [
                {"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": _QWEN_PROMPT}]}
            ]
            one = processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [Image.open(kf.image).convert("RGB") for kf in keyframes]
            inputs = processor(
                text=[one] * len(keyframes), images=images,
                padding=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            return [
                answer.strip()
                for answer in processor.batch_decode(
                    generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            ]

        mapping: dict[str, str] = {}
        bail_words = 0   # model phat ra tu bo cuoc, DU no co tim thay chu hay khong
        no_text = 0      # sau khi go, khong con chu nao
        oom_retries = 0
        width = batch_size

        index = 0
        while index < len(ctx.keyframes):
            batch = ctx.keyframes[index : index + width]
            try:
                answers = ask(batch)
            except torch.cuda.OutOfMemoryError:
                # Duong lui, va no phai o day chu khong o tay nguoi chay: lo rong
                # bao nhieu la chuyen VRAM, khong doan duoc tu truoc. Ha mot nua
                # roi thu lai CHINH lo do — khong bo qua anh nao.
                oom_retries += 1
                torch.cuda.empty_cache()
                if width == 1:
                    raise
                width = max(1, width // 2)
                logger.warning("OOM, ha lo xuong %d roi thu lai", width)
                continue
            for keyframe, answer in zip(batch, answers, strict=True):
                text, bailed = parse_qwen_ocr(answer)
                if bailed:
                    bail_words += 1
                if text:
                    mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = text
                else:
                    no_text += 1
            index += len(batch)

    return ctx.writer.write_text_part(
        "sviptr_ocr",
        ctx.video_id,
        mapping,
        model=checkpoint,
        # HAI con so, va phai tach ra. Ban truoc chi ghi `answered_none` roi dem
        # ca hai truong hop vao do, nen `"BARBER SHOP\nNone"` — mot lan model
        # THANH CONG — vao cung o voi mot lan bo cuoc that.
        #
        # `no_text` la so do can theo doi: 305/307 tren `L21_V001` la cach prompt
        # cu tu bao, trong khi easyocr tim thay chu tren ca 307 anh do.
        #
        # `batch_size` ghi lai GIA TRI CUOI, khong phai gia tri khai: neu OOM da
        # ha lo giua duong thi day la con so that, va `oom_retries` noi da ha may
        # lan. Ghi gia tri khai se lam mot lan chay bi ha lo trong nhu binh thuong.
        params={"quantization": "4bit", "max_new_tokens": max_new, "do_sample": False,
                "no_text": no_text, "bail_words": bail_words, "prompt": "vung+chu v2",
                "batch_size": width, "batch_size_asked": batch_size,
                "oom_retries": oom_retries},
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N5b — hai nguon OCR chuyen dung, sinh ra de DO chu chua de dung
# ---------------------------------------------------------------------------
#
# Ly do ton tai, do duoc tren `L21_V001` (307 keyframe): Qwen2-VL 4-bit mat
# **10,8 giay/anh** khi goi tung anh, tuc hon 2 gio cho MOT video va ~532 gio GPU
# cho 177.321 keyframe cua batch 1. Gop lo ha duoc con so do nhung khong doi bac
# do lon. Cau hoi la co bo doc nao re hon ma khong te hon khong.
#
# Hai tang duoi day KHONG thay the `n5_ocr_easyocr` hay `n5_ocr_qwen` va khong
# nam trong `config/ingest_shards_batch1_btc_images.yaml`. Chung ghi ra artifact
# rieng de ba nguon dat canh nhau tren cung mot video — dung phep so ma §8.2 cua
# handoff ghi lai la da lam sai mot lan: ket luan "easyocr te" duoc rut ra tu viec
# so easyocr CUA TA voi Qwen CUA BAO, hai thu khac nhau o ca hai truc.
#
# Vi the ca hai tang deu sap xep hop chu bang CUNG mot ham `group_text_lines`, va
# ca hai deu lay vung chu tu CUNG mot bo phat hien cua PaddleOCR. Con lai giua
# `paddle_ocr` va `viet_ocr` dung mot khac biet: **bo nhan dang**.


def _quad_to_box(quad: Any) -> tuple[int, int, int, int]:
    """Da giac 4 diem (hoac 8 so phang) -> `(x0, y0, x1, y1)` bao ngoai.

    Lam tron RA NGOAI chu khong `int()` ca hai dau: `int()` cat ve 0, nen canh
    phai/duoi bi thu vao gan mot pixel va chu cao 12px mat mot phan net.
    """
    import math  # noqa: PLC0415

    points = list(quad)
    if points and not hasattr(points[0], "__len__"):
        points = [points[i : i + 2] for i in range(0, len(points), 2)]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


def group_text_lines(
    boxes: list[tuple[int, int, int, int]], tolerance_ratio: float = 0.6
) -> list[list[int]]:
    """Chi so cua `boxes`, gom thanh dong doc duoc: tren xuong, trai sang phai.

    Vi sao khong dung thu tu cua bo phat hien: moi engine tra ve mot thu tu khac
    nhau, nen hai nguon OCR se cho ra hai chuoi khac nhau **du doc y het nhau**.
    Chuoi do la thu di thang vao BM25, va mot phep so bi nhieu boi thu tu la mot
    phep so khong doc duoc. Ca hai tang duoi day goi ham nay.

    Nguong tinh theo **trung vi chieu cao hop**, khong theo so pixel co dinh:
    batch 1 khong chi co mot do phan giai, va mot dai co dinh se gop hai dong tren
    khung nho trong khi cat doi mot dong tren khung lon.
    """
    if not boxes:
        return []
    heights = sorted(y1 - y0 for _x0, y0, _x1, y1 in boxes)
    tolerance = max(1.0, tolerance_ratio * (heights[len(heights) // 2] or 1))

    def centre_of(index: int) -> float:
        return (boxes[index][1] + boxes[index][3]) / 2

    lines: list[list[int]] = []
    anchors: list[float] = []
    for index in sorted(range(len(boxes)), key=lambda i: (centre_of(i), boxes[i][0])):
        centre = centre_of(index)
        # So voi hop DAU TIEN cua dong, khong voi hop vua them: so voi hop vua
        # them thi mot chuoi hop lech dan moi cai vai pixel se troi khong gioi han
        # va nuot ca dong duoi.
        if lines and abs(centre - anchors[-1]) <= tolerance:
            lines[-1].append(index)
        else:
            lines.append([index])
            anchors.append(centre)
    for line in lines:
        line.sort(key=lambda i: boxes[i][0])
    return lines


def join_text_lines(
    boxes: list[tuple[int, int, int, int]], texts: list[str]
) -> str:
    """Ghep chu theo thu tu doc: trong dong cach nhau dau cach, giua dong xuong dong."""
    lines = group_text_lines(boxes)
    out: list[str] = []
    for line in lines:
        joined = " ".join(texts[i].strip() for i in line if texts[i].strip())
        if joined:
            out.append(joined)
    return "\n".join(out)


def _paddle_field(page: Any, key: str) -> Any:
    """Doc mot truong cua ket qua PaddleOCR, du no la mapping hay object."""
    from collections.abc import Mapping  # noqa: PLC0415

    if isinstance(page, Mapping):
        return page.get(key)
    return getattr(page, key, None)


def normalize_paddle_result(raw: Any) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """`[(hop, chu, do tin cay)]` tu ket qua PaddleOCR, ca hai the he API.

    PaddleOCR 3.x tra ve mot mapping (`rec_texts`, `rec_scores`, `rec_polys`);
    2.x tra ve `[[da_giac, (chu, diem)], …]`. Doi phien ban giua nam nguoi la
    chuyen binh thuong, va mot bo doc tra ve dung shape cua phien ban kia se cho
    ra **file rong** chu khong ném loi — nen ca hai duoc doc o day, va tang ghi
    lai `api` da dung vao marker.
    """
    page = raw
    # Ca hai the he tra ve mot phan tu cho moi anh; ta luon gui dung mot anh.
    if isinstance(page, (list, tuple)):
        page = page[0] if page else None
    if page is None:
        return []
    # 3.x doi khi boc them mot lop `{"res": {...}}`.
    if _paddle_field(page, "rec_texts") is None and _paddle_field(page, "res") is not None:
        page = _paddle_field(page, "res")

    texts = _paddle_field(page, "rec_texts")
    if texts is not None:
        polys = _paddle_field(page, "rec_polys")
        if polys is None:
            polys = _paddle_field(page, "dt_polys") or []
        scores = _paddle_field(page, "rec_scores")
        out = []
        for index, text in enumerate(texts):
            if index >= len(polys):
                continue
            score = float(scores[index]) if scores is not None and index < len(scores) else 1.0
            out.append((_quad_to_box(polys[index]), str(text), score))
        return out

    out = []
    for entry in page or []:
        quad, payload = entry[0], entry[1]
        if isinstance(payload, (list, tuple)):
            text, score = payload[0], float(payload[1])
        else:
            text, score = payload, 1.0
        out.append((_quad_to_box(quad), str(text), score))
    return out


def _unexpected_kwarg(message: str) -> str | None:
    """Ten tham so bi tu choi trong mot `TypeError` cua Python, neu doc duoc."""
    import re  # noqa: PLC0415

    found = re.search(r"unexpected keyword argument '([^']+)'", message)
    return found.group(1) if found else None


def build_paddle_ocr(**kwargs: Any) -> Any:
    """Mot `PaddleOCR`, bo di nhung tham so ban dang cai KHONG nhan.

    PaddleOCR doi ten tham so giua 2.x va 3.x (`use_gpu` -> `device`,
    `use_angle_cls` -> `use_textline_orientation`, `show_log` bien mat). Pin cung
    mot bo tham so thi notebook chet ngay o o khoi tao tren nua so phien ban; nen
    ta thu, doc ten tham so bi tu choi tu chinh `TypeError`, bo no ra roi thu lai.
    Chi bo tham so **ta da truyen vao** — mot `TypeError` khac van noi len.
    """
    from paddleocr import PaddleOCR  # noqa: PLC0415

    remaining = {key: value for key, value in kwargs.items() if value is not None}
    while True:
        try:
            return PaddleOCR(**remaining)
        except TypeError as err:
            name = _unexpected_kwarg(str(err))
            if name is None or name not in remaining:
                raise
            logger.info("paddleocr khong nhan %s, bo ra", name)
            remaining.pop(name)


def _paddle_read(ocr: Any, image: Path) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """Chay pipeline PaddleOCR tren mot anh. `predict` la ten cua 3.x, `ocr` cua 2.x."""
    run = getattr(ocr, "predict", None) or ocr.ocr
    return normalize_paddle_result(run(str(image)))


def _paddle_params(ctx: StageContext) -> dict:
    lang = ctx.params.get("lang", "vi")
    device = ctx.params.get("device", "gpu")
    kwargs = {
        "lang": lang,
        "device": device,
        "use_gpu": device != "cpu",
        "show_log": False,
        "use_angle_cls": False,
        "use_textline_orientation": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    }
    kwargs.update(ctx.params.get("paddle_kwargs", {}))
    return kwargs


def _paddle_instance(ctx: StageContext) -> Any:
    kwargs = _paddle_params(ctx)
    key = f"paddleocr lang={kwargs.get('lang')} device={kwargs.get('device')}"
    return cached_model(key, lambda: build_paddle_ocr(**kwargs))


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        return "?"


def stage_n5_ocr_paddle(ctx: StageContext) -> dict:
    """PaddleOCR tieng Viet, ca phat hien lan nhan dang. Nguon OCR thu ba.

    Doc thang tu anh, khong sinh token — nen no khong the roi vao kieu hong cua
    `n5_ocr_qwen`, noi mot prompt sai lam model tra "None" cho 305/307 anh ma
    khong ném gi. Doi lai, no khong hieu ngu canh: chu bi che mot phan thi no
    doan theo hinh chu chu khong theo nghia.
    """
    lang = _paddle_params(ctx)["lang"]
    score_min = float(ctx.params.get("rec_score_min", 0.0))

    with Stopwatch() as timer:
        ocr = _paddle_instance(ctx)
        mapping: dict[str, str] = {}
        boxes_total = 0
        dropped_low_score = 0
        silent_images = 0

        for keyframe in ctx.keyframes:
            found = _paddle_read(ocr, keyframe.image)
            boxes_total += len(found)
            kept = [(box, text) for box, text, score in found if score >= score_min]
            dropped_low_score += len(found) - len(kept)
            text = join_text_lines([b for b, _ in kept], [t for _, t in kept])
            if text:
                mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = text
            else:
                silent_images += 1

    images = len(ctx.keyframes)
    return ctx.writer.write_text_part(
        "paddle_ocr",
        ctx.video_id,
        mapping,
        model=f"paddleocr {_package_version('paddleocr')} lang={lang}",
        params={
            "lang": lang,
            "rec_score_min": score_min,
            "api": "predict" if hasattr(ocr, "predict") else "ocr",
            "order": "group_text_lines",
        },
        # Do duoc, khac nhau tung video — xem `write_text_part`.
        facts={
            "images": images,
            "boxes": boxes_total,
            "silent_images": silent_images,
            "dropped_low_score": dropped_low_score,
            "seconds_per_image": round(timer.seconds / max(1, images), 3),
        },
        elapsed_s=timer.seconds,
    )


def _crop_for_recognition(image: Any, box: tuple[int, int, int, int], pad: int) -> Any:
    """Cat hop bao ngoai, noi them `pad` pixel, kep trong khung anh."""
    width, height = image.size
    x0, y0, x1, y1 = box
    return image.crop(
        (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(width, x1 + pad),
            min(height, y1 + pad),
        )
    )


def stage_n5_ocr_vietocr(ctx: StageContext) -> dict:
    """Vung chu do PaddleOCR tim, chu do VietOCR doc.

    **VietOCR khong co bo phat hien.** No nhan mot anh DONG chu da cat va tra ve
    chuoi; dua ca khung hinh vao no se ra rac ma khong ném loi. Nen tang nay la
    mot phep ghep, va bo phat hien la cua PaddleOCR — cung bo `n5_ocr_paddle`
    dung, voi cung tham so.

    Do la ca thiet ke cua phep do: `paddle_ocr` va `viet_ocr` chay tren cung nhung
    vung chu, xep bang cung `group_text_lines`, nen chenh lech giua hai file la
    chenh lech cua **bo nhan dang tieng Viet** chu khong phai cua bo phat hien.

    Cai gia: PaddleOCR chay ca pipeline o day roi ta vut bo phan chu no doc duoc.
    Doi lai la MOT duong ma, thay vi hai duong "chi phat hien" khac nhau giua 2.x
    va 3.x — va phan bi vut bo la phan re, do bang `elapsed_s` cua hai tang.
    """
    from PIL import Image  # noqa: PLC0415

    cfg_name = ctx.params.get("vietocr_config", "vgg_transformer")
    device = ctx.params.get("vietocr_device", "cuda:0")
    beamsearch = bool(ctx.params.get("beamsearch", False))
    pad = int(ctx.params.get("crop_pad", 2))
    min_side = int(ctx.params.get("min_crop_side", 3))

    def load_vietocr():
        from vietocr.tool.config import Cfg  # noqa: PLC0415
        from vietocr.tool.predictor import Predictor  # noqa: PLC0415

        config = Cfg.load_config_from_name(cfg_name)
        config["device"] = device
        # `predictor` co the vang trong vai ban config; dat co dieu kien.
        if isinstance(config.get("predictor"), dict):
            config["predictor"]["beamsearch"] = beamsearch
        return Predictor(config)

    with Stopwatch() as timer:
        ocr = _paddle_instance(ctx)
        reader = cached_model(f"vietocr {cfg_name}", load_vietocr)
        batch_predict = getattr(reader, "predict_batch", None)

        mapping: dict[str, str] = {}
        boxes_total = 0
        crops_skipped = 0
        silent_images = 0
        batched = batch_predict is not None

        for keyframe in ctx.keyframes:
            found = _paddle_read(ocr, keyframe.image)
            boxes_total += len(found)
            page = Image.open(keyframe.image).convert("RGB")

            boxes: list[tuple[int, int, int, int]] = []
            crops = []
            for box, _text, _score in found:
                x0, y0, x1, y1 = box
                if x1 - x0 < min_side or y1 - y0 < min_side:
                    # Mot hop mong hon vai pixel khong phai dong chu. VietOCR van
                    # tra ve MOT chuoi cho no, va chuoi do di thang vao index.
                    crops_skipped += 1
                    continue
                boxes.append(box)
                crops.append(_crop_for_recognition(page, box, pad))

            if crops:
                if batched:
                    texts = list(batch_predict(crops))
                else:
                    texts = [reader.predict(crop) for crop in crops]
            else:
                texts = []

            text = join_text_lines(boxes, texts)
            if text:
                mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = text
            else:
                silent_images += 1

    images = len(ctx.keyframes)
    return ctx.writer.write_text_part(
        "viet_ocr",
        ctx.video_id,
        mapping,
        model=(
            f"vietocr {_package_version('vietocr')}/{cfg_name} "
            f"+ paddleocr {_package_version('paddleocr')} (phat hien)"
        ),
        params={
            "detector": "paddleocr",
            "vietocr_config": cfg_name,
            "beamsearch": beamsearch,
            "crop": "bbox",
            "crop_pad": pad,
            "min_crop_side": min_side,
            "order": "group_text_lines",
        },
        facts={
            "images": images,
            "boxes": boxes_total,
            "crops_skipped": crops_skipped,
            "silent_images": silent_images,
            "batched": batched,
            "seconds_per_image": round(timer.seconds / max(1, images), 3),
        },
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N6 — ReCap captions, then BGE-M3
# ---------------------------------------------------------------------------


def group_shots(
    keyframes: list[Keyframe],
    gap_seconds: float = 2.0,
    *,
    max_keyframes: int = 8,
    max_duration_s: float = 20.0,
) -> list[list[int]]:
    """Group adjacent keyframes into shots. Gap-based, but **bounded**.

    An approximation of the paper, which segments with AutoShot; installing a deep
    model purely to cut shots would drag one back into the place DAKE just removed
    it from. It goes in the manifest, and it is the first suspect if captions come
    out poor — before the prompt is blamed (§4.6).

    **The bounds are not decoration; without them the gap rule collapses.** Cutting
    only on `gap > threshold` is single-linkage chaining: one long chain absorbs
    everything. Measured on the 16-video BTC set, the threshold has a cliff rather
    than a slope — 2.0s gives 2537 shots, 6.0s gives 259, and 8.0s gives **16**,
    one per video, because BTC's largest keyframe gap is 8.4s.

    Worse, the cliff sits on the wrong side for the pass this is built for. DAKE
    *guarantees* every gap is at most 2.0s, so the 2.0s threshold §4.6 proposed —
    chosen precisely to match that guarantee — never fires at all: a 400-keyframe
    DAKE video becomes one shot. The plan's own density promise defeats the plan's
    own shot rule.

    Capping length removes the cliff and makes the Gemini bill predictable, which
    matters more here than shot purity: an over-cut shot costs one extra call, an
    under-cut one costs the whole video's caption quality.
    """
    shots: list[list[int]] = []
    current: list[int] = []
    for index, keyframe in enumerate(keyframes):
        if current:
            previous = keyframes[current[-1]]
            too_far = keyframe.pts_time - previous.pts_time > gap_seconds
            too_many = len(current) >= max_keyframes
            too_long = keyframe.pts_time - keyframes[current[0]].pts_time > max_duration_s
            if too_far or too_many or too_long:
                shots.append(current)
                current = []
        current.append(index)
    if current:
        shots.append(current)
    return shots


def build_qwen_generate(checkpoint: str = "Qwen/Qwen2-VL-7B-Instruct", **kwargs):
    """A `generate(prompts) -> list[str]` backed by a local Qwen2-VL.

    Loaded once and reused across every batch — the model is 5 GB in 4-bit and
    reloading it per video would dwarf the work. Batch size is set by the caller
    because it is VRAM-bound, not compute-bound: 7B weights plus KV cache for N
    sequences of ~5k tokens against a 15 GB T4 is tight, and the working value has
    to be measured rather than assumed.
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoProcessor, BitsAndBytesConfig  # noqa: PLC0415
    from transformers import Qwen2VLForConditionalGeneration as Qwen2VL  # noqa: PLC0415

    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.tokenizer.padding_side = "left"   # bat buoc khi generate theo lo
    model = Qwen2VL.from_pretrained(
        checkpoint,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        **kwargs,
    ).eval()

    def generate(prompts):
        texts, images = [], []
        for prompt in prompts:
            content = [{"type": "image"} for _ in prompt.images]
            content.append({"type": "text", "text": prompt.text})
            texts.append(
                processor.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True
                )
            )
            images.append([Image.open(path).convert("RGB") for path in prompt.images])

        inputs = processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    return generate


def build_gemini_generate(model_name: str = "gemini-2.5-flash"):
    """A `generate(prompts) -> list[str]` backed by the Gemini API, one call each.

    Kept alongside the local backend so the two can be compared on the same data,
    the same prompt and the same failure counting — otherwise "which is better"
    turns into comparing two runs that differ in more than the model.
    """
    from google import genai  # noqa: PLC0415

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "thieu GEMINI_API_KEY — dat trong Kaggle Secrets, khong trong o notebook"
        )
    client = genai.Client(api_key=api_key)

    def generate(prompts):
        answers = []
        for prompt in prompts:
            parts = [
                genai.types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")
                for path in prompt.images
            ]
            response = client.models.generate_content(
                model=model_name, contents=[*parts, prompt.text]
            )
            answers.append(response.text or "")
        return answers

    return generate


def stage_n6_recap(ctx: StageContext) -> dict:
    """Recurrence captioning for ONE video. See `mfe.offline.recap` for batching.

        (C_t, M_t) = LVLM(S_t, M_{t-1})

    This entry point exists for the per-video path (a handful of videos, or a
    Gemini run where the rate limit makes batching pointless). At batch-1 scale
    the local model has to run `recap.run_recap` over many videos at once — the
    recurrence serialises one video, and one video at a time leaves the GPU idle
    between short decodes (§4.6b).

    Checkpointed **per shot**: this stage is not reproducible, so a lost round is
    not re-doable work, it is different data.
    """
    from mfe.offline.recap import (  # noqa: PLC0415
        VideoState,
        run_recap,
        summarise,
    )

    backend = ctx.params.get("recap_backend", "qwen")
    language = ctx.params.get("caption_language", "vi")
    gap = ctx.params.get("shot_gap_seconds", 2.0)

    shots = build_shots(ctx, gap)
    checkpoint_path = ctx.workdir / f"recap_{ctx.video_id}.json"
    saved = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else None
    )
    state = VideoState.resume(ctx.video_id, shots, saved)

    generate = ctx.params.get("generate")
    if generate is None:
        generate = (
            build_qwen_generate(ctx.params.get("qwen_checkpoint", "Qwen/Qwen2-VL-7B-Instruct"))
            if backend == "qwen"
            else build_gemini_generate(ctx.params.get("gemini_model", "gemini-2.5-flash"))
        )

    def save(current: VideoState) -> None:
        checkpoint_path.write_text(
            json.dumps(current.to_checkpoint(), ensure_ascii=False), encoding="utf-8"
        )

    with Stopwatch() as timer:
        run_recap([state], generate, language=language, batch_videos=1, on_shot=save)

    ctx.writer.write_file(checkpoint_path, ctx.layout.recap_memory(ctx.video_id))
    report = summarise([state])
    return ctx.writer.write_text_part(
        "dense_captions",
        ctx.video_id,
        state.captions,
        model=(
            ctx.params.get("qwen_checkpoint", "Qwen/Qwen2-VL-7B-Instruct")
            if backend == "qwen"
            else ctx.params.get("gemini_model", "gemini-2.5-flash")
        ),
        params={
            "backend": backend,
            "language": language,
            "shots": report["shots"],
            "shot_segmentation": f"keyframe gap <= {gap}s (xap xi AutoShot)",
            "context_images_per_shot": 4,
            # The number that says whether a 7B model can hold this together.
            "parse_failures": report["parse_failures"],
            "parse_failure_rate": report["parse_failure_rate"],
            "empty_captions": report["empty_captions"],
        },
        elapsed_s=timer.seconds,
    )


def build_shots(ctx: StageContext, gap_seconds: float = 2.0) -> list:
    """Group this video's keyframes into shots and attach the speech they cover."""
    from mfe.offline.recap import Shot  # noqa: PLC0415

    segments_path = ctx.workdir / f"segments_{ctx.video_id}.json"
    spans = (
        json.loads(segments_path.read_text(encoding="utf-8"))["segments"]
        if segments_path.is_file()
        else []
    )
    shots = []
    for index, members in enumerate(group_shots(ctx.keyframes, gap_seconds)):
        start = ctx.keyframes[members[0]].pts_time
        end = ctx.keyframes[members[-1]].pts_time
        shots.append(
            Shot(
                index=index,
                keyframe_ids=[
                    build_keyframe_id(ctx.video_id, ctx.keyframes[i].frame_idx)
                    for i in members
                ],
                image_paths=[ctx.keyframes[i].image for i in members],
                speech=" ".join(
                    s["text"] for s in spans if s["end"] > start and s["start"] <= end + gap_seconds
                ),
                start=start,
                end=end,
            )
        )
    return shots


def stage_n6_bge_m3(ctx: StageContext) -> dict:
    """Captions -> BGE-M3. Separate from the Gemini call so re-embedding is free."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    checkpoint = "BAAI/bge-m3"
    local = ctx.workdir / f"dense_captions_{ctx.video_id}.json"
    ctx.writer.transport.get(ctx.layout.part("dense_captions", ctx.video_id), local)
    captions = json.loads(local.read_text(encoding="utf-8"))
    if not captions:
        raise RuntimeError(f"{ctx.video_id}: chua co dense_captions — chay n6_recap truoc")

    keys = sorted(captions)
    with Stopwatch() as timer:
        model = SentenceTransformer(checkpoint)
        vectors = model.encode(
            [captions[k] for k in keys],
            normalize_embeddings=True,
            batch_size=ctx.params.get("batch_size", 16),
            convert_to_numpy=True,
        )

    return ctx.writer.write_vector_part(
        "dense_vectors",
        ctx.video_id,
        keys,
        np.asarray(vectors, dtype=np.float32),
        model=checkpoint,
        params={"normalize": True},
        elapsed_s=timer.seconds,
    )


STAGES = {
    "n0_import_btc": stage_n0_import_btc,
    "n1_keyframes": stage_n1_keyframes,
    "n2_subtitles": stage_n2_subtitles,
    "n3_audio": stage_n3_audio,
    "n4_objects": stage_n4_objects,
    "n4_siglip": stage_n4_siglip,
    "n5_ocr_easyocr": stage_n5_ocr_easyocr,
    "n5_ocr_qwen": stage_n5_ocr_qwen,
    "n5_ocr_paddle": stage_n5_ocr_paddle,
    "n5_ocr_vietocr": stage_n5_ocr_vietocr,
    "n6_recap": stage_n6_recap,
    "n6_bge_m3": stage_n6_bge_m3,
}


def free_gpu() -> None:
    """Between stages, per §4.2. Cheap, and the alternative is an OOM at hour 5."""
    import gc  # noqa: PLC0415

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
