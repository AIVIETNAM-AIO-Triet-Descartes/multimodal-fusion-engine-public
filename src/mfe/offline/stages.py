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

import numpy as np

from mfe.contracts.ids import build_keyframe_id, frame_idx_from_pts
from mfe.offline.bundle import BundleLayout, BundleWriter, Stopwatch

logger = logging.getLogger(__name__)


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
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
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
        processor = ClapProcessor.from_pretrained(checkpoint)
        model = ClapModel.from_pretrained(checkpoint).eval().cuda()

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
        detector = hub.load(handle).signatures["default"]
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
        processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint).eval().cuda()
        vectors: list[np.ndarray] = []
        images = ctx.images()
        for start in range(0, len(images), batch_size):
            batch = [Image.open(p).convert("RGB") for p in images[start : start + batch_size]]
            inputs = processor(images=batch, return_tensors="pt").to("cuda")
            with torch.no_grad():
                features = model.get_image_features(**inputs)
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
        reader = easyocr.Reader(["vi", "en"], gpu=True)
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


_QWEN_PROMPT = (
    "Transcribe ALL visible text in this image exactly as written, keeping "
    "Vietnamese diacritics. If there is no text, answer exactly: None"
)


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
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import AutoProcessor, BitsAndBytesConfig  # noqa: PLC0415
    from transformers import Qwen2VLForConditionalGeneration as Qwen2VL  # noqa: PLC0415

    checkpoint = "Qwen/Qwen2-VL-7B-Instruct"
    with Stopwatch() as timer:
        processor = AutoProcessor.from_pretrained(checkpoint)
        model = Qwen2VL.from_pretrained(
            checkpoint,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto",
        ).eval()

        mapping: dict[str, str] = {}
        none_count = 0
        for keyframe in ctx.keyframes:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": _QWEN_PROMPT}],
                }
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=[prompt],
                images=[Image.open(keyframe.image).convert("RGB")],
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            answer = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0].strip()
            if answer.lower().strip(" .") == "none" or not answer:
                none_count += 1
                continue
            mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = answer

    return ctx.writer.write_text_part(
        "sviptr_ocr",
        ctx.video_id,
        mapping,
        model=checkpoint,
        params={"quantization": "4bit", "max_new_tokens": 256, "do_sample": False,
                "answered_none": none_count},
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


def stage_n6_recap(ctx: StageContext) -> dict:
    """Recurrence captioning with a carried memory string, checkpointed per shot.

        (C_t, M_t) = LVLM(S_t, M_{t-1})

    Checkpointing is finer here than anywhere else for two reasons that do not
    apply to the other stages: the calls cost money, and the stage is **not
    reproducible** — `M_t` differs between runs, so redoing a video from the start
    yields different captions for the same shot, and the bundle stops being one
    measurement (§5.3).
    """
    from google import genai  # noqa: PLC0415

    model_name = ctx.params.get("gemini_model", "gemini-2.5-flash")
    language = ctx.params.get("caption_language", "vi")
    gap = ctx.params.get("shot_gap_seconds", 2.0)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "thieu GEMINI_API_KEY — dat trong Kaggle Secrets, khong trong o notebook"
        )
    client = genai.Client(api_key=api_key)

    checkpoint_path = ctx.workdir / f"recap_{ctx.video_id}.json"
    state = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {"shot_index": -1, "memory": "", "captions": {}}
    )

    segments_path = ctx.workdir / f"segments_{ctx.video_id}.json"
    spans = (
        json.loads(segments_path.read_text(encoding="utf-8"))["segments"]
        if segments_path.is_file()
        else []
    )

    shots = group_shots(ctx.keyframes, gap)
    tongue = "tiếng Việt" if language == "vi" else "English"
    with Stopwatch() as timer:
        for shot_index, members in enumerate(shots):
            if shot_index <= state["shot_index"]:
                continue
            start = ctx.keyframes[members[0]].pts_time
            end = ctx.keyframes[members[-1]].pts_time
            speech = " ".join(
                s["text"] for s in spans if s["end"] > start and s["start"] <= end + gap
            )
            prompt = (
                f"Bạn đang mô tả các cảnh liên tiếp của một video.\n"
                f"TRÍ NHỚ (bối cảnh đã tích luỹ): {state['memory'] or '(chưa có)'}\n"
                f"LỜI THOẠI trong đoạn này: {speech or '(không có)'}\n\n"
                f"Viết mô tả bằng {tongue} cho cảnh này: thực thể, thuộc tính, tương tác. "
                f"Dùng trí nhớ để HIỂU cảnh, nhưng KHÔNG nhắc tới nó trong mô tả.\n"
                f"Sau đó cập nhật trí nhớ: giữ cái còn liên quan, bỏ cái đã cũ.\n"
                f'Trả về JSON: {{"caption": "...", "memory": "..."}}'
            )
            images = [
                genai.types.Part.from_bytes(
                    data=ctx.keyframes[i].image.read_bytes(), mime_type="image/jpeg"
                )
                for i in members[:4]
            ]
            response = client.models.generate_content(
                model=model_name, contents=[*images, prompt]
            )
            body = _parse_json_answer(response.text)
            caption = body.get("caption", "").strip()
            state["memory"] = body.get("memory", state["memory"])
            state["shot_index"] = shot_index
            for i in members:
                if caption:
                    state["captions"][
                        build_keyframe_id(ctx.video_id, ctx.keyframes[i].frame_idx)
                    ] = caption
            checkpoint_path.write_text(
                json.dumps(state, ensure_ascii=False), encoding="utf-8"
            )

    ctx.writer.write_file(checkpoint_path, ctx.layout.recap_memory(ctx.video_id))
    return ctx.writer.write_text_part(
        "dense_captions",
        ctx.video_id,
        state["captions"],
        model=model_name,
        params={
            "language": language,
            "shots": len(shots),
            "shot_segmentation": f"keyframe gap <= {gap}s (xap xi AutoShot)",
            "context_images_per_shot": 4,
        },
        elapsed_s=timer.seconds,
    )


def _parse_json_answer(text: str) -> dict:
    """Pull the JSON object out of an LLM answer, tolerating fences.

    Returns `{}` rather than raising: one unparseable shot must not lose the
    whole video's memory chain, and the caller records an empty caption, which
    shows up as a count in the manifest.
    """
    cleaned = (text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("ReCap tra ve khong phai JSON: %s", cleaned[:120])
        return {}


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
    "n1_keyframes": stage_n1_keyframes,
    "n2_subtitles": stage_n2_subtitles,
    "n3_audio": stage_n3_audio,
    "n4_objects": stage_n4_objects,
    "n4_siglip": stage_n4_siglip,
    "n5_ocr_easyocr": stage_n5_ocr_easyocr,
    "n5_ocr_qwen": stage_n5_ocr_qwen,
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
