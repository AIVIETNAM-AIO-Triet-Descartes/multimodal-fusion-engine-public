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
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mfe.contracts.ids import build_keyframe_id, frame_idx_from_pts
from mfe.offline.bundle import BundleLayout, BundleWriter, Stopwatch
from mfe.offline.caption_prompt import PROMPT_VARIANTS

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

#: Ket qua phat hien+nhan dang cua PaddleOCR, giu lai trong PHIEN.
#:
#: Ly do: `n5_ocr_paddle` va `n5_ocr_vietocr` chay CUNG mot pipeline tren CUNG bo
#: anh — tang thu hai chi can hop, nhung no phai goi lai ca pipeline de lay. Ban
#: dau tang do duoc ghi la "phan bi vut la phan re". Do la mot GIA DINH, khong
#: phai so do, va tren CPU no sai: pipeline chay hai lan la **gap doi** thoi gian
#: cua ca phep do.
#:
#: Khoa gom ca khoa cache cua model, nen doi tham so Paddle la cache truot chu
#: khong tra ve hop cu. Cung vong doi voi `_MODELS`: `unload_models()` xoa ca hai.
_DETECTIONS: dict[int, list[tuple[tuple[int, int, int, int], str, float]]] = {}

#: `(video_id, khoa cache Paddle)` ma `_DETECTIONS` dang giu. Doi mot trong hai
#: la cache bi don sach — xem `paddle_scan`.
_DETECTION_SCOPE: tuple[str, str] | None = None

#: Cau tra loi THO cua Qwen2.5-VL, giu lai trong PHIEN. `{n cua keyframe: text}`.
#:
#: Cung ly do nhu `_DETECTIONS`, va o day dat hon nhieu: `n6_caption_qwen` va
#: `n6_ocr_qwen25` doc HAI NUA cua CUNG mot cau tra loi (`[CAPTION]` va `[TEXT]`).
#: Sinh lai lan hai la tra gap doi tien GPU de lay lai dung chuoi vua co.
#:
#: Khoa scope gom ca tham so sinh, nen doi prompt hay `max_tokens` la cache truot
#: chu khong tra ve cau cu. Cung vong doi voi `_MODELS`.
_QWEN_VL_ANSWERS: dict[int, str] = {}

#: `(video_id, khoa cache sinh)` ma `_QWEN_VL_ANSWERS` dang giu.
_QWEN_VL_SCOPE: tuple[str, str] | None = None

#: Cong don `{"images": n, "seconds": s}` cua ca phien, chi de BAO CAO. Khong
#: reset theo video: cau hoi dang tra loi la "ca luot chay bao lau", va mot con
#: so tinh tren mot video thi lan nap model dau tien lam no vo nghia.
_QWEN_VL_TIMING: dict[str, float] = {}


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
    # Ket qua phat hien gan voi doi tuong model da tao ra chung; giu lai sau khi
    # model bi tha la giu mot ket qua khong con gi trong phien nay sinh lai duoc.
    _DETECTIONS.clear()
    _QWEN_VL_ANSWERS.clear()
    _QWEN_VL_TIMING.clear()
    global _DETECTION_SCOPE, _QWEN_VL_SCOPE  # noqa: PLW0603
    _DETECTION_SCOPE = None
    _QWEN_VL_SCOPE = None
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
        LOOKAHEAD,
        MAX_GAP_SECONDS,
        MIN_GAP_SECONDS,
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
    min_gap = ctx.params.get("min_gap_seconds", MIN_GAP_SECONDS)
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
            sizes,
            fps=fps,
            rho=rho,
            min_gap_seconds=min_gap,
            max_gap_seconds=max_gap,
            rank_by=rank_by,
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
        # Moi gia tri o day phai la gia tri DA DUNG, khong phai gia tri mac dinh.
        # Ban truoc ghi cung `"min_gap": 5` trong khi tang khong he truyen tham so
        # do vao `select_keyframes` — nen marker khang dinh mot con so ma khong lan
        # chay nao doc den, va no van dung nhu the sau khi mac dinh doi.
        params={
            "rho": rho,
            "lookahead": LOOKAHEAD,
            "min_gap_seconds": min_gap,
            "max_gap_seconds": max_gap,
            "rank_by": rank_by,
        },
        # Nguong theo FRAME cua chinh video nay. `params` phai giong nhau giua moi
        # shard (`check_fingerprints`), nhung 1,0s la 25 frame o 25fps va 30 frame
        # o 30fps — nen con so frame la mot phep DO cua tung video, khong phai cau
        # hinh, va no thuoc ve day.
        min_gap_frames=max(1, int(min_gap * fps)),
        max_gap_frames=max(1, int(max_gap * fps)),
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
    max_empty_ratio = float(ctx.params.get("max_empty_ratio", 0.30))

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

    keyframes_total = len(ctx.keyframes)
    keyframes_without_subtitle = keyframes_total - len(mapping)
    empty_ratio = (
        keyframes_without_subtitle / keyframes_total if keyframes_total else 0.0
    )
    if empty_ratio > max_empty_ratio and spans:
        # §4.5 stop condition, và ngưỡng đọc từ `params` — Lead chốt 21-08-2026.
        #
        # Hai lý lẽ đều đúng, nên không bên nào được thắng bằng mặc định im lặng.
        # Trọng đúng khi nói khoảng lặng, nhạc và chuyển cảnh sinh ra keyframe
        # không có chữ một cách hợp lệ, và ném lỗi ở đây giết luôn `n3_audio`
        # của video đó. §4.5 đúng khi nói một video phiên âm ra hàng chục đoạn
        # mà rơi vào gần như không keyframe nào thì hoặc sai cửa sổ, hoặc sai
        # kênh audio — và chạy tiếp sẽ chôn chuyện đó trong một manifest.
        #
        # Chuyện đó KHÔNG giả thuyết: 469 video của Quang đi lọt với 69–87%
        # rỗng vì bản cài đặt của bạn ấy không có chốt này, và mất 60–70% ánh
        # xạ subtitle cho tới khi `scripts/remap_subtitles.py` dựng lại.
        #
        # Nên mặc định vẫn NÉM, còn ai có bộ video thật sự nhiều khoảng lặng
        # thì nâng ngưỡng **có chủ ý**. Con số đó đi vào `params`, tức
        # `check_fingerprints` nhìn thấy nó: "nửa bundle này được phép thưa"
        # trở thành một sự thật được ghi lại và so được giữa các mảnh, thay vì
        # một dòng log không ai đọc. `max_empty_ratio=1.0` là tắt hẳn, và việc
        # tắt đó cũng hiện ra ở đúng chỗ ấy.
        raise RuntimeError(
            f"{ctx.video_id}: {empty_ratio:.0%} keyframe khong co subtitle du "
            f"Whisper ra {len(spans)} doan (nguong {max_empty_ratio:.0%}). "
            "Nghi sai cua so hoac sai kenh audio. Neu bo video nay that su "
            "nhieu khoang lang, dat params['max_empty_ratio'] cao hon."
        )

    return ctx.writer.write_text_part(
        "subtitles",
        ctx.video_id,
        mapping,
        model="faster-whisper large-v3",
        # GHI ĐỦ những gì tầng này thực sự đặt, kể cả thứ nằm cứng trong mã.
        #
        # Bản cũ chỉ ghi bốn khoá, còn `compute_type` và `min_silence_duration_ms`
        # thì nằm cứng ở hai dòng trên mà không vào `params`. Cái giá hiện ra ngày
        # 20-08-2026: Quang chạy một bản cài đặt khác và khai đủ tham số, nên
        # `check_fingerprints` báo lệch ở `compute_type`/`min_silence_duration_ms`
        # — trong khi hai bên chạy đúng cùng một giá trị. Sáu lần từ chối, năm
        # trong đó chỉ tồn tại trong sổ sách.
        #
        # Một tham số KHÔNG được ghi thì không so được, và "không so được" trông
        # y hệt "đã so và khớp" ở mọi báo cáo phía sau.
        params={
            "beam_size": beam,
            "language": "vi",
            "window": window,
            "vad_filter": True,
            "min_silence_duration_ms": 500,
            "compute_type": "float16",
            "max_empty_ratio": max_empty_ratio,
        },
        # Số đo của TỪNG video, nên phải là `facts` chứ không phải `params`:
        # `check_fingerprints` so `params` giữa các mảnh, nên một con số đổi theo
        # video mà nằm trong `params` sẽ khiến mọi video lệch mọi video.
        facts={
            "segments": len(spans),
            "keyframes_total": keyframes_total,
            "keyframes_with_subtitle": len(mapping),
            "keyframes_without_subtitle": keyframes_without_subtitle,
            "empty_ratio": round(empty_ratio, 6),
        },
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
    **~532 gio GPU cho batch 1**. Luc do trong repo con `build_qwen_generate()` da
    gop lo san cho ReCap, con tang nay thi khong — khong vi ly do nao.
    (`build_qwen_generate` da bi xoa 25-08-2026 cung `n6_recap`.)
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
        # ma khong ném loi.
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


#: Ten tham so `PaddleOCR` chap nhan, theo the he. 3.x doi ten gan het.
#: `enable_mkldnn` co o ca hai — xem `_paddle_params` ve vi sao no mac dinh TAT.
_PADDLE_KWARGS: dict[int, tuple[str, ...]] = {
    2: ("lang", "use_gpu", "show_log", "use_angle_cls", "enable_mkldnn"),
    3: (
        "lang",
        "device",
        "use_textline_orientation",
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "enable_mkldnn",
    ),
}


def _paddle_major(version_string: str) -> int:
    """So chinh cua phien ban paddleocr; 3 khi khong doc duoc (ban moi hon)."""
    head = version_string.split(".", 1)[0]
    return int(head) if head.isdigit() else 3


def _signature_names(target: Any) -> set[str] | None:
    """Ten tham so `target` nhan, hoac `None` khi no chi co `**kwargs`."""
    import inspect  # noqa: PLC0415

    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # pragma: no cover - built-in khong co chu ky
        return None
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return None
    return set(parameters)


def build_paddle_ocr(force: dict | None = None, **kwargs: Any) -> Any:
    """Mot `PaddleOCR`, dung **MOT** lan khoi tao.

    PaddleOCR doi ten tham so giua 2.x va 3.x (`use_gpu` -> `device`,
    `use_angle_cls` -> `use_textline_orientation`, `show_log` bien mat), nen
    khong the pin cung mot bo ten.

    **Vi sao khong "thu roi bo tham so bi tu choi rop thu lai".** Do la thiet ke
    dau tien cua ham nay va no SAI, do duoc tren Kaggle: PaddleOCR 3.x khoi tao
    PaddleX o pham vi tien trinh, nen lan dung thu hai — du lan dau da ném loi —
    chet bang `RuntimeError: PDX has already been initialized. Reinitialization
    is not supported.` Mot vong lap thu lai bien mot loi doc duoc thanh mot loi
    khac che mat loi that. Cung ly do do: **mot lan khoi tao that bai la phai
    khoi dong lai kernel**, khong phai chay lai o.

    Nen ten tham so duoc chon TRUOC khi khoi tao, theo phien ban da cai, roi loc
    them mot lan theo chu ky that neu doc duoc. `force` di thang, khong loc — no
    la duong thoat cho mot ban co ten khac ca hai bang tren.
    """
    from paddleocr import PaddleOCR  # noqa: PLC0415

    major = _paddle_major(_package_version("paddleocr"))
    names = _PADDLE_KWARGS.get(major, _PADDLE_KWARGS[3])
    chosen = {k: v for k, v in kwargs.items() if k in names and v is not None}

    accepted = _signature_names(PaddleOCR)
    if accepted is not None:
        dropped = sorted(set(chosen) - accepted)
        if dropped:
            logger.info("paddleocr %s khong nhan %s", major, ", ".join(dropped))
        chosen = {k: v for k, v in chosen.items() if k in accepted}

    chosen.update(force or {})
    logger.info("khoi tao PaddleOCR (%s) voi %s", major, sorted(chosen))
    return PaddleOCR(**chosen)


#: Ten ham da THUC SU duoc goi o lan quet gan nhat: `predict` (3.x) hay `ocr`
#: (2.x). Ghi vao marker. Doc tu day chu khong hoi lai doi tuong model, vi khi
#: `paddle_scan` tra ve ket qua da cache thi model co the chua he duoc dung.
_PADDLE_API = ""


def _paddle_read(ocr: Any, image: Path) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """Chay pipeline PaddleOCR tren mot anh. `predict` la ten cua 3.x, `ocr` cua 2.x."""
    global _PADDLE_API  # noqa: PLW0603
    run = getattr(ocr, "predict", None)
    _PADDLE_API = "predict" if run is not None else "ocr"
    return normalize_paddle_result((run or ocr.ocr)(str(image)))


def probe_keyframes(ctx: StageContext) -> list[Keyframe]:
    """Anh ma mot tang N5b se chay, sau khi ap `limit`.

    `limit` la cong cu cua phep DO, khong phai cua lan chay that: chat luong doc
    chu cua mot bo OCR doc duoc tu vai chuc anh, con 307 anh tren CPU la 25 phut
    moi vong. Marker ghi `images` va `keyframes_total`, nen mot lan chay bi cat
    khong bao gio trong giong mot lan chay du.
    """
    limit = int(ctx.params.get("limit", 0))
    return ctx.keyframes[:limit] if limit > 0 else list(ctx.keyframes)


def _progress(label: str, done: int, total: int, started: float, every: int) -> None:
    """In tien do theo dinh ky. Im lang khi `every <= 0`.

    Ton tai vi mot tang chay 307 anh tren CPU khong in gi trong 25 phut thi tu
    ngoai KHONG phan biet duoc voi treo — va lan chay dau tien cua tang nay da
    dung nhu vay.
    """
    if every <= 0 or (done % every and done != total):
        return
    import time  # noqa: PLC0415

    elapsed = time.perf_counter() - started
    rate = elapsed / max(1, done)
    logger.info(
        "%s %d/%d anh · %.2fs/anh · con ~%.1f phut",
        label, done, total, rate, rate * (total - done) / 60,
    )


def _paddle_params(ctx: StageContext) -> dict:
    """Bo tham so **hop nhat ca hai the he**; `build_paddle_ocr` loc theo ban da cai.

    `enable_mkldnn` mac dinh **False**, va do la mot phep doi cham lay chay duoc.
    Do duoc tren Kaggle CPU voi paddlepaddle 3.x:

        (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
        [pir::ArrayAttribute<pir::DoubleAttribute>]
        (at .../new_executor/instruction/onednn/onednn_instruction.cc:116)

    Duong thuc thi PIR moi cua Paddle gap mot attribute ma nhanh oneDNN cua no
    chua cai dat. Khong phai loi cua ta va khong co gi ta sua duoc trong Paddle —
    nhung tat oneDNN thi ca nhanh do bien mat. CPU cham hon, doi lai la no chay.
    Bat lai bang `params={"enable_mkldnn": True}` khi da co ban Paddle vuot qua.
    """
    lang = ctx.params.get("lang", "vi")
    device = ctx.params.get("device", "gpu")
    return {
        "lang": lang,
        "device": device,
        "use_gpu": device != "cpu",
        "show_log": False,
        "use_angle_cls": False,
        "use_textline_orientation": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "enable_mkldnn": bool(ctx.params.get("enable_mkldnn", False)),
    }


def _paddle_cache_key(ctx: StageContext) -> str:
    kwargs = _paddle_params(ctx)
    force = dict(ctx.params.get("paddle_kwargs", {}))
    # `force` vao khoa cache: doi no ma dung lai doi tuong cu la chay mot cau
    # hinh khac voi cai marker se ghi.
    return (
        f"paddleocr lang={kwargs['lang']} device={kwargs['device']} "
        f"mkldnn={kwargs['enable_mkldnn']} force={sorted(force.items())}"
    )


def _paddle_instance(ctx: StageContext) -> Any:
    kwargs = _paddle_params(ctx)
    force = dict(ctx.params.get("paddle_kwargs", {}))
    return cached_model(
        _paddle_cache_key(ctx), lambda: build_paddle_ocr(force=force, **kwargs)
    )


def paddle_scan(
    ctx: StageContext, keyframes: list[Keyframe], *, label: str
) -> list[list[tuple[tuple[int, int, int, int], str, float]]]:
    """Chay PaddleOCR tren `keyframes`, dung lai ket qua neu phien nay da chay.

    Ca hai tang N5b goi ham nay, nen tang chay sau **khong tra lai** gia cua
    pipeline. Do la mot nua thoi gian cua ca phep do khi chay tren CPU.
    """
    global _DETECTION_SCOPE  # noqa: PLW0603

    scope = (ctx.video_id, _paddle_cache_key(ctx))
    if _DETECTION_SCOPE != scope:
        # MOT video mot luc, va do la mot chan bo nho chu khong phai toi uu.
        # Hai tang luon chay lien tiep tren cung mot video, nen giu lai video
        # truoc khong cuu duoc lan goi nao — no chi tich lai: 400 video x ~300
        # keyframe x mot danh sach hop moi cai la mot phien Kaggle het RAM o
        # giua chung, sau nhieu gio da chay.
        _DETECTIONS.clear()
        _DETECTION_SCOPE = scope

    todo = [kf for kf in keyframes if kf.n not in _DETECTIONS]
    if todo:
        import time  # noqa: PLC0415

        ocr = _paddle_instance(ctx)
        every = int(ctx.params.get("progress_every", 25))
        started = time.perf_counter()
        for done, keyframe in enumerate(todo, 1):
            _DETECTIONS[keyframe.n] = _paddle_read(ocr, keyframe.image)
            _progress(label, done, len(todo), started, every)
    elif keyframes:
        logger.info("%s: dung lai ket qua Paddle da co cho %d anh", label, len(keyframes))
    return [_DETECTIONS[kf.n] for kf in keyframes]


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
    settings = _paddle_params(ctx)
    lang = settings["lang"]
    score_min = float(ctx.params.get("rec_score_min", 0.0))
    keyframes = probe_keyframes(ctx)

    with Stopwatch() as timer:
        scanned = paddle_scan(ctx, keyframes, label="paddle")
        mapping: dict[str, str] = {}
        boxes_total = 0
        dropped_low_score = 0
        silent_images = 0

        for keyframe, found in zip(keyframes, scanned, strict=True):
            boxes_total += len(found)
            kept = [(box, text) for box, text, score in found if score >= score_min]
            dropped_low_score += len(found) - len(kept)
            text = join_text_lines([b for b, _ in kept], [t for _, t in kept])
            if text:
                mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = text
            else:
                silent_images += 1

    images = len(keyframes)
    return ctx.writer.write_text_part(
        "paddle_ocr",
        ctx.video_id,
        mapping,
        model=f"paddleocr {_package_version('paddleocr')} lang={lang}",
        params={
            "lang": lang,
            "rec_score_min": score_min,
            "api": _PADDLE_API,
            "order": "group_text_lines",
        },
        # Do duoc, khac nhau tung video — xem `write_text_part`.
        facts={
            "images": images,
            # Mot lan chay bi `limit` cat phai KHONG trong giong mot lan chay du:
            # `rows` thap co the la "bo doc im lang" hoac "moi chay 40/307 anh".
            "keyframes_total": len(ctx.keyframes),
            # Duong tinh toan, KHONG phai cau hinh cua phep do. O `facts` chu
            # khong `params`: hai nguoi chay mot nguoi GPU mot nguoi CPU van gop
            # duoc bundle, nhung phai NHIN THAY duoc khi doc lai con so thoi gian.
            "device": settings["device"],
            "enable_mkldnn": settings["enable_mkldnn"],
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
    keyframes = probe_keyframes(ctx)
    settings = _paddle_params(ctx)

    def load_vietocr():
        # `Image.ANTIALIAS` bi Pillow 10 XOA (da bo tu 9.1). vietocr 0.3.x goi no
        # trong buoc resize dong chu, nen tren moi moi truong Pillow >= 10 tang
        # nay chet bang `module 'PIL.Image' has no attribute 'ANTIALIAS'` — do
        # duoc 19-08-2026, sau 486 giay chay Paddle xong roi moi ngã.
        #
        # Gan lai la KHOI PHUC CHINH XAC, khong phai thay the gan dung: o Pillow
        # <= 9, `ANTIALIAS` va `LANCZOS` la cung mot hang so (ca hai = 1). Cach
        # con lai la ha Pillow xuong < 10, ma Pillow la thu torchvision va
        # transformers cung dung — doi mot loi cua vietocr lay mot loi cua SigLIP.
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS
            logger.info("da gan lai PIL.Image.ANTIALIAS = LANCZOS cho vietocr")

        from vietocr.tool.config import Cfg  # noqa: PLC0415
        from vietocr.tool.predictor import Predictor  # noqa: PLC0415

        config = Cfg.load_config_from_name(cfg_name)
        config["device"] = device
        # `predictor` co the vang trong vai ban config; dat co dieu kien.
        if isinstance(config.get("predictor"), dict):
            config["predictor"]["beamsearch"] = beamsearch
        return Predictor(config)

    mapping: dict[str, str] = {}
    boxes_total = 0
    crops_skipped = 0
    silent_images = 0

    with Stopwatch() as timer:
        # PHA 1 — Paddle phat hien HET, truoc khi VietOCR (va torch) duoc nap.
        #
        # Hai pha chu khong mot vong lap xen ke, vi hai runtime nay khong hoa
        # thuan: paddle va torch cung mang theo oneDNN/OpenMP rieng, va lan chay
        # 19-08 chet ngay o anh dau cua tang nay bang mot loi NOI BO cua Paddle —
        # trong khi chinh phep phat hien do da chay tron ven o `n5_ocr_paddle` vai
        # phut truoc, cung doi tuong, cung tham so. Tach ra thi phan Paddle cua
        # tang nay co hinh dang y het tang kia: mot chuoi lien tuc, khong co gi
        # nap chen vao giua.
        import time  # noqa: PLC0415

        started = time.perf_counter()
        scanned = paddle_scan(ctx, keyframes, label="vietocr/phat hien")
        detected: list[list[tuple[int, int, int, int]]] = []
        for found in scanned:
            boxes_total += len(found)
            keep: list[tuple[int, int, int, int]] = []
            for box, _text, _score in found:
                x0, y0, x1, y1 = box
                if x1 - x0 < min_side or y1 - y0 < min_side:
                    # Mot hop mong hon vai pixel khong phai dong chu. VietOCR van
                    # tra ve MOT chuoi cho no, va chuoi do di thang vao index.
                    crops_skipped += 1
                    continue
                keep.append(box)
            detected.append(keep)
        # Tach ra vi hai nua tra loi hai cau hoi khac nhau: "phat hien co du nhanh
        # de chay 873 video khong" va "VietOCR doc mot dong het bao lau". Khi
        # `n5_ocr_paddle` da chay truoc trong cung phien, `paddle_scan` tra ve
        # ket qua da co va con so nay ~0 — dung nhu the.
        detect_seconds = round(time.perf_counter() - started, 2)

        # PHA 2 — nap VietOCR roi doc. Anh mo lai o day: giu 307 anh trong bo nho
        # de tiet kiem mot lan mo file la doi mot van de re lay mot van de dat.
        reader = cached_model(f"vietocr {cfg_name}", load_vietocr)
        batch_predict = getattr(reader, "predict_batch", None)
        batched = batch_predict is not None
        every = int(ctx.params.get("progress_every", 25))
        phase2 = time.perf_counter()

        for done, (keyframe, boxes) in enumerate(
            zip(keyframes, detected, strict=True), 1
        ):
            if boxes:
                page = Image.open(keyframe.image).convert("RGB")
                crops = [_crop_for_recognition(page, box, pad) for box in boxes]
                texts = (
                    list(batch_predict(crops)) if batched
                    else [reader.predict(crop) for crop in crops]
                )
                text = join_text_lines(boxes, texts)
            else:
                text = ""
            if text:
                mapping[build_keyframe_id(ctx.video_id, keyframe.frame_idx)] = text
            else:
                silent_images += 1
            _progress("vietocr/doc", done, len(keyframes), phase2, every)

    images = len(keyframes)
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
            # Ghi lai vi no la mot thay doi hanh vi cua thu vien ben duoi, va
            # hai lan chay tren hai ban Pillow khac nhau chi so sanh duoc khi
            # biet ban nao da phai va lai.
            "pillow_antialias_shim": True,
        },
        facts={
            "images": images,
            "boxes": boxes_total,
            "keyframes_total": len(ctx.keyframes),
            "device": settings["device"],
            "enable_mkldnn": settings["enable_mkldnn"],
            "vietocr_device": device,
            "crops_skipped": crops_skipped,
            "silent_images": silent_images,
            "batched": batched,
            "detect_seconds": detect_seconds,
            "recognize_seconds": round(timer.seconds - detect_seconds, 2),
            "seconds_per_image": round(timer.seconds / max(1, images), 3),
        },
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N6 — Qwen2.5-VL qua vLLM: caption + chu tren man hinh, MOT lan sinh
# ---------------------------------------------------------------------------

#: Anh moi lan `llm.generate()`. 32 la con so cua Bao tren 2xT4 voi
#: `gpu_memory_utilization=0.9` — model 7B chia doi con ~8 GB moi card lam KV
#: cache. vLLM tu xep lich trong lo nen day la tran, khong phai kich thuoc co dinh.
QWEN25_BATCH = 32

#: Stop token cua Qwen2.5 (`<|endoftext|>`, `<|im_end|>`).
_QWEN25_STOP_IDS = (151643, 151645)

#: Throughput da do cua bien the `bao` tren 2xT4: **~60,5 giay cho lo 32 anh**,
#: tuc **~0,52 anh/giay** — ~95 gio GPU cho 177.321 keyframe. Moi thay doi prompt
#: phai duoc do lai so voi con so nay TRUOC khi ap len ca bo (notebook 08, ô đo
#: throughput). Prompt dai hon lam prefill nang hon; `max_tokens` cao hon lam
#: decode dai hon. Ca hai deu cong vao 95 gio do.
QWEN25_BASELINE_IMAGES_PER_S = 0.52


def qwen25_prompt(ctx: StageContext) -> str:
    """Prompt cua luot chay nay. `params["prompt_variant"]` chon bien the."""
    variant = str(ctx.params.get("prompt_variant", "aic"))
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"prompt_variant={variant!r} khong co. Biet: {sorted(PROMPT_VARIANTS)}"
        )
    return PROMPT_VARIANTS[variant][0]


def qwen25_max_tokens(ctx: StageContext) -> int:
    """`max_tokens` cua luot nay — mac dinh THEO BIEN THE, khong phai mot so chung."""
    variant = str(ctx.params.get("prompt_variant", "aic"))
    return int(ctx.params.get("max_tokens", PROMPT_VARIANTS[variant][1]))


def _qwen25_cache_key(ctx: StageContext) -> str:
    """Van tay cua mot luot sinh. Doi bat ky thu gi o day la cache phai truot."""
    return json.dumps(
        {
            "checkpoint": ctx.params.get("checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct"),
            "max_tokens": qwen25_max_tokens(ctx),
            "temperature": float(ctx.params.get("temperature", 0.1)),
            # Ban than prompt, khong phai ten bien the: doi noi dung prompt ma
            # giu ten thi cache van trung, va lan chay sau tra ve cau cua prompt cu.
            "prompt": qwen25_prompt(ctx),
            "text_first": bool(ctx.params.get("text_first", True)),
        },
        sort_keys=True,
    )


def parse_qwen25_caption_ocr(text: str) -> tuple[str, str, bool]:

    """`(caption, chu doc duoc, co DU hai nhan)` tu cau tra loi tho.

    Ba thu phai xu ly, va ca ba deu quan sat duoc tren chinh 855 cau tra loi cua
    Bao:

    1. **Ngoac don bao quanh.** Prompt viet `[CAPTION] (your description here)`
       nen model thuong tra ve ca cap ngoac: `[CAPTION] (The image shows ...)`.
       Giu nguyen thi moi caption bat dau bang `(` — vo hai cho BM25 nhung ban
       ra hai chuoi khac nhau khi doi chieu voi ban cua Bao.
    2. **Cau tra loi mat nhan.** Tra ve `False` o phan tu thu ba de nguoi goi
       DEM duoc. Khong co nhan `[TEXT]` thi ta coi ca cau la caption va de OCR
       trong — an toan hon la doan, vi doan sai se do mot doan mo ta canh vat
       vao chi muc OCR.
    3. **Tu bo cuoc.** Prompt CO cho san duong thoat `'None'`, nen model se dung
       no. Loc theo TUNG DONG bang `_BAIL_WORDS`, cung ham `parse_qwen_ocr` da
       dung: cho phep loc that bai la khi model tra ca chu that VA tu bo cuoc
       trong cung mot cau.
    """
    raw = (text or "").strip()
    has_caption = "[CAPTION]" in raw
    has_text = "[TEXT]" in raw

    if has_text:
        head, _, tail = raw.partition("[TEXT]")
    else:
        head, tail = raw, ""
    caption = head.replace("[CAPTION]", "", 1).strip() if has_caption else head.strip()

    def unwrap(value: str) -> str:
        value = value.strip()
        if value.startswith("(") and value.endswith(")") and value.count("(") == 1:
            return value[1:-1].strip()
        return value

    caption = unwrap(caption)

    kept: list[str] = []
    for line in unwrap(tail).splitlines():
        cleaned = line.strip().strip("*").strip()
        if not cleaned:
            continue
        if cleaned.lower().strip(" .:\"'") in _BAIL_WORDS:
            continue
        kept.append(cleaned)
    return caption, "\n".join(kept).strip(), (has_caption and has_text)


def qwen25_scan(ctx: StageContext, keyframes: list[Keyframe], *, label: str) -> list[str]:
    """Cau tra loi tho cho `keyframes`, dung lai neu phien nay da sinh.

    Hai tang N6 goi ham nay tren cung mot video, nen tang chay sau **khong tra
    lai** tien GPU. Do la mot nua tong chi phi cua ca luot — dat hon nhieu so
    voi `paddle_scan`, vi day la mot model 7B chu khong phai mot pipeline CPU.
    """
    global _QWEN_VL_SCOPE  # noqa: PLW0603

    scope = (ctx.video_id, _qwen25_cache_key(ctx))
    if _QWEN_VL_SCOPE != scope:
        # MOT video mot luc — chan bo nho, cung ly do `paddle_scan` ghi: giu lai
        # video truoc khong cuu duoc lan goi nao, no chi tich lai cho toi khi
        # phien Kaggle het RAM sau nhieu gio da chay.
        _QWEN_VL_ANSWERS.clear()
        _QWEN_VL_SCOPE = scope

    todo = [kf for kf in keyframes if kf.n not in _QWEN_VL_ANSWERS]
    if not todo:
        if keyframes:
            logger.info("%s: dung lai %d cau tra loi da sinh", label, len(keyframes))
        return [_QWEN_VL_ANSWERS[kf.n] for kf in keyframes]

    import time  # noqa: PLC0415

    from qwen_vl_utils import process_vision_info  # noqa: PLC0415
    from vllm import SamplingParams  # noqa: PLC0415

    processor, llm = _qwen25_engine(ctx)
    prompt_text = qwen25_prompt(ctx)
    sampling = SamplingParams(
        temperature=float(ctx.params.get("temperature", 0.1)),
        max_tokens=qwen25_max_tokens(ctx),
        stop_token_ids=list(_QWEN25_STOP_IDS),
    )
    width = int(ctx.params.get("batch_size", QWEN25_BATCH))
    every = int(ctx.params.get("progress_every", 1))
    text_first = bool(ctx.params.get("text_first", True))

    started = time.perf_counter()
    for start in range(0, len(todo), width):
        batch = todo[start : start + width]
        prompts = []
        for keyframe in batch:
            # THU TU CO Y, va no la dieu kien de prefix caching co tac dung.
            #
            # vLLM tai dung KV-cache theo TIEN TO chung tinh tu token dau tien.
            # Dat anh truoc (nhu ban cua Bao) thi hang tram token thi giac —
            # khac nhau o moi anh — nam ngay dau, nen phan chi dan + few-shot
            # dai phia sau KHONG BAO GIO trung tien to va phai prefill lai tung
            # lan. Dat chi dan truoc thi ca khoi do la mot tien to co dinh dung
            # chung cho ca 177.321 anh.
            #
            # `text_first=False` giu lai de doi chieu voi ban cua Bao — bien the
            # `bao` chay dung thu tu cua ban ay.
            text_part = {"type": "text", "text": prompt_text}
            image_part = {"type": "image", "image": str(keyframe.image)}
            content = [text_part, image_part] if text_first else [image_part, text_part]
            messages = [{"role": "user", "content": content}]
            images, _ = process_vision_info(messages)
            prompts.append(
                {
                    "prompt": processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ),
                    "multi_modal_data": {"image": images[0]},
                }
            )
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        # `zip(strict=True)`: vLLM tra ve theo dung thu tu dua vao, nhung "dung
        # thu tu" la thu ta dang DUA VAO chu khong phai thu da kiem. Lech mot
        # phan tu la gan caption cua anh nay cho anh khac — sai lang le, va
        # khong bang ket qua nao lo ra.
        for keyframe, output in zip(batch, outputs, strict=True):
            _QWEN_VL_ANSWERS[keyframe.n] = output.outputs[0].text.strip()
        _progress(label, min(start + width, len(todo)), len(todo), started, every * width)

    # Throughput cua chinh luot nay, de tang ghi vao `facts`. Khong co no thi
    # cau hoi "prompt dai hon co lam cham khong" chi tra loi duoc bang cach ngoi
    # bam gio, va khong ai bam gio mot luot chay 95 tieng.
    elapsed = time.perf_counter() - started
    _QWEN_VL_TIMING["images"] = _QWEN_VL_TIMING.get("images", 0) + len(todo)
    _QWEN_VL_TIMING["seconds"] = _QWEN_VL_TIMING.get("seconds", 0.0) + elapsed
    logger.info(
        "%s: %d anh trong %.1fs = %.2f anh/giay (nen cua bien the `bao`: %.2f)",
        label, len(todo), elapsed, len(todo) / max(elapsed, 1e-9),
        QWEN25_BASELINE_IMAGES_PER_S,
    )

    return [_QWEN_VL_ANSWERS[kf.n] for kf in keyframes]


def _qwen25_engine(ctx: StageContext):
    """`(processor, LLM)` cua vLLM, nap mot lan cho ca phien.

    Tham so lay y nguyen tu notebook cua Bao, va moi dong co mot ly do nguoi
    doc khong suy ra duoc:

    * `tensor_parallel_size=2` — cua doi model 7B ra hai card T4, moi card ~7 GB,
      con ~8 GB lam KV cache. Kaggle mot GPU thi phai dat `tensor_parallel_size=1`
      va ha `max_model_len`; xem `notebooks/btc/08-*.ipynb`.
    * `dtype="half"` — T4 khong co bfloat16.
    * `enforce_eager=True` — **bat buoc** tren T4: CUDA Graph cua vLLM no tren
      kien truc Turing.
    * `enable_prefix_caching=True` — tai dung KV-cache cua TIEN TO chung. Voi
      bien the `aic`, tien to do la ca khoi chi dan + few-shot (~900 token) va no
      giong het nhau o ca 177.321 request. Chi co tac dung khi prompt dat TRUOC
      anh (`text_first`, xem `qwen25_scan`).

    Prefix caching **khong duoc phep lam do ca luot chay**: vLLM cu chua ho tro
    no cho dau vao da phuong thuc va se nem ngay o `LLM(...)`. Nen ta thu bat,
    va neu no khong nhan thi nap lai KHONG co no va GHI LAI su that do —
    `enable_prefix_caching` di vao `facts` cua marker chu khong phai mot gia dinh
    trong dau nguoi chay. Mot toi uu im lang tat di la mot con so throughput
    khong giai thich duoc.
    """
    checkpoint = ctx.params.get("checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct")
    tp_size = int(ctx.params.get("tensor_parallel_size", 2))
    max_len = int(ctx.params.get("max_model_len", 4096))
    utilisation = float(ctx.params.get("gpu_memory_utilization", 0.9))
    wants_prefix = bool(ctx.params.get("enable_prefix_caching", True))
    kv_bytes = int(ctx.params.get("kv_cache_memory_bytes", 0) or 0)

    def build():
        from transformers import AutoProcessor  # noqa: PLC0415
        from vllm import LLM  # noqa: PLC0415

        common = {
            "model": checkpoint,
            "tensor_parallel_size": tp_size,
            "dtype": "half",
            "gpu_memory_utilization": utilisation,
            "max_model_len": max_len,
            "enforce_eager": True,
            "limit_mm_per_prompt": {"image": 1},
        }
        processor = AutoProcessor.from_pretrained(checkpoint)

        # Hai co TUY CHON, va ca hai deu co the bi ban vLLM nay tu choi. Thu tu
        # bo bot di tu day du -> tran, va GHI LAI cai nao that su dinh lai.
        #
        # Vi sao phai la mot chuoi chu khong phai mot `try` long nhau: bo prefix
        # caching roi thu lai VOI kv_cache_memory_bytes se nem lai dung cho do,
        # va lan nay khong ai bat. Mot duong lui chi lui duoc mot buoc la mot
        # duong lui hong.
        attempts = []
        if wants_prefix and kv_bytes:
            attempts.append(("prefix + kv", {"enable_prefix_caching": True,
                                             "kv_cache_memory_bytes": kv_bytes}))
        if wants_prefix:
            attempts.append(("prefix", {"enable_prefix_caching": True}))
        if kv_bytes:
            attempts.append(("kv", {"kv_cache_memory_bytes": kv_bytes}))
        attempts.append(("tran", {}))

        last_error = None
        for label, extra in attempts:
            try:
                engine = LLM(**common, **extra)
            except Exception as err:  # noqa: BLE001 — ban vLLM cu khong nhan co nay
                last_error = err
                logger.warning(
                    "vLLM tu choi cau hinh `%s` (%s: %s) — thu cau hinh ke tiep.",
                    label, type(err).__name__, str(err)[:160],
                )
                continue
            _QWEN_VL_TIMING["prefix_caching"] = float("enable_prefix_caching" in extra)
            _QWEN_VL_TIMING["kv_cache_bytes"] = float(extra.get("kv_cache_memory_bytes", 0))
            logger.info(
                "vLLM: cau hinh `%s` (prefix_caching=%s, kv_cache=%s)",
                label,
                "enable_prefix_caching" in extra,
                f"{kv_bytes / 2**30:.2f} GiB" if "kv_cache_memory_bytes" in extra else "tu chon",
            )
            return processor, engine
        raise RuntimeError(
            "vLLM khong nap duoc voi bat ky cau hinh nao da thu"
        ) from last_error

    # `kv_bytes` PHAI nam trong khoa cache. Khong co no thi doi con so nay giua
    # hai luot trong CUNG mot phien se dung lai dong co cu, va bang throughput se
    # bao "khong doi gi" — mot ket luan sai ma khong gi noi ra.
    return cached_model(
        f"vllm {checkpoint} tp{tp_size} len{max_len} "
        f"prefix{int(wants_prefix)} kv{kv_bytes}",
        build,
    )


def _qwen25_common(ctx: StageContext) -> tuple[dict, dict, dict]:
    """`(params, facts, {n: (caption, ocr)})` — phan hai tang N6 dung chung."""
    answers = qwen25_scan(ctx, ctx.keyframes, label=f"qwen2.5-vl {ctx.video_id}")
    parsed: dict[int, tuple[str, str]] = {}
    unmarked = 0
    for keyframe, answer in zip(ctx.keyframes, answers, strict=True):
        caption, ocr, marked = parse_qwen25_caption_ocr(answer)
        parsed[keyframe.n] = (caption, ocr)
        unmarked += 0 if marked else 1

    prompt_text = qwen25_prompt(ctx)
    params = {
        "checkpoint": ctx.params.get("checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct"),
        "max_tokens": qwen25_max_tokens(ctx),
        "temperature": float(ctx.params.get("temperature", 0.1)),
        "engine": "vllm",
        "prompt_variant": str(ctx.params.get("prompt_variant", "aic")),
        # Van tay cua CHINH chuoi prompt, khong phai ten bien the. Sua mot chu
        # trong prompt ma giu ten thi hai nguoi sinh ra hai bo caption khac nhau
        # duoi cung mot nhan, va `check_fingerprints` se khong thay gi.
        "prompt_sha": hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
        # Thu tu text/anh doi ca dau vao cua model, nen no la tham so cua BUNDLE:
        # hai nguoi chay hai thu tu la hai bo caption khong so duoc voi nhau.
        "text_first": bool(ctx.params.get("text_first", True)),
    }
    # `tensor_parallel_size` / `batch_size` / prefix caching la lua chon cua MAY,
    # khong phai cua bundle: mot phien Kaggle 2xT4 va mot phien 1xP100 deu hop le,
    # va prefix caching chi doi TOC DO chu khong doi chuoi tra ve. Dat chung vao
    # `params` la `check_fingerprints` tu choi ca bundle vi hai nguoi dung hai
    # loai GPU — mot phep tu choi dung hinh thuc va rong ve noi dung.
    images = _QWEN_VL_TIMING.get("images", 0.0)
    seconds = _QWEN_VL_TIMING.get("seconds", 0.0)
    facts = {
        "unmarked": unmarked,
        "tensor_parallel_size": int(ctx.params.get("tensor_parallel_size", 2)),
        "batch_size": int(ctx.params.get("batch_size", QWEN25_BATCH)),
        "prefix_caching": bool(_QWEN_VL_TIMING.get("prefix_caching", 0.0)),
        # 0 = de vLLM tu chon. Con so THAT da dinh lai, khong phai con so da xin.
        "kv_cache_bytes": int(_QWEN_VL_TIMING.get("kv_cache_bytes", 0.0)),
        # Throughput CONG DON cua ca phien, khong phai cua video nay: lan nap
        # model dau tien lam con so tren mot video vo nghia.
        "images_per_s": round(images / seconds, 3) if seconds else None,
    }
    return params, facts, parsed


def stage_n6_caption_qwen(ctx: StageContext) -> dict:
    """Nua `[CAPTION]` cua mot lan sinh Qwen2.5-VL. Chay TRUOC `n6_ocr_qwen25`.

    Thu tu quan trong: hai tang doc hai nua cua CUNG mot cau tra loi, va
    `qwen25_scan` chi giu lai ket qua trong PHIEN va cho MOT video. Chay hai tang
    lien tiep tren tung video (vong lap theo VIDEO, khong theo TANG) la dieu kien
    de tang thu hai khong phai sinh lai — giong het rang buoc cua cap
    `n5_ocr_paddle` / `n5_ocr_vietocr`.
    """
    with Stopwatch() as timer:
        params, facts, parsed = _qwen25_common(ctx)
        mapping = {
            build_keyframe_id(ctx.video_id, kf.frame_idx): parsed[kf.n][0]
            for kf in ctx.keyframes
            if parsed[kf.n][0]
        }

    return ctx.writer.write_text_part(
        "qwen_captions",
        ctx.video_id,
        mapping,
        model=params["checkpoint"],
        params={**params, "field": "caption"},
        facts=facts,
        elapsed_s=timer.seconds,
    )


def stage_n6_ocr_qwen25(ctx: StageContext) -> dict:
    """Nua `[TEXT]` cua cung lan sinh. Chay NGAY SAU `n6_caption_qwen`.

    Ghi vao `qwen_ocr` (`qwen_ocr.json`, source `ocr_qwen25`) chu KHONG vao
    `sviptr_ocr`: cai ten do thuoc ve `n5_ocr_qwen` — Qwen2-**VL** 7B 4-bit, mot
    model khac, do duoc 10,8 giay/anh va da bi bo (CLAUDE.md §4). Dung lai ten
    cua no la lam song lai mot artifact vinh vien khong co du lieu.

    Arm `ocr_qwen25` da duoc noi (`TEXT_ARM_SPECS`), nhung trong so cua no CHUA
    duoc do: cau hoi "Qwen2.5 doc chu tot hon `viet_ocr` khong" can chinh du lieu
    nay moi tra loi duoc. Ba nguon OCR hien co da cho thay cau tra loi khong hien
    nhien — `ocr` (easyocr) chay mot minh 17/42 nhung dong gop AM trong hop nhat.
    """
    with Stopwatch() as timer:
        params, facts, parsed = _qwen25_common(ctx)
        mapping = {
            build_keyframe_id(ctx.video_id, kf.frame_idx): parsed[kf.n][1]
            for kf in ctx.keyframes
            if parsed[kf.n][1]
        }

    return ctx.writer.write_text_part(
        "qwen_ocr",
        ctx.video_id,
        mapping,
        model=params["checkpoint"],
        params={**params, "field": "text"},
        facts=facts,
        elapsed_s=timer.seconds,
    )


# ---------------------------------------------------------------------------
# N6 — nhung caption bang BGE-M3
# ---------------------------------------------------------------------------
#
# GO 25-08-2026: `stage_n6_recap`, `group_shots`, `build_shots`,
# `build_qwen_generate`, `build_gemini_generate` va ca module `mfe.offline.recap`
# da bi xoa. Nhom doi sang Qwen2.5-VL-7B-Instruct qua vLLM (`n6_caption_qwen` +
# `n6_ocr_qwen25`), nen duong ReCap (Gemini/Qwen2-VL, co bo nho theo chuoi shot)
# khong con nguoi dung nao. Chi tiet + so do giu lai o handoff §C.1.
#
# Hai con so tu doan da xoa, giu lai vi chung do duoc va se can neu ai do quay
# lai chuyen cat shot cho KIS (57% de hoi thu tu canh):
#
#   * Cat shot chi bang `gap > nguong` la lien ket don: mot chuoi dai nuot het.
#     Do tren 16 video BTC, nguong co VACH DUNG chu khong phai doc thoai:
#     2,0s -> 2537 shot · 6,0s -> 259 · 8,0s -> **16** (dung mot shot moi video,
#     vi khoang cach keyframe lon nhat cua BTC la 8,4s).
#   * Vach do nam SAI PHIA voi DAKE: DAKE bao dam moi khoang cach <= 2,0s, nen
#     nguong 2,0s khong bao gio kich hoat — mot video DAKE 400 keyframe thanh
#     MOT shot. Muon dung lai thi phai chan do dai, khong chi chan khoang cach.


#: Artifact chua caption ma `n6_bge_m3` nhung. MOT nguon, khong phai tham so.
#:
#: Truoc 25-08-2026 day la `dense_captions` (cua `n6_recap`, Gemini/ReCap). Tang do
#: da bi xoa, nen mot tham so `source` o day chi con dung mot gia tri — tuc la mot
#: nhanh chet cho ai do doc nham thanh "co the doi duoc".
_BGE_M3_SOURCE = "qwen_captions"


def stage_n6_bge_m3(ctx: StageContext) -> dict:
    """Caption -> BGE-M3. Tach khoi lan sinh caption, nen nhung lai la mien phi.

    Doc part `qwen_captions` **tu Volume** chu khong nhan qua `ctx`: tang nay
    khong can anh, khong can map-keyframes, khong can GPU sinh. Do la ly do no
    chay o mot LUOT RIENG chu khong noi duoi `n6_caption_qwen` — vLLM giu
    `gpu_memory_utilization=0.9` nen nap them BGE-M3 trong cung vong lap la OOM,
    con thao/nap model theo tung video thi ton hon ca phan viec no lam.

    Caption viet TIENG ANH (§4 CLAUDE.md) va BGE-M3 la mo hinh **da ngu**, nen
    truy van tieng Viet khop thang khong qua buoc dich nao.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    checkpoint = "BAAI/bge-m3"
    local = ctx.workdir / f"{_BGE_M3_SOURCE}_{ctx.video_id}.json"
    ctx.writer.transport.get(ctx.layout.part(_BGE_M3_SOURCE, ctx.video_id), local)
    captions = json.loads(local.read_text(encoding="utf-8"))
    if not captions:
        raise RuntimeError(
            f"{ctx.video_id}: chua co {_BGE_M3_SOURCE} — chay n6_caption_qwen truoc"
        )

    # `sorted` chu khong phai thu tu file: `write_vector_part` gan `ids` voi
    # `vectors` theo VI TRI, nen hai lan chay phai duyet cung mot thu tu.
    keys = sorted(captions)
    # `cached_model` chu KHONG phai `SentenceTransformer(...)` thang.
    #
    # Bge-m3 nang ~2,2 GB. Nap trong than vong lap la nap lai cho TUNG video —
    # 404 lan voi `trong`, 469 voi `quang` — trong khi phan viec that cua mot
    # video la nhung vai tram cau ngan. Do la dung cai bay
    # `tests/unit/offline/test_multipass_bundle.py::TestModelCache` do duoc o
    # cac tang khac: **5-16 gio moi nguoi chi de nap model**.
    #
    # Va no khong on ao: moi video van xong, marker van cam, chi la lau gap boi.
    model = cached_model(f"bge-m3 {checkpoint}", lambda: SentenceTransformer(checkpoint))
    with Stopwatch() as timer:
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
        params={"normalize": True, "source": _BGE_M3_SOURCE},
        elapsed_s=timer.seconds,
    )

def stage_n7_bge_m3_subtitles(ctx: StageContext) -> dict:
    """Subtitles -> BGE-M3 (Dense Vector)."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    checkpoint = "BAAI/bge-m3"
    local = ctx.workdir / f"subtitles_{ctx.video_id}.json"
    ctx.writer.transport.get(ctx.layout.part("subtitles", ctx.video_id), local)
    captions = json.loads(local.read_text(encoding="utf-8"))

    keys = sorted(captions)
    if not keys:
        vectors = np.zeros((0, 1024), dtype=np.float32)
        elapsed = 0.0
    else:
        model = cached_model(f"bge-m3 {checkpoint}", lambda: SentenceTransformer(checkpoint))
        with Stopwatch() as timer:
            vectors = model.encode(
                [captions[k] for k in keys],
                normalize_embeddings=True,
                batch_size=ctx.params.get("batch_size", 16),
                convert_to_numpy=True,
            )
        elapsed = timer.seconds

    return ctx.writer.write_vector_part(
        "subtitles_dense",
        ctx.video_id,
        keys,
        np.asarray(vectors, dtype=np.float32),
        model=checkpoint,
        params={"normalize": True, "source": "subtitles"},
        elapsed_s=elapsed,
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
    "n6_caption_qwen": stage_n6_caption_qwen,
    "n6_ocr_qwen25": stage_n6_ocr_qwen25,
    "n6_bge_m3": stage_n6_bge_m3,
    "n7_bge_m3_subtitles": stage_n7_bge_m3_subtitles,
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
