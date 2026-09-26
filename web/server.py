"""Async HTML front end for the existing multimodal-fusion-engine pipeline.

The Streamlit application remains intact.  This process owns public port 8501;
Streamlit is kept alive on the private loopback port 8502.

There are two execution lanes:

* ``shared`` has one queue and one Cascade.  Its state and results are
  broadcast to every authenticated WebSocket client.
* every account gets a private, single-worker executor and its own Cascade
  object.  Private state is sent only to that account.

The large FlatBundle and the read-only query models are shared to avoid loading
hundreds of megabytes repeatedly.  Mutable orchestration state, SQLite thread
connections and job executors are kept per lane.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

WEB_ROOT = Path("/home/ubuntu/web")
REPO_ROOT = Path("/home/ubuntu/multimodal-fusion-engine")
INDEX_PATH = WEB_ROOT / "index.html"
ACCOUNTS_PATH = WEB_ROOT / "accounts.json"
SESSION_SECRET_PATH = WEB_ROOT / ".session_secret"
SUGGESTIONS_PATH = WEB_ROOT / "suggestions.json"
DRES_SUBMISSIONS_PATH = WEB_ROOT / "dres_submissions.json"
RUNTIME_ROOT = Path("/data/multimodal-fusion-engine/runtime/btc_b12")
CONFIG_PATH = REPO_ROOT / "config" / "btc_b12.yaml"
FLAT_ROOT = RUNTIME_ROOT / "flat"
IMAGE_CACHE = RUNTIME_ROOT / "image-cache"
MAP_ROOT = RUNTIME_ROOT / "relational" / "map-keyframes"
B1_IMAGE_ROOT = Path(
    "/data/multimodal-fusion-engine/modal/keyframes-btc/bundle/aic_btc_b1_v1/keyframes"
)
B2_IMAGE_ROOT = Path(
    "/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2/bundle/aic_btc_b2_v1/keyframes"
)

sys.path.insert(0, str(REPO_ROOT / "src"))

from mfe.core.config import load_config  # noqa: E402
from mfe.eval.dres_client import (  # noqa: E402
    DresClient,
    DresError,
    apply_selected_frame,
    build_selected_payload,
    configured,
    submission_key,
)
from mfe.eval.official import to_event_hits  # noqa: E402
from mfe.mapping.relational import RelationalMap  # noqa: E402
from mfe.offline.transport import load_dotenv  # noqa: E402
from mfe.online import dante  # noqa: E402
from mfe.online.dense_arms import preload_query_encoders  # noqa: E402
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle  # noqa: E402
from mfe.online.flat_evidence import FlatTextStore  # noqa: E402
from mfe.online.gemini_caption_rerank import rerank_candidates  # noqa: E402
from mfe.online.keyframe_images import KeyframeImages  # noqa: E402
from mfe.online.quest import (  # noqa: E402
    QuestClient,
    QuestKIS,
    quest_clauses,
    quest_decompose,
    quest_expand,
    split_trake_events,
)
from mfe.online.video_source import resolve_video  # noqa: E402
from mfe.online.vqa_answer import read_answer_hybrid  # noqa: E402

dres_configured = configured

for _key, _value in load_dotenv(REPO_ROOT / ".env").items():
    os.environ.setdefault(_key, _value)


logging.basicConfig(
    level=os.getenv("MFE_WEB_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [mfe-web] %(message)s",
)
logger = logging.getLogger("mfe-web")


_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_VIDEO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")

# The browser only receives these three safe, named choices.  Keeping the
# mapping server-side prevents a client from selecting an arbitrary upstream
# model and makes QUEST/rerank cache keys unambiguous.
TRAMMAY_MODEL_TIERS = {
    "high": "ag/gemini-3.8-flash-high",
    "medium": "ag/gemini-3.8-flash-medium",
    "low": "ag/gemini-3.8-flash-low",
}


def _default_trammay_tier() -> str:
    current = os.getenv("MFE_TRAMMAY_MODEL", TRAMMAY_MODEL_TIERS["high"]).strip()
    return next(
        (tier for tier, model in TRAMMAY_MODEL_TIERS.items() if model == current),
        "high",
    )


def _resolve_trammay_tier(raw: Any) -> tuple[str, str]:
    tier = str(raw or _default_trammay_tier()).strip().lower()
    model = TRAMMAY_MODEL_TIERS.get(tier)
    if model is None:
        raise ValueError("trammay_tier phai la high, medium hoac low")
    return tier, model


def _tier_for_trammay_model(model: str | None) -> str:
    selected = str(model or "").strip()
    return next(
        (tier for tier, tier_model in TRAMMAY_MODEL_TIERS.items() if tier_model == selected),
        _default_trammay_tier(),
    )


def _safe_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _search_cache_key(
    *,
    query: str,
    query_en: str | None,
    top_k: int,
    query_type: str,
    events: list[str],
    events_en: list[str],
    answer: str,
    auto_answer: bool,
    penalty: float,
    ocr_keywords: str | None,
    arm_weights: dict[str, float] | None,
    arm_enabled: dict[str, bool] | None,
    gemini_rerank: bool,
    trammay_model: str | None,
) -> str:
    """Stable key for the RAM result cache.

    A result is only reusable when all inputs that can change ranking or the QA
    answer are identical. The process clears this cache on restart, so a new
    bundle or config can never reuse stale results.
    """
    payload = {
        "query": query,
        "query_en": query_en or "",
        "top_k": int(top_k),
        "query_type": query_type,
        "events": events,
        "events_en": events_en,
        "answer": answer,
        "auto_answer": bool(auto_answer),
        "penalty": float(penalty),
        "ocr_keywords": ocr_keywords or "",
        "arm_weights": arm_weights or {},
        "arm_enabled": arm_enabled or {},
        "gemini_rerank": bool(gemini_rerank),
        "trammay_model": trammay_model or "",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _gemini_rerank_default() -> bool:
    """Prefer the accuracy pass while allowing an operator to opt out."""
    return os.getenv("MFE_GEMINI_RERANK_DEFAULT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _visible_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def _physical_cpu_count(cpus: list[int]) -> int:
    pairs: set[tuple[str, str]] = set()
    for cpu in cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = base / "core_id"
        package = base / "physical_package_id"
        if core.is_file() and package.is_file():
            try:
                pairs.add((package.read_text().strip(), core.read_text().strip()))
            except OSError:
                pass
    return len(pairs) or max(1, len(cpus) // 2)


@dataclass(frozen=True)
class HardwareInfo:
    logical_cpus: int
    physical_cpus: int
    torch_threads: int
    torch_interop_threads: int
    max_parallel_lanes: int
    accelerator: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_cpus": self.logical_cpus,
            "physical_cpus": self.physical_cpus,
            "torch_threads": self.torch_threads,
            "torch_interop_threads": self.torch_interop_threads,
            "max_parallel_lanes": self.max_parallel_lanes,
            "accelerator": self.accelerator,
        }


def detect_hardware() -> HardwareInfo:
    cpus = _visible_cpus()
    logical = max(1, len(cpus))
    physical = _physical_cpu_count(cpus)
    # This host has 23 physical / 46 logical CPUs.  Eight independent lanes
    # keep the shared lane and several private accounts responsive while the
    # process-wide torch pool still bounds the actual numerical work.  The
    # environment variable remains available for a smaller machine or a
    # deliberate operator override.
    torch_threads = _safe_int_env("MFE_WEB_TORCH_THREADS", min(16, max(4, physical)))
    interop = _safe_int_env("MFE_WEB_TORCH_INTEROP_THREADS", min(4, max(1, physical // 8)))
    default_parallel = max(1, min(8, (physical + 2) // 3))
    parallel = _safe_int_env("MFE_WEB_MAX_PARALLEL_LANES", default_parallel)
    accelerator = "cpu"
    try:
        import torch

        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            # Another imported component may have already initialized the
            # inter-op pool.  The process still uses the configured intra-op
            # pool and remains healthy.
            pass
        if torch.cuda.is_available():
            accelerator = f"cuda:{torch.cuda.device_count()}"
    except Exception as exc:  # noqa: BLE001 - health still works without torch
        logger.warning("khong cau hinh duoc torch pool: %s", exc)
    return HardwareInfo(logical, physical, torch_threads, interop, parallel, accelerator)


def _read_session_secret() -> str:
    if not SESSION_SECRET_PATH.is_file():
        SESSION_SECRET_PATH.write_text(secrets.token_urlsafe(48) + "\n", encoding="utf-8")
        SESSION_SECRET_PATH.chmod(0o600)
    return SESSION_SECRET_PATH.read_text(encoding="utf-8").strip()


def _load_accounts() -> dict[str, dict[str, Any]]:
    if not ACCOUNTS_PATH.is_file():
        raise RuntimeError(f"Thieu {ACCOUNTS_PATH}; chay /home/ubuntu/web/create_accounts.py")
    payload = json.loads(ACCOUNTS_PATH.read_text(encoding="utf-8"))
    users = payload.get("users")
    if not isinstance(users, dict) or not users:
        raise RuntimeError("accounts.json khong co users hop le")
    return users


def _verify_password(password: str, record: dict[str, Any]) -> bool:
    try:
        salt = bytes.fromhex(str(record["salt"]))
        expected = bytes.fromhex(str(record["hash"]))
        n = int(record.get("n", 2**14))
        r = int(record.get("r", 8))
        p = int(record.get("p", 1))
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p)
    except (KeyError, TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, expected)


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


class Lane:
    """One serialized job lane backed by a dedicated OS thread."""

    def __init__(self, manager: EngineManager, name: str) -> None:
        self.manager = manager
        self.name = name
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", f"mfe-{name}")[:48]
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=prefix)
        self.cascade: Any = None
        self.active = False
        self.lock = asyncio.Lock()

    def _ensure_cascade(self) -> Any:
        if self.cascade is None:
            if self.name == "shared":
                self.cascade = self.manager.base_cascade
            else:
                self.cascade = self.manager.new_cascade()
        return self.cascade

    def run_sync(
        self,
        query: str,
        query_en: str | None,
        top_k: int,
        run_id: str,
        query_type: str = "KIS",
        events: list[str] | None = None,
        events_en: list[str] | None = None,
        answer: str = "",
        auto_answer: bool = False,
        penalty: float = dante.PENALTY,
        ocr_keywords: str | None = None,
        arm_weights: dict[str, float] | None = None,
        arm_enabled: dict[str, bool] | None = None,
        gemini_rerank: bool = True,
        trammay_model: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cache_key = _search_cache_key(
            query=query,
            query_en=query_en,
            top_k=top_k,
            query_type=query_type,
            events=events or [],
            events_en=events_en or [],
            answer=answer,
            auto_answer=auto_answer,
            penalty=penalty,
            ocr_keywords=ocr_keywords,
            arm_weights=arm_weights,
            arm_enabled=arm_enabled,
            gemini_rerank=gemini_rerank,
            trammay_model=trammay_model,
        )
        cached = self.manager.cached_query(cache_key)
        if cached is not None:
            cached["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
            cached["cache_hit"] = True
            logger.info("query cache hit (%s/%s)", self.name, query_type)
            return cached

        cascade = self._ensure_cascade()
        rerank_metrics: dict[str, Any] = {
            "enabled": bool(gemini_rerank),
            "ran": False,
            "cached": False,
            "moved": 0,
            "duration_ms": 0.0,
            "reason": None,
        }

        def search_one(text: str, text_en: str | None, search_run_id: str):
            with cascade.overriding(weights=arm_weights, enabled=arm_enabled):
                # QueryTranslator owns a small mutable usage/cache object. Keep
                # that object shared, but guard the complete search only when
                # an English arm actually needs translation.
                needs_translation = (
                    not text_en
                    and cascade.translator is not None
                    and cascade.arms.needs_english()
                )
                if needs_translation:
                    with self.manager.translation_lock:
                        return cascade.search(
                            query_text=text,
                            top_k=top_k,
                            run_id=search_run_id,
                            query_text_en=text_en,
                            ocr_keywords=ocr_keywords,
                        )
                return cascade.search(
                    query_text=text,
                    top_k=top_k,
                    run_id=search_run_id,
                    query_text_en=text_en,
                    ocr_keywords=ocr_keywords,
                )

        def candidates_of(raw: Any) -> list[Any]:
            return raw[0] if isinstance(raw, tuple) else raw

        def maybe_rerank(items: list[Any], text: str, text_en: str | None) -> list[Any]:
            if not gemini_rerank or self.manager.bundle is None:
                return items
            reordered, outcome = rerank_candidates(
                items,
                text,
                self.manager.bundle,
                query_en=text_en,
                depth=_safe_int_env("MFE_GEMINI_RERANK_DEPTH", 30),
                model=trammay_model,
            )
            rerank_metrics["ran"] = rerank_metrics["ran"] or outcome.ran
            rerank_metrics["cached"] = rerank_metrics["cached"] or outcome.cached
            rerank_metrics["moved"] += outcome.moved
            rerank_metrics["duration_ms"] = round(
                float(rerank_metrics["duration_ms"]) + outcome.duration_ms, 1
            )
            if outcome.reason and not rerank_metrics["reason"]:
                rerank_metrics["reason"] = outcome.reason
            return reordered

        if query_type == "TRAKE":
            event_texts = [text.strip() for text in (events or []) if text.strip()]
            if not event_texts:
                raise ValueError("TRAKE can it nhat mot su kien")
            per_event: list[list[Any]] = []
            for index, event_text in enumerate(event_texts):
                event_en = (
                    events_en[index].strip()
                    if events_en and index < len(events_en) and events_en[index].strip()
                    else None
                )
                raw = search_one(event_text, event_en, f"{run_id}-e{index}")
                per_event.append(
                    maybe_rerank(candidates_of(raw), event_text, event_en)[:top_k]
                )
            alignments = dante.align(
                [to_event_hits(candidates, self.manager.relational) for candidates in per_event],
                penalty=penalty,
                limit=top_k,
            )
            results = self.manager.alignment_payload(alignments)
            answer_payload = None
        else:
            raw = search_one(query, query_en, run_id)
            candidates = maybe_rerank(candidates_of(raw), query, query_en)
            results = self.manager.candidate_payload(candidates[:top_k])
            answer_payload = None
            if query_type == "QA":
                if answer.strip():
                    answer_payload = {
                        "text": answer.strip(),
                        "source": "manual",
                        "ok": True,
                    }
                elif auto_answer:
                    if self.manager.images is None:
                        answer_payload = {
                            "text": "",
                            "source": "vlm_failed",
                            "ok": False,
                            "reason": "image cache chua san sang",
                        }
                    else:
                        vqa = read_answer_hybrid(
                            query,
                            candidates,
                            self.manager.images.cached,
                            text_store=self.manager.text_store,
                            relational=self.manager.relational,
                            logic_question=query,
                            ocr_keywords=ocr_keywords or "",
                            model=trammay_model,
                        )
                        answer_payload = {
                            "text": vqa.text,
                            "frame": vqa.frame,
                            "confidence": vqa.confidence,
                            "why": vqa.why,
                            "source": "vlm" if vqa.ok else "vlm_failed",
                            "ok": vqa.ok,
                            "reason": vqa.reason,
                            "looked_at": vqa.looked_at,
                            "seconds": round(vqa.seconds, 3),
                        }
                else:
                    answer_payload = {
                        "text": "",
                        "source": "missing",
                        "ok": False,
                        "reason": "QA chua co dap an; nhap dap an hoac bat VLM",
                    }
        result = {
            "results": results,
            "query_type": query_type,
            "answer": answer_payload,
            "events": events or [],
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "cache_hit": False,
            "gemini_rerank": rerank_metrics,
            "trammay_model": trammay_model or TRAMMAY_MODEL_TIERS[_default_trammay_tier()],
            "trammay_tier": _tier_for_trammay_model(trammay_model),
        }
        self.manager.cache_query(cache_key, result)
        return result

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class EngineManager:
    def __init__(self) -> None:
        self.hardware = detect_hardware()
        self.config: Any = None
        self.bundle: Any = None
        self.base_cascade: Any = None
        self.relational: RelationalMap | None = None
        self.images: KeyframeImages | None = None
        self.text_store: FlatTextStore | None = None
        self.batch2_video_ids: set[str] = set()
        self.ready = False
        self.initializing = False
        self.init_error: str | None = None
        self.init_started = time.time()
        self.init_task: asyncio.Task[None] | None = None
        self.cpu_gate: asyncio.Semaphore | None = None
        self.query_cache_max = _safe_int_env("MFE_WEB_QUERY_CACHE_SIZE", 128)
        self.query_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.query_cache_lock = threading.RLock()
        self.translation_lock = threading.RLock()
        self.dres_lock = threading.RLock()
        self.dres_client: DresClient | None = None
        self.dres_submissions: set[str] = self._load_dres_submissions()
        self.suggestions_lock = threading.RLock()
        self.suggestions: list[dict[str, Any]] = self._load_suggestions()
        self.shared_lane: Lane | None = None
        self.private_lanes: dict[str, Lane] = {}
        self.clients: dict[WebSocket, str] = {}
        self.shared_state: dict[str, Any] = self._idle_state("shared", None)
        self.private_states: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _idle_state(scope: str, owner: str | None) -> dict[str, Any]:
        return {
            "scope": scope,
            "status": "idle",
            "owner": owner,
            "query": None,
            "query_type": "KIS",
            "events": [],
            "ocr_keywords": None,
            "arm_weights": {},
            "arm_enabled": {},
            "gemini_rerank": {
                "enabled": False,
                "ran": False,
                "cached": False,
                "moved": 0,
                "duration_ms": 0.0,
                "reason": None,
            },
            "answer": None,
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "cache_hit": False,
            "trammay_tier": _default_trammay_tier(),
            "trammay_model": TRAMMAY_MODEL_TIERS[_default_trammay_tier()],
            "error": None,
            "results": [],
        }

    def start(self) -> None:
        self.shared_lane = Lane(self, "shared")
        self.cpu_gate = asyncio.Semaphore(self.hardware.max_parallel_lanes)
        self.initializing = True
        self.init_task = asyncio.create_task(asyncio.to_thread(self._initialize_sync))
        self.init_task.add_done_callback(self._initialization_done)

    def _initialize_sync(self) -> None:
        logger.info(
            "hardware: %d logical / %d physical CPU, torch=%d threads, max_lanes=%d, %s",
            self.hardware.logical_cpus,
            self.hardware.physical_cpus,
            self.hardware.torch_threads,
            self.hardware.max_parallel_lanes,
            self.hardware.accelerator,
        )
        logger.info("dang nap config + FlatBundle B1+B2; cache mmap duoc giu lai")
        config = load_config(str(CONFIG_PATH))
        # Keep derived mmap/BM25 caches beside the combined runtime.  Reusing
        # the old default cache can silently serve a B1-only index after the
        # B1+B2 flat files have changed.
        flat_cache = RUNTIME_ROOT / "flat-cache"
        bundle = load_flat_bundle(
            str(FLAT_ROOT),
            cache_dir=flat_cache,
            map_keyframes=config.paths.map_keyframes,
        )
        cascade = build_flat_cascade(config, str(FLAT_ROOT), flat_bundle=bundle)
        active = [arm.name for arm in cascade.arms.active()]
        logger.info("preload query encoders: %s", active)
        preload_query_encoders(active)
        relational = RelationalMap(config.paths.relational_db)
        batch2_video_ids = {path.stem for path in B2_IMAGE_ROOT.iterdir() if path.is_dir()}
        image_roots = {
            video_id: (B2_IMAGE_ROOT if video_id in batch2_video_ids else B1_IMAGE_ROOT)
            for path in MAP_ROOT.glob("*.csv")
            for video_id in (path.stem,)
        }
        images = KeyframeImages(
            cache_dir=IMAGE_CACHE,
            maps=MAP_ROOT,
            bundle="aic_btc_b12_runtime",
            volume="keyframes-batch-2-v2",
            offline=True,
            image_roots=image_roots,
        )
        self.config = config
        self.bundle = bundle
        self.base_cascade = cascade
        self.relational = relational
        self.images = images
        self.text_store = FlatTextStore(bundle)
        self.batch2_video_ids = batch2_video_ids
        logger.info("engine san sang; active arms=%s; %s", active, bundle.report())

    def _initialization_done(self, task: asyncio.Task[None]) -> None:
        self.initializing = False
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - exposed through health endpoint
            self.init_error = f"{type(exc).__name__}: {exc}"
            logger.exception("engine initialization failed")
        else:
            self.ready = True
            logger.info("HTML backend ready on port 8501")
        asyncio.create_task(broadcast_health(self))

    def new_cascade(self) -> Any:
        if not self.ready and self.base_cascade is None:
            raise RuntimeError("engine dang khoi dong")
        return build_flat_cascade(
            self.config,
            str(FLAT_ROOT),
            flat_bundle=self.bundle,
            # The translator has mutable cache accounting; all lanes use the
            # lock in Lane.run_sync around the only section that can mutate it.
            translator=self.base_cascade.translator,
        )

    def private_lane(self, username: str) -> Lane:
        lane = self.private_lanes.get(username)
        if lane is None:
            lane = Lane(self, f"private-{username}")
            self.private_lanes[username] = lane
        return lane

    def state_for(self, username: str) -> dict[str, Any]:
        private = self.private_states.get(username)
        if private is None:
            private = self._idle_state("private", username)
            self.private_states[username] = private
        return {"shared": _copy_state(self.shared_state), "private": _copy_state(private)}

    def dres(self) -> DresClient | None:
        """Return the process-local DRES client without exposing its session."""
        with self.dres_lock:
            if self.dres_client is None:
                self.dres_client = DresClient.from_env()
            return self.dres_client

    @staticmethod
    def _load_dres_submissions() -> set[str]:
        if not DRES_SUBMISSIONS_PATH.is_file():
            return set()
        try:
            payload = json.loads(DRES_SUBMISSIONS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("khong doc duoc %s; bat dau khoa DRES rong", DRES_SUBMISSIONS_PATH)
            return set()
        if not isinstance(payload, list):
            return set()
        return {str(item) for item in payload if isinstance(item, str) and item}

    def _persist_dres_submissions(self) -> None:
        temporary = DRES_SUBMISSIONS_PATH.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(sorted(self.dres_submissions), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(DRES_SUBMISSIONS_PATH)
            DRES_SUBMISSIONS_PATH.chmod(0o600)
        except OSError:
            logger.warning("khong ghi duoc khoa DRES %s", DRES_SUBMISSIONS_PATH)

    def suggestions_snapshot(self) -> list[dict[str, Any]]:
        with self.suggestions_lock:
            return _copy_state(self.suggestions)

    def cached_query(self, key: str) -> dict[str, Any] | None:
        """Return a deep copy of a completed search held in RAM."""
        with self.query_cache_lock:
            value = self.query_cache.get(key)
            if value is None:
                return None
            self.query_cache.move_to_end(key)
            return _copy_state(value)

    def cache_query(self, key: str, value: dict[str, Any]) -> None:
        with self.query_cache_lock:
            self.query_cache[key] = _copy_state(value)
            self.query_cache.move_to_end(key)
            while len(self.query_cache) > self.query_cache_max:
                self.query_cache.popitem(last=False)

    def cache_stats(self) -> dict[str, int]:
        with self.query_cache_lock:
            return {"query_entries": len(self.query_cache), "query_capacity": self.query_cache_max}

    def add_suggestion(self, suggestion: dict[str, Any]) -> dict[str, Any]:
        with self.suggestions_lock:
            self.suggestions.insert(0, suggestion)
            del self.suggestions[200:]
            self._persist_suggestions()
            return _copy_state(suggestion)

    def clear_suggestions(self) -> None:
        with self.suggestions_lock:
            self.suggestions.clear()
            self._persist_suggestions()

    @staticmethod
    def _load_suggestions() -> list[dict[str, Any]]:
        if not SUGGESTIONS_PATH.is_file():
            return []
        try:
            payload = json.loads(SUGGESTIONS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("khong doc duoc %s; bat dau kho de xuat rong", SUGGESTIONS_PATH)
            return []
        return payload if isinstance(payload, list) else []

    def _persist_suggestions(self) -> None:
        temporary = SUGGESTIONS_PATH.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(self.suggestions, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(SUGGESTIONS_PATH)
            SUGGESTIONS_PATH.chmod(0o600)
        except OSError:
            logger.warning("khong ghi duoc kho de xuat %s", SUGGESTIONS_PATH)

    def _keyframe_payload(
        self,
        keyframe_id: str,
        video_id: str,
        score: float,
        rank: int,
    ) -> dict[str, Any] | None:
        if self.relational is None or self.images is None or self.bundle is None:
            return None
        try:
            row = self.relational.get_keyframe(keyframe_id)
            fps = float(self.relational.get_video(row.video_id).fps)
        except Exception as exc:  # noqa: BLE001 - one stale hit must not kill a job
            logger.warning("bo qua keyframe %s: %s", keyframe_id, exc)
            return None
        image = self.images.cached(keyframe_id)
        image_url = None
        resolved_video_id = row.video_id or video_id
        if image is not None and _VIDEO_RE.fullmatch(resolved_video_id):
            image_url = f"/media/{quote(resolved_video_id, safe='')}/{row.frame_idx}"
        video_url = None
        if os.getenv("MFE_VIDEO_BASE_URL", "").strip() and _VIDEO_RE.fullmatch(resolved_video_id):
            # The actual SAS URL stays server-side in `.env`; the browser follows
            # this authenticated redirect when the operator opens a result.
            video_url = f"/video/{quote(resolved_video_id, safe='')}"

        caption = ""
        for source in ("caption_gemini", "caption_qwen", "caption"):
            caption = str(self.bundle.texts.get(source, {}).get(keyframe_id, ""))
            if caption.strip():
                break
        ocr = ""
        for source in ("ocr_gemini", "ocr_qwen25", "ocr", "ocr_vietocr"):
            ocr = str(self.bundle.texts.get(source, {}).get(keyframe_id, ""))
            if ocr.strip():
                break
        return {
            "rank": rank,
            "keyframe_id": keyframe_id,
            "video_id": resolved_video_id,
            "batch": "batch2" if resolved_video_id in self.batch2_video_ids else "batch1",
            "frame_idx": row.frame_idx,
            "keyframe_index": row.keyframe_index,
            "timestamp_ms": row.timestamp_ms,
            "seconds": round(float(row.timestamp_ms) / 1000.0, 3),
            "fps": fps,
            "shot_id": row.shot_id,
            "score": round(float(score), 8),
            "caption": caption,
            "ocr": ocr,
            "image_url": image_url,
            "video_url": video_url,
        }

    def candidate_payload(self, candidates: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            item = self._keyframe_payload(
                candidate.keyframe_id, candidate.video_id, candidate.score, rank
            )
            if item is not None:
                out.append(item)
        return out

    def alignment_payload(self, alignments: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rank, alignment in enumerate(alignments, start=1):
            events: list[dict[str, Any]] = []
            for index, hit in enumerate(alignment.hits, start=1):
                item = self._keyframe_payload(hit.keyframe_id, hit.video_id, hit.score, index)
                if item is not None:
                    item["event_index"] = index
                    item["frame_id"] = hit.frame_id
                    events.append(item)
            out.append(
                {
                    "rank": rank,
                    "video_id": alignment.video_id,
                    "score": round(float(alignment.score), 8),
                    "support": alignment.support,
                    "frame_ids": [event["frame_idx"] for event in events],
                    "events": events,
                }
            )
        return out

    def health(self) -> dict[str, Any]:
        if self.ready:
            status = "ready"
        elif self.init_error:
            status = "error"
        else:
            status = "warming"
        return {
            "status": status,
            "ready": self.ready,
            "dataset": "btc_b12",
            "batches": {"batch1": "ready", "batch2": "ready"},
            "streamlit_preserved": True,
            "streamlit_internal_port": 8502,
            "hardware": self.hardware.as_dict(),
            "bundle": self.bundle.report() if self.bundle is not None else None,
            "active_arms": [arm.name for arm in self.base_cascade.arms.active()]
            if self.base_cascade is not None
            else [],
            "cache": self.cache_stats(),
            "video_source_configured": bool(os.getenv("MFE_VIDEO_BASE_URL", "").strip()),
            "error": self.init_error,
            "uptime_s": round(max(0.0, time.time() - self.init_started), 1),
        }

    def arm_config(self) -> list[dict[str, Any]]:
        """Return the operator-facing arm switches without exposing secrets."""
        if self.base_cascade is None:
            return []
        return [
            {
                "name": arm.name,
                "available": arm.available,
                "enabled": arm.enabled,
                "active": arm.active,
                "weight": arm.weight,
                "status": arm.status(),
                "description": arm.description,
                "query_language": arm.query_language,
                "unavailable_reason": arm.unavailable_reason,
            }
            for arm in self.base_cascade.arms
        ]

    def close(self) -> None:
        if self.init_task and not self.init_task.done():
            self.init_task.cancel()
        if self.shared_lane:
            self.shared_lane.close()
        for lane in self.private_lanes.values():
            lane.close()
        if self.relational:
            self.relational.close()


def _session_user(request: Request) -> str | None:
    username = request.session.get("user")
    return username if isinstance(username, str) else None


def _csrf_ok(request: Request) -> bool:
    expected = request.session.get("csrf")
    provided = request.headers.get("x-csrf-token")
    return (
        isinstance(expected, str)
        and isinstance(provided, str)
        and hmac.compare_digest(expected, provided)
    )


async def _json_body(request: Request) -> dict[str, Any] | None:
    try:
        payload = await request.json()
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


async def broadcast_state(manager: EngineManager, scope: str, owner: str | None) -> None:
    payload: dict[str, Any]
    if scope == "shared":
        payload = {"type": "state", "scope": "shared", "state": _copy_state(manager.shared_state)}
    else:
        if owner is None:
            return
        state = manager.private_states.get(owner)
        if state is None:
            return
        payload = {"type": "state", "scope": "private", "state": _copy_state(state)}

    stale: list[WebSocket] = []
    for websocket, username in list(manager.clients.items()):
        if scope == "shared" or username == owner:
            try:
                await websocket.send_json(payload)
            except Exception:  # noqa: BLE001 - disconnect cleanup is best effort
                stale.append(websocket)
    for websocket in stale:
        manager.clients.pop(websocket, None)


async def broadcast_suggestions(manager: EngineManager) -> None:
    payload = {"type": "suggestions", "suggestions": manager.suggestions_snapshot()}
    stale: list[WebSocket] = []
    for websocket in list(manager.clients):
        try:
            await websocket.send_json(payload)
        except Exception:  # noqa: BLE001
            stale.append(websocket)
    for websocket in stale:
        manager.clients.pop(websocket, None)


async def broadcast_health(manager: EngineManager) -> None:
    payload = {"type": "health", "health": manager.health()}
    stale: list[WebSocket] = []
    for websocket in list(manager.clients):
        try:
            await websocket.send_json(payload)
        except Exception:  # noqa: BLE001
            stale.append(websocket)
    for websocket in stale:
        manager.clients.pop(websocket, None)


async def _execute_job(
    manager: EngineManager,
    lane: Lane,
    scope: str,
    owner: str | None,
    query: str,
    query_en: str | None,
    top_k: int,
    run_id: str,
    query_type: str,
    events: list[str],
    events_en: list[str],
    answer: str,
    auto_answer: bool,
    penalty: float,
    ocr_keywords: str | None,
    arm_weights: dict[str, float],
    arm_enabled: dict[str, bool],
    gemini_rerank: bool,
    trammay_model: str,
) -> None:
    state = manager.shared_state if scope == "shared" else manager.private_states[owner or ""]
    state["status"] = "queued"
    await broadcast_state(manager, scope, owner)
    assert manager.cpu_gate is not None
    try:
        await manager.cpu_gate.acquire()
        state["status"] = "running"
        state["started_at"] = time.time()
        await broadcast_state(manager, scope, owner)
        loop = asyncio.get_running_loop()
        async with lane.lock:
            result = await loop.run_in_executor(
                lane.executor,
                lane.run_sync,
                query,
                query_en,
                top_k,
                run_id,
                query_type,
                events,
                events_en,
                answer,
                auto_answer,
                penalty,
                ocr_keywords,
                arm_weights,
                arm_enabled,
                gemini_rerank,
                trammay_model,
            )
        state["status"] = "done"
        state["results"] = result["results"]
        state["query_type"] = result["query_type"]
        state["events"] = result["events"]
        state["answer"] = result["answer"]
        state["duration_ms"] = result["duration_ms"]
        state["cache_hit"] = bool(result.get("cache_hit", False))
        state["trammay_tier"] = result.get("trammay_tier", _tier_for_trammay_model(trammay_model))
        state["trammay_model"] = result.get("trammay_model", trammay_model)
        state["gemini_rerank"] = result.get("gemini_rerank", state.get("gemini_rerank"))
        state["finished_at"] = time.time()
        state["error"] = None
    except Exception as exc:  # noqa: BLE001 - surface a useful job error to UI
        logger.exception("job failed (%s/%s)", scope, owner or "shared")
        state["status"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_at"] = time.time()
    finally:
        manager.cpu_gate.release()
        lane.active = False
        await broadcast_state(manager, scope, owner)


async def _schedule_job(
    manager: EngineManager,
    scope: str,
    owner: str | None,
    query: str,
    query_en: str | None,
    top_k: int,
    query_type: str,
    events: list[str],
    events_en: list[str],
    answer: str,
    auto_answer: bool,
    penalty: float,
    ocr_keywords: str | None,
    arm_weights: dict[str, float],
    arm_enabled: dict[str, bool],
    gemini_rerank: bool,
    trammay_model: str,
) -> JSONResponse:
    if not manager.ready:
        return JSONResponse(
            {"ok": False, "error": manager.init_error or "engine dang khoi dong"},
            status_code=503,
        )
    lane = manager.shared_lane if scope == "shared" else manager.private_lane(owner or "")
    if lane is None:
        return JSONResponse({"ok": False, "error": "shared lane chua san sang"}, status_code=503)
    if lane.active:
        return JSONResponse(
            {"ok": False, "error": "lane dang co mot job; hay cho job hien tai ket thuc"},
            status_code=409,
        )

    lane.active = True
    state = (
        manager.shared_state
        if scope == "shared"
        else manager.private_states.setdefault(owner or "", manager._idle_state("private", owner))
    )
    state.update(
        {
            "scope": scope,
            "status": "queued",
            "owner": owner,
            "query": query,
            "query_type": query_type,
            "events": events,
            "ocr_keywords": ocr_keywords,
            "arm_weights": arm_weights,
            "arm_enabled": arm_enabled,
            "trammay_tier": _tier_for_trammay_model(trammay_model),
            "trammay_model": trammay_model,
            "gemini_rerank": {
                "enabled": bool(gemini_rerank),
                "ran": False,
                "cached": False,
                "moved": 0,
                "duration_ms": 0.0,
                "reason": None,
            },
            "answer": None,
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "cache_hit": False,
            "error": None,
            "results": [],
        }
    )
    run_id = f"web_{scope}_{uuid.uuid4().hex}"
    asyncio.create_task(
        _execute_job(
            manager,
            lane,
            scope,
            owner,
            query,
            query_en,
            top_k,
            run_id,
            query_type,
            events,
            events_en,
            answer,
            auto_answer,
            penalty,
            ocr_keywords,
            arm_weights,
            arm_enabled,
            gemini_rerank,
            trammay_model,
        )
    )
    await broadcast_state(manager, scope, owner)
    return JSONResponse({"ok": True, "run_id": run_id, "scope": scope}, status_code=202)


async def homepage(request: Request) -> FileResponse:
    return FileResponse(
        INDEX_PATH,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


async def favicon(request: Request) -> PlainTextResponse:
    return PlainTextResponse("", status_code=204)


async def health(request: Request) -> JSONResponse:
    return JSONResponse(request.app.state.manager.health())


async def config(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    manager: EngineManager = request.app.state.manager
    return JSONResponse(
        {
            "ok": True,
            "arms": manager.arm_config(),
            "quest": {
                "provider": "llm-gateway.invalid / Gemini"
                if os.getenv("TRAMMAY_API_KEY")
                else "chua cau hinh Trammay",
                "model": TRAMMAY_MODEL_TIERS[_default_trammay_tier()],
                "tier": _default_trammay_tier(),
            },
            "gemini_rerank": {
                "available": bool(os.getenv("TRAMMAY_API_KEY", "").strip()),
                "model": TRAMMAY_MODEL_TIERS[_default_trammay_tier()],
                "tier": _default_trammay_tier(),
                "depth": _safe_int_env("MFE_GEMINI_RERANK_DEPTH", 30),
                "default_enabled": _gemini_rerank_default(),
            },
            "trammay": {
                "tiers": [
                    {"value": tier, "model": model}
                    for tier, model in TRAMMAY_MODEL_TIERS.items()
                ],
                "default_tier": _default_trammay_tier(),
                "default_model": TRAMMAY_MODEL_TIERS[_default_trammay_tier()],
            },
            "dres": {
                "configured": dres_configured(),
                "base_url": os.getenv("MFE_DRES_BASE_URL", "https://eventretrieval.one"),
                "evaluation_id": os.getenv("MFE_DRES_EVALUATION_ID", "").strip() or None,
            },
            "video": {
                "configured": bool(os.getenv("MFE_VIDEO_BASE_URL", "").strip()),
                "proxy": "/video/{video_id}",
            },
        }
    )


def _payload_lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def _trake_input(query: str, events: list[str]) -> tuple[str, list[str]]:
    """Return a preamble and ordered TRAKE events from either UI input form."""
    source = query.strip()
    preamble, parsed = split_trake_events(source)
    if parsed:
        return preamble, parsed
    return source, [item.strip() for item in events if item.strip()]


def _quest_analysis_sync(
    query_type: str, query: str, events: list[str], trammay_model: str
) -> dict[str, Any]:
    if query_type == "TRAKE":
        preamble, parsed_events = _trake_input(query, events)
        if parsed_events:
            return {
                "ok": True,
                "query_type": query_type,
                "provider": "backend parser",
                "query": query,
                "trake_preamble": preamble,
                "clauses": parsed_events,
                "parsed_format": True,
                "fallback": False,
            }
        source = "\n".join(
            ([preamble] if preamble else [])
            + [f"E{index}: {event}" for index, event in enumerate(parsed_events, 1)]
        )
    else:
        source = query.strip()

    client = QuestClient(model_override=trammay_model)
    available, reason = client.available()
    if not available:
        raise RuntimeError(reason or "QUEST Trammay chua san sang")
    provider = "llm-gateway.invalid / Gemini" if client.trammay_client is not None else "legacy"

    if query_type == "KIS":
        kis = QuestKIS()
        kis.client = client
        # Translation and implicit-term expansion are independent requests.
        # Run them together so KIS pays for the slower gateway response once,
        # rather than waiting for two serial Trammay round trips.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mfe-quest") as pool:
            translation_future = pool.submit(kis.translate, query)
            expansion_future = pool.submit(quest_expand, query, client)
            text_en = translation_future.result().strip()
            terms = expansion_future.result()
        expanded_vi = " ".join(term.vi for term in terms if term.vi).strip()
        expanded_en = " ".join(term.en for term in terms if term.en).strip()
        return {
            "ok": True,
            "query_type": query_type,
            "provider": provider,
            "query": query,
            "text_en": text_en,
            "expanded_vi": expanded_vi,
            "expanded_en": expanded_en,
            "expanded_query_vi": f"{query} {expanded_vi}".strip(),
            "expanded_query_en": f"{text_en} {expanded_en}".strip(),
            "terms": [
                {"vi": term.vi, "en": term.en, "why": term.why}
                for term in terms
            ],
        }

    if query_type == "QA":
        parts = quest_decompose(query, client=client)
        fallback = (
            str(parts.get("search_hint", "")).strip() == query
            and str(parts.get("logic_question", "")).strip() == query
        )
        return {
            "ok": True,
            "query_type": query_type,
            "provider": provider,
            "search_hint": str(parts.get("search_hint", "")).strip(),
            "ocr_keywords": str(parts.get("ocr_keywords", "")).strip(),
            "logic_question": str(parts.get("logic_question", "")).strip(),
            "fallback": fallback,
        }

    if query_type == "TRAKE" and not source:
        source = "\n".join(events)
    clauses = quest_clauses(source, client=client) if source else []
    return {
        "ok": True,
        "query_type": query_type,
        "provider": provider,
        "source": source,
        "clauses": clauses,
        "trake_preamble": "",
        "parsed_format": False,
        "fallback": not bool(clauses),
    }


async def quest(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    if not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    payload = await _json_body(request)
    query_type = str(payload.get("query_type", "KIS")).strip().upper() if payload else "KIS"
    if query_type not in {"KIS", "QA", "TRAKE"}:
        return JSONResponse({"ok": False, "error": "query_type khong hop le"}, status_code=400)
    query = str(payload.get("query", "")).strip() if payload else ""
    events = _payload_lines(payload.get("events", [])) if payload else []
    source = query or "\n".join(events)
    if not source:
        return JSONResponse({"ok": False, "error": "query khong duoc rong"}, status_code=400)
    if len(source) > 8000 or any(len(item) > 4000 for item in events):
        return JSONResponse({"ok": False, "error": "query qua dai"}, status_code=400)
    try:
        _, trammay_model = _resolve_trammay_tier(
            payload.get("trammay_tier") if payload else None
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        result = await asyncio.to_thread(
            _quest_analysis_sync, query_type, query, events, trammay_model
        )
    except Exception as exc:  # noqa: BLE001 - report gateway errors to the operator
        logger.warning("QUEST failed for %s: %s: %s", username, type(exc).__name__, exc)
        return JSONResponse(
            {"ok": False, "error": f"QUEST Trammay loi: {type(exc).__name__}: {exc}"},
            status_code=502,
        )
    return JSONResponse(result)


async def me(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"authenticated": False})
    return JSONResponse(
        {
            "authenticated": True,
            "username": username,
            "role": request.session.get("role", "operator"),
            "csrf": request.session.get("csrf"),
        }
    )


async def login(request: Request) -> JSONResponse:
    payload = await _json_body(request)
    username = str(payload.get("username", "")).strip() if payload else ""
    password = str(payload.get("password", "")) if payload else ""
    if not _USERNAME_RE.fullmatch(username):
        return JSONResponse(
            {"ok": False, "error": "tai khoan hoac mat khau khong dung"}, status_code=401
        )
    try:
        accounts = _load_accounts()
        record = accounts.get(username)
    except Exception as exc:  # noqa: BLE001
        logger.exception("cannot load account database")
        return JSONResponse(
            {"ok": False, "error": f"loi cau hinh tai khoan: {exc}"}, status_code=500
        )
    if not isinstance(record, dict) or not _verify_password(password, record):
        return JSONResponse(
            {"ok": False, "error": "tai khoan hoac mat khau khong dung"}, status_code=401
        )
    request.session.clear()
    request.session.update(
        {
            "user": username,
            "role": str(record.get("role", "operator")),
            "csrf": secrets.token_urlsafe(24),
        }
    )
    return JSONResponse(
        {
            "ok": True,
            "username": username,
            "role": str(record.get("role", "operator")),
            "csrf": request.session["csrf"],
        }
    )


async def logout(request: Request) -> JSONResponse:
    if _session_user(request) is not None and not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    request.session.clear()
    return JSONResponse({"ok": True})


async def state(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    return JSONResponse({"ok": True, **request.app.state.manager.state_for(username)})


def _dres_evaluation_view(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(item.get("id", "")),
            "name": str(item.get("name", "")),
            "type": str(item.get("type", "")),
            "status": str(item.get("status", "")),
        }
        for item in items
        if str(item.get("id", ""))
    ]


async def dres_evaluations(request: Request) -> JSONResponse:
    if _session_user(request) is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    manager: EngineManager = request.app.state.manager
    client = manager.dres()
    if client is None:
        return JSONResponse(
            {
                "ok": True,
                "configured": False,
                "evaluations": [],
                "error": "thieu user/pass DRES trong .env",
            }
        )
    try:
        with manager.dres_lock:
            evaluations = await asyncio.to_thread(client.evaluations)
    except DresError as exc:
        return JSONResponse({"ok": False, "configured": True, "error": str(exc)}, status_code=502)
    return JSONResponse(
        {
            "ok": True,
            "configured": True,
            "evaluations": _dres_evaluation_view(evaluations),
            "default": os.getenv("MFE_DRES_EVALUATION_ID", "").strip() or None,
        }
    )


def _manual_optional_int(payload: dict[str, Any], key: str, label: str) -> int | None:
    """Read a non-negative integer from the manual DRES form."""
    raw = payload.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise DresError(f"{label} phai la so nguyen khong am") from None
    if value < 0:
        raise DresError(f"{label} phai la so nguyen khong am")
    return value


def _manual_time_ms(
    payload: dict[str, Any], *, time_key: str, frame_key: str, label: str
) -> int:
    """Use an explicitly entered millisecond value or convert a frame index."""
    time_ms = _manual_optional_int(payload, time_key, f"{label} ms")
    if time_ms is not None:
        return time_ms
    frame = _manual_optional_int(payload, frame_key, f"{label} frame")
    if frame is None:
        raise DresError(f"can nhap {label} ms hoac {label} frame")
    try:
        fps = float(payload.get("fps", 25))
    except (TypeError, ValueError):
        raise DresError("FPS phai la so duong") from None
    if not math.isfinite(fps) or fps <= 0:
        raise DresError("FPS phai la so duong")
    return int(round(frame * 1000.0 / fps))


def _manual_frames(payload: dict[str, Any]) -> list[dict[str, int]]:
    """Parse TRAKE frame numbers pasted as lines, commas, or spaces."""
    raw = payload.get("frames", "")
    values = raw if isinstance(raw, list) else re.split(r"[,;\s]+", str(raw))
    frames: list[dict[str, int]] = []
    for index, value in enumerate(values, 1):
        if value is None or not str(value).strip():
            continue
        try:
            frame = int(value)
        except (TypeError, ValueError):
            raise DresError(f"TRAKE event {index} phai la frame so nguyen") from None
        if frame < 0:
            raise DresError("frame TRAKE phai khong am")
        frames.append({"event_index": len(frames) + 1, "frame_idx": frame})
    if not frames:
        raise DresError("TRAKE can it nhat mot frame")
    return frames


def _manual_submission(payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Build a DRES payload and its synthetic state from direct operator input."""
    kind = str(payload.get("query_type", "KIS")).strip().upper()
    if kind not in {"KIS", "QA", "TRAKE"}:
        raise DresError("query_type phai la KIS, QA hoac TRAKE")
    video = str(payload.get("video_id", "")).strip()
    if not video:
        raise DresError("video_id khong duoc rong")

    if kind == "KIS":
        start_ms = _manual_time_ms(
            payload, time_key="start_ms", frame_key="start_frame", label="start"
        )
        end_ms = start_ms
        if payload.get("end_ms") not in (None, "") or payload.get("end_frame") not in (None, ""):
            end_ms = _manual_time_ms(
                payload, time_key="end_ms", frame_key="end_frame", label="end"
            )
        item = {
            "video_id": video,
            "timestamp_ms": start_ms,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        events: list[str] = []
    elif kind == "QA":
        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise DresError("QA can nhap dap an")
        timestamp_ms = _manual_time_ms(
            payload, time_key="qa_time_ms", frame_key="qa_frame", label="QA"
        )
        item = {"video_id": video, "timestamp_ms": timestamp_ms}
        events = []
    else:
        trake_events = _manual_frames(payload)
        item = {"video_id": video, "events": trake_events}
        events = [str(event["frame_idx"]) for event in trake_events]

    submission = build_selected_payload(
        kind,
        item,
        answer=str(payload.get("answer", "")).strip(),
        events=item.get("events"),
    )
    state = {
        "query_type": kind,
        "query": f"manual:{kind}:{video}",
        "events": events,
    }
    return kind, state, submission


async def dres_submit_manual(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    if not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    payload = await _json_body(request)
    if not payload:
        return JSONResponse({"ok": False, "error": "payload rong"}, status_code=400)
    try:
        kind, state, submission = _manual_submission(payload)
    except DresError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    evaluation_id = str(payload.get("evaluation_id", "")).strip() or os.getenv(
        "MFE_DRES_EVALUATION_ID", ""
    ).strip()
    manager: EngineManager = request.app.state.manager
    client = manager.dres()
    if client is None:
        return JSONResponse(
            {"ok": False, "error": "thieu user/pass DRES trong .env"}, status_code=503
        )
    try:
        with manager.dres_lock:
            if not evaluation_id:
                evaluations = await asyncio.to_thread(client.evaluations)
                active = [
                    item
                    for item in evaluations
                    if str(item.get("status", "")).upper() == "ACTIVE"
                ]
                if len(active) != 1:
                    return JSONResponse(
                        {
                            "ok": False,
                            "error": "can chon mot evaluation ACTIVE",
                            "evaluations": _dres_evaluation_view(active),
                        },
                        status_code=409,
                    )
                evaluation_id = str(active[0]["id"])
            duplicate_key = submission_key(evaluation_id, state, submission)
            if duplicate_key in manager.dres_submissions:
                return JSONResponse(
                    {"ok": False, "error": "ket qua nay da bam nop"}, status_code=409
                )
            response = await asyncio.to_thread(client.submit, evaluation_id, submission)
            manager.dres_submissions.add(duplicate_key)
            manager._persist_dres_submissions()
    except DresError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    verdict = response.get("submission") if isinstance(response, dict) else None
    description = response.get("description") if isinstance(response, dict) else None
    return JSONResponse(
        {
            "ok": True,
            "manual": True,
            "evaluation_id": evaluation_id,
            "query_type": kind,
            "verdict": verdict,
            "description": description,
            "response": response
            if isinstance(response, (dict, list, str, int, float, bool))
            else {},
        }
    )


async def dres_submit(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    if not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    payload = await _json_body(request)
    if not payload:
        return JSONResponse({"ok": False, "error": "payload rong"}, status_code=400)
    manager: EngineManager = request.app.state.manager
    scope = str(payload.get("scope", "")).strip()
    if scope not in {"shared", "private"}:
        return JSONResponse({"ok": False, "error": "scope khong hop le"}, status_code=400)
    if scope == "shared":
        selected_state = manager.shared_state
    else:
        selected_state = manager.private_states.get(username)
    if not selected_state or selected_state.get("status") != "done":
        return JSONResponse({"ok": False, "error": "chua co ket qua de nop"}, status_code=409)
    try:
        result_index = int(payload.get("result_index", 0))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "result_index khong hop le"}, status_code=400)
    results = selected_state.get("results") or []
    if result_index < 0 or result_index >= len(results):
        return JSONResponse(
            {"ok": False, "error": "ket qua da chon khong ton tai"}, status_code=400
        )
    item = results[result_index]
    answer = str(payload.get("answer", "")).strip()
    if not answer and isinstance(selected_state.get("answer"), dict):
        answer = str(selected_state["answer"].get("text", "")).strip()
    try:
        selected_frame = payload.get("selected_frame")
        if selected_frame is not None:
            item = apply_selected_frame(item, selected_frame)
        selected_events = None
        if str(selected_state.get("query_type", "")).upper() == "TRAKE":
            raw_selected_events = payload.get("selected_events")
            if raw_selected_events is not None:
                if not isinstance(raw_selected_events, list):
                    raise DresError("selected_events phai la danh sach")
                by_index: dict[int, dict[str, Any]] = {}
                for selected_event in raw_selected_events:
                    if not isinstance(selected_event, dict):
                        raise DresError("selected_events co phan tu khong hop le")
                    try:
                        event_index = int(selected_event["event_index"])
                    except (KeyError, TypeError, ValueError):
                        raise DresError("selected_events thieu event_index") from None
                    by_index[event_index] = selected_event
                selected_events = []
                for event in item.get("events", []):
                    if not isinstance(event, dict):
                        continue
                    event_index = int(event.get("event_index", len(selected_events) + 1))
                    selected_events.append(
                        apply_selected_frame(event, by_index[event_index])
                        if event_index in by_index
                        else event
                    )
        submission = build_selected_payload(
            str(selected_state.get("query_type", "KIS")),
            item,
            answer=answer,
            events=selected_events,
        )
    except (DresError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    evaluation_id = str(payload.get("evaluation_id", "")).strip() or os.getenv(
        "MFE_DRES_EVALUATION_ID", ""
    ).strip()
    client = manager.dres()
    if client is None:
        return JSONResponse(
            {"ok": False, "error": "thieu user/pass DRES trong .env"}, status_code=503
        )
    try:
        with manager.dres_lock:
            if not evaluation_id:
                evaluations = await asyncio.to_thread(client.evaluations)
                active = [
                    item
                    for item in evaluations
                    if str(item.get("status", "")).upper() == "ACTIVE"
                ]
                if len(active) != 1:
                    return JSONResponse(
                        {
                            "ok": False,
                            "error": "can chon mot evaluation ACTIVE",
                            "evaluations": _dres_evaluation_view(active),
                        },
                        status_code=409,
                    )
                evaluation_id = str(active[0]["id"])
            duplicate_key = submission_key(evaluation_id, selected_state, submission)
            if duplicate_key in manager.dres_submissions:
                return JSONResponse(
                    {"ok": False, "error": "ket qua nay da bam nop"}, status_code=409
                )
            response = await asyncio.to_thread(client.submit, evaluation_id, submission)
            manager.dres_submissions.add(duplicate_key)
            manager._persist_dres_submissions()
    except DresError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    verdict = response.get("submission") if isinstance(response, dict) else None
    description = response.get("description") if isinstance(response, dict) else None
    return JSONResponse(
        {
            "ok": True,
            "evaluation_id": evaluation_id,
            "query_type": selected_state.get("query_type"),
            "result_rank": item.get("rank"),
            "verdict": verdict,
            "description": description,
            "response": response
            if isinstance(response, (dict, list, str, int, float, bool))
            else {},
        }
    )


def _suggestion_source(
    manager: EngineManager,
    username: str,
    scope: str,
    result_index: int,
    event_index: int | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    state = manager.shared_state if scope == "shared" else manager.private_states.get(username)
    if not state or not isinstance(state.get("results"), list):
        return None
    results = state["results"]
    if not 0 <= result_index < len(results):
        return None
    result = results[result_index]
    selected = result
    if event_index is not None and isinstance(result.get("events"), list):
        selected = next(
            (
                event
                for event in result["events"]
                if int(event.get("event_index", -1)) == event_index
            ),
            None,
        )
        if selected is None:
            return None
    return state, selected


def _suggestion_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the complete TRAKE chain in a shared suggestion safely."""
    fields = (
        "event_index",
        "video_id",
        "keyframe_id",
        "frame_id",
        "frame_idx",
        "timestamp_ms",
        "seconds",
        "fps",
        "video_url",
        "image_url",
    )
    events = result.get("events")
    if not isinstance(events, list):
        return []
    return [
        {field: event.get(field) for field in fields if field in event}
        for event in events
        if isinstance(event, dict)
    ]


async def suggestions(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    manager: EngineManager = request.app.state.manager
    if request.method == "GET":
        return JSONResponse({"ok": True, "suggestions": manager.suggestions_snapshot()})
    if not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    payload = await _json_body(request)
    if not payload:
        return JSONResponse({"ok": False, "error": "payload rong"}, status_code=400)
    scope = str(payload.get("scope", "")).strip()
    try:
        result_index = int(payload.get("result_index", 0))
        event_index = payload.get("event_index")
        event_index = int(event_index) if event_index is not None else None
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "index khong hop le"}, status_code=400)
    source = _suggestion_source(manager, username, scope, result_index, event_index)
    if source is None:
        return JSONResponse({"ok": False, "error": "khong tim thay ket qua"}, status_code=404)
    state, item = source
    try:
        selected_frame = payload.get("selected_frame")
        if selected_frame is not None:
            item = apply_selected_frame(item, selected_frame)
    except DresError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    note = str(payload.get("note", "")).strip()
    if not note or len(note) > 2000:
        return JSONResponse(
            {"ok": False, "error": "ghi chu phai tu 1 den 2000 ky tu"}, status_code=400
        )
    suggestion = {
        "id": uuid.uuid4().hex,
        "created_at": time.time(),
        "user": username,
        "scope": scope,
        "query_type": state.get("query_type"),
        "query": state.get("query"),
        "note": note,
        "result_rank": item.get("rank"),
        "video_id": item.get("video_id"),
        "keyframe_id": item.get("keyframe_id"),
        "frame_idx": item.get("frame_idx"),
        "timestamp_ms": item.get("timestamp_ms"),
        "video_url": item.get("video_url"),
        "score": item.get("score"),
        "event_index": event_index,
    }
    if event_index is None and str(state.get("query_type", "")).upper() == "TRAKE":
        suggestion["events"] = _suggestion_events(item)
    manager.add_suggestion(suggestion)
    await broadcast_suggestions(manager)
    return JSONResponse({"ok": True, "suggestion": suggestion}, status_code=201)


async def clear_suggestions(request: Request) -> JSONResponse:
    if _session_user(request) is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    if not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    manager: EngineManager = request.app.state.manager
    manager.clear_suggestions()
    await broadcast_suggestions(manager)
    return JSONResponse({"ok": True})


async def run_shared(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    if not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    return await _run_request(request, "shared", username)


async def run_private(request: Request) -> JSONResponse:
    username = _session_user(request)
    if username is None:
        return JSONResponse({"ok": False, "error": "can dang nhap"}, status_code=401)
    if not _csrf_ok(request):
        return JSONResponse({"ok": False, "error": "csrf token khong dung"}, status_code=403)
    return await _run_request(request, "private", username)


def _parse_arm_config(
    manager: EngineManager, payload: dict[str, Any] | None
) -> tuple[dict[str, float], dict[str, bool]]:
    """Validate per-job arm settings before a lane mutates its Cascade."""
    known = {arm.name for arm in (manager.base_cascade.arms if manager.base_cascade else [])}
    weights_raw = payload.get("arm_weights", {}) if payload else {}
    enabled_raw = payload.get("arm_enabled", {}) if payload else {}
    if not isinstance(weights_raw, dict) or not isinstance(enabled_raw, dict):
        raise TypeError("arm_weights va arm_enabled phai la object")
    unknown = (set(weights_raw) | set(enabled_raw)) - known
    if unknown:
        raise ValueError(f"arm khong ton tai: {', '.join(sorted(map(str, unknown)))}")

    weights: dict[str, float] = {}
    for name, raw in weights_raw.items():
        if isinstance(raw, bool):
            raise TypeError(f"trong so arm {name} khong hop le")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise TypeError(f"trong so arm {name} khong hop le") from None
        if not math.isfinite(value) or value < 0.0 or value > 4.0:
            raise TypeError(f"trong so arm {name} phai nam trong [0, 4]")
        weights[str(name)] = value

    enabled: dict[str, bool] = {}
    for name, raw in enabled_raw.items():
        if not isinstance(raw, bool):
            raise TypeError(f"enabled cua arm {name} phai la boolean")
        enabled[str(name)] = raw
    return weights, enabled


async def _run_request(request: Request, scope: str, username: str) -> JSONResponse:
    payload = await _json_body(request)
    query_type = str(payload.get("query_type", "KIS")).strip().upper() if payload else "KIS"
    if query_type not in {"KIS", "QA", "TRAKE"}:
        return JSONResponse(
            {"ok": False, "error": "query_type phai la KIS, QA hoac TRAKE"}, status_code=400
        )
    query = str(payload.get("query", "")).strip() if payload else ""
    query_en = str(payload.get("query_en", "")).strip() if payload else ""
    events_raw = payload.get("events", []) if payload else []
    events = (
        [str(item).strip() for item in events_raw if str(item).strip()]
        if isinstance(events_raw, list)
        else [line.strip() for line in str(events_raw).splitlines() if line.strip()]
    )
    events_en_raw = payload.get("events_en", []) if payload else []
    events_en = (
        [str(item).strip() for item in events_en_raw if str(item).strip()]
        if isinstance(events_en_raw, list)
        else [line.strip() for line in str(events_en_raw).splitlines() if line.strip()]
    )
    trake_preamble = str(payload.get("trake_preamble", "")).strip() if payload else ""
    if query_type == "TRAKE":
        # When QUEST fields were edited, the explicit `events` list is newer
        # than the original pasted query. Otherwise parse the E<n> lines from
        # the query itself so a full organiser-style paste also works.
        if not (trake_preamble and events):
            parsed_preamble, parsed_events = _trake_input(query, events)
            if parsed_events:
                trake_preamble = trake_preamble or parsed_preamble
                events = parsed_events
        if trake_preamble:
            events = [f"{trake_preamble} {event}".strip() for event in events]
            query = trake_preamble
    answer = str(payload.get("answer", "")).strip() if payload else ""
    auto_answer = bool(payload.get("auto_answer", False)) if payload else False
    gemini_rerank = (
        bool(payload.get("gemini_rerank", _gemini_rerank_default()))
        if payload
        else _gemini_rerank_default()
    )
    try:
        _, trammay_model = _resolve_trammay_tier(payload.get("trammay_tier") if payload else None)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    ocr_keywords = str(payload.get("ocr_keywords", "")).strip() if payload else ""
    try:
        arm_weights, arm_enabled = _parse_arm_config(request.app.state.manager, payload)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        top_k = int(payload.get("top_k", 50)) if payload else 50
    except (TypeError, ValueError):
        top_k = 50
    if query_type == "TRAKE" and not events:
        return JSONResponse(
            {
                "ok": False,
                "error": "TRAKE can it nhat mot su kien; dung dang E1: ..., E2: ...",
            },
            status_code=400,
        )
    if query_type != "TRAKE" and not query:
        return JSONResponse({"ok": False, "error": "query khong duoc rong"}, status_code=400)
    if query_type == "TRAKE" and len(events_en) not in {0, len(events)}:
        return JSONResponse(
            {"ok": False, "error": "events_en phai co cung so dong voi events"}, status_code=400
        )
    if (
        len(query) > 4000
        or len(query_en) > 4000
        or len(ocr_keywords) > 4000
        or any(len(item) > 4000 for item in events + events_en)
    ):
        return JSONResponse({"ok": False, "error": "query qua dai"}, status_code=400)
    if len(answer) > 1000:
        return JSONResponse({"ok": False, "error": "dap an QA qua dai"}, status_code=400)
    top_k = max(1, min(100, top_k))
    return await _schedule_job(
        request.app.state.manager,
        scope,
        username,
        query,
        query_en or None,
        top_k,
        query_type,
        events,
        events_en,
        answer,
        auto_answer if query_type == "QA" else False,
        dante.PENALTY,
        ocr_keywords or None,
        arm_weights,
        arm_enabled,
        gemini_rerank,
        trammay_model,
    )


async def media(request: Request) -> FileResponse | JSONResponse:
    if _session_user(request) is None:
        return JSONResponse({"error": "can dang nhap"}, status_code=401)
    video_id = request.path_params["video_id"]
    frame_raw = request.path_params["frame_idx"]
    if not _VIDEO_RE.fullmatch(video_id) or not str(frame_raw).isdigit():
        return JSONResponse({"error": "invalid media path"}, status_code=404)
    manager: EngineManager = request.app.state.manager
    if manager.images is None:
        return JSONResponse({"error": "engine warming"}, status_code=503)
    keyframe_id = f"{video_id}#{int(frame_raw):06d}"
    path = manager.images.cached(keyframe_id)
    if path is None or not path.is_file():
        return JSONResponse({"error": "image not found"}, status_code=404)
    if not manager.images.is_allowed_local_path(path):
        return JSONResponse({"error": "invalid image path"}, status_code=404)
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


async def video(request: Request) -> RedirectResponse | JSONResponse:
    """Authenticate a result click, then redirect the browser to Azure Blob."""
    if _session_user(request) is None:
        return JSONResponse({"error": "can dang nhap"}, status_code=401)
    video_id = request.path_params["video_id"]
    if not _VIDEO_RE.fullmatch(video_id):
        return JSONResponse({"error": "invalid video path"}, status_code=404)
    source = resolve_video(video_id)
    if source is None:
        return JSONResponse({"error": "video source not configured"}, status_code=404)
    if not source.is_url:
        return JSONResponse(
            {"error": "video source is local; configure MFE_VIDEO_BASE_URL for browser playback"},
            status_code=503,
        )
    return RedirectResponse(
        source.location,
        status_code=307,
        headers={"Cache-Control": "private, no-store"},
    )


async def websocket_endpoint(websocket: WebSocket) -> None:
    username = websocket.session.get("user")
    if not isinstance(username, str):
        await websocket.close(code=4401)
        return
    manager: EngineManager = websocket.app.state.manager
    await websocket.accept()
    manager.clients[websocket] = username
    await websocket.send_json({"type": "health", "health": manager.health()})
    snapshot = manager.state_for(username)
    await websocket.send_json({"type": "snapshot", **snapshot})
    await websocket.send_json(
        {"type": "suggestions", "suggestions": manager.suggestions_snapshot()}
    )
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("websocket closed", exc_info=True)
    finally:
        manager.clients.pop(websocket, None)


@asynccontextmanager
async def lifespan(app: Starlette):
    manager = EngineManager()
    app.state.manager = manager
    manager.start()
    try:
        yield
    finally:
        manager.close()


routes = [
    Route("/", homepage),
    Route("/favicon.ico", favicon),
    Route("/api/health", health),
    Route("/api/config", config),
    Route("/api/quest", quest, methods=["POST"]),
    Route("/api/me", me),
    Route("/api/login", login, methods=["POST"]),
    Route("/api/logout", logout, methods=["POST"]),
    Route("/api/state", state),
    Route("/api/dres/evaluations", dres_evaluations),
    Route("/api/dres/submit-manual", dres_submit_manual, methods=["POST"]),
    Route("/api/dres/submit", dres_submit, methods=["POST"]),
    Route("/api/suggestions", suggestions, methods=["GET", "POST"]),
    Route("/api/suggestions/clear", clear_suggestions, methods=["POST"]),
    Route("/api/shared/run", run_shared, methods=["POST"]),
    Route("/api/private/run", run_private, methods=["POST"]),
    Route("/media/{video_id}/{frame_idx:int}", media),
    Route("/video/{video_id}", video),
    WebSocketRoute("/ws", websocket_endpoint),
]

app = Starlette(
    debug=False,
    routes=routes,
    middleware=[
        Middleware(
            SessionMiddleware,
            secret_key=_read_session_secret(),
            session_cookie="mfe_session",
            max_age=86400,
            same_site="lax",
            https_only=False,
        )
    ],
    lifespan=lifespan,
)
