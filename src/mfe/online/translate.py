"""VI→EN query translation for arms whose model has no Vietnamese.

    text_en điền tay  →  cache trên đĩa  →  MT chạy máy  →  [sau này] LLM viết lại

`siglip` is the arm with the largest marginal contribution on the pilot set
(mean rank 2.7 → 4.1 when it is removed), and **all** of that value sits behind
this step: the same arm scores 9/9 at rank 2.0 on hand-written English and 7/9
at rank 16.7 on the untranslated Vietnamese. Until now the English came from a
hand-filled `text_en` on each query, which does not scale to 115 queries × 5
people. This module is what replaces the hand.

Three constraints, none of them negotiable (docs/triet/handoff.md §5.1):

**1. Hand-written wins.** A `text_en` on the query short-circuits every tier.
The test set has to stay reproducible, and whoever reads a bad translation must
be able to override it by editing one line.

**2. The cache is keyed by `(text, tier, model)` and lives on disk.** Re-running
an eval must be free and deterministic — §3.6 of the merge plan requires it, and
it is precisely where Bảo's chain leaks: two runs on different days produce two
score tables that cannot be compared. `envit5` decodes greedily and is already
deterministic, so for the MT tier the cache is only a speed-up; for an LLM tier
it *is* the determinism, which is why the cache file is tracked in git rather
than parked under `stores/` with the other build artifacts.

**3. There is no "pass the Vietnamese through" tier, and there cannot be one.**
`Tier` is a closed set of `mt` and `llm`. When every tier fails, `translate`
returns None, and `Arm.query_language == "en"` makes the arm sit out with a
logged reason. Feeding Vietnamese to an English text tower does not raise — it
returns rubbish, and rubbish still votes in fusion: a noisy arm measured on this
same pilot set dragged `kis_02` from rank 1 to rank 15. An arm that abstains
costs one arm; an arm that guesses costs the query.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

#: Tiers that can *produce* a translation, cheapest and most deterministic
#: first. `manual` is not listed: it is not a tier but the caller's own input,
#: and it is consulted before any of these regardless of configuration.
Tier = Literal["mt", "llm"]

#: Marks a translation that came from the query file rather than a model.
MANUAL_TIER = "manual"

#: The default machine-translation checkpoint: Vietnamese↔English T5, runs on
#: CPU, no API key, no network after the first download.
DEFAULT_MT_CHECKPOINT = "VietAI/envit5-translation"


@dataclass(frozen=True)
class Translation:
    """One English form of a query, and where it came from.

    `tier` and `model` travel with the text because a score is only comparable
    to another score produced the same way — the same reason `RunProvenance`
    records which arms ran (`mfe.eval.official`).
    """

    text: str
    text_en: str
    tier: str
    model: str

    @property
    def is_manual(self) -> bool:
        return self.tier == MANUAL_TIER


class TranslationBackend(Protocol):
    """Anything that turns Vietnamese into English."""

    #: Identifies *what produced the text*, including any decode setting that
    #: changes the output. It is part of the cache key, so two settings can
    #: never share an entry.
    model_id: str

    def translate(self, text: str) -> str: ...


def _normalize(text: str) -> str:
    """The form a cache key is built from.

    NFC for the same reason every other boundary in this project normalizes
    (`StorageRecord.validate_text`): the identical-looking decomposed string is a
    different dict key, so a Unikey "Unicode tổ hợp" query would miss a cache
    entry written from a precomposed one and quietly pay for a second inference.
    """
    return unicodedata.normalize("NFC", text).strip()


class TranslationCache:
    """Disk-backed `(text, tier, model) -> text_en`, written as sorted JSON.

    Sorted and re-serialized in full on every save so the file is stable under
    git: an eval run that adds one query produces a one-line diff, not a
    reshuffled file. Entries are records rather than a flat `key -> value` map
    so the file stays readable by whoever has to check a suspect translation.
    """

    VERSION = 1

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: dict[tuple[str, str, str], str] = {}
        self._dirty = False
        if self.path is not None and self.path.is_file():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for entry in payload.get("entries", []):
                key = (_normalize(entry["text"]), entry["tier"], entry["model"])
                self._entries[key] = entry["text_en"]
        except (OSError, ValueError, KeyError) as err:
            # A corrupt cache must not take an eval run down with it — the
            # entries are recoverable by re-translating, the run is not.
            logger.warning("translation cache %s unreadable (%s); starting empty", self.path, err)
            self._entries = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, text: str, tier: str, model: str) -> str | None:
        return self._entries.get((_normalize(text), tier, model))

    def put(self, text: str, tier: str, model: str, text_en: str) -> None:
        key = (_normalize(text), tier, model)
        if self._entries.get(key) == text_en:
            return
        self._entries[key] = text_en
        self._dirty = True

    def save(self) -> None:
        """Write the cache back, atomically. No-op without a path or a change."""
        if self.path is None or not self._dirty:
            return
        payload = {
            "version": self.VERSION,
            "entries": [
                {"text": text, "tier": tier, "model": model, "text_en": self._entries[key]}
                for key in sorted(self._entries)
                for text, tier, model in [key]
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-replace: a run interrupted mid-write must not leave a
        # truncated cache behind, because the next run would silently re-pay for
        # every entry it lost.
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False, suffix=".tmp"
        )
        try:
            with handle as out:
                json.dump(payload, out, ensure_ascii=False, indent=2, sort_keys=False)
                out.write("\n")
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        self._dirty = False


class MachineTranslator:
    """`VietAI/envit5-translation` on the local machine. Lazy, deterministic.

    Greedy decoding by default (`num_beams=1`, no sampling) because a query set
    scored twice must produce the same two numbers. The beam count is part of
    `model_id` for the same reason — changing it changes the output, so it must
    invalidate the cache rather than silently mix two decoders' results in one
    score table.

    The checkpoint is ~1.2 GB and loads on first use only, so a run whose
    English arms are all off never pays for it.

    **Chạy trên GPU khi máy có GPU** (25-09-2026). Đo được **560 → 268 ms**
    (2,1×), và đó không phải con số trang trí: sau khi tầng văn bản chuyển sang
    ma trận thưa, bước dịch trở thành **khoản lớn nhất** của một truy vấn.

    ⚠ Và nó chỉ hiện ra ở chung kết. `measure_latency.py` báo *"DỊCH 0 ms"* vì
    `eval/translations.json` đã cache sẵn mọi câu dev; mỗi câu BTC thả xuống là
    một lần **trượt cache**.

    ⚠ **`device` KHÔNG vào `model_id`, và đó là chủ ý.** Ràng buộc 2 ở đầu
    module nói cache phải xác định; nếu CPU và GPU cho hai bản dịch khác nhau
    thì cùng một khoá cache sẽ trỏ vào hai kết quả. Đã kiểm 25-09 trên 6 câu:
    **6/6 trùng khít từng ký tự** — giải mã greedy không phụ thuộc thiết bị ở
    độ dài này. Nên đưa `device` vào khoá chỉ làm mọi bản dịch đã cache mất
    hiệu lực mà không mua được tính đúng đắn nào. Nếu một ngày thấy hai thiết bị
    ra khác nhau thì **thêm `device` vào `model_id`**, đừng bỏ cache.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_MT_CHECKPOINT,
        num_beams: int = 1,
        max_new_tokens: int = 128,
        device: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        #: None = tự dò. `MFE_MT_DEVICE=cpu` để ép về CPU khi cần so hai lượt.
        self.device = device or os.getenv("MFE_MT_DEVICE") or None
        self._model = None
        self._tokenizer = None

    @property
    def model_id(self) -> str:
        return f"{self.checkpoint}|beams={self.num_beams}|max_new={self.max_new_tokens}"

    def _chon_thiet_bi(self) -> str:
        """`cuda` nếu có, `cpu` nếu không. Hỏng ở đây phải rơi về CPU, không ném.

        Một máy có driver lệch bản vẫn phải dịch được — mất 300 ms còn hơn mất
        arm `siglip` (trọng số 2,0, bỏ nó là −0,093 Final).
        """
        if self.device:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as err:  # noqa: BLE001 — thiếu torch/driver lệch đều về CPU
            logger.info("khong do duoc GPU (%s); dich tren CPU", err)
            return "cpu"

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info("Loading MT checkpoint %s (first call downloads ~1.2 GB)", self.checkpoint)
            self._tokenizer = self._load_tokenizer(AutoTokenizer)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
            model.eval()
            thiet_bi = self._chon_thiet_bi()
            if thiet_bi != "cpu":
                try:
                    model = model.to(thiet_bi)
                except Exception as err:  # noqa: BLE001
                    logger.warning("khong dua model len %s (%s); dung CPU", thiet_bi, err)
                    thiet_bi = "cpu"
            self.device = thiet_bi
            logger.info("MT chay tren %s", thiet_bi)
            self._model = model
        return self._tokenizer, self._model

    def _load_tokenizer(self, auto_tokenizer):
        """`AutoTokenizer`, falling back to reading `tokenizer.json` directly.

        envit5 needs the fallback, and the reason is worth writing down because
        the error names neither the model nor the cause: the checkpoint is a T5
        *architecture* carrying a **BPE** tokenizer, so `AutoTokenizer` routes it
        through T5's sentencepiece→Unigram converter and dies on
        `argument 'vocab': 'dict' object cannot be converted to 'Sequence'`
        (transformers 5.14.1 / tokenizers 0.22.2). The repo ships a perfectly
        good `tokenizer.json`; loading it as a plain fast tokenizer skips the
        conversion entirely.

        Ordered this way round so every other checkpoint keeps its normal path.
        """
        try:
            return auto_tokenizer.from_pretrained(self.checkpoint)
        except Exception as err:  # noqa: BLE001 - moi loi deu roi ve tokenizer.json
            from transformers import PreTrainedTokenizerFast

            logger.info(
                "AutoTokenizer could not build %s (%s); reading tokenizer.json directly",
                self.checkpoint,
                err,
            )
            return PreTrainedTokenizerFast.from_pretrained(self.checkpoint)

    def translate(self, text: str) -> str:
        """Vietnamese in, English out. Raises if the model cannot be loaded.

        envit5 is trained with an explicit language prefix and answers with one:
        the input must be `"vi: …"` and the output arrives as `"en: …"`. Both are
        handled here, at the one place that knows this checkpoint's convention —
        leaking `"en: "` into a query would put a literal token in front of every
        SigLIP search.
        """
        tokenizer, model = self._load()
        import torch

        batch = tokenizer([f"vi: {text}"], return_tensors="pt", padding=True)
        # Tensor phải đi theo model: `_load` có thể đã đẩy model lên CUDA, và
        # một batch còn nằm trên CPU thì `generate` ném "Expected all tensors to
        # be on the same device" chứ không tự chuyển.
        if self.device and self.device != "cpu":
            batch = batch.to(self.device)
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_language_prefix(decoded)


def _strip_language_prefix(text: str) -> str:
    """Drop envit5's `"en: "` answer prefix, tolerating its absence."""
    stripped = text.strip()
    for prefix in ("en:", "En:", "EN:"):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


class QueryTranslator:
    """The chain: hand-written → cache → each configured tier in order.

    Returns None when nothing produced English, which is a supported outcome and
    not an error: the caller leaves `query_text_en` unset and every
    `query_language="en"` arm sits out with a reason (`ArmRegistry.run`).
    """

    def __init__(
        self,
        backends: dict[str, TranslationBackend],
        tiers: list[str],
        cache: TranslationCache | None = None,
    ) -> None:
        missing = [tier for tier in tiers if tier not in backends]
        if missing:
            # Loudly, at construction. A tier named in config with nothing behind
            # it would otherwise degrade to "no translation" at query time, which
            # looks exactly like a query nobody wrote English for.
            raise ValueError(
                f"translation tier(s) {missing} have no backend; "
                f"configured backends: {sorted(backends)}"
            )
        self.backends = backends
        self.tiers = list(tiers)
        self.cache = cache if cache is not None else TranslationCache(None)
        #: How each answer was obtained this run, e.g. `{"manual": 7, "mt": 2}`,
        #: with `"none"` counting the queries left untranslated. Recorded into
        #: the eval report: a score built on hand-written English and one built
        #: on machine English are two different measurements, and the difference
        #: has to be visible in the file rather than remembered.
        self.usage: dict[str, int] = {}

    def _count(self, tier: str) -> None:
        self.usage[tier] = self.usage.get(tier, 0) + 1

    def translate(self, text: str, manual: str | None = None) -> Translation | None:
        if manual and manual.strip():
            self._count(MANUAL_TIER)
            return Translation(
                text=text, text_en=manual.strip(), tier=MANUAL_TIER, model=""
            )

        for tier in self.tiers:
            backend = self.backends[tier]
            model_id = backend.model_id
            cached = self.cache.get(text, tier, model_id)
            if cached is not None:
                self._count(tier)
                return Translation(text=text, text_en=cached, tier=tier, model=model_id)

            try:
                text_en = backend.translate(text).strip()
            except Exception as err:  # noqa: BLE001 — try the next tier, then abstain
                # Ghi ca KIEU loi, khong chi thong diep. Ngay 21-08-2026 tren
                # Kaggle no in ra "translation tier mt failed (0)" — mot chuoi
                # "0" khong noi duoc gi, va phai them mot vong len Kaggle nua
                # moi biet do la KeyError hay IndexError hay gi khac. Mot dong
                # log ve loi ma khong dinh danh duoc loi thi bang khong co.
                logger.warning(
                    "translation tier %s failed (%s: %s); trying the next",
                    tier, type(err).__name__, err,
                )
                continue
            if not text_en:
                logger.warning("translation tier %s returned an empty string", tier)
                continue

            self.cache.put(text, tier, model_id, text_en)
            self._count(tier)
            logger.info("translated via %s: %r -> %r", tier, text, text_en)
            return Translation(text=text, text_en=text_en, tier=tier, model=model_id)

        self._count("none")
        logger.warning(
            "no English form for %r (tiers tried: %s). English-only arms will sit out — "
            "this is deliberate; the Vietnamese is NOT passed through.",
            text,
            ", ".join(self.tiers) or "none",
        )
        return None

    def save_cache(self) -> None:
        self.cache.save()


def build_translator(
    config,
    extra_backends: dict[str, TranslationBackend] | None = None,
    repo_root: str | Path | None = None,
) -> QueryTranslator | None:
    """Build the translator named in `retrieval.translation`, or None if off.

    `extra_backends` is how the LLM tier arrives without this module importing an
    API client: whoever owns the agent layer passes one in. A tier that is
    configured but not supplied raises here rather than at query time.
    """
    settings = getattr(getattr(config, "retrieval", None), "translation", None)
    if settings is None or not settings.enabled:
        return None

    backends: dict[str, TranslationBackend] = {
        "mt": MachineTranslator(
            checkpoint=settings.mt_checkpoint,
            num_beams=settings.mt_num_beams,
            max_new_tokens=settings.mt_max_new_tokens,
        )
    }
    backends.update(extra_backends or {})

    cache_path: Path | None = None
    if settings.cache_path:
        if repo_root is not None:
            cache_path = Path(repo_root) / settings.cache_path
        elif hasattr(config, "resolve_path"):
            cache_path = Path(config.resolve_path(settings.cache_path))
        else:
            cache_path = Path(settings.cache_path)

    cache = TranslationCache(cache_path)
    logger.info(
        "translation on: tiers=%s, cache=%s (%d entr%s)",
        ", ".join(settings.tiers),
        cache_path,
        len(cache),
        "y" if len(cache) == 1 else "ies",
    )
    return QueryTranslator(backends=backends, tiers=list(settings.tiers), cache=cache)


#: A backend built from a plain callable — for tests, and for anyone wiring an
#: LLM tier from the agent layer without writing a class.
def callable_backend(model_id: str, fn: Callable[[str], str]) -> TranslationBackend:
    @dataclass
    class _Backend:
        model_id: str

        def translate(self, text: str) -> str:
            return fn(text)

    return _Backend(model_id=model_id)
