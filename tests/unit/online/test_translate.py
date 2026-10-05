"""The translation chain — `mfe.online.translate`.

No model is loaded anywhere here: every tier is a callable that counts its own
calls, which is what makes "the cache spared a second inference" an assertion
rather than a hope.
"""

from __future__ import annotations

import json
import unicodedata

import pytest

from mfe.online.arms import Arm, ArmRegistry
from mfe.online.cascade import Cascade
from mfe.online.translate import (
    MANUAL_TIER,
    QueryTranslator,
    TranslationCache,
    _strip_language_prefix,
    callable_backend,
)


class CountingBackend:
    """A tier that records how often it was actually asked to work."""

    def __init__(self, model_id: str = "fake-mt|beams=1", answer: str = "ENGLISH") -> None:
        self.model_id = model_id
        self.answer = answer
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return self.answer


def test_manual_wins_and_no_backend_is_touched() -> None:
    backend = CountingBackend()
    translator = QueryTranslator(backends={"mt": backend}, tiers=["mt"])

    result = translator.translate("một con lừa hoang", manual="a wild ass")

    assert result is not None
    assert result.text_en == "a wild ass"
    assert result.tier == MANUAL_TIER
    assert result.is_manual
    assert backend.calls == [], "hand-written English must short-circuit the tiers"


def test_falls_through_to_the_machine_tier() -> None:
    backend = CountingBackend(answer="a wild ass foal")
    translator = QueryTranslator(backends={"mt": backend}, tiers=["mt"])

    result = translator.translate("một con lừa hoang con")

    assert result is not None
    assert (result.text_en, result.tier) == ("a wild ass foal", "mt")
    assert result.model == "fake-mt|beams=1"
    assert backend.calls == ["một con lừa hoang con"]


def test_no_translation_returns_none_never_the_vietnamese() -> None:
    """The constraint the whole module exists to enforce (handoff §5.1)."""

    def explode(_text: str) -> str:
        raise RuntimeError("no network")

    translator = QueryTranslator(
        backends={"mt": callable_backend("dead-mt", explode)}, tiers=["mt"]
    )

    result = translator.translate("cờ Anh trên nóc toà nhà")

    assert result is None, "a failed chain must abstain, not pass Vietnamese through"
    assert translator.usage == {"none": 1}


def test_empty_translation_is_treated_as_a_failure() -> None:
    translator = QueryTranslator(
        backends={"mt": callable_backend("blank-mt", lambda _t: "   ")}, tiers=["mt"]
    )
    assert translator.translate("bờ kè bê tông") is None


def test_tiers_run_in_order_and_a_failure_falls_through() -> None:
    llm = CountingBackend(model_id="fake-llm", answer="a collapsed embankment")
    translator = QueryTranslator(
        backends={
            "mt": callable_backend("dead-mt", _raise),
            "llm": llm,
        },
        tiers=["mt", "llm"],
    )

    result = translator.translate("bờ kè sập")

    assert result is not None
    assert (result.text_en, result.tier) == ("a collapsed embankment", "llm")
    assert llm.calls == ["bờ kè sập"]


def _raise(_text: str) -> str:
    raise RuntimeError("tier down")


def test_a_configured_tier_without_a_backend_fails_at_construction() -> None:
    """Loudly, and before a query — otherwise it looks like an untranslated query."""
    with pytest.raises(ValueError, match="llm"):
        QueryTranslator(backends={"mt": CountingBackend()}, tiers=["mt", "llm"])


def test_empty_tier_list_means_hand_written_only() -> None:
    translator = QueryTranslator(backends={}, tiers=[])
    assert translator.translate("xe máy ban đêm") is None
    assert translator.translate("xe máy ban đêm", manual="motorbikes at night") is not None


# --- cache ---------------------------------------------------------------


def test_cache_spares_the_second_inference(tmp_path) -> None:
    backend = CountingBackend(answer="motorbikes at night")
    cache = TranslationCache(tmp_path / "translations.json")
    translator = QueryTranslator(backends={"mt": backend}, tiers=["mt"], cache=cache)

    first = translator.translate("xe máy ban đêm")
    second = translator.translate("xe máy ban đêm")

    assert first is not None and second is not None
    assert first.text_en == second.text_en
    assert backend.calls == ["xe máy ban đêm"], "the second call must come from cache"


def test_cache_survives_a_new_process(tmp_path) -> None:
    path = tmp_path / "translations.json"
    backend = CountingBackend(answer="a grey cloudy sky")
    first = QueryTranslator(
        backends={"mt": backend}, tiers=["mt"], cache=TranslationCache(path)
    )
    first.translate("bầu trời nhiều mây xám")
    first.save_cache()

    reloaded = QueryTranslator(
        backends={"mt": backend}, tiers=["mt"], cache=TranslationCache(path)
    )
    result = reloaded.translate("bầu trời nhiều mây xám")

    assert result is not None and result.text_en == "a grey cloudy sky"
    assert len(backend.calls) == 1, "a reloaded cache must not re-translate"


def test_cache_key_includes_the_model_so_a_swap_invalidates(tmp_path) -> None:
    """Two decoders' outputs must never share one score table."""
    path = tmp_path / "translations.json"
    cache = TranslationCache(path)
    greedy = CountingBackend(model_id="envit5|beams=1", answer="greedy output")
    beams = CountingBackend(model_id="envit5|beams=4", answer="beam output")

    QueryTranslator(backends={"mt": greedy}, tiers=["mt"], cache=cache).translate("cờ Anh")
    result = QueryTranslator(
        backends={"mt": beams}, tiers=["mt"], cache=cache
    ).translate("cờ Anh")

    assert result is not None and result.text_en == "beam output"
    assert beams.calls == ["cờ Anh"], "a different model must miss the cache"


def test_cache_key_is_nfc_so_a_decomposed_query_still_hits(tmp_path) -> None:
    backend = CountingBackend(answer="a landslide")
    translator = QueryTranslator(
        backends={"mt": backend}, tiers=["mt"], cache=TranslationCache(tmp_path / "c.json")
    )
    precomposed = "sạt lở bờ sông"

    translator.translate(precomposed)
    translator.translate(unicodedata.normalize("NFD", precomposed))

    assert len(backend.calls) == 1, "NFD and NFC of one query are one cache entry"


def test_saved_cache_is_sorted_and_readable(tmp_path) -> None:
    path = tmp_path / "translations.json"
    translator = QueryTranslator(
        backends={"mt": callable_backend("m", lambda t: t.upper())},
        tiers=["mt"],
        cache=TranslationCache(path),
    )
    for text in ("bờ kè", "ao dài", "con lừa"):
        translator.translate(text)
    translator.save_cache()

    payload = json.loads(path.read_text(encoding="utf-8"))
    texts = [entry["text"] for entry in payload["entries"]]
    assert texts == sorted(texts), "a stable order keeps the git diff to one line"
    assert payload["entries"][0].keys() == {"text", "tier", "model", "text_en"}


def test_a_corrupt_cache_does_not_take_the_run_down(tmp_path) -> None:
    path = tmp_path / "translations.json"
    path.write_text("{not json", encoding="utf-8")

    cache = TranslationCache(path)

    assert len(cache) == 0
    assert cache.get("bất kỳ", "mt", "m") is None


def test_usage_counts_how_the_english_was_obtained() -> None:
    translator = QueryTranslator(
        backends={"mt": callable_backend("m", lambda t: "EN")}, tiers=["mt"]
    )
    translator.translate("một", manual="one")
    translator.translate("hai")
    translator.translate("ba")

    assert translator.usage == {MANUAL_TIER: 1, "mt": 2}


def test_strip_language_prefix() -> None:
    """envit5 answers with its own `en:` marker; it must not reach a query."""
    assert _strip_language_prefix("en: a wild ass") == "a wild ass"
    assert _strip_language_prefix("  En:  a wild ass ") == "a wild ass"
    assert _strip_language_prefix("a wild ass") == "a wild ass"


# --- cascade wiring ------------------------------------------------------


def _registry(*arms: Arm) -> ArmRegistry:
    return ArmRegistry(list(arms))


def _cascade_with(arms: ArmRegistry, translator) -> Cascade:
    cascade = Cascade(config=None, translator=translator)
    cascade.arms = arms
    return cascade


def test_cascade_does_not_translate_when_no_arm_reads_english() -> None:
    """Nothing would consume it, and the checkpoint costs 1.2 GB to find out."""
    backend = CountingBackend()
    translator = QueryTranslator(backends={"mt": backend}, tiers=["mt"])
    cascade = _cascade_with(
        _registry(Arm(name="bm25", search=lambda q, k, r: [], query_language="vi")),
        translator,
    )

    assert cascade.resolve_english("một câu tiếng Việt", None) is None
    assert backend.calls == []


def test_cascade_translates_when_an_english_arm_is_active() -> None:
    backend = CountingBackend(answer="a wild ass")
    translator = QueryTranslator(backends={"mt": backend}, tiers=["mt"])
    cascade = _cascade_with(
        _registry(Arm(name="siglip", search=lambda q, k, r: [], query_language="en")),
        translator,
    )

    assert cascade.resolve_english("con lừa hoang", None) == "a wild ass"
    assert backend.calls == ["con lừa hoang"]


def test_cascade_skips_translation_for_a_disabled_english_arm() -> None:
    backend = CountingBackend()
    translator = QueryTranslator(backends={"mt": backend}, tiers=["mt"])
    arms = _registry(Arm(name="siglip", search=lambda q, k, r: [], query_language="en"))
    arms.set_enabled("siglip", False)
    cascade = _cascade_with(arms, translator)

    assert cascade.resolve_english("con lừa hoang", None) is None
    assert backend.calls == []


def test_cascade_prefers_the_hand_written_form() -> None:
    backend = CountingBackend(answer="machine output")
    translator = QueryTranslator(backends={"mt": backend}, tiers=["mt"])
    cascade = _cascade_with(
        _registry(Arm(name="siglip", search=lambda q, k, r: [], query_language="en")),
        translator,
    )

    assert cascade.resolve_english("con lừa hoang", "a wild ass foal") == "a wild ass foal"
    assert backend.calls == []


def test_cascade_without_a_translator_still_passes_manual_english_through() -> None:
    cascade = _cascade_with(
        _registry(Arm(name="siglip", search=lambda q, k, r: [], query_language="en")),
        None,
    )
    assert cascade.resolve_english("con lừa hoang", "a wild ass foal") == "a wild ass foal"
    assert cascade.resolve_english("con lừa hoang", None) is None
