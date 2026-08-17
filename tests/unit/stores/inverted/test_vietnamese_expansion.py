"""Unit tests for Vietnamese query preprocessing, NFC normalization, and compound extraction."""

from __future__ import annotations

import unicodedata

from mfe.stores.production.vietnamese import (
    VietnameseQueryExpander,
    normalize_vietnamese_query,
    remove_accents,
)


def test_remove_accents_comprehensive() -> None:
    assert remove_accents("áo dài màu đỏ") == "ao dai mau do"
    assert remove_accents("Đà Nẵng & Thành phố Hồ Chí Minh") == "Da Nang & Thanh pho Ho Chi Minh"
    assert remove_accents("Người đi bộ qua đường") == "Nguoi di bo qua duong"
    assert remove_accents("Hello World 123!") == "Hello World 123!"


def test_normalize_vietnamese_query_nfc() -> None:
    # Test decomposed NFD string normalized to NFC
    decomposed = unicodedata.normalize("NFD", "thành phố")
    normalized = normalize_vietnamese_query(decomposed)
    assert unicodedata.is_normalized("NFC", normalized)
    assert normalized == "thành phố"


def test_extract_compounds_detection() -> None:
    query = "người phụ nữ mặc áo dài đi xe máy gần đám cháy"
    compounds = VietnameseQueryExpander.extract_compounds(query)
    assert "áo dài" in compounds
    assert "xe máy" in compounds
    assert "đám cháy" in compounds


def test_expand_query_structure() -> None:
    res = VietnameseQueryExpander.expand_query("cảnh sát giao thông áo dài", phrase_boost=2.5)
    assert res["original_nfc"] == "cảnh sát giao thông áo dài"
    assert res["unaccented"] == "canh sat giao thong ao dai"
    assert "cảnh sát" in res["compounds"]
    assert "giao thông" in res["compounds"]
    assert "áo dài" in res["compounds"]
    assert res["phrase_boost"] == 2.5
    assert res["has_accents"] is True


def test_expand_query_pure_ascii() -> None:
    res = VietnameseQueryExpander.expand_query("car accident road", phrase_boost=2.0)
    assert res["original_nfc"] == "car accident road"
    assert res["unaccented"] is None
    assert res["has_accents"] is False


def test_extract_compounds_sorting_and_whitespace() -> None:
    # Multiple compounds with redundant whitespace and uppercase
    query = "  XE MÁY   chạy qua   vạch kẻ đường  "
    compounds = VietnameseQueryExpander.extract_compounds(query)
    assert "vạch kẻ đường" in compounds
    assert "xe máy" in compounds
    # Longer compounds appear first in sorted list
    assert compounds.index("vạch kẻ đường") < compounds.index("xe máy")


def test_expand_query_disabled_unaccented() -> None:
    res = VietnameseQueryExpander.expand_query("áo dài đỏ", include_unaccented=False)
    assert res["original_nfc"] == "áo dài đỏ"
    assert res["unaccented"] is None
    assert res["has_accents"] is True

