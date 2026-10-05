"""Vietnamese query preprocessing, accent-folding, and compound phrase extraction.

Optimized for AIC (Ho Chi Minh City AI Challenge) text search on Elasticsearch:
1. Normalizes Unicode to standard NFC (handling both Unicode dựng sẵn & tổ hợp).
2. Generates unaccented (không dấu) variants for query expansion against typos / Telex issues.
3. Identifies compound terms (từ ghép) to enforce match_phrase boost in BM25.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Bảng chuyển đổi ký tự tiếng Việt có dấu -> không dấu
_VIETNAMESE_ACCENTS_MAP = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "đ": "d",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
}

# Các từ ghép phổ biến trong video surveillance/news/traffic AIC
COMMON_VIETNAMESE_COMPOUNDS = {
    "áo dài", "xe máy", "xe đạp", "xe buýt", "ô tô", "tai nạn",
    "cảnh sát", "giao thông", "chữa cháy", "cứu hỏa", "nhà xưởng",
    "đám cháy", "trường học", "bệnh viện", "công viên", "sân bay",
    "nông dân", "học sinh", "giáo viên", "bài giảng", "người đi bộ",
    "đèn đỏ", "đèn xanh", "vạch kẻ đường", "vỉa hè", "cột điện",
    "trực thăng", "tên lửa", "xe tải", "xe cứu thương", "xe cảnh sát",
}


def remove_accents(text: str) -> str:
    """Remove Vietnamese diacritics and convert to ASCII-friendly representation."""
    text_nfc = unicodedata.normalize("NFC", text)
    result = []
    for ch in text_nfc:
        lower_ch = ch.lower()
        if lower_ch in _VIETNAMESE_ACCENTS_MAP:
            mapped = _VIETNAMESE_ACCENTS_MAP[lower_ch]
            result.append(mapped.upper() if ch.isupper() else mapped)
        else:
            result.append(ch)
    return "".join(result)


def normalize_vietnamese_query(text: str) -> str:
    """Normalize whitespace and ensure consistent Unicode NFC."""
    text_nfc = unicodedata.normalize("NFC", text.strip())
    # Clean multiple spaces
    return re.sub(r"\s+", " ", text_nfc)


class VietnameseQueryExpander:
    """Query preprocessor for Elasticsearch multi-match retrieval in AIC."""

    @classmethod
    def extract_compounds(cls, text: str) -> list[str]:
        """Extract multi-word compound terms present in the query."""
        normalized = normalize_vietnamese_query(text).lower()
        found: list[str] = []
        for compound in COMMON_VIETNAMESE_COMPOUNDS:
            if compound in normalized:
                found.append(compound)
        return sorted(found, key=len, reverse=True)

    @classmethod
    def expand_query(
        cls,
        query: str,
        *,
        include_unaccented: bool = True,
        phrase_boost: float = 2.0,
    ) -> dict[str, Any]:
        """Generate structured query payload with exact, compound phrase, and unaccented terms.

        Returns a dictionary suitable for building hybrid Elasticsearch queries.
        """
        nfc_query = normalize_vietnamese_query(query)
        unaccented_query = remove_accents(nfc_query)
        compounds = cls.extract_compounds(nfc_query)

        has_accents = nfc_query != unaccented_query

        return {
            "original_nfc": nfc_query,
            "unaccented": unaccented_query if (include_unaccented and has_accents) else None,
            "compounds": compounds,
            "phrase_boost": phrase_boost,
            "has_accents": has_accents,
        }
