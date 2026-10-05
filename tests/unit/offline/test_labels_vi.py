"""Vietnamese synonyms for the OpenImages labels — mfe.offline.labels_vi."""

from __future__ import annotations

import pytest

from mfe.offline.labels_vi import LABELS_VI, missing_labels, translate_labels


def test_english_labels_are_kept() -> None:
    """Append, never replace.

    An English name still carries information — "Segway" reads the same in both
    languages — and dropping it would silently change what every already-built
    index contains.
    """
    out = translate_labels(["Motorcycle", "Person"])

    assert "Motorcycle" in out
    assert "Person" in out


def test_vietnamese_synonyms_are_appended() -> None:
    out = translate_labels(["Motorcycle"])

    assert "xe máy" in out
    assert "mô tô" in out


def test_several_synonyms_per_label() -> None:
    """One term is not enough: an operator may type any of them.

    BM25 scores a document by which query tokens it contains, so listing the
    alternatives in the document is what makes all the phrasings hit.
    """
    assert len(LABELS_VI["Motorcycle"]) >= 2
    assert len(LABELS_VI["Woman"]) >= 2


def test_unmapped_label_passes_through_unchanged() -> None:
    """A class we have not translated must not stop an ingest.

    It degrades to English-only — the status quo before this table existed —
    rather than raising in the middle of a long run.
    """
    out = translate_labels(["Zorkmid", "Motorcycle"])

    assert "Zorkmid" in out
    assert "xe máy" in out


def test_no_duplicates_in_output() -> None:
    """Two labels sharing a synonym must not index the term twice.

    "Tire" and "Wheel" both map to "bánh xe"; a duplicated term inflates its
    term frequency and BM25 would score the keyframe as if it saw two wheels.
    """
    out = translate_labels(["Wheel", "Tire"])

    assert len(out) == len(set(out))
    assert out.count("bánh xe") == 1


def test_duplicate_input_labels_collapse() -> None:
    out = translate_labels(["Person", "Person"])

    assert len(out) == len(set(out))


def test_empty_input() -> None:
    assert translate_labels([]) == []


def test_missing_labels_reports_only_the_gaps() -> None:
    assert missing_labels(["Motorcycle", "Zorkmid", "Person"]) == ["Zorkmid"]
    assert missing_labels(["Motorcycle"]) == []


def test_missing_labels_is_sorted_and_deduplicated() -> None:
    assert missing_labels(["Zorkmid", "Aardvark", "Zorkmid"]) == ["Aardvark", "Zorkmid"]


# --- sanity of the table itself --------------------------------------------


def test_no_label_maps_to_nothing() -> None:
    empty = [label for label, terms in LABELS_VI.items() if not terms]
    assert not empty, f"these labels have an empty synonym tuple: {empty}"


def test_no_synonym_is_blank() -> None:
    blank = [
        (label, term)
        for label, terms in LABELS_VI.items()
        for term in terms
        if not term.strip()
    ]
    assert not blank


def test_no_synonym_repeats_within_a_label() -> None:
    """Two entries differing only by case are one entry written twice.

    Not a blanket lowercase rule: "mì Ý" and "cây thông Noel" carry proper nouns
    and are spelled correctly as they are. The index casefolds at both ends, so
    case never affects matching — it only affects whether a duplicate is
    visible to a reader of this table.
    """
    for label, terms in LABELS_VI.items():
        folded = [term.casefold() for term in terms]
        assert len(folded) == len(set(folded)), f"{label} lists a synonym twice: {terms}"


def test_diacritics_are_preserved() -> None:
    """Tones are kept deliberately — stripping them collapses distinct words.

    `InvertedIndex` normalizes with NFKC + casefold at both ends, so case is
    already handled; removing tones would additionally merge "má" with "ma".
    """
    assert "xe máy" in LABELS_VI["Motorcycle"]
    assert "người phụ nữ" in LABELS_VI["Woman"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Person", "người"),
        ("Woman", "người phụ nữ"),
        ("Bicycle", "xe đạp"),
        ("Umbrella", "cái ô"),
        ("Tree", "cái cây"),
        ("Flag", "lá cờ"),
    ],
)
def test_spot_check_common_labels(label: str, expected: str) -> None:
    """The labels that actually dominate this corpus, checked by hand."""
    assert expected in LABELS_VI[label]
