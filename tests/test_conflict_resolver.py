"""Tests for the word deduplication and conflict-detection machinery."""

from collections.abc import Sequence

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import conflict_resolver


def make_word(
    word: str,
    part_of_speech: str,
    meaning: str = "meaning",
    rank: int | None = None,
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=(),
    )


def group_of(lookup: conflict_resolver.WordLookup, word: types.Word) -> Sequence[str]:
    return tuple(str(w.part_of_speech) for w in lookup.lookup(word))


@pytest.mark.parametrize(
    ("pos1", "pos2"),
    [
        # An unspecified characteristic is a wildcard and must match a
        # specified one; a hash-based key cannot express this.
        ("n", "nm"),
        ("n", "nf"),
        ("nm", "n"),
        ("v", "vtr"),
        ("v", "vi"),
        ("vtr", "v"),
        # Phrasality is excluded from equality entirely.
        ("nm", "nm phr"),
        ("vtr", "vtr phr"),
        ("adj", "adj phr"),
    ],
)
def test_compatible_parts_of_speech_share_a_group(pos1: str, pos2: str) -> None:
    first = make_word("palabra", pos1)
    second = make_word("palabra", pos2)
    lookup = conflict_resolver.WordLookup([first, second])

    assert len(lookup) == 1
    assert lookup.lookup(first) == (first, second)
    assert lookup.lookup(second) == (first, second)


@pytest.mark.parametrize(
    ("pos1", "pos2"),
    [
        # "el cura" (priest) and "la cura" (cure) are genuinely different cards.
        ("nm", "nf"),
        ("nf", "nm"),
        # "vi" and "vtr" both specify transitivity, and disagree.
        ("vi", "vtr"),
        # "el perdido" (loss) and "perdido/a" (lost) must not merge.
        ("nm", "adj"),
        ("adj", "nm"),
        ("conj", "adv"),
        ("prep", "v phr"),
        ("n", "adj"),
    ],
)
def test_incompatible_parts_of_speech_stay_apart(pos1: str, pos2: str) -> None:
    first = make_word("palabra", pos1)
    second = make_word("palabra", pos2)
    lookup = conflict_resolver.WordLookup([first, second])

    assert len(lookup) == 2
    assert lookup.lookup(first) == (first,)
    assert lookup.lookup(second) == (second,)


@pytest.mark.parametrize(
    ("word1", "word2"),
    [
        # Two spellings of one gendered pair are one card.
        ("mirón/ona", "mirón/a"),
        ("francés/esa", "francés/a"),
        ("Perro/a", "perro/a"),
    ],
)
def test_equivalent_headword_spellings_share_a_group(word1: str, word2: str) -> None:
    first = make_word(word1, "nm/f")
    second = make_word(word2, "nm/f")
    lookup = conflict_resolver.WordLookup([first, second])

    assert len(lookup) == 1
    assert lookup.lookup(second) == (first, second)


def test_headword_canonicalization_does_not_merge_unrelated_words() -> None:
    lookup = conflict_resolver.WordLookup(
        [make_word("actor/actriz", "nm/f"), make_word("actor/a", "nm/f")]
    )
    assert len(lookup) == 2


def test_different_headwords_stay_apart() -> None:
    first = make_word("perro", "nm")
    second = make_word("gato", "nm")
    lookup = conflict_resolver.WordLookup([first, second])

    assert len(lookup) == 2


def test_wildcard_does_not_bridge_incompatible_groups() -> None:
    """A wildcard must not merge "nm" and "nf" transitively through "n"."""
    masculine = make_word("cura", "nm", meaning="priest")
    unspecified = make_word("cura", "n", meaning="unspecified")
    feminine = make_word("cura", "nf", meaning="cure")

    lookup = conflict_resolver.WordLookup([masculine, unspecified, feminine])

    assert len(lookup) == 2
    assert group_of(lookup, masculine) == ("nm", "n")
    assert group_of(lookup, feminine) == ("nf",)


def test_membership_requires_compatibility_with_every_member() -> None:
    """ "nf" must not join a group merely because it matches the "n" in it."""
    lookup = conflict_resolver.WordLookup(
        [make_word("cura", "nm"), make_word("cura", "n")]
    )
    feminine = make_word("cura", "nf")

    assert feminine not in lookup

    lookup.add(feminine)
    assert len(lookup) == 2
    assert group_of(lookup, feminine) == ("nf",)


def test_contains_uses_compatibility_not_identity() -> None:
    lookup = conflict_resolver.WordLookup([make_word("palabra", "nm")])

    assert make_word("palabra", "n") in lookup
    assert make_word("palabra", "nm phr") in lookup
    assert make_word("palabra", "nf") not in lookup
    assert make_word("palabra", "adj") not in lookup
    assert make_word("otra", "nm") not in lookup


def test_iteration_yields_every_group_exactly_once() -> None:
    words = [
        make_word("cura", "nm"),
        make_word("cura", "nf"),
        make_word("perro", "nm"),
        make_word("perro", "n"),
    ]
    lookup = conflict_resolver.WordLookup(words)

    groups = list(lookup)
    assert len(groups) == 3
    assert sum(len(group) for group in groups) == len(words)
    assert len(lookup) == len(groups)


def test_merge_preserves_grouping() -> None:
    left = conflict_resolver.WordLookup([make_word("cura", "nm")])
    right = conflict_resolver.WordLookup(
        [make_word("cura", "nf"), make_word("cura", "n")]
    )

    merged = left.merge(right)

    # "n" attaches to whichever of "nm"/"nf" it meets first, which depends on
    # insertion order, but the two gendered entries must never end up together.
    assert len(merged) == 2
    for group in merged:
        genders = {str(w.part_of_speech) for w in group}
        assert not {"nm", "nf"} <= genders


def test_lookup_of_absent_word_is_empty() -> None:
    lookup = conflict_resolver.WordLookup([make_word("perro", "nm")])
    assert lookup.lookup(make_word("gato", "nm")) == ()


def test_find_conflicting_word_sets_ignores_identical_words() -> None:
    existing = make_word("perro", "nm", meaning="dog", rank=1)
    new = make_word("perro", "nm", meaning="dog", rank=1)

    conflicts = list(
        conflict_resolver.find_conflicting_word_sets(
            [new], conflict_resolver.WordLookup([existing])
        )
    )
    assert conflicts == []


def test_find_conflicting_word_sets_reports_wildcard_matches() -> None:
    """The bug this replaced: "nm" vs "n" never reached the conflict list."""
    existing = make_word("trepa", "nm", meaning="social climber", rank=8576)
    new = make_word("trepa", "n", meaning="climber", rank=9615)

    conflicts = list(
        conflict_resolver.find_conflicting_word_sets(
            [new], conflict_resolver.WordLookup([existing])
        )
    )
    assert len(conflicts) == 1
    assert {w.meaning for w in conflicts[0]} == {"social climber", "climber"}


def test_find_conflicting_word_sets_leaves_gender_splits_alone() -> None:
    existing = make_word("cura", "nm", meaning="priest", rank=1790)
    new = make_word("cura", "nf", meaning="cure", rank=9000)

    conflicts = list(
        conflict_resolver.find_conflicting_word_sets(
            [new], conflict_resolver.WordLookup([existing])
        )
    )
    assert conflicts == []
