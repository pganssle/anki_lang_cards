"""Normalization of headwords for the purposes of deduplication.

The `word` field of a note is not a bare lemma; it carries notation that the
card template understands. Two notes can therefore be written differently and
still produce the same card front, which makes them duplicates even though the
strings differ.

This module turns a headword into a canonical form so that those cases collide.
It is deliberately conservative: it only rewrites notation whose meaning is
fixed by the card template, never anything that depends on what a word means.
"""

import enum
import functools
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Final

# The card template's own de-accenting table (see `word_front.html`).
_ACCENT_MAP: Final[Mapping[str, str]] = {
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
}
_VOWELS: Final[frozenset[str]] = frozenset(_ACCENT_MAP) | frozenset(
    _ACCENT_MAP.values()
)

_PLURAL_ENDINGS: Final[Sequence[str]] = ("os/as", "as/os")
_SINGULAR_ENDINGS: Final[Sequence[str]] = ("o/a", "a/o")

_PARENTHETICAL: Final = re.compile(r"\s*\(([^()]*)\)")

# A parenthetical made of nothing but these describes how a word is
# constructed, not what it means.
_GRAMMATICAL_WORDS: Final[frozenset[str]] = frozenset(
    {
        # Prepositions that a verb can govern.
        "a",
        "ante",
        "bajo",
        "con",
        "contra",
        "de",
        "desde",
        "en",
        "entre",
        "hacia",
        "hasta",
        "para",
        "por",
        "según",
        "sin",
        "sobre",
        "tras",
        # The complementizer, as in "quedar en (que)".
        "que",
        # Placeholders standing in for whatever the complement happens to be.
        "algo",
        "alguien",
        "alguno",
        "alguna",
        "uno",
        "una",
        "mismo",
        "misma",
        "sí",
    }
)


def normalize(word: str, /) -> str:
    """Apply Unicode normalization to a headword.

    Composed and decomposed spellings of the same accented word are
    indistinguishable on screen but compare unequal, so everything that reaches
    a lookup key needs to agree on one of them.
    """
    return unicodedata.normalize("NFC", word)


def _deaccent_final_syllable(stem: str, /) -> str:
    """Strip the written accent that adding a feminine ending makes redundant.

    Ported from the card template, including its scan order: walk backwards
    replacing accented vowels, and stop at the first unaccented one. "soplón"
    becomes "soplon" so that the feminine renders as "soplona" and not the
    misaccented "soplóna".
    """
    characters = list(stem)
    for index in range(len(characters) - 1, -1, -1):
        character = characters[index]
        if character not in _VOWELS:
            continue
        if character not in _ACCENT_MAP:
            break
        characters[index] = _ACCENT_MAP[character]
    return "".join(characters)


def _split_suffix_notation(stem: str, suffix: str) -> str | None:
    """Resolve dictionary-style suffix notation such as "mirón/ona".

    The suffix replaces however much of the stem it restates, so "mirón/ona"
    means "mirona" (the "on" of the suffix restates the "ón" of the stem) while
    "actor/actriz" restates nothing and simply means "actriz".

    Returns the feminine form, or None when the suffix restates no part of the
    stem and is therefore a whole word rather than a suffix.
    """
    for overlap in range(min(len(stem), len(suffix)), 0, -1):
        tail = _deaccent_final_syllable(stem[-overlap:])
        if tail == suffix[:overlap]:
            return stem[:-overlap] + suffix
    return None


def gendered_forms(word: str, /) -> tuple[str, ...]:
    """Expand slash notation into the forms the card template will display.

    Mirrors `word_front.html` so that dedup agrees with what the deck shows:

        perro/a     -> ("perro", "perra")
        perros/as   -> ("perros", "perras")
        soplón/a    -> ("soplón", "soplona")
        mirón/ona   -> ("mirón", "mirona")
        actor/actriz-> ("actor", "actriz")

    A headword with no slash expands to itself.
    """
    word = normalize(word).strip()
    if "/" not in word:
        return (word,)

    for ending in _PLURAL_ENDINGS:
        if word.endswith(ending):
            base = word[: -len(ending)]
            return (base + ending[:2], base + ending[3:])

    for ending in _SINGULAR_ENDINGS:
        if word.endswith(ending):
            base = word[: -len(ending)]
            return (base + ending[0], base + ending[2])

    if word.endswith("/a"):
        stem = word[:-2]
        return (stem, _deaccent_final_syllable(stem) + "a")

    stem, _, suffix = word.partition("/")
    stem, suffix = stem.strip(), suffix.strip()
    if (feminine := _split_suffix_notation(stem, suffix)) is not None:
        return (stem, feminine)
    return (stem, suffix)


@functools.cache
def canonical_form(word: str, /) -> str:
    """Rewrite a headword into the shortest notation with the same expansion.

    Two spellings of one gendered pair should not produce two cards, so the
    longhand collapses onto the deck's prevailing shorthand:

        mirón/ona   -> mirón/a
        francés/esa -> francés/a
        perro/a     -> perro/a   (already canonical)
        actor/actriz-> actor/actriz  (not a suffix pair; left alone)

    Anything whose expansion the shorthand cannot reproduce is returned
    unchanged rather than guessed at.
    """
    word = normalize(word).strip()
    forms = gendered_forms(word)
    if len(forms) != 2:
        return word

    masculine, feminine = forms
    if masculine.endswith("o") and masculine[:-1] + "a" == feminine:
        return f"{masculine}/a"
    if _deaccent_final_syllable(masculine) + "a" == feminine:
        return f"{masculine}/a"
    return word


class MarkerKind(enum.Enum):
    """What a parenthetical in a headword is doing.

    SENSE markers pick out one meaning of a word that has several, so they are
    the whole reason two notes exist separately: "llave (lucha)" is a card
    about wrestling holds and "llave" is a card about keys, and merging them
    would defeat the point.

    VALENCY markers record the preposition a word governs. "percatarse (de)"
    and "percatarse" are one word written two ways, so they belong on one card.
    """

    SENSE = "sense"
    VALENCY = "valency"


def _classify(contents: str, /) -> MarkerKind:
    tokens = [token for token in re.split(r"[\s,]+", contents.casefold()) if token]
    if all(token in _GRAMMATICAL_WORDS for token in tokens):
        return MarkerKind.VALENCY
    return MarkerKind.SENSE


def parenthetical_markers(word: str, /) -> Sequence[tuple[str, MarkerKind]]:
    """Return each parenthetical in a headword along with what it is doing.

    percatarse (de)   -> (("de", VALENCY),)
    llave (lucha)     -> (("lucha", SENSE),)
    perro             -> ()
    """
    return tuple(
        (match.group(1).strip(), _classify(match.group(1)))
        for match in _PARENTHETICAL.finditer(normalize(word))
    )


def strip_valency_markers(word: str, /) -> str:
    """Remove the parentheticals that only record a governed preposition.

    Sense markers are left in place, because dropping one would merge two cards
    that exist precisely because they mean different things.
    """

    def replace(match: re.Match[str]) -> str:
        if _classify(match.group(1)) is MarkerKind.VALENCY:
            return ""
        return match.group(0)

    return re.sub(r"\s+", " ", _PARENTHETICAL.sub(replace, normalize(word))).strip()


def lookup_key(word: str, /) -> str:
    """Return the form of a headword used to decide whether two notes collide.

    Unlike `canonical_form` this is allowed to discard information, because
    nothing is displayed from it. Valency markers are dropped so that
    "percatarse (de)" and "percatarse" are considered for merging; sense
    markers are kept so that "llave (lucha)" and "llave" are not.

    Collapsing two headwords here only means they get compared, not that they
    get merged — they still reach the conflict finder as a set for review.

    Case and surrounding whitespace are already ignored by `Word.__eq__`, so
    they are ignored here too; bucketing more strictly than words compare would
    hide collisions from the conflict finder.
    """
    return canonical_form(strip_valency_markers(word)).casefold()
