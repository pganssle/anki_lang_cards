"""Measuring how much two definitions say the same thing.

Two notes with the same headword are sometimes one word written twice and
sometimes two genuinely different cards, and the part of speech does not
reliably tell them apart — the model is happy to call the same word "nm/f" one
run and "nf" the next, while "el cura"/"la cura" really are two words. What
does separate them is whether the definitions agree.

This is a bag-of-words comparison, weighted so that agreeing on a rare word
("fogey", "climber") counts for much more than agreeing on a common one
("person", "thing"). It is meant to rank a list a human will read, not to
decide anything on its own.
"""

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence, Set
from typing import Final

import attrs

# Dropped before comparing: function words, and the register and usage labels
# that decorate definitions without saying what a word means.
_IGNORED_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "one",
        "ones",
        "or",
        "oneself",
        "someone",
        "something",
        "that",
        "the",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "to",
        "with",
        # Register and usage labels.
        "colloquial",
        "colloquially",
        "derogatory",
        "especially",
        "esp",
        "figurative",
        "figuratively",
        "formal",
        "informal",
        "literally",
        "offensive",
        "pejorative",
        "slang",
        "spain",
        "usually",
        "vulgar",
    }
)

_PARENTHETICAL: Final = re.compile(r"\([^()]*\)")
_GLOSS_SEPARATOR: Final = re.compile(r"[,;/]")
_WORD: Final = re.compile(r"[a-z']+")


def _fold(text: str) -> str:
    """Lowercase and strip accents so that tokens compare on their letters."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )


def glosses(definition: str, /) -> Set[str]:
    """Split a definition into the individual senses it lists.

    Parentheticals are dropped and a leading "to " is removed so that "to
    pity" and "pity" are recognized as the same gloss.
    """
    stripped = _PARENTHETICAL.sub(" ", _fold(definition))
    out = set()
    for gloss in _GLOSS_SEPARATOR.split(stripped):
        gloss = re.sub(r"\s+", " ", gloss).strip()
        gloss = re.sub(r"^to\s+", "", gloss)
        if gloss:
            out.add(gloss)
    return out


def tokens(definition: str, /) -> Set[str]:
    """Split a definition into the content words it uses."""
    stripped = _PARENTHETICAL.sub(" ", _fold(definition))
    return {
        word
        for word in _WORD.findall(stripped)
        if word not in _IGNORED_WORDS and len(word) > 2
    }


@attrs.frozen
class Comparison:
    """The result of comparing two definitions."""

    score: float
    """Weighted overlap, from 0.0 (nothing in common) to 1.0 (identical)."""

    shared_glosses: frozenset[str]
    """Whole senses that both definitions list verbatim."""

    @property
    def is_strong(self) -> bool:
        """Whether the two definitions agree enough to propose merging them.

        Either they share a whole sense outright, or enough weighted vocabulary
        that they are describing the same thing in different words. The
        threshold sits in the empty band between the duplicate pairs and the
        deliberate sense splits observed in the deck.
        """
        return bool(self.shared_glosses) or self.score >= _STRONG_SCORE


_STRONG_SCORE: Final[float] = 0.45


class DefinitionComparer:
    """Compares definitions against the background of a corpus of definitions.

    The corpus supplies the inverse document frequencies, so words that are
    rare across the deck dominate the score. Build it once from the existing
    deck and reuse it.
    """

    def __init__(self, corpus: Iterable[str]):
        document_frequency: dict[str, int] = {}
        total = 0
        for definition in corpus:
            total += 1
            for token in tokens(definition):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self._document_frequency: Mapping[str, int] = document_frequency
        self._total: Final = total

    def weight(self, token: str, /) -> float:
        """How much agreeing on a token counts for."""
        seen = self._document_frequency.get(token, 0)
        if not self._total:
            return 1.0
        return math.log((self._total + 1) / (seen + 1))

    def compare(self, first: str, second: str, /) -> Comparison:
        """Compare two definitions."""
        first_tokens, second_tokens = tokens(first), tokens(second)
        shared = frozenset(glosses(first) & glosses(second))
        if not first_tokens or not second_tokens:
            return Comparison(score=0.0, shared_glosses=shared)

        overlap = sum(self.weight(t) for t in first_tokens & second_tokens)
        total = sum(self.weight(t) for t in first_tokens) + sum(
            self.weight(t) for t in second_tokens
        )
        score = 0.0 if not total else (2 * overlap) / total
        return Comparison(score=score, shared_glosses=shared)

    def compare_all(self, definitions: Sequence[str], /) -> Comparison:
        """Compare several definitions, reporting the weakest agreement.

        A group only counts as agreeing if every member agrees, so the lowest
        pairwise score is the one that matters.
        """
        if len(definitions) < 2:
            return Comparison(score=1.0, shared_glosses=frozenset())

        comparisons = [
            self.compare(first, second)
            for index, first in enumerate(definitions)
            for second in definitions[index + 1 :]
        ]
        weakest = min(comparisons, key=lambda comparison: comparison.score)
        shared = frozenset.intersection(
            *(comparison.shared_glosses for comparison in comparisons)
        )
        return Comparison(score=weakest.score, shared_glosses=shared)
