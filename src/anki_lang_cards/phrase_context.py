"""Finding words that only ever occur inside one fixed phrase.

Some headwords are not really words. "reojo" exists only in "de reojo",
"birlibirloque" only in "por arte de birlibirloque"; a card asking for either
on its own teaches a fragment. Others look the same from a distance but are
genuinely independent — "genio" and "sepultura" each stand alone as well as
appearing in "genio y figura hasta la sepultura" — and those deserve their own
cards alongside the phrase.

What separates the two is whether the word's surroundings vary. This module
looks for the longest run of words that recurs around a headword across all of
its example sentences: for "reojo" that run is "de reojo" every time, while for
"genio" nothing recurs. No dictionary, definition text or lemmatizer is
involved, so nothing here depends on how a definition happens to be worded.

The known limits are worth stating, since both are inherent to counting
surface forms. Discontinuous phrases are invisible — "apañárselas" surfaces as
"se las van a apañar" and no run of adjacent words recurs. So are headwords
that inflect, since the surface form has to repeat to be counted. Both failures
are silent misses, never false accusations, so the cost of one is a phrase that
goes unflagged rather than a card wrongly merged.
"""

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import attrs

from . import _types as types
from . import headword

_WORD_CHARACTERS: Final = re.compile(r"[^\w\s]|_", re.UNICODE)

DEFAULT_MAX_PHRASE_LENGTH: Final[int] = 6
"""Longest run of words considered. Spanish idioms rarely run longer."""

DEFAULT_MIN_SENTENCES: Final[int] = 3
"""Fewest occurrences that can say anything about how a word is used."""

DEFAULT_MIN_COVERAGE: Final[float] = 0.9
"""How much of the evidence a run must cover to count as the word's context."""

_MIN_OCCURRENCES: Final[int] = 2
"""One sentence is a sentence, not a pattern."""

# Spanish nouns take a determiner almost obligatorily, so "la lumbre" recurring
# across every example says nothing about "lumbre". Prepositions carry no such
# obligation, which is why "de reojo" recurring does say something. Note that
# "al" and "del" are absent deliberately: they contract a preposition and so
# count as real context.
_DETERMINERS: Final[frozenset[str]] = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "lo",
        "un",
        "una",
        "unos",
        "unas",
        "mi",
        "mis",
        "tu",
        "tus",
        "su",
        "sus",
        "nuestro",
        "nuestra",
        "nuestros",
        "nuestras",
        "vuestro",
        "vuestra",
        "vuestros",
        "vuestras",
        "este",
        "esta",
        "estos",
        "estas",
        "ese",
        "esa",
        "esos",
        "esas",
        "aquel",
        "aquella",
        "aquellos",
        "aquellas",
    }
)


def _is_informative(run: Sequence[str], target: str) -> bool:
    """Whether a run says anything beyond "this word takes a determiner".

    Two things qualify. Something other than a determiner standing before the
    headword means the word only turns up as the object of that something,
    which is what being stuck in a phrase looks like. Failing that, material on
    both sides is a frame rather than an accident.

    Material only after the headword is not enough: "manojo de" and "enjambre
    de" recur perfectly, but taking a complement is what those nouns do, not a
    phrase they are trapped in.
    """
    position = run.index(target)
    before, after = run[:position], run[position + 1 :]
    if any(token not in _DETERMINERS for token in before):
        return True
    return bool(before) and bool(after)


def tokenize(sentence: str, /) -> Sequence[str]:
    """Split a sentence into comparable lowercase words.

    Punctuation is dropped rather than treated as a boundary, so that a phrase
    is still recognized when a comma or dash falls inside it.
    """
    folded = unicodedata.normalize("NFC", sentence).casefold()
    return tuple(_WORD_CHARACTERS.sub(" ", folded).split())


def _matches(token: str, target: str) -> bool:
    """Whether a token is the headword, allowing for a plural ending."""
    if token == target:
        return True
    return token.startswith(target) and len(token) - len(target) <= 2


@attrs.frozen(order=False)
class ContextRun:
    """A run of words that recurs around a headword."""

    phrase: str
    sentence_count: int
    coverage: float
    """Fraction of the sentences containing the headword that show this run."""

    @property
    def length(self) -> int:
        return len(self.phrase.split())


def _runs_in_sentence(
    tokens: Sequence[str], target: str, max_length: int
) -> frozenset[tuple[str, ...]]:
    """Every run of words in one sentence that contains the headword.

    The headword itself is rewritten to its dictionary form wherever it occurs,
    so that "balas de fogueo" and "balas de fogueos" are recognized as the same
    context rather than two unrelated ones.
    """
    positions = [index for index, token in enumerate(tokens) if _matches(token, target)]
    tokens = [
        target if index in set(positions) else token
        for index, token in enumerate(tokens)
    ]
    runs: set[tuple[str, ...]] = set()
    for position in positions:
        first = max(0, position - max_length + 1)
        for start in range(first, position + 1):
            last = min(len(tokens), start + max_length)
            for end in range(position + 1, last + 1):
                if end - start >= 2:
                    runs.add(tuple(tokens[start:end]))
    return frozenset(runs)


def _is_contained_in(inner: Sequence[str], outer: Sequence[str]) -> bool:
    """Whether one run appears contiguously inside another."""
    if len(inner) >= len(outer):
        return False
    return any(
        tuple(outer[offset : offset + len(inner)]) == tuple(inner)
        for offset in range(len(outer) - len(inner) + 1)
    )


def recurring_context(
    word: str,
    sentences: Iterable[str],
    *,
    max_length: int = DEFAULT_MAX_PHRASE_LENGTH,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> Sequence[ContextRun]:
    """Find the runs of words that recur around a headword.

    Only sentences that actually contain the headword count toward coverage, so
    a sentence using an inflected form neither helps nor hurts.

    A run seen in a single sentence is not recurring, however much of the
    evidence that one sentence happens to be, so two occurrences are always
    required. Runs wholly contained in a longer run that covers as much
    evidence are dropped, leaving the longest description of the context rather
    than all of its fragments. Results are ordered longest first.
    """
    target = tokenize(word)
    if len(target) != 1:
        return ()

    per_sentence = [
        runs
        for sentence in sentences
        if (runs := _runs_in_sentence(tokenize(sentence), target[0], max_length))
    ]
    if not per_sentence:
        return ()

    counts: dict[tuple[str, ...], int] = {}
    for runs in per_sentence:
        for run in runs:
            counts[run] = counts.get(run, 0) + 1

    total = len(per_sentence)
    frequent = {
        run: count
        for run, count in counts.items()
        if count >= _MIN_OCCURRENCES
        and count / total >= min_coverage
        and _is_informative(run, target[0])
    }
    maximal = [
        run
        for run in frequent
        if not any(
            _is_contained_in(run, other) and frequent[other] >= frequent[run]
            for other in frequent
            if other != run
        )
    ]
    return tuple(
        sorted(
            (
                ContextRun(
                    phrase=" ".join(run),
                    sentence_count=frequent[run],
                    coverage=frequent[run] / total,
                )
                for run in maximal
            ),
            key=lambda run: (-run.length, -run.coverage, run.phrase),
        )
    )


@attrs.frozen(order=False)
class PhraseBoundWord:
    """A single-word note whose examples only ever show it inside a phrase."""

    word: types.Word
    context: ContextRun
    existing_phrase: types.Word | None
    """The note for the phrase itself, when the deck already has one."""

    @property
    def is_redundant(self) -> bool:
        """Whether the phrase is already a card, making this one a fragment of it."""
        return self.existing_phrase is not None


def _spanish_sentences(word: types.Word) -> Sequence[str]:
    return tuple(sentence.spanish for sentence in word.sentences if sentence.spanish)


def find_phrase_bound_words(
    words: Iterable[types.Word],
    *,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_length: int = DEFAULT_MAX_PHRASE_LENGTH,
) -> Sequence[PhraseBoundWord]:
    """Report single-word notes that never escape one phrase.

    A note is flagged when enough of its examples exist to say anything, and
    effectively all of them show the headword inside the same run of words.
    Whether the phrase is already its own note is reported alongside, since
    that decides what to do: if it is, this note is a fragment of that one; if
    it is not, the phrase is probably what the note should have been.
    """
    words = tuple(words)
    by_headword: Mapping[str, types.Word] = {
        headword.lookup_key(word.word): word for word in words
    }

    found: list[PhraseBoundWord] = []
    for word in words:
        if len(tokenize(word.word)) != 1:
            continue
        sentences = _spanish_sentences(word)
        if len(sentences) < min_sentences:
            continue

        runs = recurring_context(
            word.word, sentences, max_length=max_length, min_coverage=min_coverage
        )
        if not runs:
            continue

        best = runs[0]
        if best.sentence_count < min_sentences:
            continue

        existing = by_headword.get(headword.lookup_key(best.phrase))
        found.append(PhraseBoundWord(word=word, context=best, existing_phrase=existing))

    return tuple(sorted(found, key=lambda f: (-f.context.coverage, -f.context.length)))
