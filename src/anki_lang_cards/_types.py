import functools
from collections.abc import Sequence
from typing import NewType, Self, TypeVar

import attrs

Url = NewType("Url", str)
_T = TypeVar("_T")


def _non_nulls(v1: _T | None, v2: _T | None) -> Sequence[_T]:
    if v1 and not v2:
        return (v1,)
    if v2 and not v1:
        return (v2,)
    if not v1 and not v2:
        return ()
    return (v1, v2)


def _compare_normalized(s1: str, s2: str) -> bool:
    return s1.strip().casefold() == s2.strip().casefold()


@attrs.define
class Word:
    rank: int | None
    word: str
    meaning: str
    part_of_speech: str
    article: str | None
    spanish_sentence: str | None
    english_sentence: str | None
    example_sentences: Sequence[dict[str, str]]
    model_used: str | None = None

    def update(self, other: Self) -> None:
        """Update the current word with the contents of `other`."""
        if self is other:
            return
        for attr in ("word", "meaning", "part_of_speech"):
            if not _compare_normalized(getattr(self, attr), getattr(other, attr)):
                raise ValueError(f"{self!r} is not the same word as {other!r}")

        ranks = _non_nulls(self.rank, other.rank)
        if ranks:
            rank = min(ranks)
        else:
            rank = None

        example_sentences = functools.reduce(
            lambda a, b: a + b,
            map(list, _non_nulls(self.example_sentences, other.example_sentences)),
            [],
        )

        # The complicated part: Merge the two sets of example sentences,
        # without adding any duplicates.
        if self.spanish_sentence and other.spanish_sentence:
            example_sentences.append(
                {"english": other.english_sentence, "spanish": other.spanish_sentence}
            )
        elif other.spanish_sentence:
            self.spanish_sentence = other.spanish_sentence
            self.english_sentence = other.english_sentence

        spanish_examples = set()
        if self.spanish_sentence:
            spanish_examples.add(self.spanish_sentence.casefold().strip())

        self.example_sentences = []
        for example_sentence in example_sentences:
            normalized_spanish = example_sentence["spanish"].casefold().strip()
            if normalized_spanish not in spanish_examples:
                spanish_examples.add(normalized_spanish)
                self.example_sentences.append(example_sentence)

        self.model_used = ", ".join(set(_non_nulls(self.model_used, other.model_used)))
