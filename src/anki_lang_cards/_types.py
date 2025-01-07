import functools
from collections.abc import Sequence
from pathlib import Path
from typing import NewType, Self, TypedDict, TypeVar

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


def _normalize_str(s: str) -> str:
    return s.strip().casefold()


def _compare_normalized(s1: str, s2: str) -> bool:
    return _normalize_str(s1) == _normalize_str(s2)


def _compare_normalized_sequences(seq1: Sequence[str], seq2: Sequence[str]) -> bool:
    if isinstance(seq1, str) or isinstance(seq2, str):
        raise TypeError("Cannot compare strings")

    for s1, s2 in zip(seq1, seq2, strict=True):
        if not _compare_normalized(s1, s2):
            return False
    return True


def _base_pos(s: str) -> str:
    if s[0] in "nv":
        return s[0]

    return s


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

    def __eq__(self, other):
        return _compare_normalized_sequences(
            (self.word, _base_pos(self.part_of_speech), self.meaning),
            (other.word, _base_pos(other.part_of_speech), other.meaning),
        )

    def __hash__(self):
        return hash((self.word, _base_pos(self.part_of_speech)))

    def update(self, other: Self) -> None:
        """Update the current word with the contents of `other`."""
        if self is other:
            return
        if self != other:
            raise ValueError(f"{self!r}\n\n is not the same word as\n\n{other!r}")

        if self.part_of_speech != other.part_of_speech:
            if _base_pos(self.part_of_speech) == "v":
                if self.part_of_speech == "v":
                    self.part_of_speech = other.part_of_speech
                    should_raise = False
                elif other.part_of_speech == "v":
                    should_raise = False
                else:
                    should_raise = True
            else:
                should_raise = True

            if should_raise:
                raise ValueError(
                    f"Part of speech does not match: {self.part_of_speech} != "
                    f"{other.part_of_speech} for word {self.word}"
                )

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


#####
# Annotation-related types
class BookData(TypedDict):
    title: str
    author: str
    path: Path
    language: str


class Annotation(TypedDict, total=False):
    text: str
    context: str
    note: str | None
    chapter: str
    page_number: int


class CombinedAnnotation(TypedDict, total=False):
    highlights: Sequence[str]
    context: str
    chapters: Sequence[str]
    page_numbers: Sequence[int]


class BookAnnotations(TypedDict):
    book: BookData
    annotations: Sequence[Annotation]

class CombinedBookAnnotations(TypedDict):
    book: BookData
    annotations: Sequence[CombinedAnnotation]
