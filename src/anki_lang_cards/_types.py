import contextlib
import csv
import datetime
import functools
import io
import json
import threading
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NewType, Protocol, Self, TypedDict, TypeVar

import attrs
import cattrs

from . import part_of_speech as pos

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


ID_LOCK = threading.Lock()


def _get_next_id() -> int:
    with ID_LOCK:
        if getattr(_get_next_id, "_last_id", None) is None:
            _get_next_id._last_id = 0

        _get_next_id._last_id += 1
        return _get_next_id._last_id


@attrs.frozen
class Citation(Protocol):
    id: int = attrs.field(factory=_get_next_id, kw_only=True)

    @property
    def source_type(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return _cattrs_converter().unstructure(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:

        source_type = data.get("source_type")
        if source_type == "book":
            klass: type = BookCitation
        elif source_type == "model":
            klass = ModelCitation
        else:
            if source_type is None:
                message_preamble = "Source type not found in citation:\n"
            else:
                message_preamble = (
                    f"Unknown citation source type {source_type} in citation:\n"
                )
            raise ValueError(f"{message_preamble}{data}")
        return _cattrs_converter().structure(data, klass)


@attrs.frozen(slots=True)
class BookCitation(Citation):
    title: str
    authors: Sequence[str]
    pages: Sequence[int] | None = None
    chapters: Sequence[str] | None = None

    source_type: Final[str] = attrs.field(
        default="book", validator=attrs.validators.in_(("book",))
    )


@attrs.frozen(slots=True)
class ModelCitation(Citation):
    model: str
    date: datetime.date = attrs.field(factory=lambda: datetime.datetime.now().date())

    source_type: Final[str] = attrs.field(
        default="model", validator=attrs.validators.in_(("model",))
    )


def _convert_pos(
    value: str | pos.PartOfSpeech,
) -> pos.PartOfSpeech:
    if isinstance(value, pos.PartOfSpeech):
        return value
    return pos.parse_part_of_speech(value)


class Word:
    rank: int | None
    word: str
    meaning: str
    part_of_speech: pos.PartOfSpeech = attrs.field(converter=_convert_pos)
    article: str | None
    spanish_sentence: str | None
    english_sentence: str | None
    example_sentences: Sequence[dict[str, str]]
    model_used: str | None = None
    sentence_source: Citation | None = None

    def __eq__(self, other):
        return (
            self.part_of_speech == other.part_of_speech
        ) and _compare_normalized_sequences(
            (self.word, self.meaning),
            (other.word, other.meaning),
        )

    def __hash__(self):
        return hash((self.word, self.part_of_speech.base_pos))

    def update(self, other: Self) -> None:
        """Update the current word with the contents of `other`."""
        if self is other:
            return
        if self != other:
            raise ValueError(f"{self!r}\n\n is not the same word as\n\n{other!r}")

        if self.part_of_speech != other.part_of_speech:
            raise ValueError(
                f"Part of speech does not match: {self.part_of_speech} != "
                f"{other.part_of_speech} for word {self.word}"
            )

        self.part_of_speech = self.part_of_speech.merge(other.part_of_speech)
        self.article = self.article or other.article

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
            sentence_obj = {
                "english": other.english_sentence,
                "spanish": other.spanish_sentence,
            }
            if other.sentence_source:
                sentence_obj["source"] = other.sentence_source
            example_sentences.append(sentence_obj)
        elif other.spanish_sentence:
            self.spanish_sentence = other.spanish_sentence
            self.english_sentence = other.english_sentence
            self.sentence_source = other.sentence_source

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

    def to_csv_line(self, require_rank: bool = False) -> Sequence[str]:
        if self.example_sentences:
            example_sentences = json.dumps(
                _cattrs_converter().unstructure(self.example_sentences)
            )
        else:
            example_sentences = ""

        if not require_rank:
            rank = self.rank or ""
        elif not self.rank:
            raise ValueError(f"Rank required but not specified for {self}")
        else:
            rank = self.rank

        if self.sentence_source is not None:
            sentence_source = self.sentence_source.to_json()
        else:
            sentence_source = ""

        return (
            rank,
            self.word,
            str(self.part_of_speech),
            self.meaning,
            self.spanish_sentence or "",
            self.english_sentence or "",
            "",  # Frequency
            "",  # Image
            self.article or "",
            sentence_source,  # Citation for sentence
            "",  # Note
            example_sentences,
            self.model_used or "",
            "",
        )

    @classmethod
    def make_csv_header(cls) -> Sequence[str]:
        return (
            "rank",
            "word",
            "part_of_speech",
            "meaning",
            "spanish_sentence",
            "english_sentence",
            "frequency",
            "image",
            "article",
            "sentence_source",
            "note",
            "example_sentences",
            "model_used",
            "tags",
        )

    @classmethod
    def words_from_csv(cls, csv_path: Path | str) -> "Sequence[Self]":
        out = []
        if isinstance(csv_path, str):
            string_obj = io.StringIO(csv_path)
            cm = contextlib.nullcontext(string_obj)
        else:
            cm = open(csv_path, "rt")

        with cm as f:
            reader = csv.DictReader(f)
            for line in reader:
                if not (line["word"] and line["meaning"]):
                    continue

                sentences = []
                spanish_sentence = line["spanish_sentence"]
                if spanish_sentence:
                    english_sentence = line["english_sentence"]
                    if not english_sentence:
                        raise ValueError(
                            f"Spanish sentence exists without english sentence: {line}" +                        )

                    if "sentence_source" in line and line["sentence_source"]:
                        sentence_source = Citation.from_json(line["sentence_source"])
                    else:
                        sentence_source = None
                    sentences.append(
                        Sentence(
                            spanish=spanish_sentence,
                            english=english_sentence,
                            source=sentence_source,
                        )
                    )

                if line["example_sentences"]:
                    example_sentences = json.loads(line["example_sentences"])
                    sentences.extend(map(Sentence.from_dict, example_sentences))

                if "rank" not in line or not line["rank"]:
                    rank: int | None = None
                else:
                    rank = int(line["rank"])

                out.append(
                    cls(
                        rank=rank,
                        word=line["word"],
                        meaning=line["meaning"],
                        part_of_speech=line["part_of_speech"],
                        article=line["article"] or None,
                        sentences=tuple(sentences),
                        model_used=line["model_used"] or None,
                    )
                )

        return out



#####
# Annotation-related types
class CattrsSerializable(Protocol):
    def to_dict(self) -> dict:
        return _cattrs_converter().unstructure(self)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return _cattrs_converter().structure(data, cls)

    @classmethod
    def sequence_from_json(cls, path: Path) -> Sequence[Self]:
        json_dict = json.loads(path.read_text())
        return tuple(map(cls.from_dict, json_dict))

    @classmethod
    def sequence_to_serializable(cls, sequence: Sequence[Self]) -> Sequence[dict]:
        return [obj.to_dict() for obj in sequence]


@attrs.frozen
class BookData:
    title: str
    author: str
    path: Path
    language: str


@attrs.frozen(kw_only=True)
class Annotation:
    text: str
    context: str
    chapter: str
    page_number: int
    note: str | None = None


@attrs.frozen
class CombinedAnnotation:
    highlights: Sequence[str]
    context: str
    chapters: Sequence[str]
    page_numbers: Sequence[int]


@attrs.frozen
class BookAnnotations(CattrsSerializable):
    book: BookData
    annotations: Sequence[Annotation]


@attrs.frozen
class CombinedBookAnnotations(CattrsSerializable):
    book: BookData
    annotations: Sequence[CombinedAnnotation]


@functools.cache
def _cattrs_converter() -> cattrs.Converter:
    converter = cattrs.Converter()
    converter.register_structure_hook(Path, lambda d, _: Path(d))
    converter.register_unstructure_hook(Path, lambda p: str(p))

    converter.register_unstructure_hook_factory(
        lambda cl: cl in (Citation, ModelCitation, BookCitation),
        lambda cl: cattrs.gen.make_dict_unstructure_fn(
            cl, converter, id=cattrs.gen.override(omit=True)
        ),
    )

    converter.register_unstructure_hook(datetime.date, lambda d: d.isoformat())
    converter.register_structure_hook(
        datetime.date, lambda s, _: datetime.date.fromisoformat(s)
    )

    return converter
