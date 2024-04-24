import attrs
from collections.abc import Sequence
from typing import NewType

Url = NewType("Url", str)

@attrs.define
class Word:
    rank: int | None
    word: str
    meaning: str
    part_of_speech: str
    article: str | None
    spanish_sentence: str | None
    english_sentence: str | None
    example_sentences: Sequence[str]
    model_used: str | None = None
