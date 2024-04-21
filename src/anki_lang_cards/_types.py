import attrs
from collections.abc import Sequence

@attrs.define
class Word:
    word: str
    meaning: str
    part_of_speech: str
    article: str | None
    spanish_sentence: str | None
    english_sentence: str | None
    example_sentences: Sequence[str]
    model_used: str | None = None
