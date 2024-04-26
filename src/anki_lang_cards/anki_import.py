import csv
import json
import pathlib
from collections.abc import Sequence

from . import _types as types


def _load_anki_word(row: Sequence[str]) -> types.Word:
    (
        rank,
        word,
        part_of_speech,
        meaning,
        spanish_sentence,
        english_sentence,
        frequency,
        _image,
        _word_audio,
        _spanish_audio,
        _note,
        extra_sentences,
        model_used,
        _tags,
    ) = row

    if extra_sentences:
        extra_sentences = tuple(json.loads(extra_sentences))
    return types.Word(
        rank=int(rank),
        word=word,
        part_of_speech=part_of_speech,
        meaning=meaning,
        article=None,
        spanish_sentence=spanish_sentence or None,
        english_sentence=english_sentence or None,
        example_sentences=extra_sentences or (),
        model_used=model_used,
    )


def load_anki_words(in_path: pathlib.Path) -> Sequence[types.Word]:
    """Loads from an Anki .txt export"""
    out = []
    with open(in_path, "rt") as f:
        # Find the first entry in the tab-separated file
        while True:
            line_start = f.tell()
            line = f.readline()
            if not line.startswith("#"):
                f.seek(line_start)
                break

        reader = csv.reader(f, dialect="excel-tab")
        for row in reader:
            try:
                out.append(_load_anki_word(row))
            except ValueError:
                print(row)
                raise

        return out
