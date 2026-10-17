"""Tests for building prompt inputs from combined annotations."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import annotations_to_csv, dictionary_entries


def _entry(headword: str, meaning: str) -> dictionary_entries.DictionaryEntry:
    return dictionary_entries.DictionaryEntry(
        headword=headword,
        definitions=(dictionary_entries.Definition(meaning=meaning, pos="n"),),
    )


class FakeEngine:
    """Just enough dictionary for lookup_words."""

    def __init__(
        self, entries: Mapping[str, Sequence[dictionary_entries.DictionaryEntry]]
    ) -> None:
        self.entries = entries

    def lookup(self, word: str) -> Sequence[dictionary_entries.DictionaryEntry]:
        return self.entries.get(word, ())


ENGINE: Final = FakeEngine(
    {
        "zanja": (_entry("zanja", "trench"),),
        "llama": (_entry("llama", "flame"),),
    }
)


def _book(
    highlights: Sequence[str | types.HighlightWithNote],
) -> Sequence[types.CombinedBookAnnotations]:
    return (
        types.CombinedBookAnnotations(
            book=types.BookInfo(
                title="Libro", author="Autora", path=Path("libro.epub"), language="es"
            ),
            annotations=(
                types.CombinedAnnotation(
                    highlights=highlights,
                    context="Saltó la zanja hacia la llama.",
                    chapters=("1",),
                    page_numbers=(3,),
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "highlights",
    [
        ("zanja", "llama"),
        (types.HighlightWithNote(highlight="zanja", note="K"), "llama"),
        (
            types.HighlightWithNote(highlight="zanja", note="K"),
            types.HighlightWithNote(highlight="llama", note="K: fuego"),
        ),
    ],
)
def test_every_highlight_gets_its_dictionary_entries(
    highlights: Sequence[str | types.HighlightWithNote],
) -> None:
    books = _book(highlights)
    word_map = annotations_to_csv.lookup_words(ENGINE, books)
    (prompt_input,) = annotations_to_csv.prepare_prompt_inputs(books, word_map)
    assert set(prompt_input.rag_entries) == {"zanja", "llama"}


def test_highlights_without_entries_are_skipped() -> None:
    books = _book((types.HighlightWithNote(highlight="desconocida", note="K"), "zanja"))
    word_map = annotations_to_csv.lookup_words(ENGINE, books)
    assert set(word_map) == {"zanja"}
    (prompt_input,) = annotations_to_csv.prepare_prompt_inputs(books, word_map)
    assert set(prompt_input.rag_entries) == {"zanja"}
