"""Tests for filtering annotations with a book skip list."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import annotations_to_csv


def _annotation(word: str, chapter: str, page: int) -> types.CombinedAnnotation:
    return types.CombinedAnnotation(
        highlights=[word], context=word, chapters=[chapter], page_numbers=[page]
    )


def _book(title: str) -> types.CombinedBookAnnotations:
    return types.CombinedBookAnnotations(
        book=types.BookInfo(
            title=title, author="Autora", path=Path(f"{title}.epub"), language="es"
        ),
        annotations=[
            _annotation("prólogo", "Prologue", 1),
            _annotation("varita", "Chapter 1", 10),
            _annotation("escoba", "Chapter 2", 20),
            _annotation("gracias", "Acknowledgements", 30),
        ],
    )


def _skip(tmp_path: Path, rules: Sequence[dict[str, Any]]) -> dict[str, Sequence[str]]:
    skip_list = tmp_path / "skip.json"
    skip_list.write_text(json.dumps(rules))
    result = annotations_to_csv.handle_book_skipping(
        skip_list, [_book("Harry Potter"), _book("Otro libro")]
    )
    return {
        book.book.title: [
            highlight
            for annotation in book.annotations
            for highlight in annotation.highlights
        ]
        for book in result
    }


_ALL_WORDS = ["prólogo", "varita", "escoba", "gracias"]


@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        pytest.param([], {"Harry Potter": _ALL_WORDS, "Otro libro": _ALL_WORDS}),
        pytest.param(
            [{"title": "Harry Potter"}],
            {"Otro libro": _ALL_WORDS},
            id="whole-book",
        ),
        pytest.param(
            [
                {
                    "title": {"pattern": "Harry Potter"},
                    "chapters": {"pattern": "Acknowledgements|Prologue"},
                }
            ],
            {"Harry Potter": ["varita", "escoba"], "Otro libro": _ALL_WORDS},
            id="chapters-only",
        ),
        pytest.param(
            [{"title": "Harry Potter", "page_numbers": [{"range": [5, 25]}]}],
            {"Harry Potter": ["prólogo", "gracias"], "Otro libro": _ALL_WORDS},
            id="page-range",
        ),
        pytest.param(
            [{"title": "Harry Potter", "chapters": {"pattern": ".*"}}],
            {"Otro libro": _ALL_WORDS},
            id="all-chapters-drops-book",
        ),
    ],
)
def test_handle_book_skipping(
    tmp_path: Path,
    rules: Sequence[dict[str, Any]],
    expected: dict[str, Sequence[str]],
) -> None:
    assert _skip(tmp_path, rules) == expected
