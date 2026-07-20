"""Tests for the HTML-inert JSON serialization of Anki-bound fields.

The JSON-bearing fields (``Sentence Source``, ``Extra Sentences``) live in
Anki note fields, which are rendered as HTML, so the serializer must emit
JSON with no HTML-active characters; and historically-mangled fields (HTML
entities from imports, hand-doubled ``\\"`` escapes from the old
template-literal workaround) must be repairable back to the original data.
"""

import csv
import io
import json
from typing import Any

import hypothesis
import pytest
from hypothesis import strategies as st

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import

HTML_ACTIVE = frozenset("<>&")


@pytest.mark.parametrize(
    "obj",
    [
        {"english": '"What a stumble," he said, laughing.'},
        {"text": "1 < 2 & 4 > 3"},
        {"html": '<div class="x">&amp;</div>'},
        {"spanish": "—Dime algo que no sepa —dijo Ramón."},
        [{"source": {"title": "Libro & Cosas", "pages": [1, 2]}}],
        {"empty": "", "backslash": "a\\b", "newline": "a\nb"},
    ],
)
def test_json_for_anki_field(obj: Any) -> None:
    serialized = types.json_for_anki_field(obj)

    assert HTML_ACTIVE.isdisjoint(serialized)
    assert json.loads(serialized) == obj


@hypothesis.given(st.dictionaries(st.text(), st.text()))
def test_json_for_anki_field_round_trips(obj: dict[str, str]) -> None:
    serialized = types.json_for_anki_field(obj)

    assert HTML_ACTIVE.isdisjoint(serialized)
    assert json.loads(serialized) == obj


def _nasty_word() -> types.Word:
    citation = types.BookCitation(
        title="Libro & Cosas <ilustrado>",
        authors=["A. Autor"],
        pages=[3, 4],
    )
    model_citation = types.ModelCitation(model="fake-cards")
    return types.Word(
        rank=1401,
        word="gato",
        meaning='cat, "male cat"',
        part_of_speech="nm",
        article="el",
        sentences=(
            types.Sentence(
                spanish='La palabra "gato" apareció.',
                english='The word "gato" appeared.',
                source=citation,
            ),
            types.Sentence(
                spanish="Dijo: «1 < 2 & 3 > 2», raro pero válido.",
                english='"One is less than two," she said.',
                source=model_citation,
            ),
        ),
        model_used="fake-cards",
    )


def test_to_csv_line_json_fields_are_inert() -> None:
    word = _nasty_word()
    line = word.to_csv_line()
    header = types.Word.make_csv_header()
    fields = dict(zip(header, line, strict=True))

    for name in ("sentence_source", "example_sentences"):
        assert HTML_ACTIVE.isdisjoint(fields[name]), name
        assert "\n" not in fields[name], name
        json.loads(fields[name])  # must be valid JSON as-is


def test_to_csv_line_round_trips_through_words_from_csv() -> None:
    word = _nasty_word()
    output = io.StringIO()
    csv.writer(output).writerow(word.to_csv_line())

    (loaded,) = types.Word.words_from_csv(output.getvalue())

    assert [(s.spanish, s.english) for s in loaded.sentences] == [
        (s.spanish, s.english) for s in word.sentences
    ]
    # The citation id is a session-local counter and is not round-tripped,
    # so compare the meaningful fields rather than the whole object.
    assert loaded.sentence_source.title == word.sentence_source.title
    assert tuple(loaded.sentence_source.authors) == tuple(word.sentence_source.authors)


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda s: s, id="verbatim"),
        pytest.param(lambda s: s.replace('"', "&quot;"), id="html-escaped"),
        pytest.param(lambda s: s.replace('\\"', '\\\\"'), id="hand-doubled"),
        pytest.param(
            lambda s: s.replace('\\"', '\\\\"').replace('"', "&quot;"),
            id="hand-doubled-then-html-escaped",
        ),
        pytest.param(lambda s: s.replace('\\"', '"'), id="escapes-stripped"),
    ],
)
def test_parse_json_field_with_repair(mangle) -> None:
    obj = [{"english": '"What a stumble," he said.', "spanish": "¡Qué traspié!"}]
    field = mangle(json.dumps(obj))

    assert anki_import.parse_json_field_with_repair(field) == obj


def test_repair_does_not_touch_valid_json() -> None:
    """Destructive repairs only apply to fields that fail to parse."""
    obj = {"empty": "", "content": 'a \\" b'}
    field = json.dumps(obj)

    assert anki_import.parse_json_field_with_repair(field) == obj


def test_repair_failure_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        anki_import.parse_json_field_with_repair("not json at all")


def test_canonicalize_sentence_source() -> None:
    source = {"source_type": "model", "model": "claude-x", "id": 1}
    mangled = json.dumps(source, indent=2).replace('"', "&quot;")

    canonical = anki_import.canonicalize_sentence_source(mangled)

    assert json.loads(canonical) == source
    assert HTML_ACTIVE.isdisjoint(canonical)


def test_canonicalize_extra_sentences_unwraps_nested_source() -> None:
    nested = [
        {
            "spanish": "Una frase.",
            "english": '"A sentence," he said.',
            "source": json.dumps({"source_type": "model", "model": "m", "id": 2}),
        }
    ]
    field = json.dumps(nested).replace('\\"', '\\\\"')

    canonical = anki_import.canonicalize_extra_sentences(field)

    parsed = json.loads(canonical)
    assert parsed[0]["english"] == '"A sentence," he said.'
    assert parsed[0]["source"] == {"source_type": "model", "model": "m", "id": 2}


def test_canonicalize_extra_sentences_is_idempotent() -> None:
    field = json.dumps(
        [{"spanish": "s & t", "english": '"quoted" <text>', "source": None}]
    )

    once = anki_import.canonicalize_extra_sentences(field)
    twice = anki_import.canonicalize_extra_sentences(once)

    assert once == twice
    assert HTML_ACTIVE.isdisjoint(once)
