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


@pytest.mark.parametrize(
    ("mangled", "expected"),
    [
        ("Alice, I didn&#x27;t mean it", "Alice, I didn't mean it"),
        ("Alice, I didn&#39;t mean it", "Alice, I didn't mean it"),
        ("&quot;Hola&quot;, dijo María", '"Hola", dijo María'),
        # Each bad re-import stacks another &amp; layer.
        ("Tom &amp;amp; Jerry", "Tom & Jerry"),
        ("Tom & Jerry", "Tom & Jerry"),
        ("una&nbsp;frase", "una frase"),
        ("uno<br>dos", "uno dos"),
        # A literal < stays escaped so the field remains inert as HTML.
        ("1 &lt; 2", "1 &lt; 2"),
        ("ya limpio", "ya limpio"),
    ],
)
def test_canonicalize_text_field(mangled: str, expected: str) -> None:
    assert anki_import.canonicalize_text_field(mangled) == expected


@hypothesis.given(st.text())
def test_canonicalize_text_field_is_idempotent(text: str) -> None:
    once = anki_import.canonicalize_text_field(text)

    assert anki_import.canonicalize_text_field(once) == once


def test_make_anki_directives() -> None:
    directives = types.Word.make_anki_directives("My::Deck")

    assert "#separator:Comma" in directives
    assert "#html:true" in directives
    assert "#deck:My::Deck" in directives
    columns = next(d for d in directives if d.startswith("#columns:"))
    assert columns == "#columns:" + ",".join(types.ANKI_IMPORT_COLUMNS)

    assert not any(d.startswith("#deck:") for d in types.Word.make_anki_directives())


def test_anki_directives_leave_unwritten_fields_alone() -> None:
    """Anki's importer overwrites every mapped field (and the tags, if a tags
    column is set) of a note it updates, so a column that to_csv_line always
    leaves empty must map onto nothing, or importing the diff wipes the
    deck's Freq, Image, Note and tags."""
    directives = types.Word.make_anki_directives("My::Deck")
    columns = next(d for d in directives if d.startswith("#columns:"))
    names = columns.removeprefix("#columns:").split(",")
    tags_column = next(
        int(d.removeprefix("#tags column:"))
        for d in directives
        if d.startswith("#tags column:")
    )
    # Every attribute of this word is populated, so its empty cells are the
    # ones the pipeline never writes.
    row = _nasty_word().to_csv_line(require_rank=True)
    assert len(names) == len(row)

    unwritten = {name for name, cell in zip(names, row, strict=True) if cell == ""}
    written = {name for name, cell in zip(names, row, strict=True) if cell != ""}
    assert unwritten.isdisjoint(types.ANKI_NOTETYPE_FIELDS)
    assert written <= set(types.ANKI_NOTETYPE_FIELDS)
    assert {"Freq", "Image", "Note"}.isdisjoint(names)
    # 1-based; past the last column means no column is imported as tags.
    assert tags_column > len(names)


def test_words_from_csv_skips_anki_directives() -> None:
    word = _nasty_word()
    output = io.StringIO()
    output.write("\n".join(types.Word.make_anki_directives("My::Deck")) + "\n")
    csv.writer(output).writerow(word.to_csv_line())

    (loaded,) = types.Word.words_from_csv(output.getvalue())

    assert loaded.word == word.word
    assert [s.spanish for s in loaded.sentences] == [s.spanish for s in word.sentences]


def test_canonicalize_extra_sentences_is_idempotent() -> None:
    field = json.dumps(
        [{"spanish": "s & t", "english": '"quoted" <text>', "source": None}]
    )

    once = anki_import.canonicalize_extra_sentences(field)
    twice = anki_import.canonicalize_extra_sentences(once)

    assert once == twice
    assert HTML_ACTIVE.isdisjoint(once)


def test_canonicalize_repairs_entities_inside_valid_json() -> None:
    """Entity damage inside string values leaves the JSON parseable, so it
    survives the structural repairs; the content still needs unescaping."""
    field = json.dumps(
        [
            {
                "spanish": "Una frase.",
                "english": "Alice, I didn&#x27;t mean it",
                "source": {"title": "Tom &amp; Jerry", "source_type": "book"},
            }
        ]
    )
    assert json.loads(field)  # the mangled field parses as-is

    parsed = json.loads(anki_import.canonicalize_extra_sentences(field))

    assert parsed[0]["english"] == "Alice, I didn't mean it"
    assert parsed[0]["source"]["title"] == "Tom & Jerry"


def test_repair_br_inside_string_becomes_newline() -> None:
    """A ``<br>`` inside a string value must become the two-character
    ``\\n`` escape, not a raw newline (an invalid control character)."""
    obj = [{"spanish": "–¿Duele mucho?<br>–Sí, es nefasta.", "english": "Dialogue."}]
    field = json.dumps(obj).replace('"', "&quot;")

    parsed = anki_import.parse_json_field_with_repair(field)

    assert parsed[0]["spanish"] == "–¿Duele mucho?\n–Sí, es nefasta."


def test_repair_br_between_tokens_is_whitespace() -> None:
    """Editing a pretty-printed field in the Anki GUI turns its structural
    newlines into ``<br>``; those must become plain whitespace."""
    obj = {"title": "Todo Arde", "pages": [17]}
    field = json.dumps(obj, indent=2).replace("\n", "<br>").replace('"', "&quot;")

    assert anki_import.parse_json_field_with_repair(field) == obj


def test_repair_br_after_escaped_quote() -> None:
    """An escaped quote must not flip the scanner's in-string state."""
    obj = [{"english": 'She said "no."<br>Then she left.'}]
    field = json.dumps(obj).replace('"', "&quot;")

    parsed = anki_import.parse_json_field_with_repair(field)

    assert parsed[0]["english"] == 'She said "no."\nThen she left.'


def test_repair_br_with_doubled_backslashes() -> None:
    """Doubled-backslash damage makes quote tracking lie until it is
    undone, so the ``<br>`` replacement must run per candidate, after
    each candidate's structural repairs."""
    obj = [{"english": '"What?"<br>A shout.', "spanish": "¿Qué?"}]
    field = json.dumps(obj).replace('\\"', '\\\\"').replace('"', "&quot;")

    parsed = anki_import.parse_json_field_with_repair(field)

    assert parsed[0]["english"] == '"What?"\nA shout.'


def test_canonicalize_sentence_source_repairs_in_string_entities() -> None:
    source = {"source_type": "book", "title": "El se&#xF1;or &quot;X&quot;"}
    field = json.dumps(source)

    parsed = json.loads(anki_import.canonicalize_sentence_source(field))

    assert parsed["title"] == 'El señor "X"'
