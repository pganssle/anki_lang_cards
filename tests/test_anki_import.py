"""End-to-end tests of importing pipeline CSVs with Anki's real importer.

These use the ``anki`` Python package (the same Rust-backed importer the
desktop import dialog drives) against a scratch collection, and verify that
the JSON-bearing fields written by ``Word.to_csv_line`` survive the trip into
Anki — or document exactly how they get mangled when the import is
misconfigured. The historical damage repaired by
``parse_json_field_with_repair`` and ``scripts/repair_json_fields.py``
(``&quot;``, ``<br>``, ``&amp;``) is reproduced here by importing with
"allow HTML" disabled.
"""

import csv
import io
import json
import pathlib
from collections.abc import Sequence
from typing import Final

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import

anki_collection = pytest.importorskip(
    "anki.collection", reason="the anki package is required for import tests"
)
from anki.import_export_pb2 import CsvMetadata, ImportCsvRequest  # noqa: E402

pytestmark = pytest.mark.e2e

# Field names as they exist in the real deck's notetype (shared with the
# #columns: directive writer). Note that these are the names the AnkiConnect
# *read* path expects (see NoteFields / word_from_anki_connect_response).
NOTETYPE_FIELDS: Final[Sequence[str]] = types.ANKI_NOTETYPE_FIELDS

CSV_HEADER: Final = types.Word.make_csv_header()


def _nasty_word(rank: int) -> types.Word:
    """A word whose serialized fields contain historically-mangled characters."""
    citation = types.BookCitation(
        title="Libro & Cosas",
        authors=["A. Autor"],
        pages=[3, 4],
        chapters=["Capítulo «uno»"],
    )
    model_citation = types.ModelCitation(model="fake-cards")
    return types.Word(
        rank=rank,
        word="gato",
        meaning='cat, "male cat"',
        part_of_speech="nm",
        article="el",
        sentences=(
            types.Sentence(
                spanish='La palabra "gato" apareció.',
                english='The word "gato" didn\'t appear.',
                source=citation,
            ),
            types.Sentence(
                spanish="Los expertos dicen: «gato» es fundamental.",
                english='They say \\"gato\\" is key.',
                source=model_citation,
            ),
        ),
        model_used="fake-cards",
    )


def _word_csv(words: Sequence[types.Word], *, header: bool = True) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    if header:
        writer.writerow(CSV_HEADER)
    for word in words:
        writer.writerow(word.to_csv_line(require_rank=True))
    return output.getvalue()


@pytest.fixture
def collection(tmp_path: pathlib.Path):
    col = anki_collection.Collection(str(tmp_path / "collection.anki2"))
    notetype = col.models.new("Spanish Vocab Test")
    for name in NOTETYPE_FIELDS:
        col.models.add_field(notetype, col.models.new_field(name))
    template = col.models.new_template("Card 1")
    template["qfmt"] = "{{Word}}"
    template["afmt"] = "{{Definition}}"
    col.models.add_template(notetype, template)
    col.models.add(notetype)
    yield col
    col.close()


def _import_csv(
    col: "anki_collection.Collection",
    csv_path: pathlib.Path,
    deck: str,
    *,
    is_html: bool,
):
    metadata = col.get_csv_metadata(str(csv_path), None)
    metadata.global_notetype.id = col.models.by_name("Spanish Vocab Test")["id"]
    # CSV columns 1..13 map onto the notetype fields; the last column is tags.
    metadata.global_notetype.field_columns[:] = list(range(1, len(NOTETYPE_FIELDS) + 1))
    metadata.tags_column = len(NOTETYPE_FIELDS) + 1
    metadata.deck_id = col.decks.id(deck)
    metadata.delimiter = CsvMetadata.Delimiter.COMMA
    metadata.force_delimiter = True
    metadata.is_html = is_html
    metadata.force_is_html = True
    return col.import_csv(ImportCsvRequest(path=str(csv_path), metadata=metadata))


def _notes_by_word(
    col: "anki_collection.Collection", deck: str
) -> dict[str, dict[str, str]]:
    notes = {}
    for note_id in col.find_notes(f'deck:"{deck}"'):
        note = col.get_note(note_id)
        fields = dict(zip(NOTETYPE_FIELDS, note.fields, strict=False))
        notes[fields["Word"]] = fields
    return notes


def test_import_with_html_enabled_is_faithful(
    collection, tmp_path: pathlib.Path
) -> None:
    """With "allow HTML" enabled, every field survives byte-for-byte."""
    word = _nasty_word(rank=1401)
    csv_path = tmp_path / "words.csv"
    csv_path.write_text(_word_csv([word], header=False), encoding="utf-8")

    log = _import_csv(collection, csv_path, "html-on", is_html=True)
    assert len(log.log.new) == 1

    fields = _notes_by_word(collection, "html-on")["gato"]
    expected = dict(zip(NOTETYPE_FIELDS, word.to_csv_line(require_rank=True)))
    for name in NOTETYPE_FIELDS:
        assert fields[name] == str(expected[name]), f"field {name!r} mangled"

    # The JSON-bearing fields parse directly, with no repair step needed.
    source = json.loads(fields["Sentence Source"])
    assert source["title"] == "Libro & Cosas"
    extra = json.loads(fields["Extra Sentences"])
    assert extra[0]["spanish"] == "Los expertos dicen: «gato» es fundamental."
    assert extra[0]["english"] == 'They say \\"gato\\" is key.'
    assert isinstance(extra[0]["source"], dict)


def test_import_with_html_disabled_mangles_json(
    collection, tmp_path: pathlib.Path
) -> None:
    """With "allow HTML" disabled the importer HTML-escapes the fields.

    Because the serializer writes ``<``, ``>`` and ``&`` as ``\\uXXXX``
    escapes, the only remaining damage is quotes becoming &quot; — which
    the repair chain undoes losslessly. Imports should still use "allow
    HTML", but a misconfigured import is no longer permanently corrupting.
    """
    word = _nasty_word(rank=1402)
    csv_path = tmp_path / "words.csv"
    csv_path.write_text(_word_csv([word], header=False), encoding="utf-8")

    _import_csv(collection, csv_path, "html-off", is_html=False)
    fields = _notes_by_word(collection, "html-off")["gato"]

    assert "&quot;" in fields["Extra Sentences"]
    assert "&quot;" in fields["Sentence Source"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(fields["Sentence Source"])

    source = anki_import.parse_json_field_with_repair(fields["Sentence Source"])
    assert source["title"] == "Libro & Cosas"
    extra = anki_import.parse_json_field_with_repair(fields["Extra Sentences"])
    assert extra[0]["english"] == 'They say \\"gato\\" is key.'


def test_header_row_imports_as_junk_note(collection, tmp_path: pathlib.Path) -> None:
    """Documents that importing the pipeline CSV as-is turns the header row
    into a junk note: Anki's importer has no header-skipping option."""
    csv_path = tmp_path / "words.csv"
    csv_path.write_text(_word_csv([_nasty_word(rank=1403)]), encoding="utf-8")

    log = _import_csv(collection, csv_path, "with-header", is_html=True)
    assert len(log.log.new) == 2  # the header note and the real note

    notes = _notes_by_word(collection, "with-header")
    assert set(notes) == {"word", "gato"}
    assert notes["word"]["Rank"] == "rank"


def test_directive_header_imports_cleanly(collection, tmp_path: pathlib.Path) -> None:
    """Prefixing the CSV with Anki file directives instead of a header row
    imports only the real notes, with no junk and no field mapping needed."""
    word = _nasty_word(rank=1404)
    # The #columns: names must match the notetype's field names for the
    # importer to map them automatically.
    directives = (
        "#separator:Comma\n"
        "#html:true\n"
        "#notetype:Spanish Vocab Test\n"
        "#deck:directives\n"
        f"#columns:{','.join(NOTETYPE_FIELDS)},tags\n"
        f"#tags column:{len(NOTETYPE_FIELDS) + 1}\n"
    )
    csv_path = tmp_path / "words.csv"
    csv_path.write_text(directives + _word_csv([word], header=False), encoding="utf-8")

    metadata = collection.get_csv_metadata(str(csv_path), None)
    log = collection.import_csv(ImportCsvRequest(path=str(csv_path), metadata=metadata))
    assert len(log.log.new) == 1
    fields = _notes_by_word(collection, "directives")["gato"]
    assert fields["Rank"] == "1404"
    assert json.loads(fields["Sentence Source"])["title"] == "Libro & Cosas"


def test_make_anki_directives_import_is_faithful(
    collection, tmp_path: pathlib.Path
) -> None:
    """A CSV prefixed with Word.make_anki_directives imports byte-for-byte,
    with no junk note and no manual importer configuration beyond the
    notetype — in particular #html:true prevents the entity mangling
    (&quot;, &#x27;) that a misconfigured GUI import applies."""
    word = _nasty_word(rank=1406)
    directives = "\n".join(
        (
            *types.Word.make_anki_directives("generated-directives"),
            "#notetype:Spanish Vocab Test",
        )
    )
    csv_path = tmp_path / "words.csv"
    csv_path.write_text(
        directives + "\n" + _word_csv([word], header=False), encoding="utf-8"
    )

    metadata = collection.get_csv_metadata(str(csv_path), None)
    log = collection.import_csv(ImportCsvRequest(path=str(csv_path), metadata=metadata))
    assert len(log.log.new) == 1

    fields = _notes_by_word(collection, "generated-directives")["gato"]
    expected = dict(zip(NOTETYPE_FIELDS, word.to_csv_line(require_rank=True)))
    for name in NOTETYPE_FIELDS:
        assert fields[name] == str(expected[name]), f"field {name!r} mangled"
    assert "&quot;" not in fields["Definition"]
    assert "&#" not in fields["English"]


def test_update_keeps_unmanaged_fields_and_tags(
    collection, tmp_path: pathlib.Path
) -> None:
    """Updating an existing note through the generated directives leaves
    the fields the pipeline never fills in (Freq, Image, Note) and the
    note's tags alone, even though the CSV carries empty columns for them.

    The #tags column: directive points one past the last column, which is
    undocumented importer behaviour, so this guards against an anki bump
    quietly going back to wiping tags.
    """
    seeded = {
        "Rank": "1407",
        "Word": "gato",
        "Definition": "stale definition",
        "Freq": "99",
        "Image": '<img src="gato.png">',
        "Note": "my note",
    }
    note = collection.new_note(collection.models.by_name("Spanish Vocab Test"))
    for index, name in enumerate(NOTETYPE_FIELDS):
        note.fields[index] = seeded.get(name, "")
    note.tags = ["keep"]
    collection.add_note(note, collection.decks.id("update"))

    word = _nasty_word(rank=1407)
    directives = "\n".join(
        (*types.Word.make_anki_directives("update"), "#notetype:Spanish Vocab Test")
    )
    csv_path = tmp_path / "words.csv"
    csv_path.write_text(
        directives + "\n" + _word_csv([word], header=False), encoding="utf-8"
    )

    metadata = collection.get_csv_metadata(str(csv_path), None)
    log = collection.import_csv(ImportCsvRequest(path=str(csv_path), metadata=metadata))
    assert not log.log.new
    assert len(log.log.first_field_match) == 1

    updated = collection.get_note(note.id)
    fields = dict(zip(NOTETYPE_FIELDS, updated.fields, strict=True))
    assert fields["Definition"] == word.meaning
    for name in ("Freq", "Image", "Note"):
        assert fields[name] == seeded[name], f"field {name!r} blanked"
    assert updated.tags == ["keep"]


def test_reimport_is_idempotent(collection, tmp_path: pathlib.Path) -> None:
    """Importing the same file twice must not double-escape anything."""
    word = _nasty_word(rank=1405)
    csv_path = tmp_path / "words.csv"
    csv_path.write_text(_word_csv([word], header=False), encoding="utf-8")

    _import_csv(collection, csv_path, "reimport", is_html=True)
    first = _notes_by_word(collection, "reimport")

    log = _import_csv(collection, csv_path, "reimport", is_html=True)
    assert not log.log.new
    second = _notes_by_word(collection, "reimport")
    assert second == first
