"""End-to-end tests against a real Anki instance running AnkiConnect.

A real Anki desktop process is started under Xvfb with the AnkiConnect addon
installed (the ``ANKI_CONNECT_PLUGIN`` environment variable points at the
addon source), and the tests drive it over the same HTTP API the production
tooling uses. The focus is field fidelity: JSON-bearing fields must survive
the add/read/update paths byte-for-byte, with no HTML escaping and no double
escaping.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Final

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import

ANKI_CONNECT_URL: Final = "http://127.0.0.1:8765"
DECK: Final = "Import Test"
MODEL: Final = "Spanish Vocab Test"

# Field names as the AnkiConnect read path (word_from_anki_connect_response)
# expects them.
NOTETYPE_FIELDS: Final[Sequence[str]] = (
    "Rank",
    "Word",
    "Part-of-Speech",
    "Definition",
    "Spanish",
    "English",
    "Freq",
    "Image",
    "Article",
    "Sentence Source",
    "Note",
    "Extra Sentences",
    "model_used",
)

_MISSING_REQUIREMENTS = [
    requirement
    for requirement, present in (
        ("the anki executable", shutil.which("anki") is not None),
        ("xvfb-run", shutil.which("xvfb-run") is not None),
        ("ANKI_CONNECT_PLUGIN", bool(os.environ.get("ANKI_CONNECT_PLUGIN"))),
    )
    if not present
]

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        bool(_MISSING_REQUIREMENTS),
        reason=f"Missing: {', '.join(_MISSING_REQUIREMENTS)}",
    ),
]


def _invoke(action: str, **params: Any) -> Any:
    payload = json.dumps({"action": action, "version": 6, "params": params})
    # The URL is the fixed local AnkiConnect endpoint.
    request = urllib.request.Request(  # noqa: S310
        ANKI_CONNECT_URL,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        result = json.loads(response.read())
    if result.get("error"):
        raise RuntimeError(f"AnkiConnect error: {result['error']}")
    return result["result"]


# Pre-create the profile so that Anki's first-run language dialog (which
# would block startup, since nothing can click it) never appears.
_PROFILE_BOOTSTRAP: Final = (
    "import sys\n"
    "from pathlib import Path\n"
    "from aqt.profiles import ProfileManager\n"
    "pm = ProfileManager(base=Path(sys.argv[1]))\n"
    "pm.setupMeta()\n"
    "pm.create('User 1')\n"
    "pm.setLang('en_US')\n"
)


@pytest.fixture(scope="session")
def anki_instance(tmp_path_factory: pytest.TempPathFactory) -> Any:
    base = tmp_path_factory.mktemp("anki-base")
    addon_target = base / "addons21" / "ankiconnect"
    addon_target.parent.mkdir(parents=True)
    shutil.copytree(os.environ["ANKI_CONNECT_PLUGIN"], addon_target)
    subprocess.run(
        (sys.executable, "-c", _PROFILE_BOOTSTRAP, str(base)),
        check=True,
        capture_output=True,
    )

    log_path = base / "anki.log"
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            ("xvfb-run", "-a", "anki", "-b", str(base)),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    pytest.fail(
                        "Anki exited during startup:\n"
                        + log_path.read_text(errors="replace")
                    )
                try:
                    _invoke("version")
                    break
                except (urllib.error.URLError, ConnectionError, OSError):
                    if time.monotonic() > deadline:
                        pytest.fail(
                            "AnkiConnect did not come up:\n"
                            + log_path.read_text(errors="replace")
                        )
                    time.sleep(1)

            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


@pytest.fixture(scope="session")
def anki_setup(anki_instance: Any) -> None:
    if MODEL not in _invoke("modelNames"):
        _invoke(
            "createModel",
            modelName=MODEL,
            inOrderFields=list(NOTETYPE_FIELDS),
            cardTemplates=[
                {"Name": "Card 1", "Front": "{{Word}}", "Back": "{{Definition}}"}
            ],
        )
    _invoke("createDeck", deck=DECK)


def _nasty_word(rank: int, word: str = "gato") -> types.Word:
    citation = types.BookCitation(
        title="Libro & Cosas",
        authors=["A. Autor"],
        pages=[3, 4],
        chapters=["Capítulo «uno»"],
    )
    return types.Word(
        rank=rank,
        word=word,
        meaning='cat, "male cat"',
        part_of_speech="nm",
        article="el",
        sentences=(
            types.Sentence(
                spanish=f'La palabra "{word}" apareció.',
                english=f'The word "{word}" appeared.',
                source=citation,
            ),
            types.Sentence(
                spanish=f"Los expertos dicen: «{word}» es fundamental.",
                english=f'They say \\"{word}\\" is key.',
                source=types.ModelCitation(model="fake-cards"),
            ),
        ),
        model_used="fake-cards",
    )


def _note_fields(word: types.Word) -> Mapping[str, str]:
    line = word.to_csv_line(require_rank=True)
    return dict(zip(NOTETYPE_FIELDS, map(str, line), strict=False))


def _add_note(word: types.Word) -> int:
    note_id = _invoke(
        "addNote",
        note={
            "deckName": DECK,
            "modelName": MODEL,
            "fields": _note_fields(word),
            "options": {"allowDuplicate": True},
        },
    )
    # At least with this Anki/AnkiConnect combination, addNote ignores
    # deckName and files the card in the current deck, so move it explicitly.
    _invoke("changeDeck", cards=_invoke("findCards", query=f"nid:{note_id}"), deck=DECK)
    return note_id


def _fields_of(note_id: int) -> Mapping[str, str]:
    (info,) = _invoke("notesInfo", notes=[note_id])
    return {name: field["value"] for name, field in info["fields"].items()}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_added_fields_stored_verbatim(anki_setup: None) -> None:
    """addNote stores every field byte-for-byte: no HTML escaping."""
    word = _nasty_word(2001, word="verbatim")
    note_id = _add_note(word)

    stored = _fields_of(note_id)
    for name, value in _note_fields(word).items():
        assert stored[name] == value, f"field {name!r} was mangled"

    # The JSON fields parse directly.
    assert json.loads(stored["Sentence Source"])["title"] == "Libro & Cosas"
    extra = json.loads(stored["Extra Sentences"])
    assert extra[0]["spanish"] == "Los expertos dicen: «verbatim» es fundamental."


def test_word_roundtrip_through_client(anki_setup: None) -> None:
    """A Word pushed through addNote comes back equal through the client."""
    word = _nasty_word(2002, word="redondo")
    _add_note(word)

    async def retrieve() -> Mapping[int, types.Word]:
        async with anki_import.create_client() as client:
            return await client.retrieve_matching_words(DECK, [word])

    matches = [w for w in _run(retrieve()).values() if w.word == "redondo"]
    (retrieved,) = matches

    assert retrieved == word  # Compares word, meaning and part of speech
    assert retrieved.rank == word.rank
    assert retrieved.article == word.article
    assert [s.spanish for s in retrieved.sentences] == [
        s.spanish for s in word.sentences
    ]
    assert [s.english for s in retrieved.sentences] == [
        s.english for s in word.sentences
    ]
    assert isinstance(retrieved.sentences[0].source, types.BookCitation)
    assert retrieved.sentences[0].source.title == "Libro & Cosas"
    assert isinstance(retrieved.sentences[1].source, types.ModelCitation)


def test_update_note_fields_is_idempotent(anki_setup: None) -> None:
    """Updating a JSON-bearing field twice must not escalate escaping."""
    word = _nasty_word(2003, word="doble")
    note_id = _add_note(word)

    new_extra = json.dumps(
        [
            {
                "spanish": 'Dijo: "doble" y «doble».',
                "english": 'He said "doble" & more.',
                "source": {"model": "fake-cards", "source_type": "model"},
            }
        ],
        ensure_ascii=False,
    )

    async def update() -> None:
        async with anki_import.create_client() as client:
            await client.update_note_info(
                {"id": note_id, "fields": {"Extra Sentences": new_extra}}
            )

    _run(update())
    after_first = _fields_of(note_id)["Extra Sentences"]
    assert after_first == new_extra

    _run(update())
    after_second = _fields_of(note_id)["Extra Sentences"]
    assert after_second == new_extra


@pytest.mark.xfail(
    strict=True,
    reason="_word_to_note_data writes a 'Part of Speech' field but the "
    "notetype (and the read path) use 'Part-of-Speech'; AnkiConnect "
    "silently ignores unknown fields, so the part of speech never updates.",
)
def test_update_from_word_updates_part_of_speech(anki_setup: None) -> None:
    """Updating a note from a Word updates its part of speech."""
    word = _nasty_word(2004, word="renombre")
    note_id = _add_note(word)

    replacement = types.Word(
        rank=2004,
        word="renombre",
        meaning="a corrected meaning",
        part_of_speech="adj",
        article=None,
        sentences=word.sentences,
        model_used="fake-cards",
    )

    async def update() -> None:
        async with anki_import.create_client() as client:
            await client.update_note_info_from_words({note_id: replacement})

    _run(update())
    fields = _fields_of(note_id)
    assert fields["Definition"] == "a corrected meaning"
    assert fields["Part-of-Speech"] == "adj"


@pytest.mark.xfail(
    strict=True,
    reason="The reader's blanket rewrite of doubled quotes corrupts any "
    "legitimately empty JSON string, so the extra sentences are silently "
    "dropped. Valid JSON should be parsed as-is, with repair only as a "
    "fallback.",
)
def test_empty_json_string_sentences_preserved(anki_setup: None) -> None:
    """A stored sentence with an empty translation survives the read path."""
    word = _nasty_word(2005, word="vacio")
    note_id = _add_note(word)
    empty_english = json.dumps(
        [{"spanish": "Sin traducción.", "english": ""}], ensure_ascii=False
    )
    _invoke(
        "updateNoteFields",
        note={"id": note_id, "fields": {"Extra Sentences": empty_english}},
    )
    assert _fields_of(note_id)["Extra Sentences"] == empty_english

    async def retrieve() -> Mapping[int, types.Word]:
        async with anki_import.create_client() as client:
            return await client.retrieve_matching_words(DECK, [word])

    (retrieved,) = [w for w in _run(retrieve()).values() if w.word == "vacio"]
    assert len(retrieved.sentences) == 2
    assert retrieved.sentences[1].spanish == "Sin traducción."
    assert retrieved.sentences[1].english == ""


def test_find_highest_rank_against_real_search(anki_setup: None) -> None:
    """The Rank:re: queries generated by find_highest_rank work against
    Anki's real search engine."""
    for rank in range(2101, 2113):
        _add_note(_nasty_word(rank, word=f"rango{rank}"))

    async def find() -> int:
        async with anki_import.create_client() as client:
            return await client.find_highest_rank(DECK, max_range=50)

    assert _run(find()) == 2112
