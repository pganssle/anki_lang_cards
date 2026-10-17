"""Tests for the review helper behind the full-run harness skill."""

import contextlib
import csv
import importlib.util
import json
import pathlib
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import

SCRIPT: Final = (
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "review_words.py"
)
BOOK: Final = types.BookCitation(
    title="Libro", authors=("Autora",), pages=(12, 13), id=1
)
MODEL: Final = types.ModelCitation(model="claude-sonnet-5")


@pytest.fixture(scope="module")
def script() -> Any:
    spec = importlib.util.spec_from_file_location("review_words_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sentence(spanish: str, source: types.Citation | None = MODEL) -> types.Sentence:
    return types.Sentence(spanish=spanish, english=f"<{spanish}>", source=source)


def make_word(
    word: str,
    *,
    rank: int | None = None,
    meaning: str = "a meaning",
    part_of_speech: str = "nm",
    sentences: Sequence[types.Sentence] = (),
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article="el" if part_of_speech == "nm" else None,
        sentences=tuple(sentences),
    )


WORDS: Final = (
    make_word("gato", sentences=(sentence("El gato.", BOOK), sentence("Un gato."))),
    make_word("despedir", part_of_speech="vtr", sentences=(sentence("Lo despidió."),)),
    make_word("despedirse", part_of_speech="vi"),
    make_word("seco/a", part_of_speech="adj"),
)


def write_words(
    path: pathlib.Path, words: Sequence[types.Word], *, directives: bool = False
) -> pathlib.Path:
    with path.open("w", newline="") as f:
        if directives:
            f.write("\n".join(types.Word.make_anki_directives("Deck")) + "\n")
        writer = csv.writer(f)
        writer.writerows(word.to_csv_line() for word in words)
    return path


#####
# Recognizing probable duplicates


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("apañárselas", "apañarselas"),
        ("despedir", "despedirse"),
        ("seco/a", "seco"),
        ("sortear (obstáculo)", "sortear"),
        ("Gato", "gato"),
        ("seguir el rollo", "seguirse el rollo"),
        ("gatos", "gato"),
        ("comillas", "comilla"),
        ("ingleses", "inglés"),
    ],
)
def test_fold_keys_match(script: Any, first: str, second: str) -> None:
    assert script.fold_keys(first) & script.fold_keys(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("gato", "gata"),
        ("pase", "pasar"),
        ("casa", "cosa"),
        ("mes", "me"),
        ("gatos negros", "gato negro"),
    ],
)
def test_fold_keys_differ(script: Any, first: str, second: str) -> None:
    assert not script.fold_keys(first) & script.fold_keys(second)


def test_export_rows_points_out_duplicates(script: Any) -> None:
    deck = (
        make_word("gato", rank=100),
        make_word("despedirse", rank=200, part_of_speech="vi"),
        make_word("perro", rank=300),
    )

    rows = script.export_rows(WORDS, deck)

    assert [row["row"] for row in rows] == [0, 1, 2, 3]
    assert rows[1]["csv_matches"] == [2]
    assert rows[2]["csv_matches"] == [1]
    assert rows[0]["csv_matches"] == []
    assert [match["rank"] for match in rows[0]["deck_matches"]] == [100]
    assert [match["word"] for match in rows[1]["deck_matches"]] == ["despedirse"]
    assert rows[3]["deck_matches"] == []


def test_export_rows_ignores_the_note_it_updates(script: Any) -> None:
    """In a diff, a row updating a deck note has that note's rank."""
    (row,) = script.export_rows(
        [make_word("gato", rank=100)], [make_word("gato", rank=100)]
    )

    assert row["deck_matches"] == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (BOOK, "book: Libro, p. 12-13"),
        (
            types.BookCitation(title="Libro", authors=(), pages=(20,), id=3),
            "book: Libro, p. 20",
        ),
        (types.BookCitation(title="Libro", authors=(), id=2), "book: Libro"),
        (MODEL, "model: claude-sonnet-5"),
        (None, None),
    ],
)
def test_export_rows_describes_sources(
    script: Any, source: types.Citation | None, expected: str | None
) -> None:
    (row,) = script.export_rows(
        [make_word("gato", sentences=(sentence("Gato.", source),))], ()
    )

    assert row["sentences"] == [
        {"index": 0, "spanish": "Gato.", "english": "<Gato.>", "source": expected}
    ]


#####
# Applying corrections


def apply(
    script: Any,
    corrections: Sequence[Mapping[str, Any]],
    words: Sequence[types.Word] = WORDS,
) -> tuple[Sequence[types.Word], Sequence[Any]]:
    return script.apply_corrections(words, corrections, "claude-opus-5-5")


def test_set_fields(script: Any) -> None:
    revised, (change,) = apply(
        script,
        [
            {
                "row": 1,
                "word": "despedir",
                "reason": "reflexive in context",
                "set": {
                    "word": "despedirse",
                    "part_of_speech": "vi",
                    "meaning": "to say goodbye",
                },
            }
        ],
    )

    assert revised[1].word == "despedirse"
    assert str(revised[1].part_of_speech) == "vi"
    assert revised[1].meaning == "to say goodbye"
    assert revised[1].sentences == WORDS[1].sentences
    assert "word: despedir -> despedirse" in change.notes


def test_drop_and_rank_lookup(script: Any) -> None:
    words = [make_word("gato", rank=7), make_word("perro", rank=8)]

    revised, (change,) = apply(
        script,
        [{"rank": 8, "word": "perro", "reason": "a coinage", "drop": True}],
        words,
    )

    assert [word.word for word in revised] == ["gato"]
    assert change.result is None


def test_sentence_edits(script: Any) -> None:
    (gato, *rest), _ = apply(
        script,
        [
            {
                "row": 0,
                "word": "gato",
                "reason": "weak sentences",
                "sentences": [
                    {"index": 1, "remove": True},
                    {"index": 0, "spanish": "El gato duerme."},
                    {
                        "add": True,
                        "spanish": "Mi gato ronronea.",
                        "english": "My cat purrs.",
                    },
                    {
                        "add": True,
                        "first": True,
                        "spanish": "Tengo un gato.",
                        "english": "I have a cat.",
                    },
                ],
            }
        ],
    )

    assert [s.spanish for s in gato.sentences] == [
        "Tengo un gato.",
        "El gato duerme.",
        "Mi gato ronronea.",
    ]
    # An edit keeps the original source; a new sentence is the reviewer's.
    assert gato.sentences[1].source == BOOK
    assert gato.sentences[1].english == "<El gato.>"
    added = gato.sentences[0].source
    assert isinstance(added, types.ModelCitation)
    assert added.model == "claude-opus-5-5"
    assert rest == list(WORDS[1:])


@pytest.mark.parametrize(
    ("correction", "problem"),
    [
        ({"word": "gato", "reason": "x"}, "exactly one of 'row' and 'rank'"),
        ({"row": 0, "rank": 1, "word": "gato", "reason": "x"}, "exactly one"),
        ({"row": 99, "word": "gato", "reason": "x"}, "there is no row 99"),
        ({"rank": 99, "word": "gato", "reason": "x"}, "no row with rank 99"),
        ({"row": 0, "word": "perro", "reason": "x"}, "different version"),
        ({"row": 0, "word": "gato"}, "give a reason"),
        (
            {"row": 0, "word": "gato", "reason": "x", "set": {"tags": "a"}},
            "cannot set ['tags']",
        ),
        (
            {"row": 0, "word": "gato", "reason": "x", "set": {"part_of_speech": "zz"}},
            "correction 0",
        ),
        (
            {
                "row": 0,
                "word": "gato",
                "reason": "x",
                "sentences": [{"index": 5, "remove": True}],
            },
            "there is no sentence 5",
        ),
        (
            {"row": 0, "word": "gato", "reason": "x", "sentences": [{"index": 0}]},
            "has no change to make",
        ),
        (
            {
                "row": 0,
                "word": "gato",
                "reason": "x",
                "sentences": [{"add": True, "spanish": "Sí."}],
            },
            "needs spanish and english",
        ),
        (
            {
                "row": 0,
                "word": "gato",
                "reason": "x",
                "sentences": [
                    {"index": 0, "remove": True},
                    {"index": 0, "remove": True},
                ],
            },
            "already removed",
        ),
    ],
)
def test_bad_corrections(
    script: Any, correction: Mapping[str, Any], problem: str
) -> None:
    with pytest.raises(script.CorrectionError) as error:
        apply(script, [correction])

    assert any(problem in line for line in error.value.problems)


def test_every_problem_is_reported(script: Any) -> None:
    with pytest.raises(script.CorrectionError) as error:
        apply(
            script,
            [
                {"row": 0, "word": "gato", "reason": "x", "drop": True},
                {"row": 0, "word": "gato", "reason": "again", "drop": True},
                {"row": 1, "word": "wrong", "reason": "x"},
            ],
        )

    assert len(error.value.problems) == 2
    assert "already dropped" in error.value.problems[0]


#####
# The command line


def run(script: Any, *argv: object) -> int:
    return script.main([str(arg) for arg in argv])


def rows(path: pathlib.Path) -> Sequence[Sequence[str]]:
    with path.open(newline="") as f:
        return list(csv.reader(line for line in f if not line.startswith("#")))


@pytest.mark.parametrize("directives", [False, True])
def test_export_then_apply(
    script: Any, tmp_path: pathlib.Path, directives: bool
) -> None:
    source = write_words(tmp_path / "words.csv", WORDS, directives=directives)
    chunks = tmp_path / "chunks"

    assert run(script, "export", source, "--out-dir", chunks, "--chunk-size", 3) == 0

    index = json.loads((chunks / "index.json").read_text())
    assert index["chunks"] == ["chunk-001.json", "chunk-002.json"]
    first = json.loads((chunks / "chunk-001.json").read_text())
    assert [row["word"] for row in first] == ["gato", "despedir", "despedirse"]

    corrections = tmp_path / "corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "row": 3,
                    "word": "seco/a",
                    "reason": "meaning",
                    "set": {"meaning": "dry"},
                }
            ]
        )
    )
    out = tmp_path / "reviewed.csv"
    changelog = tmp_path / "changelog.md"
    code = run(
        script,
        "apply",
        source,
        corrections,
        "--out",
        out,
        "--changelog",
        changelog,
        "--reviewer-model",
        "claude-opus-5-5",
        "--index",
        chunks / "index.json",
    )

    assert code == 0
    assert out.read_text().startswith("#separator:Comma") is directives
    revised = types.Word.words_from_csv(out)
    assert revised[3].meaning == "dry"
    assert "`seco/a` (unranked): meaning" in changelog.read_text()
    assert "meaning: a meaning -> dry" in changelog.read_text()


def test_apply_with_no_corrections_changes_nothing(
    script: Any, tmp_path: pathlib.Path
) -> None:
    source = write_words(tmp_path / "words.csv", WORDS, directives=True)
    corrections = tmp_path / "none.json"
    corrections.write_text("[]")
    out = tmp_path / "reviewed.csv"

    run(script, "apply", source, corrections, "--out", out, "--reviewer-model", "m")

    assert out.read_text() == source.read_text()


@given(
    st.lists(
        st.text(alphabet=st.characters(categories=["L", "Zs", "P"]), min_size=1).filter(
            str.strip
        ),
        min_size=1,
        max_size=5,
    )
)
def test_round_trip_keeps_sentences(script: Any, spanish: Sequence[str]) -> None:
    """Whatever the sentences contain, an untouched row comes back intact."""
    word = make_word("gato", sentences=[sentence(text.strip()) for text in spanish])

    (revised,), _ = apply(script, [], [word])

    assert [s.spanish for s in revised.sentences] == [s.spanish for s in word.sentences]


def test_header_row_is_kept(script: Any, tmp_path: pathlib.Path) -> None:
    source = tmp_path / "words.csv"
    with source.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(types.Word.make_csv_header())
        writer.writerows(word.to_csv_line() for word in WORDS)
    corrections = tmp_path / "none.json"
    corrections.write_text("[]")
    out = tmp_path / "reviewed.csv"

    run(script, "apply", source, corrections, "--out", out, "--reviewer-model", "m")

    assert rows(out)[0] == list(types.Word.make_csv_header())


def test_apply_refuses_a_changed_file(
    script: Any, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_words(tmp_path / "words.csv", WORDS)
    run(script, "export", source, "--out-dir", tmp_path / "chunks")
    write_words(source, WORDS[:2])
    corrections = tmp_path / "none.json"
    corrections.write_text("[]")

    code = run(
        script,
        "apply",
        source,
        corrections,
        "--out",
        tmp_path / "out.csv",
        "--reviewer-model",
        "m",
        "--index",
        tmp_path / "chunks" / "index.json",
    )

    assert code == 2
    assert "has changed since it was exported" in capsys.readouterr().err
    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ('{"row": 0}', "expected a JSON list"),
        ('[{"row": 0, "word": "perro", "reason": "x"}]', "different version"),
    ],
)
def test_apply_reports_problems(
    script: Any,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    problem: str,
) -> None:
    source = write_words(tmp_path / "words.csv", WORDS)
    corrections = tmp_path / "bad.json"
    corrections.write_text(content)

    code = run(
        script,
        "apply",
        source,
        corrections,
        "--out",
        tmp_path / "out.csv",
        "--reviewer-model",
        "m",
    )

    assert code == 2
    assert problem in capsys.readouterr().err
    assert not (tmp_path / "out.csv").exists()


def test_export_with_an_exported_deck(
    script: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deck_file = tmp_path / "deck.txt"
    deck_file.write_text("")
    monkeypatch.setattr(
        anki_import, "load_anki_words", lambda path: [make_word("gato", rank=100)]
    )
    source = write_words(tmp_path / "words.csv", WORDS)

    run(
        script,
        "export",
        source,
        "--out-dir",
        tmp_path / "chunks",
        "--exported-deck",
        deck_file,
    )

    first, *_ = json.loads((tmp_path / "chunks" / "chunk-001.json").read_text())
    assert [match["rank"] for match in first["deck_matches"]] == [100]


class FakeAnkiClient:
    """Enough of Anki Connect for the export, with Anki's exact-match search.

    A ``Word:`` search term only finds that spelling, so a fake that handed
    back whatever it liked would not notice an export asking Anki for the
    words under review instead of fetching the deck.
    """

    def __init__(self, deck: Mapping[int, types.Word]):
        self.deck = dict(deck)
        self.queries: list[tuple[str, str | None]] = []

    async def get_note_ids(
        self, deck_name: str, query: str | None = None
    ) -> Sequence[int]:
        self.queries.append((deck_name, query))
        return [
            note_id
            for note_id, word in self.deck.items()
            if query is None or "Word:" not in query or f'"Word:{word.word}"' in query
        ]

    async def notes_mod_times(self, note_ids: Sequence[int]) -> Mapping[int, int]:
        return dict.fromkeys(note_ids, 1)

    async def get_anki_notes(
        self, note_ids: Sequence[int], batch_size: int = 1
    ) -> AsyncIterator[Mapping[str, Any]]:
        for note_id in note_ids:
            yield {"noteId": note_id}

    def word_from_anki_connect_response(self, note: Mapping[str, Any]) -> types.Word:
        return self.deck[note["noteId"]]


def test_export_with_the_live_deck(
    script: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeAnkiClient(
        {
            1: make_word("gatos", rank=100),
            2: make_word("despedirse", rank=200, part_of_speech="vi"),
        }
    )

    @contextlib.asynccontextmanager
    async def create_client() -> AsyncIterator[FakeAnkiClient]:
        yield client

    monkeypatch.setattr(anki_import, "create_client", create_client)
    monkeypatch.chdir(tmp_path)
    source = write_words(tmp_path / "words.csv", WORDS)

    run(script, "export", source, "--out-dir", "chunks", "--anki", "--deck-name", "D")

    chunk = json.loads((tmp_path / "chunks" / "chunk-001.json").read_text())
    assert [deck_name for deck_name, _ in client.queries] == ["D"]
    # Variant spellings are only found by fetching the whole deck.
    assert [match["rank"] for match in chunk[0]["deck_matches"]] == [100]
    assert [match["rank"] for match in chunk[1]["deck_matches"]] == [200]
    assert (tmp_path / anki_import.DEFAULT_NOTE_CACHE).exists()


def test_export_rows_names_other_sources(script: Any) -> None:
    class OtherCitation:
        source_type = "dictionary"

    other = types.Sentence(spanish="Gato.", english="Cat.", source=OtherCitation())  # type: ignore[arg-type]

    (row,) = script.export_rows([make_word("gato", sentences=(other,))], ())

    assert row["sentences"][0]["source"] == "dictionary"


def test_changelog_names_drops_and_renames(script: Any) -> None:
    words = [make_word("gato", rank=7), make_word("despedir", part_of_speech="vtr")]
    _, changes = apply(
        script,
        [
            {"rank": 7, "word": "gato", "reason": "duplicate of 100", "drop": True},
            {
                "row": 1,
                "word": "despedir",
                "reason": "reflexive",
                "set": {"word": "despedirse"},
            },
        ],
        words,
    )

    changelog = script.format_changelog(pathlib.Path("words.csv"), changes)

    assert "- **dropped** `gato` (rank 7): duplicate of 100" in changelog
    assert "- `despedir` -> `despedirse` (unranked): reflexive" in changelog


def test_edit_can_reattribute_a_sentence(script: Any) -> None:
    (gato, *_), (change,) = apply(
        script,
        [
            {
                "row": 0,
                "word": "gato",
                "reason": "rewritten",
                "sentences": [
                    {"index": 0, "english": "The cat.", "source": "reviewer"}
                ],
            }
        ],
    )

    assert gato.sentences[0].spanish == "El gato."
    assert gato.sentences[0].english == "The cat."
    assert isinstance(gato.sentences[0].source, types.ModelCitation)
    assert gato.sentences[0].source.model == "claude-opus-5-5"
    # A translation-only edit says what changed, not "El gato. -> El gato."
    assert list(change.notes) == [
        "translation of 'El gato.': <El gato.> -> The cat.",
        "attributed to the reviewer: El gato.",
    ]


def test_edit_rejects_other_sources(script: Any) -> None:
    with pytest.raises(script.CorrectionError, match="can only be"):
        apply(
            script,
            [
                {
                    "row": 0,
                    "word": "gato",
                    "reason": "x",
                    "sentences": [{"index": 0, "source": "book"}],
                }
            ],
        )


ANNOTATIONS: Final = (
    pathlib.Path(__file__).parent.parent
    / "data"
    / "cards"
    / "combined_annotations.json"
)


def test_export_attaches_reading_passages(script: Any, tmp_path: pathlib.Path) -> None:
    from_book = types.BookCitation(
        title="Libro de Pruebas", authors=(), pages=(13, 12), id=9
    )
    words = [
        make_word(
            "gato",
            sentences=(sentence("Los gatos comían.", from_book), sentence("Otro.")),
        ),
        make_word("perro", sentences=(sentence("Un perro."),)),
    ]
    source = write_words(tmp_path / "words.csv", words)

    run(
        script,
        "export",
        source,
        "--out-dir",
        tmp_path / "chunks",
        "--annotations",
        ANNOTATIONS,
    )

    gato, perro = json.loads((tmp_path / "chunks" / "chunk-001.json").read_text())
    (passage,) = gato["contexts"]
    assert passage.startswith("Los gatos comían fiambres en la cocina.")
    assert perro["contexts"] == []


def test_export_without_annotations_has_no_contexts(script: Any) -> None:
    (row,) = script.export_rows([make_word("gato")], ())

    assert "contexts" not in row
