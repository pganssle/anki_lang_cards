"""Tests for how the pipeline fetches deck notes over Anki Connect.

The client is replaced with a fake that answers the way Anki would: an
exact-spelling search misses a deck note written with a different
shorthand, while the related-words search finds it. The run stops at the
duplicate check, before anything is sent to a model, so the model backend is
stubbed out rather than requiring a plugin to be installed.
"""

import asyncio
import contextlib
import json
import pathlib
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final, Self

import pytest

from anki_lang_cards import __main__ as main_module
from anki_lang_cards import _types as types
from anki_lang_cards import anki_import, model_selection, triage

DECK_NOTE: Final = types.Word(
    rank=500,
    word="mirón/ona",
    meaning="onlooker",
    part_of_speech="nm/f",
    article="el/la",
    sentences=(),
)


class FakeAnkiClient:
    """Answers deck searches the way Anki does for `DECK_NOTE`."""

    async def retrieve_matching_words(
        self,
        deck_name: str,
        words: Sequence[types.Word],
        *,
        cache: anki_import.NoteCache | None = None,
    ) -> Mapping[int, types.Word]:
        # The pipeline no longer calls this; it is here so that a regression
        # to the exact-spelling search fails on the assertion below rather
        # than with an AttributeError. Exact spellings only: "mirón/a" is not
        # "mirón/ona".
        spellings = {word.word for word in words}
        return {1: DECK_NOTE} if DECK_NOTE.word in spellings else {}

    async def retrieve_related_words(
        self,
        deck_name: str,
        words: Sequence[types.Word],
        *,
        cache: anki_import.NoteCache | None = None,
    ) -> Mapping[int, types.Word]:
        return {1: DECK_NOTE}


@contextlib.asynccontextmanager
async def fake_create_client(
    *args: object, **kwargs: object
) -> AsyncIterator[FakeAnkiClient]:
    yield FakeAnkiClient()


class UnreachableRunner:
    """Stands in for the query runner on a run that must stop before it."""

    async def __aenter__(self) -> Self:
        pytest.fail("The run reached the model; the deck note was not a duplicate")

    async def __aexit__(self, *exc_info: object) -> None:
        pass


def make_unreachable_runner(*args: object, **kwargs: object) -> tuple[None, object]:
    return None, UnreachableRunner()


def test_variant_spelling_in_deck_reaches_dedup(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An input spelled differently from its deck note is flagged as a
    duplicate rather than becoming a second card."""
    inputs = tmp_path / "words.csv"
    inputs.write_text(
        "rank,word,part_of_speech,meaning,spanish_sentence,english_sentence,"
        "frequency,image,article,sentence_source,note,example_sentences,"
        "model_used,tags\n"
        ",mirón/a,nm/f,onlooker,,,,,el/la,,,,,\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    config.write_text('[models]\ndefault_model="stub"\n', encoding="utf-8")
    pending = tmp_path / "pending.json"

    monkeypatch.setattr(anki_import, "create_client", fake_create_client)
    monkeypatch.setattr(model_selection, "make_runner", make_unreachable_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "anki_lang_cards",
            "--inputs",
            str(inputs),
            "--output",
            str(tmp_path / "output.csv"),
            "--config",
            str(config),
            "--no-ui",
            "--note-cache",
            str(tmp_path / "notes.json"),
            "--decision-cache",
            str(tmp_path / "decisions.json"),
            "--conflict-cache-path",
            str(tmp_path / "conflicts.json"),
            "--pending-file",
            str(pending),
            # Given a rank, the run has no reason to ask the deck for one.
            "--start-rank-at",
            "600",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(main_module.main())

    assert excinfo.value.code == main_module.EXIT_DECISIONS_NEEDED
    # The deck note is one side of the single unanswered duplicate group.
    (entry,) = json.loads(pending.read_text(encoding="utf-8"))
    assert {w.word for w in triage.words_from_csv(entry["words"])} == {
        "mirón/a",
        "mirón/ona",
    }
