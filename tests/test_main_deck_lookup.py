"""Tests for how the pipeline fetches deck notes over Anki Connect.

The client is replaced with a fake that answers the way Anki would: an
exact-spelling search misses a deck note written with a different
shorthand, while the related-words search finds it.
"""

import asyncio
import contextlib
import json
import pathlib
import sys
from collections.abc import AsyncIterator, Mapping, Sequence

import pytest

from anki_lang_cards import __main__ as main_module
from anki_lang_cards import _types as types
from anki_lang_cards import anki_import

DECK_NOTE = types.Word(
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
        # Exact spellings only: "mirón/a" is not "mirón/ona".
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

    async def find_highest_rank(self, deck_name: str, max_range: int) -> int:
        return DECK_NOTE.rank or 0


@contextlib.asynccontextmanager
async def fake_create_client(*args: object, **kwargs: object) -> AsyncIterator:
    yield FakeAnkiClient()


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
    config.write_text('[models]\ndefault_model="fake-cards"\n', encoding="utf-8")
    pending = tmp_path / "pending.json"

    monkeypatch.setattr(anki_import, "create_client", fake_create_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "anki_lang_cards",
            "--inputs",
            str(inputs),
            "--output",
            str(tmp_path / "output.csv"),
            "--model-name",
            "fake-cards",
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
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(main_module.main())

    assert excinfo.value.code == main_module.EXIT_DECISIONS_NEEDED
    # The deck note is one side of an unanswered duplicate group.
    entries = json.loads(pending.read_text(encoding="utf-8"))
    assert "mirón/ona" in json.dumps(entries, ensure_ascii=False)
