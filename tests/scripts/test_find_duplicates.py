"""Tests for the whole-deck duplicate fixer.

The deck is supplied through --cache-file and Anki Connect is faked, so these
cover the wiring: the shared flow finds and remembers answers, and the Anki
half plans updates, deletions and the data-loss guard.
"""

import asyncio
import contextlib
import csv
import importlib.util
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Final

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import, triage

SCRIPT: Final = (
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "find_duplicates.py"
)


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("find_duplicates_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_word(rank: int, word: str, part_of_speech: str, meaning: str) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=(),
    )


def write_cache_file(path: pathlib.Path, words: Sequence[types.Word]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(types.Word.make_csv_header())
        writer.writerows(word.to_csv_line(require_rank=True) for word in words)


class FakeAnki:
    """Just enough of AnkiConnectClient for the script's Anki half."""

    def __init__(self, words: Sequence[types.Word]):
        self.notes: dict[int, dict] = {}
        self.suspended: dict[int, bool] = {}
        self.deleted: list[int] = []
        self.updated: dict[int, types.Word] = {}
        self.unsuspended: list[int] = []
        for offset, word in enumerate(words):
            note_id = 1000 + offset
            card_id = 5000 + offset
            self.notes[note_id] = {
                "noteId": note_id,
                "cards": [card_id],
                "fields": {
                    "Rank": {"value": str(word.rank), "order": 0},
                    "Freq": {"value": "", "order": 6},
                    "Image": {"value": "", "order": 7},
                    "Note": {"value": "", "order": 10},
                },
            }
            self.suspended[card_id] = False

    def set_field(self, rank: int, field: str, value: str) -> None:
        for note in self.notes.values():
            if note["fields"]["Rank"]["value"] == str(rank):
                note["fields"][field] = {"value": value, "order": 99}

    async def get_note_ids(self, deck_name, query=None):
        wanted = {
            token.split(":", 1)[1]
            for token in (query or "").replace("(", " ").replace(")", " ").split()
            if token.startswith("Rank:")
        }
        return [
            note_id
            for note_id, note in self.notes.items()
            if note["fields"]["Rank"]["value"] in wanted
        ]

    async def get_anki_notes(self, note_ids, batch_size=1):
        for note_id in note_ids:
            yield self.notes[note_id]

    async def are_suspended(self, card_ids):
        return {card_id: self.suspended[card_id] for card_id in card_ids}

    async def unsuspend_cards(self, card_ids):
        # Anki Connect wants a flat list of known card ids
        assert all(card_id in self.suspended for card_id in card_ids), card_ids
        self.unsuspended.extend(card_ids)
        return True

    async def update_note_info_from_words(self, replacements):
        for note_id, replacement in replacements.items():
            word = replacement[0] if isinstance(replacement, tuple) else replacement
            self.updated[note_id] = word

    async def delete_notes(self, note_ids):
        self.deleted.extend(note_ids)


@pytest.fixture
def fake_anki(monkeypatch):
    state: dict[str, FakeAnki] = {}

    @contextlib.asynccontextmanager
    async def fake_create_client(*args, **kwargs):
        yield state["client"]

    monkeypatch.setattr(anki_import, "create_client", fake_create_client)
    return state


def run_script(script, tmp_path: pathlib.Path, *extra: str) -> int:
    sys.argv = [
        "find_duplicates.py",
        "--cache-file",
        str(tmp_path / "deck_cache.csv"),
        "--decision-cache",
        str(tmp_path / "decisions.json"),
        "--merge-cache",
        str(tmp_path / "merges.json"),
        "--pending-file",
        str(tmp_path / "pending.json"),
        "--no-ui",
        *extra,
    ]
    return asyncio.run(script.amain())


def test_identical_duplicates_are_merged_and_deleted(
    script, tmp_path, fake_anki
) -> None:
    words = [
        make_word(100, "zanja", "nf", "trench, ditch"),
        make_word(200, "zanja", "nf", "trench, ditch"),
        make_word(300, "perro", "nm", "dog"),
    ]
    write_cache_file(tmp_path / "deck_cache.csv", words)
    fake_anki["client"] = FakeAnki(words)

    code = run_script(script, tmp_path, "--accept")

    assert code == 0
    client = fake_anki["client"]
    # Rank 100 survives with the merged content; rank 200's note is deleted.
    (updated,) = client.updated.values()
    assert updated.rank == 100
    assert len(client.deleted) == 1
    assert client.notes[client.deleted[0]]["fields"]["Rank"]["value"] == "200"


def test_an_unanswered_group_exits_3_and_touches_nothing(
    script, tmp_path, fake_anki
) -> None:
    words = [
        make_word(100, "zanja", "nf", "trench"),
        make_word(200, "zanja", "nf", "ditch"),
    ]
    write_cache_file(tmp_path / "deck_cache.csv", words)
    fake_anki["client"] = FakeAnki(words)

    code = run_script(script, tmp_path, "--accept")

    assert code == script.EXIT_DECISIONS_NEEDED
    client = fake_anki["client"]
    assert client.deleted == []
    assert client.updated == {}
    (entry,) = json.loads((tmp_path / "pending.json").read_text())
    assert "zanja" in entry["words"]


def test_an_answered_decision_file_is_applied_to_anki(
    script, tmp_path, fake_anki
) -> None:
    words = [
        make_word(100, "zanja", "nf", "trench"),
        make_word(200, "zanja", "nf", "ditch"),
    ]
    write_cache_file(tmp_path / "deck_cache.csv", words)
    fake_anki["client"] = FakeAnki(words)
    assert run_script(script, tmp_path, "--accept") == script.EXIT_DECISIONS_NEEDED

    pending = tmp_path / "pending.json"
    data = json.loads(pending.read_text())
    data[0]["disposition"] = "merge"
    pending.write_text(json.dumps(data))

    code = run_script(script, tmp_path, "--accept", "--decision-file", str(pending))

    assert code == 0
    client = fake_anki["client"]
    (updated,) = client.updated.values()
    assert updated.rank == 100
    assert updated.meaning == "trench, ditch"
    assert len(client.deleted) == 1


def test_keep_separate_is_remembered_and_never_reapplied(
    script, tmp_path, fake_anki
) -> None:
    words = [
        make_word(100, "zanja", "nf", "trench"),
        make_word(200, "zanja", "nf", "ditch"),
    ]
    write_cache_file(tmp_path / "deck_cache.csv", words)
    fake_anki["client"] = FakeAnki(words)
    decisions = triage.DecisionCache(tmp_path / "decisions.json")
    decisions.remember(
        triage.Candidate(
            words=tuple(words),
            reason=triage.Reason.COMPATIBLE,
            comparison=_comparison(),
            disposition=triage.Disposition.KEEP_SEPARATE,
        )
    )
    decisions.save()

    code = run_script(script, tmp_path, "--accept")

    assert code == 0
    client = fake_anki["client"]
    assert client.deleted == []
    assert client.updated == {}


def test_data_loss_stops_that_group(script, tmp_path, fake_anki, capsys) -> None:
    """A superseded note with an Image survives; nothing of it is deleted."""
    words = [
        make_word(100, "zanja", "nf", "trench, ditch"),
        make_word(200, "zanja", "nf", "trench, ditch"),
    ]
    write_cache_file(tmp_path / "deck_cache.csv", words)
    client = FakeAnki(words)
    client.set_field(200, "Image", "<img src='zanja.png'>")
    fake_anki["client"] = client

    code = run_script(script, tmp_path, "--accept")

    assert code == 0
    assert client.deleted == []
    assert client.updated == {}
    assert "data loss" in capsys.readouterr().out


def test_dry_run_plans_without_touching_anki(
    script, tmp_path, fake_anki, capsys
) -> None:
    words = [
        make_word(100, "zanja", "nf", "trench, ditch"),
        make_word(200, "zanja", "nf", "trench, ditch"),
    ]
    write_cache_file(tmp_path / "deck_cache.csv", words)
    fake_anki["client"] = FakeAnki(words)

    code = run_script(script, tmp_path, "--dry-run")

    assert code == 0
    client = fake_anki["client"]
    assert client.deleted == []
    assert client.updated == {}
    assert "Merging into 100" in capsys.readouterr().out


def _comparison():
    from anki_lang_cards import definition_similarity

    return definition_similarity.Comparison(0.5, frozenset())
