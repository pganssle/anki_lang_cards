"""Tests for the unified find-and-merge script.

These drive `amain()` with real files the way a user would, exported-deck and
faked-live alike. The decision-file loop is exercised end to end: run, get
exit 3 and a pending file, answer it, run again, get the import.
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
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "apply_merges.py"
)


@pytest.fixture
def script():
    """Import the script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("apply_merges_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_word(
    rank: int | None, word: str, part_of_speech: str, meaning: str
) -> types.Word:
    return types.Word(
        rank=rank,
        word=word,
        meaning=meaning,
        part_of_speech=part_of_speech,
        article=None,
        sentences=(),
    )


def write_inputs(path: pathlib.Path, words: Sequence[types.Word]) -> None:
    with path.open("w", newline="") as handle:
        handle.write("\n".join(types.Word.make_anki_directives(None)) + "\n")
        csv.writer(handle).writerows(word.to_csv_line() for word in words)


def write_export(path: pathlib.Path, words: Sequence[types.Word]) -> None:
    with path.open("w", newline="") as handle:
        handle.write("#separator:tab\n#html:true\n")
        csv.writer(handle, dialect="excel-tab").writerows(
            word.to_csv_line() for word in words
        )


def run_script(script, tmp_path: pathlib.Path, *extra: str) -> int:
    sys.argv = [
        "apply_merges.py",
        "--decision-cache",
        str(tmp_path / "decisions.json"),
        "--merge-cache",
        str(tmp_path / "merges.json"),
        "--output",
        str(tmp_path / "merged_import.csv"),
        "--notes-to-delete",
        str(tmp_path / "notes_to_delete.csv"),
        "--pending-file",
        str(tmp_path / "pending.json"),
        *extra,
    ]
    return asyncio.run(script.amain())


@pytest.fixture
def exported(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "deck.txt"
    write_export(
        path,
        [
            make_word(100, "zanja", "nf", "trench, ditch"),
            make_word(300, "perro", "nm", "dog"),
        ],
    )
    return path


def test_no_inputs_is_an_error(script, tmp_path) -> None:
    with pytest.raises(SystemExit):
        run_script(script, tmp_path, "--auto")


class TestAuto:
    def test_a_proposed_merge_lands_in_the_import(
        self, script, tmp_path, exported
    ) -> None:
        inputs = tmp_path / "added.csv"
        write_inputs(inputs, [make_word(9000, "zanja", "nf", "ditch, trench")])

        code = run_script(
            script,
            tmp_path,
            "--auto",
            "--exported-deck",
            str(exported),
            "--inputs",
            str(inputs),
        )

        assert code == 0
        written = types.Word.words_from_csv(tmp_path / "merged_import.csv")
        assert [w.rank for w in written] == [100]
        assert written[0].meaning == "trench, ditch"

    def test_superseded_deck_notes_are_listed_for_deletion(
        self, script, tmp_path
    ) -> None:
        """A merge that keeps rank 100 leaves deck rank 200 behind."""
        deck = tmp_path / "deck.txt"
        write_export(
            deck,
            [
                make_word(100, "zanja", "nf", "trench, ditch"),
                make_word(200, "zanja", "nf", "ditch, trench"),
            ],
        )
        inputs = tmp_path / "added.csv"
        write_inputs(inputs, [make_word(200, "zanja", "nf", "ditch, trench")])

        code = run_script(
            script,
            tmp_path,
            "--auto",
            "--exported-deck",
            str(deck),
            "--inputs",
            str(inputs),
        )

        assert code == 0
        with (tmp_path / "notes_to_delete.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row["rank"] for row in rows] == ["200"]

    def test_dry_run_writes_nothing(self, script, tmp_path, exported) -> None:
        inputs = tmp_path / "added.csv"
        write_inputs(inputs, [make_word(9000, "zanja", "nf", "ditch, trench")])

        code = run_script(
            script,
            tmp_path,
            "--auto",
            "--dry-run",
            "--exported-deck",
            str(exported),
            "--inputs",
            str(inputs),
        )

        assert code == 0
        assert not (tmp_path / "merged_import.csv").exists()


class TestDecisionFileLoop:
    def ask(self, script, tmp_path, exported, *extra: str) -> int:
        inputs = tmp_path / "added.csv"
        if not inputs.exists():
            write_inputs(inputs, [make_word(9000, "zanja", "nf", "ditch")])
        return run_script(
            script,
            tmp_path,
            "--no-ui",
            "--exported-deck",
            str(exported),
            "--inputs",
            str(inputs),
            *extra,
        )

    def test_an_unanswered_group_exits_3_with_a_pending_file(
        self, script, tmp_path, exported
    ) -> None:
        code = self.ask(script, tmp_path, exported)

        assert code == script.EXIT_DECISIONS_NEEDED
        (entry,) = json.loads((tmp_path / "pending.json").read_text())
        assert entry["index"] == 0
        assert "proposed" in entry
        assert not (tmp_path / "merged_import.csv").exists()

    def test_an_answered_file_completes_the_run(
        self, script, tmp_path, exported
    ) -> None:
        assert self.ask(script, tmp_path, exported) == script.EXIT_DECISIONS_NEEDED

        pending = tmp_path / "pending.json"
        data = json.loads(pending.read_text())
        data[0]["disposition"] = "merge"
        pending.write_text(json.dumps(data))

        code = self.ask(script, tmp_path, exported, "--decision-file", str(pending))

        assert code == 0
        written = types.Word.words_from_csv(tmp_path / "merged_import.csv")
        assert [w.rank for w in written] == [100]
        assert written[0].meaning == "trench, ditch"

    def test_a_answered_run_does_not_ask_again(
        self, script, tmp_path, exported
    ) -> None:
        """The decision cache carries the answer; the file is only needed once."""
        assert self.ask(script, tmp_path, exported) == script.EXIT_DECISIONS_NEEDED
        pending = tmp_path / "pending.json"
        data = json.loads(pending.read_text())
        data[0]["disposition"] = "merge"
        pending.write_text(json.dumps(data))
        assert (
            self.ask(script, tmp_path, exported, "--decision-file", str(pending)) == 0
        )

        assert self.ask(script, tmp_path, exported) == 0

    def test_the_json_report_says_what_happened(
        self, script, tmp_path, exported
    ) -> None:
        report = tmp_path / "report.json"
        code = self.ask(script, tmp_path, exported, "--report-json", str(report))

        assert code == script.EXIT_DECISIONS_NEEDED
        payload = json.loads(report.read_text())
        assert payload["status"] == "decisions-needed"
        assert payload["pending"] == 1
        assert payload["files"]["pending"] == str(tmp_path / "pending.json")


class TestLiveDeck:
    @pytest.fixture
    def fake_anki(self, monkeypatch) -> dict[str, object]:
        state: dict[str, object] = {
            "notes": {
                1: make_word(100, "zanja", "nf", "trench, ditch"),
            },
            "highest": 4999,
            "queries": [],
        }

        class FakeClient:
            async def retrieve_related_words(self, deck_name, words):
                state["queries"].append((deck_name, tuple(w.word for w in words)))
                return dict(state["notes"])

            async def find_highest_rank(self, deck_name, **kwargs):
                return state["highest"]

        @contextlib.asynccontextmanager
        async def fake_create_client(*args, **kwargs):
            yield FakeClient()

        monkeypatch.setattr(anki_import, "create_client", fake_create_client)
        return state

    def test_the_live_deck_is_the_default_source(
        self, script, tmp_path, fake_anki
    ) -> None:
        inputs = tmp_path / "added.csv"
        write_inputs(inputs, [make_word(9000, "zanja", "nf", "ditch, trench")])

        code = run_script(script, tmp_path, "--auto", "--inputs", str(inputs))

        assert code == 0
        assert fake_anki["queries"] == [("Spanish::Vocab by Frequency", ("zanja",))]
        written = types.Word.words_from_csv(tmp_path / "merged_import.csv")
        assert [w.rank for w in written] == [100]

    def test_split_cards_get_ranks_past_the_live_deck(
        self, script, tmp_path, fake_anki
    ) -> None:
        """A rank-less split card must not collide with unseen deck ranks."""
        inputs = tmp_path / "added.csv"
        write_inputs(inputs, [make_word(9000, "zanja", "nf", "ditch")])

        assert (
            run_script(script, tmp_path, "--no-ui", "--inputs", str(inputs))
            == script.EXIT_DECISIONS_NEEDED
        )
        pending = tmp_path / "pending.json"
        data = json.loads(pending.read_text())
        data[0]["disposition"] = "split"
        data[0]["replacements"] = triage.words_to_csv(
            [
                make_word(100, "zanja", "nf", "trench"),
                make_word(None, "zanja (obra)", "nf", "ditch"),
            ]
        )
        pending.write_text(json.dumps(data))

        code = run_script(
            script,
            tmp_path,
            "--no-ui",
            "--inputs",
            str(inputs),
            "--decision-file",
            str(pending),
        )

        assert code == 0
        written = types.Word.words_from_csv(tmp_path / "merged_import.csv")
        by_word = {w.word: w for w in written}
        assert by_word["zanja (obra)"].rank == 5000, "past the deck's highest"
