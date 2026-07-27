"""Tests for the triage script's wiring.

The modules underneath are well covered, which is exactly why these exist: an
option can be declared, documented and never actually passed on, and every
other test still passes. `--sentences` shipped that way. These drive `main()`
so the arguments have to reach the code that uses them.
"""

import csv
import importlib.util
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any, Final

import attrs
import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import triage

SCRIPT: Final = (
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "triage_duplicates.py"
)


@pytest.fixture
def script():
    """Import the script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("triage_duplicates", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_words(path: pathlib.Path, rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="") as handle:
        handle.write("\n".join(types.Word.make_anki_directives(None)) + "\n")
        writer = csv.writer(handle)
        for row in rows:
            rank, word, part_of_speech, meaning = row
            writer.writerow(
                types.Word(
                    rank=int(rank),
                    word=word,
                    meaning=meaning,
                    part_of_speech=part_of_speech,
                    article=None,
                    sentences=(),
                ).to_csv_line()
            )


@pytest.fixture
def inputs(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "added.csv"
    write_words(
        path,
        [
            ("9087", "no obstante", "conj", "nevertheless"),
            ("9265", "no obstante", "adv", "nevertheless, however"),
        ],
    )
    return path


def run_script(script, tmp_path: pathlib.Path, *extra: str) -> None:
    sys.argv = [
        "triage_duplicates.py",
        "--triage-file",
        str(tmp_path / "triage.json"),
        "--decision-cache",
        str(tmp_path / "decisions.json"),
        "--revisions-file",
        str(tmp_path / "revisions.csv"),
        *extra,
    ]
    script.main()


def test_generating_writes_a_triage_file(script, tmp_path, inputs, capsys) -> None:
    run_script(script, tmp_path, "--inputs", str(inputs))

    written = json.loads((tmp_path / "triage.json").read_text())
    assert len(written) == 1
    assert written[0]["reason"] == "base-part-of-speech"


def test_no_inputs_is_an_error(script, tmp_path) -> None:
    with pytest.raises(SystemExit):
        run_script(script, tmp_path)


def test_applying_a_missing_file_is_an_error(script, tmp_path, inputs) -> None:
    with pytest.raises(SystemExit):
        run_script(script, tmp_path, "--inputs", str(inputs), "--apply")


class TestResolveWiring:
    """`--resolve` reaches the UI with the options it was given."""

    @pytest.fixture
    def recorded(self, script, monkeypatch) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        async def fake_resolve(candidates, per_page=None, **kwargs):
            seen["candidates"] = tuple(candidates)
            seen["per_page"] = per_page
            return tuple(candidates)

        monkeypatch.setattr(script.triage_ui, "resolve_candidates", fake_resolve)
        return seen

    def test_resolve_calls_the_ui(self, script, tmp_path, inputs, recorded) -> None:
        run_script(script, tmp_path, "--inputs", str(inputs), "--resolve")
        assert len(recorded["candidates"]) == 1

    def test_a_fixed_page_size_is_passed_through(
        self, script, tmp_path, inputs, recorded
    ) -> None:
        """The bug this file exists for: --sentences was declared and dropped."""
        run_script(
            script, tmp_path, "--inputs", str(inputs), "--resolve", "--sentences", "7"
        )
        assert recorded["per_page"] == 7

    def test_the_page_size_defaults_to_whatever_fits(
        self, script, tmp_path, inputs, recorded
    ) -> None:
        run_script(script, tmp_path, "--inputs", str(inputs), "--resolve")
        assert recorded["per_page"] is None

    def test_only_undecided_groups_are_offered_by_default(
        self, script, tmp_path, recorded
    ) -> None:
        path = tmp_path / "added.csv"
        write_words(
            path,
            [
                ("1", "trepa", "nm/f", "social climber"),
                ("2", "trepa", "nf", "social climber"),
                ("3", "cura", "nm", "priest"),
                ("4", "cura", "nf", "cure, remedy"),
            ],
        )
        run_script(script, tmp_path, "--inputs", str(path), "--resolve")

        offered = recorded["candidates"]
        assert len(offered) == 1
        assert offered[0].disposition is triage.Disposition.REVIEW

    def test_include_proposed_offers_everything(
        self, script, tmp_path, recorded
    ) -> None:
        path = tmp_path / "added.csv"
        write_words(
            path,
            [
                ("1", "trepa", "nm/f", "social climber"),
                ("2", "trepa", "nf", "social climber"),
                ("3", "cura", "nm", "priest"),
                ("4", "cura", "nf", "cure, remedy"),
            ],
        )
        run_script(
            script, tmp_path, "--inputs", str(path), "--resolve", "--include-proposed"
        )
        assert len(recorded["candidates"]) == 2


class TestPersistence:
    def test_answers_are_remembered_between_runs(
        self, script, tmp_path, inputs, monkeypatch, capsys
    ) -> None:
        async def answer_merge(candidates, per_page=None, **kwargs):
            return tuple(
                attrs.evolve(c, disposition=triage.Disposition.MERGE)
                for c in candidates
            )

        monkeypatch.setattr(script.triage_ui, "resolve_candidates", answer_merge)
        run_script(script, tmp_path, "--inputs", str(inputs), "--resolve")

        async def fail_if_asked(candidates, per_page=None, **kwargs):
            raise AssertionError(
                f"asked again about {[c.summarize() for c in candidates]}"
            )

        monkeypatch.setattr(script.triage_ui, "resolve_candidates", fail_if_asked)
        run_script(script, tmp_path, "--inputs", str(inputs), "--resolve")

        written = json.loads((tmp_path / "triage.json").read_text())
        assert written[0]["disposition"] == "merge"

    def test_revisions_are_written_for_import(
        self, script, tmp_path, inputs, monkeypatch
    ) -> None:
        async def rewrite(candidates, per_page=None, **kwargs):
            out = []
            for candidate in candidates:
                revised = tuple(
                    attrs.evolve(word, word=f"{word.word} ({word.part_of_speech})")
                    for word in candidate.words
                )
                out.append(
                    attrs.evolve(
                        candidate,
                        disposition=triage.Disposition.KEEP_SEPARATE,
                        replacements=revised,
                    )
                )
            return tuple(out)

        monkeypatch.setattr(script.triage_ui, "resolve_candidates", rewrite)
        run_script(script, tmp_path, "--inputs", str(inputs), "--resolve")

        revisions = tmp_path / "revisions.csv"
        assert revisions.exists()
        # It has to be importable, which means directives and no header row.
        reloaded = types.Word.words_from_csv(revisions)
        assert {w.word for w in reloaded} == {
            "no obstante (conj)",
            "no obstante (adv)",
        }

    def test_nothing_is_written_when_nothing_was_rewritten(
        self, script, tmp_path, inputs, monkeypatch
    ) -> None:
        async def answer_merge(candidates, per_page=None, **kwargs):
            return tuple(
                attrs.evolve(c, disposition=triage.Disposition.MERGE)
                for c in candidates
            )

        monkeypatch.setattr(script.triage_ui, "resolve_candidates", answer_merge)
        run_script(script, tmp_path, "--inputs", str(inputs), "--resolve")

        assert not (tmp_path / "revisions.csv").exists()
