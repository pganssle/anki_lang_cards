"""Tests for scripts/compare_outputs.py."""

import csv
import pathlib
import subprocess
import sys
from collections.abc import Sequence
from typing import Final

import pytest

from anki_lang_cards import _types as types

SCRIPT: Final = pathlib.Path(__file__).parent.parent / "scripts" / "compare_outputs.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(),
    reason="scripts/ is not present (e.g. inside the Docker test image)",
)


def _make_word(
    headword: str, spanish_sentences: Sequence[str], model: str
) -> types.Word:
    return types.Word(
        rank=None,
        word=headword,
        meaning=f"meaning of {headword}",
        part_of_speech="nf",
        article="la",
        sentences=tuple(
            types.Sentence(
                spanish=spanish, english=f"translation of {spanish}", source=None
            )
            for spanish in spanish_sentences
        ),
        model_used=model,
    )


def _write_csv(path: pathlib.Path, words: Sequence[types.Word]) -> pathlib.Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for word in words:
            writer.writerow(word.to_csv_line())
    return path


@pytest.fixture
def comparison(tmp_path: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Run the script on two overlapping CSVs; returns (stdout, html_path)."""
    csv_a = _write_csv(
        tmp_path / "a.csv",
        [
            _make_word("casa", ["Una frase compartida.", "Solo del modelo A."], "m-a"),
            _make_word("perro", ["El perro ladra."], "m-a"),
            _make_word("solo-en-a", ["No aparece en B."], "m-a"),
        ],
    )
    csv_b = _write_csv(
        tmp_path / "b.csv",
        [
            _make_word("casa", ["Una frase compartida.", "Solo del modelo B."], "m-b"),
            _make_word("perro", ["El perro come."], "m-b"),
        ],
    )
    html_path = tmp_path / "comparison.html"
    result = subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            str(csv_a),
            str(csv_b),
            "--labels",
            "Model A",
            "Model B",
            "-o",
            str(html_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout, html_path


def test_common_words_compared(comparison: tuple[str, pathlib.Path]) -> None:
    stdout, html_path = comparison
    content = html_path.read_text(encoding="utf-8")

    assert "casa" in content
    assert "Solo del modelo A." in content
    assert "Solo del modelo B." in content
    assert "Model A" in content
    assert "Model B" in content


def test_word_missing_from_one_file_is_skipped_and_reported(
    comparison: tuple[str, pathlib.Path],
) -> None:
    stdout, html_path = comparison
    content = html_path.read_text(encoding="utf-8")

    assert "solo-en-a" not in content
    assert "2 words common to all files." in stdout
    assert "solo-en-a" in stdout


def test_shared_sentences_are_marked(comparison: tuple[str, pathlib.Path]) -> None:
    _, html_path = comparison
    content = html_path.read_text(encoding="utf-8")

    shared = '<div class="sentence shared"><p class="es">Una frase compartida.</p>'
    unshared = '<div class="sentence"><p class="es">Solo del modelo A.</p>'
    assert content.count(shared) == 2  # dimmed in both columns
    assert unshared in content


def test_no_overlap_fails(tmp_path: pathlib.Path) -> None:
    csv_a = _write_csv(tmp_path / "a.csv", [_make_word("uno", ["Frase uno."], "m-a")])
    csv_b = _write_csv(tmp_path / "b.csv", [_make_word("dos", ["Frase dos."], "m-b")])

    result = subprocess.run(
        (sys.executable, str(SCRIPT), str(csv_a), str(csv_b)),
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "No words appear in all of the input files." in result.stderr
