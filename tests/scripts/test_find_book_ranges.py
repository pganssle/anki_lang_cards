"""Unit tests for the page-range filter in scripts/find_book_ranges.py."""

import csv
import importlib.util
import json
import pathlib
from collections.abc import Sequence
from types import ModuleType
from typing import Final

import pytest

SCRIPT: Final = (
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "find_book_ranges.py"
)


@pytest.fixture
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("find_book_ranges_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: pathlib.Path, rows: Sequence[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("word", "sentence_source"))
        for word, title, page in rows:
            writer.writerow((word, json.dumps({"title": title, "pages": [page]})))


def _words(path: pathlib.Path) -> Sequence[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row["word"] for row in csv.DictReader(f)]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [
                ("uno", "Libro", 1),
                ("dos", "Libro", 10),
                ("tres", "Libro", 11),
                ("cuatro", "Libro", 12),
                ("cinco", "Libro", 30),
            ],
            ["uno", "cinco"],
            id="consecutive",
        ),
        pytest.param(
            [
                ("uno", "Libro", 1),
                ("dos", "Libro", 10),
                ("tres", "Otro", 10),
                ("cuatro", "Libro", 12),
                ("cinco", "Libro", 13),
            ],
            ["uno", "tres"],
            id="separated",
        ),
    ],
)
def test_filter_csv_excludes_every_row_in_range(
    script: ModuleType,
    tmp_path: pathlib.Path,
    rows: Sequence[tuple[str, str, int]],
    expected: Sequence[str],
) -> None:
    input_csv = tmp_path / "words.csv"
    output_csv = tmp_path / "filtered.csv"
    _write_csv(input_csv, rows)

    all_books = script.process_book_data(input_csv)
    script.filter_csv(input_csv, output_csv, ("^Libro$", 5, 20), all_books)

    assert _words(output_csv) == expected
