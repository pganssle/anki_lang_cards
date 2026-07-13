import json
import pathlib
import subprocess
import sys
from typing import Any, Final

import pytest

EXPECTED_OUTPUT: Final = pathlib.Path(__file__).parent / "data" / "annotations.json"


def _run_extractor(
    calibre_library: pathlib.Path,
    output_directory: pathlib.Path,
    language: str,
) -> None:
    subprocess.run(
        (
            sys.executable,
            "-m",
            "anki_lang_cards.annotation_extractor",
            "--calibre-directory",
            str(calibre_library),
            "--language",
            language,
            "--context-sentences",
            "1",
            "--output",
            str(output_directory),
            "--json",
        ),
        check=True,
    )


@pytest.mark.e2e
def test_annotation_extractor_cli(
    calibre_library: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    _run_extractor(calibre_library, tmp_path, "Spanish")

    with (tmp_path / "annotations.json").open(encoding="utf-8") as output_file:
        actual: Any = json.load(output_file)
    with EXPECTED_OUTPUT.open(encoding="utf-8") as expected_file:
        expected: Any = json.load(expected_file)

    assert actual == expected
    assert len(actual) == 1
    assert len(actual[0]["annotations"]) == 3


@pytest.mark.e2e
def test_annotation_extractor_filters_other_languages(
    calibre_library: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    _run_extractor(calibre_library, tmp_path, "English")

    with (tmp_path / "annotations.json").open(encoding="utf-8") as output_file:
        assert json.load(output_file) == []
