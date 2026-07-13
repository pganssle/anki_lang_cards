"""Shared pytest fixtures."""

import json
import os
import pathlib
import re
import shutil
import subprocess
from collections.abc import Sequence
from typing import Final

import pytest

BOOKS_DIR: Final = pathlib.Path(__file__).parent / "data" / "books"


def _run_calibre(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a Calibre command outside any active Python virtual environment."""
    environment = os.environ.copy()
    virtual_environment = environment.pop("VIRTUAL_ENV", None)
    if virtual_environment is not None:
        virtual_bin = pathlib.Path(virtual_environment) / "bin"
        environment["PATH"] = os.pathsep.join(
            entry
            for entry in environment["PATH"].split(os.pathsep)
            if pathlib.Path(entry) != virtual_bin
        )

    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )


def _add_book(library: pathlib.Path, book_directory: pathlib.Path) -> None:
    book_files = tuple(
        path for path in book_directory.iterdir() if path.suffix.lower() == ".epub"
    )
    if len(book_files) != 1:
        raise ValueError(f"Expected exactly one EPUB in {book_directory}")

    result = _run_calibre(
        (
            "calibredb",
            "add",
            "--library-path",
            os.fspath(library),
            os.fspath(book_files[0]),
        )
    )
    match = re.search(r"Added book ids:\s*(?P<id>\d+)", result.stdout)
    if match is None:
        raise ValueError(f"Could not determine Calibre book ID from {result.stdout!r}")

    book_id = match.group("id")
    _run_calibre(
        (
            "calibredb",
            "set_metadata",
            "--library-path",
            os.fspath(library),
            book_id,
            os.fspath(book_directory / "metadata.opf"),
        )
    )

    with (book_directory / "sidecar.json").open(encoding="utf-8") as sidecar_file:
        sidecar = json.load(sidecar_file)
    sidecar_json = json.dumps(sidecar, ensure_ascii=False, separators=(",", ":"))
    _run_calibre(
        (
            "calibredb",
            "set_metadata",
            "--library-path",
            os.fspath(library),
            "--field",
            f"#raw_sidecar:{sidecar_json}",
            book_id,
        )
    )


@pytest.fixture(scope="session")
def calibre_library(
    tmp_path_factory: pytest.TempPathFactory,
) -> pathlib.Path:
    """Build a fresh Calibre library from the source fixtures."""
    if shutil.which("calibredb") is None:
        pytest.fail("calibredb is required for tests marked e2e")

    library = tmp_path_factory.mktemp("calibre-library")
    _run_calibre(
        (
            "calibredb",
            "add_custom_column",
            "--library-path",
            os.fspath(library),
            "raw_sidecar",
            "Raw sidecar JSON",
            "comments",
        )
    )

    for author_directory in sorted(BOOKS_DIR.iterdir()):
        if author_directory.is_dir():
            for book_directory in sorted(author_directory.iterdir()):
                if book_directory.is_dir():
                    _add_book(library, book_directory)

    return library
