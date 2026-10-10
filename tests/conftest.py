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


#####
# Recorded HTTP interactions (VCR.py, through pytest-recording)
#
# Tests marked ``@pytest.mark.vcr`` replay real API exchanges checked in
# under ``tests/cassettes/<test module>/``. By default they never touch the
# network. To record a cassette that is missing, run the test with
# credentials and ``--record-mode=once``; to re-record, ``--record-mode=rewrite``;
# to check the recorded behavior still matches the live API without touching
# the cassettes, ``--disable-recording``.
_SECRET_REQUEST_HEADERS: Final = ("authorization", "x-api-key", "cookie")
_DROPPED_RESPONSE_HEADERS: Final = frozenset(
    {"anthropic-organization-id", "set-cookie", "cf-ray"}
)


def _scrub_response(response: dict) -> dict:
    response["headers"] = {
        name: value
        for name, value in response["headers"].items()
        if name.lower() not in _DROPPED_RESPONSE_HEADERS
    }
    return response


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    return {
        "filter_headers": _SECRET_REQUEST_HEADERS,
        "before_record_response": _scrub_response,
        "decode_compressed_response": True,
        # Matching on the body means a changed prompt or request parameter
        # needs a fresh recording rather than silently replaying a stale one.
        "match_on": ("method", "scheme", "host", "port", "path", "query", "body"),
    }


@pytest.fixture(scope="session")
def live_api(record_mode: str, disable_recording: bool) -> bool:
    """Whether requests in recorded tests reach the real API this run."""
    return disable_recording or record_mode != "none"


@pytest.fixture(autouse=True)
def _require_recording(request: pytest.FixtureRequest) -> None:
    """Fail a replay-only run of a recorded test that has no recording yet,
    saying how to make one, rather than with VCR's cryptic mismatch error."""
    # Looked up only for recorded tests, so that environments without
    # pytest-recording (like the calibre end-to-end image) can still run the
    # rest of the suite.
    if request.node.get_closest_marker("vcr") is None:
        return
    if request.getfixturevalue("disable_recording"):
        return
    if request.getfixturevalue("record_mode") != "none":
        return
    cassette = request.getfixturevalue("vcr")
    if cassette is not None and not cassette.data:
        pytest.fail(
            f"No recording for {request.node.nodeid}. Record one with real "
            f"credentials:\n    pytest '{request.node.nodeid}' --record-mode=once"
        )


def _caused_by_unmatched_request(error: BaseException) -> bool:
    """Whether VCR refused a request somewhere in an exception's chain.

    HTTP clients like the Anthropic SDK re-raise transport errors as their
    own connection errors, so VCR's exception is usually the cause.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if type(current).__name__ == "CannotOverwriteExistingCassetteException":
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    """Say how to fix a recording that no longer matches the requests."""
    report = yield
    if call.excinfo is not None and _caused_by_unmatched_request(call.excinfo.value):
        report.sections.append(
            (
                "recording",
                "A request no longer matches the recording, probably because a "
                "prompt or request parameter changed. Re-record it with real "
                f"credentials:\n    pytest '{item.nodeid}' --record-mode=rewrite",
            )
        )
    return report
