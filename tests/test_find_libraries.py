"""Tests for anki_lang_cards.find_libraries.

These tests build minimal ``metadata.db`` files by hand containing only the
tables the queries touch, so they do not require a Calibre installation.
"""

import json
import pathlib
import sqlite3
from collections.abc import Sequence
from typing import Final

import pytest

from anki_lang_cards import find_libraries

_SCHEMA: Final[str] = """
    CREATE TABLE custom_columns (id INTEGER PRIMARY KEY, label TEXT);
    CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT);
    CREATE TABLE languages (id INTEGER PRIMARY KEY, lang_code TEXT);
    CREATE TABLE books_languages_link (book INTEGER, lang_code INTEGER);
"""

_SIDECAR_COLUMN_ID: Final[int] = 2


def make_sidecar(annotation_count: int) -> str:
    annotations = {
        str(i): {"text": f"word{i}", "pageno": i}
        for i in range(1, annotation_count + 1)
    }
    return json.dumps({"annotations": annotations})


def make_library(
    library: pathlib.Path,
    books: Sequence[tuple[str | None, str | None]] = (),
    sidecar_column: str | None = "raw_sidecar",
) -> pathlib.Path:
    """Create a minimal Calibre-shaped library.

    ``books`` is a sequence of ``(lang_code, sidecar_value)`` pairs; either
    element may be ``None`` to leave the book without a language or without a
    sidecar value.
    """
    library.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(library / "metadata.db") as conn:
        conn.executescript(_SCHEMA)
        if sidecar_column is not None:
            conn.execute(
                "INSERT INTO custom_columns (id, label) VALUES (?, ?)",
                (_SIDECAR_COLUMN_ID, sidecar_column),
            )
            conn.execute(
                f"CREATE TABLE custom_column_{_SIDECAR_COLUMN_ID} "
                "(book INTEGER, value TEXT)"
            )

        language_ids: dict[str, int] = {}
        for book_id, (lang_code, sidecar_value) in enumerate(books, start=1):
            conn.execute(
                "INSERT INTO books (id, title, path) VALUES (?, ?, ?)",
                (book_id, f"Book {book_id}", f"Author/Book {book_id} ({book_id})"),
            )
            if lang_code is not None:
                if lang_code not in language_ids:
                    language_ids[lang_code] = len(language_ids) + 1
                    conn.execute(
                        "INSERT INTO languages (id, lang_code) VALUES (?, ?)",
                        (language_ids[lang_code], lang_code),
                    )
                conn.execute(
                    "INSERT INTO books_languages_link (book, lang_code) "
                    "VALUES (?, ?)",
                    (book_id, language_ids[lang_code]),
                )
            if sidecar_value is not None and sidecar_column is not None:
                conn.execute(
                    f"INSERT INTO custom_column_{_SIDECAR_COLUMN_ID} "  # noqa: S608
                    "(book, value) VALUES (?, ?)",
                    (book_id, sidecar_value),
                )
    conn.close()
    return library


class TestFindLibraryRoots:
    def test_finds_nested_libraries_in_sorted_order(self, tmp_path):
        make_library(tmp_path / "b" / "Spanish Library")
        make_library(tmp_path / "a" / "English Library")
        make_library(tmp_path / "root_library")

        found = list(find_libraries.find_library_roots([tmp_path]))

        assert found == [
            tmp_path / "a" / "English Library",
            tmp_path / "b" / "Spanish Library",
            tmp_path / "root_library",
        ]

    def test_root_itself_is_a_library(self, tmp_path):
        make_library(tmp_path)
        make_library(tmp_path / "nested")

        found = list(find_libraries.find_library_roots([tmp_path]))

        assert found == [tmp_path]

    def test_skips_hidden_directories(self, tmp_path):
        make_library(tmp_path / ".hidden" / "library")
        make_library(tmp_path / "visible")

        found = list(find_libraries.find_library_roots([tmp_path]))

        assert found == [tmp_path / "visible"]

    def test_does_not_descend_into_libraries(self, tmp_path):
        library = make_library(tmp_path / "library")
        make_library(library / "inner")

        found = list(find_libraries.find_library_roots([tmp_path]))

        assert found == [library]

    def test_overlapping_roots_deduplicated(self, tmp_path):
        library = make_library(tmp_path / "outer" / "library")

        found = list(find_libraries.find_library_roots([tmp_path, tmp_path / "outer"]))

        assert found == [library]

    def test_no_libraries(self, tmp_path):
        (tmp_path / "not_a_library").mkdir()

        assert list(find_libraries.find_library_roots([tmp_path])) == []


class TestInspectLibrary:
    def test_missing_sidecar_column(self, tmp_path):
        library = make_library(tmp_path / "library", sidecar_column=None)

        report = find_libraries.inspect_library(library)

        assert report == find_libraries.LibraryReport(
            path=library, has_sidecar_column=False
        )

    def test_counts_books_and_annotations(self, tmp_path):
        library = make_library(
            tmp_path / "library",
            books=[
                ("spa", make_sidecar(3)),
                ("spa", make_sidecar(2)),
                ("eng", make_sidecar(1)),
                ("spa", None),  # No sidecar: not annotated
            ],
        )

        report = find_libraries.inspect_library(library)

        assert report == find_libraries.LibraryReport(
            path=library,
            has_sidecar_column=True,
            annotated_books=3,
            annotation_count=6,
        )

    @pytest.mark.parametrize("language", ["spa", "es", "spanish", "SPANISH"])
    def test_language_filter(self, tmp_path, language):
        library = make_library(
            tmp_path / "library",
            books=[
                ("spa", make_sidecar(3)),
                ("eng", make_sidecar(1)),
                (None, make_sidecar(5)),  # No language: excluded by the filter
            ],
        )

        report = find_libraries.inspect_library(library, language=language)

        assert report.annotated_books == 1
        assert report.annotation_count == 3

    def test_unsupported_language(self, tmp_path):
        library = make_library(tmp_path / "library")

        with pytest.raises(ValueError, match="Unsupported language"):
            find_libraries.inspect_library(library, language="klingon")

    def test_custom_sidecar_column_name(self, tmp_path):
        library = make_library(
            tmp_path / "library",
            books=[("spa", make_sidecar(1))],
            sidecar_column="koreader_sidecar",
        )

        report = find_libraries.inspect_library(
            library, sidecar_column="koreader_sidecar"
        )
        default_report = find_libraries.inspect_library(library)

        assert report.annotated_books == 1
        assert not default_report.has_sidecar_column

    @pytest.mark.parametrize(
        "sidecar_value",
        [
            "not json",
            json.dumps({"no_annotations_key": True}),
            json.dumps(["not", "a", "mapping"]),
        ],
    )
    def test_unparseable_sidecar_counts_as_zero_annotations(
        self, tmp_path, sidecar_value
    ):
        library = make_library(tmp_path / "library", books=[("spa", sidecar_value)])

        report = find_libraries.inspect_library(library)

        assert report.annotated_books == 1
        assert report.annotation_count == 0

    def test_corrupt_database(self, tmp_path):
        library = tmp_path / "library"
        library.mkdir()
        (library / "metadata.db").write_text("this is not a sqlite database")

        with pytest.raises(sqlite3.Error):
            find_libraries.inspect_library(library)

    def test_database_not_modified(self, tmp_path):
        library = make_library(tmp_path / "library", books=[("spa", make_sidecar(1))])
        before = (library / "metadata.db").read_bytes()

        find_libraries.inspect_library(library)

        assert (library / "metadata.db").read_bytes() == before


class TestLibraryReportDescribe:
    @pytest.mark.parametrize(
        ("report_kwargs", "expected_suffix"),
        [
            (
                {"has_sidecar_column": False},
                "no 'raw_sidecar' column",
            ),
            (
                {
                    "has_sidecar_column": True,
                    "annotated_books": 1,
                    "annotation_count": 1,
                },
                "1 annotated book, 1 annotation",
            ),
            (
                {
                    "has_sidecar_column": True,
                    "annotated_books": 2,
                    "annotation_count": 5,
                },
                "2 annotated books, 5 annotations",
            ),
            (
                {"has_sidecar_column": True},
                "0 annotated books, 0 annotations",
            ),
        ],
    )
    def test_describe(self, report_kwargs, expected_suffix):
        library = pathlib.Path("/library")
        report = find_libraries.LibraryReport(path=library, **report_kwargs)

        assert report.describe() == f"{library}: {expected_suffix}"


class TestMain:
    def test_reports_all_libraries(self, tmp_path, capsys):
        make_library(tmp_path / "annotated", books=[("spa", make_sidecar(2))])
        make_library(tmp_path / "empty")
        make_library(tmp_path / "no_column", sidecar_column=None)

        find_libraries.main([str(tmp_path)])

        output = capsys.readouterr().out
        assert f"{tmp_path / 'annotated'}: 1 annotated book, 2 annotations" in output
        assert f"{tmp_path / 'empty'}: 0 annotated books" in output
        assert f"{tmp_path / 'no_column'}: no 'raw_sidecar' column" in output
        assert "Found 3 Calibre libraries (1 with annotations)." in output

    def test_annotated_only(self, tmp_path, capsys):
        make_library(tmp_path / "annotated", books=[("spa", make_sidecar(2))])
        make_library(tmp_path / "empty")

        find_libraries.main([str(tmp_path), "--annotated-only"])

        output = capsys.readouterr().out
        assert str(tmp_path / "annotated") in output
        assert str(tmp_path / "empty") not in output

    def test_language_filter(self, tmp_path, capsys):
        make_library(
            tmp_path / "library",
            books=[("spa", make_sidecar(2)), ("eng", make_sidecar(7))],
        )

        find_libraries.main([str(tmp_path), "--language", "es"])

        output = capsys.readouterr().out
        assert "1 annotated book, 2 annotations" in output

    def test_invalid_language_rejected_before_scanning(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported language"):
            find_libraries.main([str(tmp_path), "--language", "klingon"])

    def test_unreadable_database_reported(self, tmp_path, capsys):
        library = tmp_path / "corrupt"
        library.mkdir()
        (library / "metadata.db").write_text("this is not a sqlite database")
        make_library(tmp_path / "good", books=[("spa", make_sidecar(1))])

        find_libraries.main([str(tmp_path)])

        output = capsys.readouterr().out
        assert f"[ERROR] {library}" in output
        assert "Found 1 Calibre library (1 with annotations)." in output

    def test_no_libraries_found(self, tmp_path, capsys):
        find_libraries.main([str(tmp_path)])

        assert "No Calibre libraries found." in capsys.readouterr().out
