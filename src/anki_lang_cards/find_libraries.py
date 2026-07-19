"""Find Calibre libraries containing KOReader annotations.

Scans one or more directory trees for Calibre libraries (directories
containing a ``metadata.db``) and reports, for each library found, whether it
has a KOReader sidecar column and how many annotated books and annotations it
contains. This is useful for remembering which library to point
``anki_lang_cards.annotation_extractor`` at.

Databases are opened read-only, so it is safe to run this against a library
that Calibre currently has open.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from typing import Final

import attrs

from . import annotation_extractor

DEFAULT_SIDECAR_COLUMN: Final[str] = "raw_sidecar"

# Unlike annotation_extractor's GET_BOOKS query, this skips the author join
# (not needed for counting) so it also works on sparsely-populated libraries.
_COUNT_QUERY: Final[str] = """
    SELECT custom_column_{0}.value AS annotations
    FROM custom_column_{0}
    JOIN books ON books.id = custom_column_{0}.book
    LEFT JOIN books_languages_link ON books.id = books_languages_link.book
    LEFT JOIN languages ON books_languages_link.lang_code = languages.id
    WHERE custom_column_{0}.value IS NOT NULL
    {1}
"""


@attrs.frozen
class LibraryReport:
    """Summary of the annotation contents of a single Calibre library."""

    path: pathlib.Path
    has_sidecar_column: bool
    annotated_books: int = 0
    annotation_count: int = 0

    def describe(self, sidecar_column: str = DEFAULT_SIDECAR_COLUMN) -> str:
        if not self.has_sidecar_column:
            return f"{self.path}: no {sidecar_column!r} column"
        book_plural = "book" if self.annotated_books == 1 else "books"
        annotation_plural = (
            "annotation" if self.annotation_count == 1 else "annotations"
        )
        return (
            f"{self.path}: {self.annotated_books} annotated {book_plural}, "
            f"{self.annotation_count} {annotation_plural}"
        )


def find_library_roots(roots: Iterable[pathlib.Path]) -> Iterator[pathlib.Path]:
    """Yield every Calibre library at or below the given roots.

    A Calibre library is any directory containing a ``metadata.db`` file.
    Hidden directories are not descended into, and libraries are not searched
    for nested libraries. Results are deterministic: libraries under each root
    are yielded in sorted order, and a library reachable from several of the
    given roots is yielded only once.
    """
    seen: set[pathlib.Path] = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            current = pathlib.Path(dirpath)
            if "metadata.db" in filenames:
                # Do not descend into the library looking for nested libraries.
                dirnames.clear()
                resolved = current.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield current
            else:
                # Sorting in place makes the walk order deterministic.
                dirnames[:] = sorted(
                    dirname for dirname in dirnames if not dirname.startswith(".")
                )


def _count_annotations(sidecar_value: str) -> int:
    try:
        return len(json.loads(sidecar_value)["annotations"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return 0


def inspect_library(
    library: pathlib.Path,
    sidecar_column: str = DEFAULT_SIDECAR_COLUMN,
    language: str | None = None,
) -> LibraryReport:
    """Summarize the annotations stored in a single Calibre library.

    Raises ``sqlite3.Error`` if ``metadata.db`` is not a readable SQLite
    database with the expected Calibre schema, and ``ValueError`` if
    ``language`` is not a recognized language name.
    """
    db_path = library / "metadata.db"
    lang_clause, lang_params = annotation_extractor.get_language_clause(language)

    db_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with contextlib.closing(sqlite3.connect(db_uri, uri=True)) as conn:
        cursor = conn.execute(
            annotation_extractor.DatabaseQueries.GET_COLUMN_ID, (sidecar_column,)
        )
        column_row = cursor.fetchone()
        if column_row is None:
            return LibraryReport(path=library, has_sidecar_column=False)

        annotated_books = 0
        annotation_count = 0
        query = _COUNT_QUERY.format(column_row[0], lang_clause)
        for (sidecar_value,) in conn.execute(query, lang_params):
            annotated_books += 1
            annotation_count += _count_annotations(sidecar_value)

    return LibraryReport(
        path=library,
        has_sidecar_column=True,
        annotated_books=annotated_books,
        annotation_count=annotation_count,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Find Calibre libraries containing KOReader annotations"
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=pathlib.Path,
        default=(pathlib.Path("."),),
        help="Directories to search for Calibre libraries (default: current directory)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Only count books in this language (e.g. 'es' or 'spanish')",
    )
    parser.add_argument(
        "--sidecar-column",
        type=str,
        default=DEFAULT_SIDECAR_COLUMN,
        help="Label of the Calibre custom column holding KOReader sidecars",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Only show libraries with at least one annotated book",
    )
    args = parser.parse_args(argv)

    # Validate the language before walking any directory trees.
    annotation_extractor.get_language_clause(args.language)

    libraries_found = 0
    libraries_with_annotations = 0
    for library in find_library_roots(args.roots):
        try:
            report = inspect_library(
                library, sidecar_column=args.sidecar_column, language=args.language
            )
        except sqlite3.Error as e:
            print(f"[ERROR] {library}: cannot read metadata.db ({e})")
            continue

        libraries_found += 1
        if report.annotated_books:
            libraries_with_annotations += 1
        if report.annotated_books or not args.annotated_only:
            print(report.describe(sidecar_column=args.sidecar_column))

    if libraries_found == 0:
        print("No Calibre libraries found.")
    else:
        print(
            f"\nFound {libraries_found} Calibre "
            f"{'library' if libraries_found == 1 else 'libraries'} "
            f"({libraries_with_annotations} with annotations)."
        )


if __name__ == "__main__":
    main()
