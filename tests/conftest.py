"""Pytest fixtures for testing applications that use Calibre."""

import itertools
import json
import os
import pathlib
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any, Final

import attrs
import pytest

DEFAULT_CALIBRE_LOCATION: Final[pathlib.Path] = pathlib.Path(".cache/testing/calibre")
DATA_DIR: Final[pathlib.Path] = pathlib.Path(__file__).parent / "data"
BOOKS_DIR: Final[pathlib.Path] = DATA_DIR / "books"


@attrs.frozen(eq=True)
class CustomColumnMetadata:
    """Represents metadata for a Calibre custom column."""

    table: str
    label: str
    kind: str
    name: str
    datatype: str
    column: str = attrs.field(eq=False)
    is_multiple: bool | None = attrs.field(default=None, eq=False)
    display: Mapping[str, Any] = attrs.field(factory=dict, eq=False)

    def to_calibredb_command(self, library_path: pathlib.Path) -> str:
        """Generate calibredb add_custom_column command for this column."""
        # Build the basic command
        cmd = [
            "calibredb",
            "--ignore-plugins",
            "add_custom_column",
            "--library-path",
            os.fspath(library_path),
            self.label,
            self.name,
            self.datatype,
        ]

        # Add optional parameters
        if self.is_multiple:
            cmd.append("--is-multiple")

        # Handle display options
        if (
            self.display
            and "date_format" in self.display
            and self.display["date_format"]
        ):
            cmd.extend(("--display", json.dumps(self.display)))

        return cmd


def parse_opf_file(opf_path: pathlib.Path) -> Sequence[CustomColumnMetadata]:
    """
    Parse a metadata.opf file and extract custom column metadata.

    Args:
        opf_path: Path to the metadata.opf file

    Returns:
        Sequence of CustomColumnMetadata objects

    Raises:
        FileNotFoundError: If the OPF file doesn't exist
        ET.ParseError: If the XML is malformed
        json.JSONDecodeError: If the JSON content in meta tags is invalid
    """
    if not opf_path.exists():
        raise FileNotFoundError(f"OPF file not found: {opf_path}")

    # Parse the XML
    tree = ET.parse(opf_path)  # noqa: S314
    root = tree.getroot()

    # Find all meta tags with calibre:user_metadata names
    custom_columns = []

    # Handle XML namespaces - OPF files typically use namespaces
    namespaces = {"opf": "http://www.idpf.org/2007/opf"}

    # Look for meta tags (they might be in the default namespace or opf namespace)
    meta_tags = root.findall(".//meta[@name]") + root.findall(
        ".//opf:meta[@name]", namespaces
    )

    valid_field_names = {field.name for field in attrs.fields(CustomColumnMetadata)}
    for meta in meta_tags:
        name = meta.get("name", "")
        content = meta.get("content", "")

        # Check if this is a custom column metadata entry
        if name.startswith("calibre:user_metadata:#"):
            try:
                # Parse the JSON content
                metadata_dict = json.loads(content)

                # Convert to our data class
                constructor_kwargs = {
                    key: value
                    for key, value in metadata_dict.items()
                    if key in valid_field_names
                }
                column = CustomColumnMetadata(**constructor_kwargs)
                custom_columns.append(column)

            except (json.JSONDecodeError, TypeError, ValueError) as e:
                # Log the error but continue processing other columns
                print(f"Warning: Failed to parse metadata for {name}: {e}")
                continue

    return tuple(custom_columns)


def _run_without_venv(cmd: Sequence[str], *args, **kwargs) -> None:
    # Get current PATH
    if "env" in kwargs:
        new_env = kwargs["env"].copy()
    else:
        new_env = os.environ.copy()

    current_path = new_env.get("PATH", "")

    # Remove virtualenv paths (they typically contain 'bin' and virtualenv name)
    path_parts = current_path.split(os.pathsep)
    filtered_path = []

    for path_part in path_parts:
        # Skip paths that look like virtualenv bin directories
        if not (
            (
                path_part.endswith("/bin")
                and any(
                    venv_indicator in path_part.lower()
                    for venv_indicator in ["tox", ".tox", "venv", "env", ".venv"]
                )
            )
            or (path_part.endswith("/shims") and ".pyenv" in path_part.lower())
        ):
            filtered_path.append(path_part)

    # Create new environment with filtered PATH
    new_env["PATH"] = os.pathsep.join(filtered_path)
    kwargs["env"] = new_env
    return subprocess.run(cmd, *args, **kwargs)


@pytest.fixture(scope="session")
def calibre_available() -> None:
    """Check if Calibre is installed and skip test if not available.

    Raises:
        pytest.skip: If Calibre is not installed or not accessible.
    """
    try:
        result = _run_without_venv(
            ["calibre", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip("Calibre is not installed or not accessible")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Calibre is not installed or not accessible")


@pytest.fixture(scope="session")
def calibre_library(calibre_available) -> Generator[pathlib.Path, None, None]:
    """Create or use existing Calibre library for testing.

    Creates a Calibre library in the specified directory (defaulting to
    .cache/testing/calibre) and populates it with books from tests/data/books.
    The library location can be overridden with the CALIBRE_LIBRARY_LOCATION
    environment variable.

    Each book directory should contain:
    - The book file(s)
    - metadata.opf file for metadata
    - sidecar.json file for custom column data

    Yields:
        pathlib.Path: The root directory of the Calibre library.
    """
    # Determine library location
    library_path = pathlib.Path(
        os.environ.get("CALIBRE_LIBRARY_LOCATION", DEFAULT_CALIBRE_LOCATION)
    )

    # Check if library already exists
    if library_path.exists() and (library_path / "metadata.db").exists():
        yield library_path
        return

    books: list[pathlib.Path] = []
    for author_dir in BOOKS_DIR.iterdir():
        if not author_dir.is_dir():
            continue
        for book_dir in author_dir.iterdir():
            if not book_dir.is_dir():
                continue
            books.append(book_dir)

    all_custom_columns = _get_custom_columns_from_books(books)

    # Create library directory
    library_path.mkdir(parents=True, exist_ok=True)

    try:
        # Add custom column for raw_sidecar
        # _run_without_venv(
        #     [
        #         "calibredb",
        #         "add_custom_column",
        #         "--library-path",
        #         os.fspath(library_path),
        #         "raw_sidecar",
        #         "Raw sidecar JSON",
        #         "text",
        #     ],
        #     check=True,
        #     capture_output=False,
        #     text=True,
        # )

        for custom_column in all_custom_columns:
            _run_without_venv(
                custom_column.to_calibredb_command(library_path),
                check=True,
                capture_output=True,
                text=True,
            )

        # Add books from tests/data/books
        for book_dir in books:
            _add_book_to_library(library_path, book_dir)

        yield library_path

    except subprocess.CalledProcessError as e:
        # Clean up on failure
        if library_path.exists():
            shutil.rmtree(library_path)
        raise RuntimeError(f"Failed to create Calibre library: {e}") from e


def _get_custom_columns_from_books(
    books: Sequence[pathlib.Path],
) -> Iterable[CustomColumnMetadata]:
    custom_columns: dict[str, CustomColumnMetadata] = {}
    for book_dir in books:
        metadata_file = book_dir / "metadata.opf"
        if not metadata_file.exists():
            continue

        book_custom_columns = parse_opf_file(metadata_file)
        for custom_column in book_custom_columns:
            if custom_column not in custom_columns:
                custom_columns[custom_column.table] = custom_column
            elif custom_columns[custom_column.table] != custom_column:
                raise ValueError(
                    f"Mismatch between {custom_columns[custom_column]} and {custom_column}"
                )

    return list(custom_columns.values())


def _add_book_to_library(library_path: pathlib.Path, book_dir: pathlib.Path) -> None:
    """Add a single book directory to the Calibre library.

    Args:
        library_path: pathlib.Path to the Calibre library root.
        book_dir: pathlib.Path to the book directory containing files and metadata.
    """
    # Find book files (exclude metadata.opf and sidecar.json)
    book_files = [
        f
        for f in book_dir.iterdir()
        if f.is_file() and f.name not in ("metadata.opf", "sidecar.json")
    ]

    if not book_files:
        return

    # Add book to library
    cmd = [
        "calibredb",
        "add",
        "--library-path",
        os.fspath(library_path),
        *map(os.fspath, book_files),
    ]

    result = _run_without_venv(cmd, check=True, capture_output=True, text=True)

    if (m := re.search(r"Added book ids:\s*(?P<id>\d+)", result.stdout)) is None:
        raise ValueError(f"Could not find book id:    {result.stdout}")

    metadata_file = book_dir / "metadata.opf"
    if metadata_file.exists():
        _run_without_venv(
            [
                "calibredb",
                "set_metadata",
                "--library-path",
                os.fspath(library_path),
                m.group("id"),
                os.fspath(metadata_file),
            ],
            check=True,
            text=True,
        )

    # # Add metadata file if it exists
    # metadata_file = book_dir / "metadata.opf"
    # if metadata_file.exists():
    #     cmd.extend(["--metadata", str(metadata_file)])

    # # Add all book files
    # cmd.extend(str(f) for f in book_files)

    # # Extract book ID from output if we have sidecar data to add
    # sidecar_file = book_dir / "sidecar.json"
    # if sidecar_file.exists():
    #     _add_sidecar_data(library_path, result.stdout, sidecar_file)


def _add_sidecar_data(
    library_path: pathlib.Path, add_output: str, sidecar_file: pathlib.Path
) -> None:
    """Add sidecar data to the raw_sidecar custom column.

    Args:
        library_path: Path to the Calibre library root.
        add_output: Output from calibredb add command containing book ID.
        sidecar_file: Path to the sidecar.json file.
    """
    try:
        # Parse book ID from calibredb output
        # Output typically contains "Added book ids: X"
        lines = add_output.strip().split("\n")
        book_id = None

        for line in lines:
            if "Added book ids:" in line:
                # Extract the ID number
                id_part = line.split("Added book ids:")[-1].strip()
                book_id = id_part.split()[0].rstrip(",")
                break

        if not book_id:
            return

        # Read sidecar data
        with sidecar_file.open() as f:
            sidecar_data = json.load(f)

        # Convert to JSON string for storage
        sidecar_json = json.dumps(sidecar_data, separators=(",", ":"))

        # Set custom column value
        _run_without_venv(
            [
                "calibredb",
                "set_metadata",
                "--library-path",
                str(library_path),
                "--field",
                f"#raw_sidecar:{sidecar_json}",
                book_id,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except (json.JSONDecodeError, subprocess.CalledProcessError, KeyError) as e:
        # Log error but don't fail
        print(f"Warning: Failed to add sidecar data from {sidecar_file}: {e}")
