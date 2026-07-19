"""End-to-end tests for the Calibre annotation extractor CLI.

The test book is an abridged Don Quijote whose sidecar contains highlights
deliberately placed on edge cases: chapter headings, front matter with no
sentence punctuation, tabular lists (the table of contents), nested and
overlapping highlights, highlights inside poems (indexed ``text()[N]``
positions), and highlights in the first and last sentences of chapter files
(context must never cross a DocFragment boundary).
"""

import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

import pytest

DATA_DIR: Final = pathlib.Path(__file__).parent / "data"
EXPECTED_OUTPUT: Final = DATA_DIR / "annotations.json"

# Distinctive strings from specific chapter files (DocFragments), used to
# detect context bleeding across chapter boundaries.
FRAGMENT_3_TEXT: Final = "Yo, Juan Gallo de Andrada"
FRAGMENT_9_TEXT: Final = "END OF THE PROJECT GUTENBERG"

with EXPECTED_OUTPUT.open(encoding="utf-8") as _expected_file:
    EXPECTED_BOOKS: Final[Sequence[Mapping[str, Any]]] = json.load(_expected_file)

EXPECTED_ANNOTATIONS: Final[Sequence[Mapping[str, Any]]] = tuple(
    EXPECTED_BOOKS[0]["annotations"]
)


class ExtractorRun(NamedTuple):
    books: Sequence[Mapping[str, Any]]
    stdout: str


def _run_extractor(
    calibre_library: pathlib.Path,
    output_directory: pathlib.Path,
    language: str,
    context_sentences: int = 1,
) -> ExtractorRun:
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "anki_lang_cards.annotation_extractor",
            "--calibre-directory",
            str(calibre_library),
            "--language",
            language,
            "--context-sentences",
            str(context_sentences),
            "--output",
            str(output_directory),
            "--json",
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    with (output_directory / "annotations.json").open(encoding="utf-8") as output_file:
        books: Sequence[Mapping[str, Any]] = json.load(output_file)
    return ExtractorRun(books=books, stdout=result.stdout)


@pytest.fixture(scope="session")
def extraction(
    calibre_library: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> ExtractorRun:
    """Extractor output for Spanish with the default single context sentence."""
    return _run_extractor(
        calibre_library, tmp_path_factory.mktemp("context-1"), "Spanish"
    )


@pytest.fixture(scope="session")
def extraction_context_3(
    calibre_library: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> ExtractorRun:
    """Extractor output for Spanish with three sentences of context."""
    return _run_extractor(
        calibre_library,
        tmp_path_factory.mktemp("context-3"),
        "Spanish",
        context_sentences=3,
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _annotations_with_text(run: ExtractorRun, text: str) -> Sequence[Mapping[str, Any]]:
    return tuple(
        annotation
        for annotation in run.books[0]["annotations"]
        if annotation["text"] == text
    )


def _annotation_with_text(run: ExtractorRun, text: str) -> Mapping[str, Any]:
    (annotation,) = _annotations_with_text(run, text)
    return annotation


@pytest.mark.e2e
def test_annotation_extractor_cli(extraction: ExtractorRun) -> None:
    assert len(extraction.books) == 1
    assert len(extraction.books[0]["annotations"]) == len(EXPECTED_ANNOTATIONS)
    assert extraction.books == EXPECTED_BOOKS


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("index", "expected"),
    tuple(enumerate(EXPECTED_ANNOTATIONS)),
    ids=[
        f"{index}-{expected['text'][:30]}"
        for index, expected in enumerate(EXPECTED_ANNOTATIONS)
    ],
)
def test_context_contains_highlight(
    extraction: ExtractorRun, index: int, expected: Mapping[str, Any]
) -> None:
    """Every extracted context must contain its own highlighted text."""
    annotation = extraction.books[0]["annotations"][index]
    assert annotation["text"] == expected["text"]
    assert _normalize(annotation["text"]) in annotation["context"]
    assert not annotation["context"].startswith((".", "?", "!", ";"))


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("outer_text", "inner_text"),
    [
        ("claras aguas destos arroyos", "destos"),
        ("Fuego soy apartado", "apartado"),
        ("espada puesta lejos", "espada"),
    ],
)
def test_nested_highlights_share_sentence(
    extraction: ExtractorRun, outer_text: str, inner_text: str
) -> None:
    """A highlight nested inside another one gets the same sentence as context."""
    outer = _annotation_with_text(extraction, outer_text)["context"]
    inner = _annotation_with_text(extraction, inner_text)["context"]
    assert inner in outer or outer in inner


@pytest.mark.e2e
def test_repeated_highlight_text_gets_distinct_contexts(
    extraction: ExtractorRun,
) -> None:
    """The same words highlighted in two places produce different contexts."""
    first, second = _annotations_with_text(extraction, "vuestra merced")
    assert first["context"] != second["context"]
    assert "vuestra merced" in first["context"]
    assert "vuestra merced" in second["context"]


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("highlight", "heading"),
    [
        # <h2> chapter heading.
        ("hidalgo", "El ingenioso hidalgo don Quijote de la Mancha"),
        # <h3> heading whose text is held in the tail of an inner <a> element.
        ("TASA", "TASA"),
    ],
)
def test_highlight_in_chapter_heading(
    extraction: ExtractorRun, highlight: str, heading: str
) -> None:
    """Highlights placed on chapter headings extract the heading and what follows."""
    context = _annotation_with_text(extraction, highlight)["context"]
    assert heading in context
    # Headings have no sentence-ending punctuation, so the context runs into
    # the first sentence of the chapter body.
    assert FRAGMENT_3_TEXT in context


@pytest.mark.e2e
def test_expansion_stops_at_tabular_list(extraction: ExtractorRun) -> None:
    """A highlight on the title page must not swallow the table of contents.

    The long, punctuation-free chapter list is classified as tabular data, so
    expansion stops at its edge; the short front-matter list stays below the
    tabular threshold and is still crossed like ordinary text."""
    context = _annotation_with_text(
        extraction, "ingenioso hidalgo don Quijote de la Mancha"
    )["context"]
    assert context.endswith("Prólogo Al libro de don Quijote de la Mancha")
    assert "Que trata de la condición" not in context
    assert FRAGMENT_3_TEXT not in context


@pytest.mark.e2e
def test_highlight_inside_tabular_list_clips_to_item(extraction: ExtractorRun) -> None:
    """A highlight inside a tabular list gets exactly its own list item."""
    context = _annotation_with_text(extraction, "graciosa")["context"]
    assert context == (
        "Donde se cuenta la graciosa manera que tuvo don Quijote en armarse "
        "caballero"
    )


@pytest.mark.e2e
def test_tabular_list_items_count_as_sentences(
    extraction_context_3: ExtractorRun,
) -> None:
    """With more context requested, tabular lists expand item by item."""
    context = _annotation_with_text(extraction_context_3, "graciosa")["context"]
    assert context == (
        "Que trata de la primera salida que de su tierra hizo el ingenioso "
        "don Quijote Donde se cuenta la graciosa manera que tuvo don Quijote "
        "en armarse caballero De lo que le sucedió a nuestro caballero "
        "cuando salió de la venta"
    )


@pytest.mark.e2e
def test_backward_expansion_ascends_out_of_nested_elements(
    extraction: ExtractorRun,
) -> None:
    """Walking backwards climbs out of nested elements instead of stopping.

    The paragraph containing "escribano" is preceded by an <h3> whose text
    lives in the tail of a nested <a>; the walker has to ascend out of the
    <a> and past the <h2> to reach the (punctuation-free) top of the file."""
    context = _annotation_with_text(extraction, "escribano")["context"]
    assert context.startswith(
        "El ingenioso hidalgo don Quijote de la Mancha TASA Yo, Juan Gallo"
    )


@pytest.mark.e2e
def test_context_starts_at_sentence_start(extraction: ExtractorRun) -> None:
    """The previous sentence's terminator is not included in the context.

    This is a regression test: the context used to come out as
    ". Acontece tener un padre..."."""
    context = _annotation_with_text(extraction, "Acontece")["context"]
    assert context == (
        "Acontece tener un padre un hijo feo y sin gracia alguna, y el amor "
        "que le tiene le pone una venda en los ojos para que no vea sus "
        "faltas, antes las juzga por discreciones y lindezas y las cuenta a "
        "sus amigos por agudezas y donaires."
    )


@pytest.mark.e2e
def test_context_does_not_cross_into_next_chapter(extraction: ExtractorRun) -> None:
    context = _annotation_with_text(extraction, "The")["context"]
    assert context == (
        "The Project Gutenberg eBook of Don Quijote This ebook is for the use "
        "of anyone anywhere in the United States and most other parts of the "
        "world at no cost and with almost no restrictions whatsoever."
    )


@pytest.mark.e2e
def test_context_does_not_cross_into_previous_chapter(
    extraction_context_3: ExtractorRun,
) -> None:
    """A highlight in the first sentence of a chapter file cannot pull context
    from the preceding chapter file, even when more sentences are requested."""
    context = _annotation_with_text(extraction_context_3, "agradezco")["context"]
    assert context == (
        "— Yo agradezco tu buena intención, amigo Sancho —respondió don "
        "Quijote—, mas quiérote hacer sabidor de que todas estas cosas que "
        "hago no son de burlas, sino muy de veras; porque de otra manera, "
        "sería contravenir a las órdenes de caballería, que nos mandan que no "
        "digamos mentira alguna, pena de relasos, y el hacer una cosa por "
        "otra lo mesmo es que mentir. Ansí que, mis calabazadas han de ser "
        "verdaderas, firmes y valederas, sin que lleven nada del sofístico ni "
        "del fantástico."
    )


@pytest.mark.e2e
def test_context_at_end_of_chapter(extraction_context_3: ExtractorRun) -> None:
    """A highlight in the last paragraph of a chapter file expands to the end
    of that file without crossing into the next one."""
    context = _annotation_with_text(extraction_context_3, "tropezando")["context"]
    assert context.endswith(
        "van ya tropezando, y han de caer del todo, " "sin duda alguna. Vale."
    )
    assert FRAGMENT_9_TEXT not in context


@pytest.mark.e2e
def test_highlight_in_poem_with_indexed_text_node(extraction: ExtractorRun) -> None:
    """Positions like text()[9] index into the text nodes between <br> tags."""
    context = _annotation_with_text(extraction, "castamente")["context"]
    assert "¡Oh, quién tan castamente se escapara" in context
    # Poem lines separated by <br> are joined with spaces.
    assert "escapara del señor Amadís como tú hiciste" in context


@pytest.mark.e2e
def test_context_sentences_expands_context(
    extraction: ExtractorRun, extraction_context_3: ExtractorRun
) -> None:
    context_1 = _annotation_with_text(extraction, "diamante")["context"]
    context_3 = _annotation_with_text(extraction_context_3, "diamante")["context"]
    assert context_3.endswith(context_1)
    assert len(context_3) > len(context_1)
    assert context_3.startswith("que a tal peña podrá llegar")


@pytest.mark.e2e
def test_note_is_preserved(extraction: ExtractorRun) -> None:
    annotation = _annotation_with_text(
        extraction, "ingenioso hidalgo don Quijote de la Mancha"
    )
    assert annotation["note"] == "Skip me"
    assert all(
        other["note"] == ""
        for other in extraction.books[0]["annotations"]
        if other is not annotation
    )


@pytest.mark.e2e
def test_annotation_without_positions_is_skipped(extraction: ExtractorRun) -> None:
    """The sidecar contains one malformed annotation with no pos0; it must be
    skipped with a warning rather than aborting the whole book."""
    assert not _annotations_with_text(extraction, "missing position")
    assert "Warning: Skipping annotation" in extraction.stdout


@pytest.mark.e2e
def test_annotation_extractor_filters_other_languages(
    calibre_library: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    run = _run_extractor(calibre_library, tmp_path, "English")
    assert run.books == []
