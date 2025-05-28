from __future__ import annotations

import argparse
import functools
import json
import os
import sqlite3
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Final, NotRequired, Self, TypedDict

import attrs
import ebooklib
import regex
from ebooklib import epub
from lxml import etree

from ._types import Annotation, BookAnnotations, BookData

LANGUAGE_MAPPINGS: Final[Mapping[str, str]] = {
    "spa": "spa",
    "es": "spa",
    "spanish": "spa",
    "esp": "spa",
    "eng": "eng",
    "en": "eng",
    "english": "eng",
}

# This regex should match most common sentence endings in English and
# Spanish. It can be expanded if needed.
_SENTENCE_END_REGEX_PATTERN: Final[str] = r"[.?!;]+[\s\"]*"
SENTENCE_END_REGEX: Final[regex.Pattern] = regex.compile(_SENTENCE_END_REGEX_PATTERN)
SENTENCE_END_REGEX_REVERSED: Final[regex.Pattern] = regex.compile(
    _SENTENCE_END_REGEX_PATTERN, flags=regex.REVERSE
)


@attrs.frozen
class Position:
    doc_id: int
    xpath: str
    text_idx: int | None
    index: int | None

    @classmethod
    def from_string(cls, pos: str) -> Self:
        m = regex.match(
            r".*DocFragment\[(?P<id>\d+)\]/(?P<path>.*?)"
            + r"(/text\(\)|/text\(\)\[(?P<text_idx>\d+)\])?"
            + r"(?P<idx>\.\d+)?$",
            pos,
        )
        if m is None:
            return f"[Invalid document path: {pos}]"

        doc_id = int(m.group("id"))
        xpath = f'.//{m.group("path")}'
        if (idx_str := m.group("idx")) is None:
            index = None
        else:
            index = int(idx_str[1:])

        if (text_idx := m.group("text_idx")) is not None:
            text_idx = int(text_idx) - 1  # xpath is using a 1-based index
        return cls(doc_id=doc_id, xpath=xpath, text_idx=text_idx, index=index)


@attrs.frozen
class PositionRange:
    start: Position
    end: Position


@attrs.frozen
class DocPosition:
    node: etree.Element
    index: int = 0
    document_beginning: bool = False
    document_end: bool = False

    @functools.cached_property
    def node_text(self) -> str:
        return (self.node.text or "") + (self.node.tail or "")

    @classmethod
    def from_position(cls, doc: etree.Element, pos: Position) -> Self:
        (node,) = doc.xpath(pos.xpath)
        index = pos.index
        if pos.text_idx is not None and pos.text_idx > 0:
            node = node.getchildren()[pos.text_idx - 1]
            if index is not None and node.text is not None:
                # When specified on a text fragment, the index is relative to
                # the "tail" of the child element we've chosen, so we need to
                # add in the length of the text contents of the child element
                # to make the indexing work out properly.
                index += len(node.text)
        return cls(node=node, index=index)


def _next_position(cpos: DocPosition) -> DocPosition:
    node_children = cpos.node.getchildren()
    if node_children:
        # Search any children of the current node. Anything between children is
        # going to be in the tail of the previous child node.
        cpos = DocPosition(node=next(iter(node_children)))
    elif (next_sibling := cpos.node.getnext()) is not None:
        cpos = DocPosition(node=next_sibling)
    else:
        # If it has no siblings, we go up the stack until we find the next thing. If we
        # don't find any sentence ends, we're presumably at the end of the document.
        parent = cpos.node.getparent()
        while parent is not None:
            next_sibling = parent.getnext()
            if next_sibling is not None:
                cpos = DocPosition(node=next_sibling)
                break
            parent = parent.getparent()
        else:
            return DocPosition(
                node=cpos.node,
                index=len(cpos.node_text),
                document_end=True,
            )
    return cpos


@attrs.frozen
class DocPositionRange:
    start: DocPosition
    end: DocPosition

    @functools.cached_property
    def text(self) -> str:
        text_out = []
        cpos = self.start
        while True:
            if cpos.node == self.end.node:
                text_out.append(cpos.node_text[cpos.index : self.end.index])
                break
            else:
                assert not cpos.document_end
                text_out.append(cpos.node_text[cpos.index :])
                cpos = _next_position(cpos)

        return regex.sub(r"\s+", " ", " ".join(text_out)).strip()


class AnnotationData(TypedDict):
    chapter: str
    datetime: str
    drawer: str
    note: NotRequired[str]
    page: str
    pageno: int
    pos0: str
    pos1: str
    text: str


class DatabaseQueries:
    GET_COLUMN_ID: Final[
        str
    ] = """
        SELECT id FROM custom_columns 
        WHERE label = ?
    """

    GET_BOOKS: Final[
        str
    ] = """
        SELECT 
            books.title,
            authors.name as author,
            books.path,
            languages.lang_code as language,
            custom_column_{}.value as annotations
        FROM books
        JOIN books_authors_link ON books.id = books_authors_link.book
        JOIN authors ON books_authors_link.author = authors.id
        LEFT JOIN books_languages_link ON books.id = books_languages_link.book
        LEFT JOIN languages ON books_languages_link.lang_code = languages.id
        LEFT JOIN custom_column_{} ON books.id = custom_column_{}.book
        WHERE custom_column_{}.value IS NOT NULL
        {}
    """


def _strip_ns_prefixes(tree: etree.Element) -> None:
    # iterate through only element nodes (skip comment node, text node, etc) :
    for element in tree.xpath("descendant-or-self::*"):
        if element.prefix:
            element.tag = etree.QName(element).localname
        element.tag = regex.sub(r"{[^}]+}", "", element.tag)


@attrs.frozen
class EPUBContext:
    """Caches and manages EPUB document parsing."""

    book: epub.EpubBook
    _doc_cache: MutableMapping[str | int, etree._Element] = attrs.field(factory=dict)

    def get_document_from_id(self, doc_id: str | int) -> etree._Element:
        if doc_id not in self._doc_cache:
            if isinstance(doc_id, int):
                doc_id -= 1
                if 0 <= doc_id < len(self.book.spine):
                    item_id = self.book.spine[doc_id][0]
                    item = self.book.get_item_with_id(item_id)
            elif isinstance(doc_id, str):
                for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                    if isinstance(item, epub.EpubHtml) and (
                        doc_id in item.id or doc_id in item.file_name
                    ):
                        item.id = item.id
                        break
            else:
                raise KeyError("Document ID not found: {doc_id}")

            item_content = item.content.replace(b"&nbsp;", b" ")
            try:
                parsed_doc = etree.fromstring(item_content)
            except etree.XMLSyntaxError:
                # Fallback for potentially malformed HTML
                parsed_doc = etree.fromstring(f"<html>{item_content}</html>")

            _strip_ns_prefixes(parsed_doc)
            self._doc_cache[doc_id] = self._doc_cache.setdefault(item_id, parsed_doc)

        return self._doc_cache[doc_id]

    def get_context_sentences(
        self, pos_range: PositionRange, max_sentences: int
    ) -> str:
        if pos_range.start.doc_id != pos_range.end.doc_id:
            raise NotImplementedError(
                "PositionRange spanning multiple documents is not supported."
            )

        # No both cases if text_idx is set, the path points to something in the
        # tail of the `text_idx-1`th element
        doc = self.get_document_from_id(pos_range.start.doc_id)
        start = DocPosition.from_position(doc, pos_range.start)
        end = DocPosition.from_position(doc, pos_range.end)
        if end.index > 0:
            end = attrs.evolve(end, index=end.index - 1)

        out_range = DocPositionRange(start=start, end=end)

        # Initial expansion to the beginning of the start node's sentence
        out_range = self._expand_range_backward(out_range)
        # Initial expansion to the end of the end node's sentence
        out_range = self._expand_range_forward(out_range)
        sentence_count = self._count_sentences(out_range)

        while sentence_count < max_sentences and not (
            out_range.start.document_beginning and out_range.end.document_end
        ):
            # Expand backwards
            if not out_range.start.document_beginning:
                out_range = self._expand_range_backward(out_range)
                if out_range.start.document_beginning:
                    sentence_count = self._count_sentences(out_range)
                else:
                    sentence_count += 1

                if sentence_count >= max_sentences:
                    break

            # Expand forwards
            if not out_range.end.document_end:
                out_range = self._expand_range_forward(out_range)
                if out_range.end.document_end:
                    sentence_count = self._count_sentences(out_range)
                else:
                    sentence_count += 1

                if sentence_count >= max_sentences:
                    break

        return out_range.text

    def _next_sentence_end(self, position: DocPosition) -> DocPosition:
        cpos = position

        while True:
            # Search the current node and any "tails". This covers situations like:
            #
            # <p>This is a sentence <span>this is a span</span> and there's some stuff after.</p>"
            #                             ^--node.text-^       ^------node.tail-------------^
            match = SENTENCE_END_REGEX.search(cpos.node_text[cpos.index :])
            if match:
                return DocPosition(node=cpos.node, index=cpos.index + match.end())

            cpos = _next_position(cpos)
            if cpos.document_end:
                return cpos

    def _expand_range_forward(
        self,
        doc_pos_range: DocPositionRange,
    ) -> DocPositionRange:
        return attrs.evolve(
            doc_pos_range,
            end=self._next_sentence_end(doc_pos_range.end),
        )

    def _previous_sentence_end(self, position: DocPosition) -> DocPosition:
        cpos = position
        last_pos = cpos
        while True:
            if cpos.index != 0:
                match = SENTENCE_END_REGEX_REVERSED.search(
                    cpos.node_text[: cpos.index - 1]
                )
                if match:
                    if match.end() == len(cpos.node_text.rstrip()):
                        return DocPosition(node=last_pos.node, index=0)
                    else:
                        return DocPosition(
                            node=cpos.node, index=max(0, match.end() - 1)
                        )

            next_sibling = cpos.node.getprevious()
            if next_sibling is None:
                next_sibling = cpos.node.getprevious()

            if next_sibling is None:
                return DocPosition(node=cpos.node, index=0, document_beginning=True)

            # We want the last child of the previous element (recursively)
            while node_children := next_sibling.getchildren():
                next_sibling = node_children[-1]

            last_pos = cpos
            cpos = DocPosition(node=next_sibling, index=-1)

    def _expand_range_backward(
        self,
        doc_pos_range: DocPositionRange,
    ) -> DocPositionRange:

        new_start = self._previous_sentence_end(doc_pos_range.start)
        return attrs.evolve(doc_pos_range, start=new_start)

    def _count_sentences(
        self,
        drange: DocPositionRange,
    ) -> int:
        count = len(SENTENCE_END_REGEX.findall(drange.text))
        return count

    def _extract_text_content(
        self, start_node: etree.Element, end_node: etree.Element
    ) -> str:
        output = []
        current_node = start_node
        while current_node is not None:
            # Remove unwanted tags and attributes, while keeping simple
            # decorative markup.
            for child in current_node.iterchildren():
                if child.tag in (
                    "a",
                    "img",
                    "span",
                    "div",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                ):
                    child.drop_tree()
                elif child.tag == "font":
                    child.drop_tag()
                else:
                    for attr in child.attrib:
                        if attr not in ("style", "class", "id"):
                            child.attrib.pop(attr)

            output.append(current_node.text or "")
            if current_node == end_node:
                break
            current_node = current_node.getnext()

        return " ".join(output)

    def extract_context(
        self, pos0: str, pos1: str, context_sentences: int, chapter: str
    ) -> str:
        """Extract context for a specific annotation."""
        # Find text containing the annotation
        pos_range = PositionRange(
            start=Position.from_string(pos0),
            end=Position.from_string(pos1),
        )

        sentences = self.get_context_sentences(
            pos_range, max_sentences=context_sentences
        )

        return sentences


def get_language_clause(language: str | None) -> tuple[str, tuple[str, ...]]:
    if not language:
        return "", ()
    normalized = LANGUAGE_MAPPINGS.get(language.lower())
    if not normalized:
        raise ValueError(f"Unsupported language: {language}")
    return "AND languages.lang_code = ?", (normalized,)


@functools.lru_cache(maxsize=16)
def parse_epub_context(book_path: Path) -> EPUBContext:
    """Cached EPUB parsing with LRU caching."""
    return EPUBContext(epub.read_epub(str(book_path)))


def process_annotations(
    calibre_dir: Path, book_data: Mapping[str, Any], context_sentences: int
) -> BookAnnotations:
    # Find the EPUB file
    book_path = calibre_dir / book_data["path"]
    epub_files = list(book_path.glob("*.epub"))
    if not epub_files:
        raise FileNotFoundError(f"No EPUB found in {book_path}")

    # Parse raw annotations (note the change in dictionary access)
    raw_annotations = json.loads(book_data["annotations"])["annotations"]

    # Prepare book info
    book_info = BookData(
        title=book_data["title"],
        author=book_data["author"],
        path=os.fspath(book_path.relative_to(calibre_dir)),
        language=book_data["language"],
    )

    # Parse EPUB context
    epub_context = parse_epub_context(epub_files[0])

    # Process annotations
    processed_annotations: list[Annotation] = []
    for annotation in raw_annotations.values():
        annotation_data = AnnotationData(**annotation)
        if "pos0" not in annotation_data:
            print("Warning: Skipping annotation")
            print(annotation_data)
            print(book_info)
            continue

        context = epub_context.extract_context(
            annotation_data["pos0"],
            annotation_data["pos1"],
            context_sentences,
            annotation_data["chapter"],
        )

        processed_annotations.append(
            Annotation(
                text=annotation_data["text"],
                context=context,
                note=annotation_data.get("note", ""),
                chapter=annotation_data["chapter"],
                page_number=annotation_data["pageno"],
            )
        )

    return BookAnnotations(book=book_info, annotations=processed_annotations)


def export_to_json(data: Sequence[BookAnnotations], output_path: Path) -> None:
    serializable_data = BookAnnotations.sequence_to_serializable(data)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)


def export_to_csv(data: Sequence[BookAnnotations], output_path: Path) -> None:
    import csv

    headers = [
        "book_title",
        "book_author",
        "book_language",
        "text",
        "context",
        "note",
        "chapter",
        "page_number",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for book_data in data:
            book = book_data.book
            for annotation in book_data.annotations:
                writer.writerow(
                    {
                        "book_title": book.title,
                        "book_author": book.author,
                        "book_language": book.language,
                        "text": annotation.text,
                        "context": annotation.context,
                        "note": annotation.note,
                        "chapter": annotation.chapter,
                        "page_number": annotation.page_number,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Calibre annotations")
    parser.add_argument("-i", "--calibre-directory", type=Path, required=True)
    parser.add_argument("-l", "--language", type=str)
    parser.add_argument("--context-sentences", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar-column", type=str, default="raw_sidecar")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--base-output-filename", type=str, default="annotations")

    args = parser.parse_args()

    db_path = args.calibre_directory / "metadata.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Calibre database not found at {db_path}")

    args.output.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get column ID
        cursor.execute(DatabaseQueries.GET_COLUMN_ID, (args.sidecar_column,))
        column_id = cursor.fetchone()
        if not column_id:
            raise ValueError(f"Column {args.sidecar_column} not found")

        # Build query with language filter if specified
        lang_clause, lang_params = get_language_clause(args.language)
        query = DatabaseQueries.GET_BOOKS.format(
            column_id[0], column_id[0], column_id[0], column_id[0], lang_clause
        )

        # Get books and annotations
        cursor.execute(query, lang_params)
        books_data = cursor.fetchall()

        # Process annotations
        results = [
            process_annotations(
                args.calibre_directory, dict(book), args.context_sentences
            )
            for book in books_data
        ]

        results = [result for result in results if result.annotations]

        # Export results
        if args.json:
            export_to_json(results, args.output / f"{args.base_output_filename}.json")

        if args.csv:
            export_to_csv(results, args.output / f"{args.base_output_filename}.csv")


if __name__ == "__main__":
    main()
