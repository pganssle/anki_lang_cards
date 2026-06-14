import functools
import pathlib

import pytest

from anki_lang_cards import annotation_extractor


@functools.cache
def get_book_data(
    calibre_dir: pathlib.Path,
    sidecar_column: str = "raw_sidecar",
    language: str = "Spanish",
) -> annotation_extractor.BookData:
    return annotation_extractor.load_annotations_from_db(
        calibre_dir / "metadata.db",
        sidecar_column=sidecar_column,
        language=language,
    )


@functools.lru_cache
def get_annotations(
    calibre_dir: pathlib.Path,
    context_sentences: int,
    sidecar_column: str = "raw_sidecar",
    language: str = "Spanish",
) -> annotation_extractor.BookData:
    book_data = get_book_data(calibre_dir, sidecar_column, language)
    return list(
        annotation_extractor.process_all_annotations(
            book_data, calibre_dir, context_sentences=context_sentences
        )
    )


def test_annotation_loading(calibre_library: pathlib.Path) -> None:
    book_data = get_book_data(calibre_library)

    assert len(book_data) == 1


def test_annotation_processing_1_sentence(calibre_library: pathlib.Path) -> None:
    all_annotations = get_annotations(calibre_library, 1)
