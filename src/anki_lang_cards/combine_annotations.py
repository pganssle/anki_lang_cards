from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ._types import (
    BookAnnotations,
    CombinedAnnotation,
    CombinedBookAnnotations,
    HighlightWithNote
)


def combine_highlights(
    input_data: Sequence[BookAnnotations],
) -> Sequence[CombinedBookAnnotations]:
    """
    Combine consecutive annotations with overlapping contexts.

    Args:
        input_data: A sequence of dictionaries containing book and annotation data.

    Returns:
        A list of dictionaries with combined highlights where possible.
    """

    def has_overlap(context1: str, context2: str) -> bool:
        """
        Check if context1 and context2 overlap based on the specified criteria.
        """
        min_overlap_length = 10
        overlap_start = max(0, len(context1) - 2000)  # Ensure max context length
        overlap_candidate = context1[overlap_start:]

        for i in range(len(overlap_candidate)):
            overlap = overlap_candidate[i:]
            if (
                context2.startswith(overlap)
                and len(overlap) >= min_overlap_length
                and " " in overlap
            ):
                return True
        return False

    def combine_contexts(context1: str, context2: str) -> str:
        """
        Merge two contexts with overlapping regions.
        """
        min_overlap_length = 10
        overlap_start = max(0, len(context1) - 2000)
        overlap_candidate = context1[overlap_start:]

        for i in range(len(overlap_candidate)):
            overlap = overlap_candidate[i:]
            if (
                context2.startswith(overlap)
                and len(overlap) >= min_overlap_length
                and " " in overlap
            ):
                return context1 + context2[len(overlap) :]
        return context1 + " " + context2

    output: list[CombinedBookAnnotations] = []

    for book_entry in input_data:
        annotations = book_entry.annotations
        combined_annotations: list[CombinedAnnotation] = []
        current_highlight_group: list[str] = []
        current_context = ""
        current_chapters: set[str] = set()
        current_page_numbers: set[int | None] = set()

        annotation_text : str |  HighlightWithNote
        for annotation in annotations:
            if annotation.note:
                if annotation.note.startswith("K:"):
                    annotation_text = HighlightWithNote(annotation.text, annotation.note.removeprefix("K:").strip())
                else:
                    # Skip anything with a note, since that signals that it was highlighted
                    # for a reason other than it being a vocabulary word.
                    continue
            else:
                annotation_text = annotation.text

            if not current_context:
                # Start a new group
                current_highlight_group = [annotation_text]
                current_context = annotation.context
                current_chapters.add(annotation.chapter or "")
                current_page_numbers.add(annotation.page_number)
            elif has_overlap(current_context, annotation.context):
                # Merge overlapping contexts
                current_context = combine_contexts(current_context, annotation.context)
                current_highlight_group.append(annotation_text)
                current_chapters.add(annotation.chapter or "")
                current_page_numbers.add(annotation.page_number)
            else:
                # Save the current group and start a new one
                combined_annotations.append(
                    CombinedAnnotation(
                        highlights=current_highlight_group,
                        context=current_context,
                        chapters=sorted(filter(None, current_chapters)),
                        page_numbers=sorted(
                            cast(set[int], current_page_numbers - {None})
                        ),
                    )
                )
                current_highlight_group = [annotation_text]
                current_context = annotation.context
                current_chapters = {annotation.chapter or ""}
                current_page_numbers = {annotation.page_number}

        # Save any remaining annotations
        if current_highlight_group:
            combined_annotations.append(
                CombinedAnnotation(
                    highlights=current_highlight_group,
                    context=current_context,
                    chapters=sorted(filter(None, current_chapters)),
                    page_numbers=sorted(cast(set[int], current_page_numbers - {None})),
                )
            )

        output.append(
            CombinedBookAnnotations(
                book=book_entry.book, annotations=combined_annotations
            )
        )

    return output


def main() -> None:
    """
    Example usage of the highlights combiner.
    """
    parser = argparse.ArgumentParser(
        description="Extract and combine Calibre annotations"
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    # Read input data from file
    # input_data = json.loads(args.input.read_text())
    input_data = BookAnnotations.sequence_from_json(args.input)

    # Process the highlights
    result = combine_highlights(input_data)
    serializable_highlights = CombinedBookAnnotations.sequence_to_serializable(result)

    # Write combined output to file
    with open(args.output, "wt") as f:
        json.dump(serializable_highlights, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
