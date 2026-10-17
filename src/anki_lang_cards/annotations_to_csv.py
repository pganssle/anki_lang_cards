"""Module to use Retrieval Augmented Generation to generate input sentences."""

import argparse
import asyncio
import csv
import itertools
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict, cast

import attrs
import tqdm

from . import (
    _types,
    caches,
    config,
    dictionary_entries,
    model_selection,
    prompts,
    query,
    runner,
)
from ._types import (
    BookCitation,
    CombinedAnnotation,
    CombinedBookAnnotations,
    HighlightWithNote,
    Sentence,
    Word,
)
from .backends import base

WordMap: TypeAlias = Mapping[str, Sequence[dictionary_entries.DictionaryEntry]]


class AnnotationEntry(TypedDict):
    highlights: Sequence[str | HighlightWithNote]
    context: str
    source_id: int


@attrs.frozen
class PromptInput:
    rag_entries: Mapping[str, str]
    annotation_entry: AnnotationEntry
    source: BookCitation


def _format_definition_list(
    definitions: Iterable[dictionary_entries.Definition], indent=2
) -> Iterable[str]:
    indent_str = " " * indent
    for ii, definition in enumerate(definitions):
        yield f"{indent_str}{ii}. {definition.meaning}"


def dictionary_entry_to_rag_input(entry: dictionary_entries.DictionaryEntry) -> str:
    if all(
        entry.definitions[0].pos == definition.pos for definition in entry.definitions
    ):
        # All the same part of speech, we can put that in with the headword and
        # not indent as much. Format looks like:
        #
        # llama (n)
        #   1. flame
        #   2. llama (animal)

        entry_pos = entry.definitions[0].pos
        if entry_pos:
            headword = f"{entry.headword} ({entry_pos})"
        else:
            headword = entry.headword
        lines: list[str] = [headword]
        lines.extend(_format_definition_list(entry.definitions))
    else:
        # Mixed part of speech, so we use a format that separtes out
        # definitions by part of speech. Format looks like:
        #
        # largo
        #   (adj)
        #     1. long
        #     2. lengthy
        #   (n)
        #     1. length
        #     2. largo (musical movement)

        # We will assume that the definitions for parts of speech are already
        # grouped together, and if they aren't it's for some good reason.
        lines = [entry.headword]
        for pos, group in itertools.groupby(entry.definitions, key=lambda x: x.pos):
            lines.append(f"  ({pos})")
            lines.extend(_format_definition_list(group, indent=4))

    return "\n".join(lines)


def lookup_words(
    engine: dictionary_entries.DictionaryEngine,
    combined_annotations: Sequence[CombinedBookAnnotations],
) -> Mapping[str, Sequence[dictionary_entries.DictionaryEntry]]:
    output: dict[str, Sequence[dictionary_entries.DictionaryEntry]] = {}
    for annotation in itertools.chain.from_iterable(
        book_annotation.annotations for book_annotation in combined_annotations
    ):
        for word in annotation.highlights:
            if hasattr(word, "highlight"):
                word = word.highlight

            entry = engine.lookup(word)
            if entry:
                output[word] = entry

    return output


StringPattern: TypeAlias = str | dict[Literal["pattern"], str]
RangePattern: TypeAlias = Sequence[int | dict[Literal["range"], tuple[int, int]]]


def handle_book_skipping(
    book_skip_list: Path, combined_book_annotations: Sequence[CombinedBookAnnotations]
) -> Sequence[CombinedBookAnnotations]:
    # This is absurdly complicated for what it does 😅

    def compile_string_match(pattern: StringPattern) -> Callable[[str], bool]:
        if isinstance(pattern, dict):
            if "pattern" not in pattern:
                raise ValueError(f"Unknown rule type: {pattern}")
            compiled_regex = re.compile(pattern["pattern"])
            return lambda value, cr=compiled_regex: cr.match(value) is not None
        else:
            return lambda value, pattern=pattern: str(value) == pattern

    def compile_range_matcher(ranges: RangePattern) -> Callable[[int], bool]:
        def make_match_range(start: int, end: int) -> Callable[[int], bool]:
            if start == 0 and end == -1:

                def match_range(page: int) -> bool:
                    return True

            elif start == -1:
                raise ValueError("Start of range cannot be -1")
            elif end == -1:

                def match_range(page: int) -> bool:
                    return page >= start

            elif start == end:

                def match_range(page: int) -> bool:
                    return page == start

            else:

                def match_range(page: int) -> bool:
                    return start <= page <= end

            return match_range

        range_matchers = []
        for range_item in ranges:
            if isinstance(range_item, dict):
                if "range" not in range_item:
                    raise ValueError(f"Unknown rule: {range_item}")
                start, end = range_item["range"]
                range_matchers.append(make_match_range(start, end))
            else:
                range_matchers.append(make_match_range(range_item, range_item))

        if len(range_matchers) == 1:
            return range_matchers[0]
        else:

            def match_ranges(page: int) -> bool:
                return any(match_range(page) for match_range in range_matchers)

            return match_ranges

    def compile_book_rule(
        rule: dict[str, StringPattern | RangePattern],
    ) -> Callable[[CombinedBookAnnotations], CombinedBookAnnotations | None]:
        rule_matchers = []
        for field in ("title", "author", "path", "language"):
            if field in rule:
                string_match = compile_string_match(rule[field])

                def rule_matcher(
                    book_annotations: CombinedBookAnnotations,
                    match=string_match,
                    field=field,
                ) -> bool:
                    return match(getattr(book_annotations.book, field))

                rule_matchers.append(
                    cast(Callable[[CombinedBookAnnotations], bool], rule_matcher)
                )

        if len(rule_matchers) == 1:
            book_filter = rule_matchers[0]
        else:

            def match_book(book_annotations: CombinedBookAnnotations) -> bool:
                return all(
                    rule_matcher(book_annotations) for rule_matcher in rule_matchers
                )

            book_filter = match_book

        if "page_numbers" not in rule and "chapter" not in rule:

            def filter_book(book_annotations: CombinedBookAnnotations):
                if book_filter(book_annotations):
                    return None
                else:
                    return book_annotations

            return filter_book

        # If we have page number or chapters, we have to compile those rules
        # and create a subset
        def subset_rule(
            rule: dict[str, StringPattern | RangePattern],
        ) -> Callable[[CombinedBookAnnotations], CombinedBookAnnotations | None]:
            annotation_matchers = []
            if "chapters" in rule:
                compiled_chapter_rule = compile_string_match(rule["chapters"])

                def match_chapter(annotation: CombinedAnnotation) -> bool:
                    return any(
                        compiled_chapter_rule(chapter)
                        for chapter in annotation.chapters
                    )

                annotation_matchers.append(match_chapter)
            if "page_numbers" in rule:
                compiled_range_rule = compile_range_matcher(rule["page_numbers"])

                def match_pages(annotation: CombinedAnnotation) -> bool:
                    return all(
                        compiled_range_rule(page) for page in annotation.page_numbers
                    )

                annotation_matchers.append(match_pages)

            if len(annotation_matchers) == 1:
                return annotation_matchers[0]
            else:

                def match_annotation(annotation: CombinedAnnotation) -> bool:
                    return any(
                        annotation_matcher(annotation)
                        for annotation_matcher in annotation_matchers
                    )

                return match_annotation

        compiled_subset_rule = subset_rule(rule)

        def filter_annotations(
            book_annotations: CombinedBookAnnotations,
        ) -> CombinedBookAnnotations | None:
            if not book_filter(book_annotations):
                return book_annotations  # Untouched by this rule
            filtered_annotations = [
                annotation
                for annotation in book_annotations.annotations
                if not compiled_subset_rule(annotation)
            ]
            if not filtered_annotations:
                return None
            return attrs.evolve(book_annotations, annotations=filtered_annotations)

        return filter_annotations

    book_skips = json.loads(book_skip_list.read_text())
    rules = [compile_book_rule(rule) for rule in book_skips]

    book_annotations = list(combined_book_annotations)
    for rule in rules:
        new_book_annotations = []
        for book_annotation in book_annotations:
            filtered = rule(book_annotation)
            if filtered:
                new_book_annotations.append(filtered)
        book_annotations = new_book_annotations
    return book_annotations


def generate_processed_pages_json(
    combined_book_annotations: Sequence[CombinedBookAnnotations],
) -> Sequence[Mapping[str, Any]]:
    """Generate a JSON object representing pages that have been processed.

    The format matches the existing book skipping syntax.
    """
    processed_books: list[dict[str, Any]] = []

    for book_annotation in combined_book_annotations:
        # Skip books with no annotations
        if not book_annotation.annotations:
            continue

        # For each book, collect all page numbers
        all_pages: Sequence[int] = sorted(
            {
                page
                for annotation in book_annotation.annotations
                for page in annotation.page_numbers
            }
        )

        if not all_pages:
            continue

        # Create a book entry with page range from first to last page
        book_entry: dict[str, StringPattern | RangePattern] = {
            "title": book_annotation.book.title,
            "author": book_annotation.book.author,
        }

        # Add page range from first to last annotated page
        book_entry["page_numbers"] = [{"range": (all_pages[0], all_pages[-1])}]

        processed_books.append(book_entry)

    return processed_books


def _prepare_prompt_inputs(
    combined_annotations: Sequence[CombinedAnnotation],
    word_map: WordMap,
    book_title: str,
    book_authors: Sequence[str],
) -> Iterable[PromptInput]:
    for annotation in combined_annotations:
        citation = BookCitation(
            source_type="book",
            title=book_title,
            authors=book_authors,
            pages=annotation.page_numbers,
            chapters=annotation.chapters,
        )

        highlights = sorted(set(annotation.highlights), key=annotation.highlights.index)

        annotation_entry = {
            "highlights": highlights,
            "context": annotation.context,
            "source_id": citation.id,
        }

        rag_entries = {}
        for highlight in annotation.highlights:
            if highlight not in word_map:
                continue

            for entry in word_map[highlight]:
                if entry.headword in rag_entries:
                    continue

                rag_entries[entry.headword] = dictionary_entry_to_rag_input(entry)

        yield PromptInput(
            rag_entries=rag_entries, annotation_entry=annotation_entry, source=citation
        )


def prepare_prompt_inputs(
    combined_book_annotations: Sequence[CombinedBookAnnotations], word_map: WordMap
) -> Sequence[PromptInput]:
    output: list[PromptInput] = []
    for book_annotations in combined_book_annotations:
        output.extend(
            _prepare_prompt_inputs(
                book_annotations.annotations,
                word_map,
                book_title=book_annotations.book.title,
                book_authors=[book_annotations.book.author],
            )
        )
    return output


def _generate_prompt(
    prompt_inputs: Sequence[PromptInput], dialect: str = config.DEFAULT_DIALECT
) -> base.Request:
    annotation_input = json.dumps(
        [prompt_input.annotation_entry for prompt_input in prompt_inputs],
        indent=2,
        ensure_ascii=False,
        default=_types._cattrs_converter().unstructure,
    )
    all_rag_entries: list[str] = []
    seen_headwords: set[str] = set()
    for prompt_input in prompt_inputs:
        for headword, rag_entry in prompt_input.rag_entries.items():
            if headword in seen_headwords:
                continue
            seen_headwords.add(headword)
            all_rag_entries.append(rag_entry)

    return base.Request(
        json_schema=prompts.BOOK_ANNOTATION_RESPONSE_SCHEMA,
        system_prompt=prompts.BOOK_ANNOTATION_SYSTEM_PROMPT % {"dialect": dialect},
        prompt=prompts.BOOK_ANNOTATION_PROMPT_TEMPLATE
        % {
            "input": annotation_input,
            "rag_entries": "\n\n".join(all_rag_entries),
        },
    )


def _estimate_tokens(request: base.Request) -> float:
    return (len(request.system_prompt) + len(request.prompt)) / 4


def pack_prompt_inputs(
    prompt_inputs: Sequence[PromptInput],
    max_tokens: int = 2048,
    dialect: str = config.DEFAULT_DIALECT,
) -> Iterable[Sequence[PromptInput]]:
    """Group annotation entries into as few prompts as fit in ``max_tokens``.

    An entry too big to share a prompt with anything gets one to itself.
    """
    if not prompt_inputs:
        raise ValueError("No prompt inputs provided")

    stack: list[PromptInput] = []
    for prompt_input in prompt_inputs:
        if stack and (
            _estimate_tokens(_generate_prompt([*stack, prompt_input], dialect=dialect))
            > max_tokens
        ):
            yield tuple(stack)
            stack = []
        stack.append(prompt_input)

    yield tuple(stack)


def generate_prompts(
    prompt_inputs: Sequence[PromptInput],
    max_tokens: int = 2048,
    dialect: str = config.DEFAULT_DIALECT,
) -> Iterable[base.Request]:
    """The prompt for each group from :func:`pack_prompt_inputs`."""
    for group in pack_prompt_inputs(prompt_inputs, max_tokens, dialect=dialect):
        yield _generate_prompt(group, dialect=dialect)


def _prepare_source_mapping(
    prompt_inputs: Sequence[PromptInput],
) -> Mapping[int, BookCitation]:
    return {
        prompt_input.source.id: prompt_input.source for prompt_input in prompt_inputs
    }


def _sentence_to_citation(
    sentence: dict[str, str | int], source_mapping: Mapping[int, BookCitation]
) -> BookCitation | None:
    if "source_id" not in sentence:
        return None
    source_id = sentence["source_id"]
    if source_id not in source_mapping:
        print(f"Unknown citation id {source_id} for sentence:\n{sentence}")
        return None
    return source_mapping[source_id]


def parse_llm_response(
    response_text: str, model_used: str, source_mapping: Mapping[int, BookCitation]
) -> Sequence[Word]:
    output: list[Word] = []
    for highlight in query.decode_words(response_text):
        try:
            headword = highlight["headword"]
            meaning = highlight["meaning"]
            part_of_speech = highlight["part_of_speech"]
            article = highlight.get("article", None) or None
            sentences = [
                Sentence(
                    spanish=sentence["spanish"],
                    english=sentence["english"],
                    source=_sentence_to_citation(sentence, source_mapping),
                )
                for sentence in highlight["examples"]
            ]
        except KeyError as e:
            raise ValueError(f"Response entry is missing {e}: {highlight}") from e

        output.append(
            Word(
                rank=None,
                word=headword,
                meaning=meaning,
                part_of_speech=part_of_speech,
                article=article,
                sentences=sentences,
                model_used=model_used,
            )
        )

    return output


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extracted and combined annotations into Anki cards"
    )
    parser.add_argument(
        "-i", "--input", type=Path, default="output/combined_annotations.json"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default="output/words_from_annotations.csv"
    )
    parser.add_argument(
        "--dictionary",
        action="append",
        type=Path,
        default=[],
        help="Path to a dictionary in STARDICT format. This can also be "
        "specified in the config file under annotations.dictionaries.",
    )
    parser.add_argument(
        "--hunspell-dict-path",
        type=Path,
        default=Path("/usr/share/hunspell/es_ES.dic"),
        help="Path to the hunspell dictionary for headword stemming",
    )
    model_selection.add_arguments(parser)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="How many times to retry a prompt whose response could not be "
        "used before giving up on it.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Config file location",
    )
    parser.add_argument(
        "--skip-list",
        type=Path,
        default=Path("input/book_skip_list.json"),
        help="List of books to skip, or page ranges within books",
    )
    parser.add_argument(
        "--processed-pages-output",
        type=Path,
        default="output/processed_pages.json",
        help="Path to output a JSON file containing processed pages",
    )
    parser.add_argument(
        "--invalidate-response-cache",
        action="store_true",
        help=(
            "If specified, delete any existing caches for this given response "
            "and replace them with the new query response."
        ),
    )

    parser.add_argument(
        "--skip-response-cache",
        action="store_true",
        help="If specified, don't use existing caches, but also don't replace them.",
    )

    parser.add_argument(
        "--csv-header",
        action="store_true",
        help="Include a header row in the output CSV. Note that Anki's CSV "
        "importer has no header-skipping option, so a header row imports "
        "as a junk note; it is useful mainly for matching up columns in "
        "the import dialog.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many annotation entries; useful for a "
        "trial run over a subset before committing to a full one. The "
        "processed-pages output is not written for a limited run.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    if args.skip_response_cache and args.invalidate_response_cache:
        raise ValueError(
            "At most one of --skip-response-cache and "
            "--invalidate-response-cache may be specified."
        )

    conf = config.Config.from_file(args.config)
    try:
        model, query_runner = model_selection.make_runner(
            args, conf, caches.CacheTypes.ANNOTATIONS
        )
    except model_selection.SelectionError as e:
        parser.exit(2, f"error: {e}\n")

    dictionaries: Sequence[Path] = args.dictionary
    if conf.annotations.dictionaries:
        dictionaries = (*dictionaries, *conf.annotations.dictionaries)

    combined_annotations = CombinedBookAnnotations.sequence_from_json(args.input)
    if args.skip_list.exists():
        combined_annotations = handle_book_skipping(
            args.skip_list, combined_annotations
        )

    engine = dictionary_entries.DictionaryEngine(
        hunspell_dict_path=args.hunspell_dict_path,
        stardict_paths=dictionaries,
    )

    word_map = lookup_words(engine, combined_annotations)
    prompt_inputs = list(prepare_prompt_inputs(combined_annotations, word_map))
    limited = args.limit is not None and args.limit < len(prompt_inputs)
    if limited:
        print(f"Limiting to the first {args.limit} of {len(prompt_inputs)} entries.")
        prompt_inputs = prompt_inputs[: args.limit]
    source_mapping = _prepare_source_mapping(prompt_inputs)

    existing: dict[Word, Word] = {}
    inputs_by_request = {
        _generate_prompt(group, dialect=conf.sentences.dialect): group
        for group in pack_prompt_inputs(prompt_inputs, dialect=conf.sentences.dialect)
    }
    requests = list(inputs_by_request)
    skipped: list[runner.Skipped] = []

    def parse(request: base.Request, response: caches.ResponseData) -> Sequence[Word]:
        return parse_llm_response(
            response["text"], model_used=model.spec, source_mapping=source_mapping
        )

    mode = (
        "as a batch job"
        if query_runner.batch
        else f"with up to {query_runner.max_concurrency} parallel queries"
    )
    print(f"Processing {len(requests)} prompts {mode}.")
    async with query_runner:
        with tqdm.tqdm(total=len(requests), desc="Processing prompts") as progress:
            async for outcome in runner.run_with_retries(
                query_runner,
                requests,
                parse,
                max_retries=args.max_retries,
                describe=lambda request: f"prompt {request.key}",
                invalidate_cache=args.invalidate_response_cache,
                skip_cache=args.skip_response_cache,
            ):
                progress.update()
                if isinstance(outcome, runner.Skipped):
                    skipped.append(outcome)
                    continue
                for word in outcome.value:
                    if word in existing:
                        existing[word].update(word)
                    else:
                        existing[word] = word

    print(f"Generated {len(existing)} entries.")
    with open(args.output, "wt") as f:
        writer = csv.writer(f)
        if args.csv_header:
            writer.writerow(Word.make_csv_header())
        for word in existing.values():
            writer.writerow(word.to_csv_line())

    print(f"Written to {args.output}")
    if skipped:
        lost = [
            entry for outcome in skipped for entry in inputs_by_request[outcome.request]
        ]
        highlights = sum(len(entry.annotation_entry["highlights"]) for entry in lost)
        print(
            f"\nWARNING: {len(skipped)} of {len(requests)} prompt(s) failed after "
            f"every retry, so {len(lost)} annotation entries with {highlights} "
            "highlight(s) produced no cards:"
        )
        for outcome in skipped:
            print(f"  prompt {outcome.request.key}: {outcome.reason}")
        print("Re-run to try them again; the responses that worked are cached.")
    # Generate and save processed pages JSON if requested
    if args.processed_pages_output:
        # The processed-pages rules are generated from every loaded
        # annotation, so after a partial run they would mark pages as
        # processed that never were.
        if limited:
            print("Not writing processed pages: --limit skipped part of the input.")
        elif skipped:
            print("Not writing processed pages: some prompts produced no cards.")
        else:
            print(f"Writing rules for processed pages to {args.processed_pages_output}")
            processed_pages = generate_processed_pages_json(combined_annotations)
            with open(args.processed_pages_output, "w") as f:
                json.dump(processed_pages, f, indent=2)
            print(f"Processed pages written to {args.processed_pages_output}")

    if skipped:
        sys.exit(runner.EXIT_REQUESTS_DROPPED)


if __name__ == "__main__":
    asyncio.run(main())
