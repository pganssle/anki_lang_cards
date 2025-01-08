"""Module to use Retrieval Augmented Generation to generate input sentences."""

import argparse
import asyncio
import csv
import datetime
import functools
import hashlib
import itertools
import json
import re
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, TypedDict, cast

import attrs
import llm

from . import config, dictionary_entries, prompts
from ._types import CombinedAnnotation, CombinedBookAnnotations, Word

WordMap: TypeAlias = Mapping[str, Sequence[dictionary_entries.DictionaryEntry]]


class AnnotationEntry(TypedDict):
    highlights: Sequence[str]
    context: str


@attrs.frozen
class PromptInput:
    rag_entries: Mapping[str, str]
    annotation_entry: AnnotationEntry


CACHE_LOC: Final[Path] = Path(".cache")


def _get_response_dir() -> Path:
    return CACHE_LOC / "responses"


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
        # All the same part of speech, we can put that in with the headword and not indent as much
        # Format looks like:
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
        # Mixed part of speech, so we use a format that separtes out definitions by part of speech
        # Format looks like:
        #
        # largo
        #   (adj)
        #     1. long
        #     2. lengthy
        #   (n)
        #     1. length
        #     2. largo (musical movement)

        # We will assume that the definitions for parts of speech are already grouped together,
        # and if they aren't it's for some good reason.
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
            entry = engine.lookup(word)
            if entry:
                output[word] = entry

    return output


def handle_book_skipping(
    book_skip_list: Path, combined_book_annotations: Sequence[CombinedBookAnnotations]
) -> Sequence[CombinedBookAnnotations]:
    # This is absurdly complicated for what it does 😅
    StringPattern: TypeAlias = str | dict[Literal["pattern"], str]
    RangePattern: TypeAlias = Sequence[int | dict[Literal["range"], tuple[int, int]]]

    def compile_string_match(pattern: StringPattern) -> Callable[[str], bool]:
        if isinstance(pattern, dict):
            if "pattern" not in pattern:
                raise ValueError(f"Unknown rule type: {pattern}")
            compiled_regex = re.compile(pattern["pattern"])
            return lambda value, cr=compiled_regex: cr.match(value) is not None
        else:
            return lambda value, pattern=pattern: value == pattern

    def compile_range_matcher(num: int, ranges: RangePattern) -> Callable[[int], bool]:
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
        rule: dict[str, StringPattern | RangePattern]
    ) -> Callable[[CombinedBookAnnotations], CombinedBookAnnotations | None]:
        rule_matchers = []
        for field in ("title", "author", "path", "language"):
            if field in rule:
                string_match = compile_string_match(rule[field])

                def rule_matcher(
                    book_annotations: CombinedBookAnnotations, match=string_match
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
            rule: dict[str, StringPattern | RangePattern]
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
                if compiled_subset_rule(annotation)
            ]
            if not filtered_annotations:
                return None
            return attrs.evolve(book_annotations, "annotations", filtered_annotations)

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


def _prepare_prompt_inputs(
    combined_annotations: Sequence[CombinedAnnotation], word_map: WordMap
) -> Iterable[PromptInput]:
    for annotation in combined_annotations:
        highlights = sorted(set(annotation.highlights), key=annotation.highlights.index)

        annotation_entry = {
            "highlights": annotation.highlights,
            "context": annotation.context,
        }

        rag_entries = {}
        for highlight in annotation.highlights:
            if highlight not in word_map:
                continue

            for entry in word_map[highlight]:
                if entry.headword in rag_entries:
                    continue

                rag_entries[entry.headword] = dictionary_entry_to_rag_input(entry)

        yield PromptInput(rag_entries=rag_entries, annotation_entry=annotation_entry)


def prepare_prompt_inputs(
    combined_book_annotations: Sequence[CombinedBookAnnotations], word_map: WordMap
) -> Sequence[PromptInput]:
    output: list[PromptInput] = []
    for book_annotations in combined_book_annotations:
        output.extend(_prepare_prompt_inputs(book_annotations.annotations, word_map))
    return output


def _generate_prompt(prompt_inputs: Sequence[PromptInput]) -> str:
    annotation_input = json.dumps(
        [prompt_input.annotation_entry for prompt_input in prompt_inputs],
        indent=2,
        ensure_ascii=False,
    )
    all_rag_entries: list[str] = []
    seen_headwords: set[str] = set()
    for prompt_input in prompt_inputs:
        for headword, rag_entry in prompt_input.rag_entries.items():
            if headword in seen_headwords:
                continue
            seen_headwords.add(headword)
            all_rag_entries.append(rag_entry)

    return prompts.BOOK_ANNOTATION_PROMPT % {
        "input": annotation_input,
        "rag_entries": "\n\n".join(all_rag_entries),
    }


def _estimate_tokens(prompt: str) -> int:
    return len(prompt) / 4


def generate_prompts(
    prompt_inputs: Sequence[PromptInput], max_tokens: int = 2048
) -> Iterable[str]:
    # TODO: Can use heuristic to estimate how many fit in the context window to make this faster

    current_prompt_stack: list[PromptInput] = []
    last_prompt: str = ""

    for prompt_input in prompt_inputs:
        current_prompt_stack.append(prompt_input)

        prompt: str = _generate_prompt(current_prompt_stack)
        if _estimate_tokens(prompt) > max_tokens:
            if not last_prompt:
                raise RuntimeError(
                    "Current prompt input is too big to fit in max_tokens"
                )

            yield last_prompt
            current_prompt_stack = [prompt_input]
        else:
            last_prompt = prompt

    yield prompt


def _get_llm_cache_file_loc(
    prompt: str,
    model_service: str,
    model_name: str,
) -> Path:
    prompt_hash: str = hashlib.md5(
        prompt.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return (
        CACHE_LOC
        / "annotations"
        / f"{model_service}"
        / f"{model_name}"
        / f"{prompt_hash}.json"
    )


async def _extract_response_data(response: llm.AsyncResponse) -> dict:
    text = await response.text()
    usage = await response.usage()

    return {
        "text": text,
        "usage": {
            "input": usage.input,
            "output": usage.output,
            "details": usage.details,
        },
    }


async def process_input_with_llm(
    prompt: str,
    model_service: str,
    model_name: str,
    conf_obj: config.Config,
    invalidate_cache: bool = False,
    skip_cache: bool = False,
) -> str:
    """Processes input using the specified LLM model service."""
    if invalidate_cache and skip_cache:
        raise ValueError("Only one of invalidate_cache and skip_cache can be true")

    cache_file_loc = _get_llm_cache_file_loc(prompt, model_service, model_name)
    if not (skip_cache or invalidate_cache) and cache_file_loc.exists():
        response_data = json.loads(cache_file_loc.read_text())["response"]
    else:
        if model_service == "anthropic":
            if not conf_obj.anthropic:
                raise ValueError("Anthropic configuration is missing.")
            model = llm.get_async_model(model_name)
            model.key = conf_obj.anthropic.api_key
        elif model_service == "tgw":
            raise NotImplementedError("Getting around to it.")
        else:
            raise ValueError(f"Unsupported model service: {model_service}")

        # Prepare and send the request
        response = await model.prompt(prompt)
        response_data = await _extract_response_data(response)

        if invalidate_cache or not cache_file_loc.exists():
            cache_data = {
                "input": {
                    "model_name": model_name,
                    "model_service": model_service,
                    "prompt": prompt,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                "response": response_data,
            }

            cache_file_loc.parent.mkdir(exist_ok=True, parents=True)
            cache_file_loc.write_text(
                json.dumps(cache_data, ensure_ascii=False, indent=2)
            )

    return response_data["text"]


def parse_llm_response(response_text: str, model_used: str) -> Sequence[Word]:
    response_data = json.loads(response_text)
    output: list[Word] = []
    for highlight in response_data:
        headword = highlight["headword"]
        meaning = highlight["meaning"]
        part_of_speech = highlight["part_of_speech"]
        article = highlight.get("article", None) or None

        example_sentences = list(highlight["examples"])
        first_example = example_sentences[0]
        del example_sentences[0]
        english_sentence = first_example["english"]
        spanish_sentence = first_example["spanish"]

        output.append(
            Word(
                rank=None,
                word=headword,
                meaning=meaning,
                part_of_speech=part_of_speech,
                article=article,
                spanish_sentence=spanish_sentence,
                english_sentence=english_sentence,
                example_sentences=example_sentences,
                model_used=model_used,
            )
        )

    return output


# TODO: Move make_csv_header and word_to_csv_line into _types
def make_csv_header() -> Sequence[str]:
    return (
        "rank",
        "word",
        "part_of_speech",
        "meaning",
        "spanish_sentence",
        "english_sentence",
        "frequency",
        "image",
        "article",
        "spanish_audio",
        "note",
        "example_sentences",
        "model_used",
        "tags",
    )


def word_to_csv_line(w: Word) -> Sequence[str]:
    if w.example_sentences:
        example_sentences = json.dumps(w.example_sentences)
    else:
        example_sentences = ""

    return (
        str(w.rank or ""),
        w.word,
        w.part_of_speech,
        w.meaning,
        w.spanish_sentence or "",
        w.english_sentence or "",
        "",  # Frequency
        "",  # Image
        w.article or "",
        "",  # Spanish Audio
        "",  # Note
        example_sentences,
        w.model_used or "",
        "",
    )


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
        default=(),
        help="Path to a dictionary in STARDICT format. This can also be "
        "specified in the config file under annotations.dictionaries.",
    )
    parser.add_argument(
        "--hunspell-dict-path",
        type=Path,
        default=Path("/usr/share/hunspell/es_ES.dic"),
        help="Path to the hunspell dictionary for headword stemming",
    )
    parser.add_argument(
        "--model-service",
        type=str,
        default="anthropic",
        help="Which model service to use. Options are:\n"
        "    tgw: Text Generation Web\n"
        "    anthropic: Anthropic",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The specific model to use. If unspecified, the default value "
        "for the chosen model is selected.",
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
        "--invalidate-response-cache",
        action="store_true",
        help="If specified, delete any existing caches for this given response and replace them with the new query response.",
    )
    parser.add_argument(
        "--skip-response-cache",
        action="store_true",
        help="If specified, don't use existing caches, but also don't replace them.",
    )
    args = parser.parse_args()

    if args.skip_response_cache and args.invalidate_response_cache:
        raise ValueError(
            "At most one of --skip-response-cache and --invalidate-response-cache may be specified."
        )

    conf = config.Config.from_file(args.config)
    if args.model_name is not None:
        model_name = args.model_name
    elif args.model_service == "anthropic":
        model_name = "claude-3-5-sonnet-20241022"
    else:
        raise NotImplementedError("No model name")

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
    prompt_inputs = prepare_prompt_inputs(combined_annotations, word_map)

    existing: dict[Word, Word] = {}
    print(f"Generating prompts for {len(prompt_inputs)} combined annotations.")
    for prompt in generate_prompts(prompt_inputs):
        # TODO: Run these queries in parallel
        response = await process_input_with_llm(
            prompt,
            model_service=args.model_service,
            model_name=model_name,
            conf_obj=conf,
            invalidate_cache=args.invalidate_response_cache,
            skip_cache=args.skip_response_cache,
        )

        for word in parse_llm_response(response, model_used=model_name):
            if word in existing:
                existing[word].update(word)
            else:
                existing[word] = word

    print(f"Generated {len(existing)} entries.")
    with open(args.output, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(make_csv_header())
        for word in existing.values():
            writer.writerow(word_to_csv_line(word))

    print(f"Written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
