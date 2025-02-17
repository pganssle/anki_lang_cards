import argparse
import asyncio
import csv
import itertools
import operator
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

from . import _types as types
from . import anki_import, caches, config, prompts, query


def _unique(s: Sequence[str], key: Callable[[str], Any] | None = None) -> Sequence[str]:
    seen: dict = {}
    out: list[str] = []
    if key is None:

        def key(x: str) -> str:
            return x

    for entry in s:
        if (entry_key := key(entry)) not in seen:
            out.append(entry)
        else:
            existing_entry = seen[entry_key]
            existing_entry.update(entry)

    return out


def _make_api_obj(
    conf_obj: config.Config, service: str, model_name: str | None
) -> query.ModelAPIBase:
    if service == "anthropic":
        if conf_obj.anthropic is None:
            raise ValueError("Anthropic model selected but no config found.")
        query_cacher = caches.QueryCacher(
            cache_type=caches.CacheTypes.SENTENCES,
            model_name=model_name or query.AnthropicAPI.DEFAULT_MODEL,
            model_service=service,
        )
        model_api = query.AnthropicAPI(
            api_key=conf_obj.anthropic.api_key,
            query_cacher=query_cacher,
            model_name=model_name,
            max_connections=1,
        )
    elif service == "tgw":
        if conf_obj.tgw is None:
            raise ValueError("tgw model selected but no config found.")

        model_api = query.TGWModelAPI(
            api_base_url=conf_obj.tgw.api_base_url,
            request_timeout=conf_obj.tgw.request_timeout,
            max_connections=conf_obj.tgw.max_connections,
            max_tokens=conf_obj.tgw.max_tokens,
            model_name=model_name,
        )
    else:
        raise ValueError(f"Unknown model service: {service}")

    return model_api


async def main():
    parser = argparse.ArgumentParser(description="Anki Language Card Creator")
    parser.add_argument(
        "--inputs",
        type=pathlib.Path,
        default="input/words.csv",
        help="Path to a CSV containing words and their meanings",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="output/output.csv",
        help="Path to an output CSV file",
    )
    parser.add_argument(
        "--model-service",
        type=str,
        default="tgw",
        help="Which model service to use. Options are:\n"
        "    tgw: Text Generation Web\n"
        "    anthropic: Anthropic",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The specific model to use. If unspecified, the default value "
        "for the chosen model is selected, which is:\n"
        f"    tgw: {query.TGWModelAPI.DEFAULT_MODEL}\n"
        + f"    anthropic: {query.AnthropicAPI.DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default="config.toml",
        help="Config file location",
    )
    parser.add_argument(
        "--start-rank-at",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of failed responses from the LLM before giving up.",
    )

    parser.add_argument(
        "--exported-deck",
        type=pathlib.Path,
        default=None,
        help="Path to a text export of the deck.",
    )
    parser.add_argument(
        "--diff-output",
        type=pathlib.Path,
        default="output/diff_from_existing.csv",
        help="Path to output a CSV only including new and updated words.",
    )

    parser.add_argument("--deck-name", type=str, default="Spanish::Vocab by Frequency")
    args = parser.parse_args()

    deck_name = args.deck_name

    word_getter = operator.attrgetter("word")
    wk_getter = operator.attrgetter("word", "part_of_speech")

    words = types.Word.words_from_csv(args.inputs)
    if args.output.exists():
        existing_words = types.Word.words_from_csv(args.output)
    else:
        existing_words = []

    if args.exported_deck is not None:
        current_deck = anki_import.load_anki_words(args.exported_deck)
        start_rank_at = args.start_rank_at
        if start_rank_at is None:
            start_rank_at = max(word.rank for word in current_deck) + 1
    else:
        print("Loading words from Anki Connect")
        async with anki_import.create_client() as client:
            current_deck_notes = await client.retrieve_matching_words(deck_name, words)
            current_deck = list(current_deck_notes.values())
        print(f"Loaded {len(current_deck)} words")

        async with anki_import.create_client() as client:
            if not args.start_rank_at:
                print("Determining next rank")
                start_rank_at = (
                    await client.find_highest_rank(deck_name, max_range=len(words)) + 1
                )
                print(f"Starting with {start_rank_at}")
            else:
                start_rank_at = args.start_rank_at

    done: set[tuple[str, str]] = set()
    to_update: set[str] = set()
    all_words = {}
    diff_words = []

    config_file = pathlib.Path(args.config)
    if not config_file.exists():
        print(f"Config file {config.abs()} does not exist.")
        raise IOError("At least one model config is required.")
    else:
        conf_obj = config.Config.from_file(config_file)

    for word in itertools.chain(current_deck, existing_words):
        if word.rank and word.rank >= start_rank_at:
            start_rank_at = word.rank + 1

        if word.example_sentences:
            done.add((word.word, word.part_of_speech))

        if (word.word, word.part_of_speech) in all_words:
            existing = all_words[(word.word, word.part_of_speech)]
            if (word.rank and word.rank < 5000) and (
                existing.rank and existing.rank < 5000
            ):
                continue

            sentences = tuple(existing.example_sentences)
            existing.update(word)
            if not sentences:
                diff_words.append(existing)
        else:
            all_words[(word.word, word.part_of_speech)] = word

    unique_words = []
    for word in words:
        if (word.word, word.part_of_speech) in done:
            if word.example_sentences or word.spanish_sentence:
                existing = all_words[(word.word, word.part_of_speech)]
                existing.update(word)
                to_update.add((word.word, word.part_of_speech))
        else:
            unique_words.append(word)

    words = unique_words
    errors = 0
    for i in range(len(words)):
        if (words[i].word, words[i].part_of_speech) in all_words:
            print(f"Word exists in deck: {words[i].word}")
            existing = all_words[(words[i].word, words[i].part_of_speech)]
            try:
                existing.update(words[i])
            except ValueError as e:
                errors += 1
                print(
                    f"{existing.rank} {existing.word} != {words[i].word}:\n"
                    + f"    {existing.meaning}\n"
                    + f"    {words[i].meaning}"
                )
            else:
                words[i] = existing
            if errors > 20:
                break
    if errors:
        raise ValueError(f"Found {errors} non-matching duplicates, stopping")

    rank = start_rank_at or 5001

    # TODO: Check size here as well
    needs_header = not args.output.exists()
    prompt_handler = query.MultiSentencePromptHandler(
        prompts.MULTI_SENTENCE_PROMPT_TEMPLATE,
        system_prompt=prompts.MULTI_SENTENCE_SYSTEM_PROMPT,
    )

    diff_words.extend(
        map(
            all_words.get,
            set(map(wk_getter, existing_words)) - set(map(wk_getter, current_deck)),
        )
    )
    diff_words = _unique(diff_words, key=id)

    if words:
        model_api = _make_api_obj(
            conf_obj=conf_obj,
            service=args.model_service,
            model_name=args.model_name,
        )

        async with model_api as api:
            query_results = api.make_queries(
                words,
                prompt=prompt_handler,
                max_invalid_output_retries=args.max_retries,
            )

            with open(args.output, "at") as f:
                writer = csv.writer(f)

                if needs_header:
                    writer.writerow(types.Word.make_csv_header())

                async for word in query_results:
                    print(f"Adding {word.word}")
                    if not word.rank:
                        word.rank = rank
                    if word.word in all_words:
                        existing = all_words[word.word]
                        if existing is not word:
                            existing.update(word)
                        word = existing

                    diff_words.append(word)
                    writer.writerow(word.to_csv_line(require_rank=True))
                    f.flush()
                    rank += 1

                for term, pos in to_update:
                    word = all_words[(term, pos)]
                    writer.writerow(word.to_csv_line(require_rank=True))
                    f.flush()

    else:
        print("No new words found.")

    print(f"Found {len(diff_words)} new or updated words.")
    print("\n".join(map(word_getter, diff_words)))
    if diff_words:
        with open(args.diff_output, "wt") as f:
            writer = csv.writer(f)
            for word in diff_words:
                writer.writerow(word.to_csv_line(word))


if __name__ == "__main__":
    asyncio.run(main())
