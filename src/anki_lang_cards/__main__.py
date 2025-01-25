import argparse
import asyncio
import csv
import itertools
import json
import operator
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

from . import _types as types
from . import config, prompts, query


def load_words(csv_path: pathlib.Path) -> Sequence[types.Word]:
    out = []
    with open(csv_path, "rt") as f:
        reader = csv.DictReader(f)
        for line in reader:
            if not (line["word"] and line["meaning"]):
                continue

            if line["example_sentences"]:
                example_sentences = tuple(json.loads(line["example_sentences"]))
            else:
                example_sentences = ()

            if "rank" not in line or not line["rank"]:
                rank: int | None = None
            else:
                rank = int(line["rank"])

            if "sentence_source" in line and line["sentence_source"]:
                sentence_source = types.Citation.from_json(line["sentence_source"])
            else:
                sentence_source = None

            out.append(
                types.Word(
                    rank=rank,
                    word=line["word"],
                    meaning=line["meaning"],
                    part_of_speech=line["part_of_speech"],
                    article=line["article"] or None,
                    spanish_sentence=line["spanish_sentence"] or None,
                    english_sentence=line["english_sentence"] or None,
                    sentence_source=sentence_source,
                    example_sentences=example_sentences,
                    model_used=line["model_used"] or None,
                )
            )

    return out


def _unique(s: Sequence[str], key: Callable[[str], Any] | None = None) -> Sequence[str]:
    seen: set = set()
    out: list[str] = []
    if key is None:

        def key(x: str) -> str:
            return x

    for entry in s:
        if (entry_key := key(entry)) not in seen:
            seen.add(entry_key)
            out.append(entry)
    return out


def _make_api_obj(
    conf_obj: config.Config, service: str, model_name: str | None
) -> query.ModelAPIBase:
    if service == "anthropic":
        if conf_obj.anthropic is None:
            raise ValueError("Anthropic model selected but no config found.")
        model_api = query.AnthropicAPI(
            api_key=conf_obj.anthropic.api_key,
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
    parser = argparse.ArgumentParser(description="My Application")
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
    args = parser.parse_args()

    word_getter = operator.attrgetter("word")
    wk_getter = operator.attrgetter("word", "part_of_speech")

    words = load_words(args.inputs)
    if args.output.exists():
        existing_words = load_words(args.output)
    else:
        existing_words = []

    if args.exported_deck is not None:
        from . import anki_import

        current_deck = anki_import.load_anki_words(args.exported_deck)
    else:
        current_deck = []

    start_rank_at = args.start_rank_at or 0
    done: set[str] = set()
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
            done.add(word.word)

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

    words = _unique([word for word in words if word.word not in done], key=word_getter)
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
            query_results = api.make_queries(words, prompt=prompt_handler)

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
