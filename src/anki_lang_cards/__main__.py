import argparse
import asyncio
import csv
import json
import operator
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

from . import _types as types
from . import prompts, query


def load_words(csv_path: pathlib.Path) -> Sequence[types.Word]:
    out = []
    with open(csv_path, "rt") as f:
        reader = csv.DictReader(f)
        header = next(reader, None)
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

            out.append(
                types.Word(
                    rank=rank,
                    word=line["word"],
                    meaning=line["meaning"],
                    part_of_speech=line["part_of_speech"],
                    article=line["article"] or None,
                    spanish_sentence=line["spanish_sentence"] or None,
                    english_sentence=line["english_sentence"] or None,
                    example_sentences=example_sentences,
                )
            )

    return out


def make_csv_header() -> Sequence[str]:
    return (
        "word",
        "meaning",
        "part_of_speech",
        "article",
        "spanish_sentence",
        "english_sentence",
        "example_sentences",
        "model_used",
    )


def word_to_csv_line(w: types.Word) -> Sequence[str]:
    if w.example_sentences:
        example_sentences = json.dumps(w.example_sentences)
    else:
        example_sentences = ""

    return (
        w.rank,
        w.word,
        w.meaning,
        w.part_of_speech,
        w.article or "",
        w.spanish_sentence or "",
        w.english_sentence or "",
        example_sentences,
        w.model_used or "",
    )


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
        "--start-rank-at",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    word_getter = operator.attrgetter("word")

    words = load_words(args.inputs)
    if args.output.exists():
        existing_words = load_words(args.output)
    else:
        existing_words = []

    done = set(map(word_getter, existing_words))
    words = _unique([word for word in words if word.word not in done], key=word_getter)

    start_rank_at = args.start_rank_at or 0
    for word in existing_words:
        if word.rank and word.rank >= start_rank_at:
            start_rank_at = word.rank + 1

    rank = start_rank_at or 5001

    # TODO: Check size here as well
    needs_header = not args.output.exists()

    with open(args.output, "at") as f:
        query_results = query.make_queries(words, prompts.SENTENCE_PROMPT)
        writer = csv.writer(f)

        if needs_header:
            writer.writerow(make_csv_header())

        async for word in query_results:
            print(f"Adding {word.word}")
            word.rank = rank
            writer.writerow(word_to_csv_line(word))
            f.flush()
            rank += 1


if __name__ == "__main__":
    asyncio.run(main())
