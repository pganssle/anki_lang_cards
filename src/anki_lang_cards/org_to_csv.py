from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import io
import pathlib
import sys
from collections.abc import Sequence
from typing import Final

import attrs

from . import caches, config, prompts
from .backends import base, runner, selection

CACHE_LOC: Final[pathlib.Path] = pathlib.Path(".cache")


@attrs.frozen
class WordResponse:
    word: str
    meaning: str
    part_of_speech: str
    article: str

    @classmethod
    def csv_header(cls) -> Sequence[str]:
        return (
            "rank",
            "word",
            "meaning",
            "part_of_speech",
            "article",
            "spanish_sentence",
            "english_sentence",
            "example_sentences",
            "model_used",
        )

    def to_csv_row(self) -> Sequence[str]:
        return (
            "",
            self.word,
            self.meaning,
            self.part_of_speech,
            self.article,
            "",
            "",
            "",
            "",
        )


async def process_input_with_llm(prompt: str, query_runner: runner.QueryRunner) -> str:
    """Send one prompt; this tool keeps its own cache of the responses."""
    async with query_runner:
        async for outcome in query_runner.run([base.Request(prompt)], skip_cache=True):
            match outcome:
                case runner.Success(response=response):
                    return response["text"]
                case runner.Failure(reason=reason):
                    raise RuntimeError(f"The model did not answer: {reason}")
    raise AssertionError("The runner produced no outcome.")


def _get_response_dir() -> pathlib.Path:
    return CACHE_LOC / "responses"


def cache_response(response_text: str, filename: str | None = None) -> None:
    if not CACHE_LOC.exists():
        CACHE_LOC.mkdir()

    response_dir = _get_response_dir()
    if not response_dir.exists():
        response_dir.mkdir()

    if filename is None:
        filename = (
            f"response-{datetime.datetime.now().isoformat(timespec='seconds')}.csv"
        )

    cache_path = response_dir / filename

    with open(cache_path, "wt") as f:
        f.write(response_text)


def load_latest_cache() -> str:
    response_dir = _get_response_dir()
    if not response_dir.exists() or not any(response_dir.iterdir()):
        raise RuntimeError("No cached responses found")

    # Get all files in the cache directory.
    files = list(response_dir.glob("*.csv"))

    if not files:
        return None

    # Sort files by last modified time, in descending order.
    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    # Read and return the contents of the most recent file.
    with open(latest_file, "rt") as f:
        return f.read()


def parse_response(csv_response: str) -> Sequence[WordResponse]:
    reader = csv.reader(io.StringIO(csv_response))
    riter = iter(reader)
    header = next(riter)
    idxes = {value: i for i, value in enumerate(header)}

    output = []

    def get_kwarg(row, kwarg) -> str:
        idx = idxes[kwarg]
        if idx >= len(row) and kwarg == "article":
            return ""

        return row[idx]

    for row in riter:
        if not row:
            continue
        wr = WordResponse(**{kwarg: get_kwarg(row, kwarg) for kwarg in header})
        if "/" in wr.meaning and "," not in wr.meaning:
            wr = attrs.evolve(wr, meaning=", ".join(wr.meaning.split("/")))
        output.append(wr)

    return output


def words_to_csv(response: Sequence[WordResponse], write_header: bool = True) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    if write_header:
        writer.writerow(WordResponse.csv_header())
    for word in response:
        writer.writerow(word.to_csv_row())

    output.seek(0)
    return output.read()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Converts org-mode text to CSVs")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("input/words_from_org_mode.csv"),
        help="Path to an output CSV file",
    )
    selection.add_arguments(parser)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("config.toml"),
        help="Config file location",
    )
    parser.add_argument(
        "--if-exists",
        type=str,
        default="append",
        help="What to do if the file exists. Options are [append, overwrite, fail]",
    )
    parser.add_argument(
        "--use-latest-cache",
        action="store_true",
        help="If specified, ignore inputs and read the latest response from the cache",
    )
    args = parser.parse_args()

    if args.if_exists not in {"fail", "overwrite", "append"}:
        raise ValueError(f"Invalid argument for --if-exists: {args.if_exists}")

    if args.output.exists() and args.if_exists == "fail":
        raise RuntimeError(f"{args.output} exists and --if-exists is fail")

    if args.use_latest_cache:
        csv_output: str = load_latest_cache()
    else:
        conf = config.Config.from_file(args.config)
        # A single interactive request gains nothing from batching.
        args.batch = False
        _, query_runner = selection.make_runner(args, conf, caches.CacheTypes.RESPONSES)

        # Read piped input
        input_data = await asyncio.to_thread(lambda: sys.stdin.read().strip())
        if not input_data:
            raise ValueError("No input data provided via stdin.")

        prompt = prompts.ORG_TO_CSV_PROMPT_TEMPLATE % {"org_text": input_data}

        # Process with the LLM
        csv_output = await process_input_with_llm(prompt, query_runner)

        cache_response(csv_output)

    response_words = parse_response(csv_output)

    if not args.output.exists() or (args.if_exists == "overwrite"):
        csv_text = words_to_csv(response_words)
        args.output.write_text(csv_text)
    else:
        csv_text = words_to_csv(response_words, write_header=False)
        with open(args.output, "at+") as f:
            f.seek(0, 2)
            t = f.tell()
            f.seek(t - 1)
            char = f.read(1)
            if char != "\n":
                f.write("\n")

            f.write(csv_text)


if __name__ == "__main__":
    asyncio.run(main())
