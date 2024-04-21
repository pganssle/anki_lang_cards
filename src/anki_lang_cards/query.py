import asyncio
import json
from collections.abc import AsyncIterable, Mapping, Sequence
from typing import Any, Final

import aiohttp

from . import _types as types

# Define the API endpoint and headers
LOCAL_API_ENDPOINT: Final[str] = "https://tgw-api.a23.x78.be/v1/completions"
HEADERS = {"Content-Type": "application/json"}

# Semaphore to limit concurrent requests
API_SEMAPHORE: Final[asyncio.Semaphore] = asyncio.Semaphore(1)


async def query_model(
    session: aiohttp.ClientSession,
    prompt: str,
    params: Mapping[str, str | int | float] | None = None,
) -> dict:
    """
    Fetch data from the API endpoint.
    """
    if params is None:
        params = {}

    data = {"prompt": prompt, **params}
    async with API_SEMAPHORE:
        async with session.post(
            LOCAL_API_ENDPOINT, headers=HEADERS, json=data
        ) as response:
            response.raise_for_status()
            return await response.json()


def parse_results(resp: Mapping[str, Any]) -> tuple[Mapping[str, str], str]:
    choices = resp["choices"]
    choice = choices[0]
    if "text" not in choice:
        raise ValueError(
            f"Failed to retrieve message for reason {choice['finish_reason']}"
        )
    elif choice["finish_reason"] != "stop":
        print(f"Query finished for reason {choice['finish_reason']}")
    try:
        return json.loads(choice["text"]), resp["model"]
    except json.decoder.JSONDecodeError:
        print(f"Failed to decode text:\n{choice['text']}")
        raise


def result_to_word(word: types.Word, parsed: Mapping[str, str]) -> types.Word:
    if not word.spanish_sentence:
        example_sentence = parsed["examples"].pop()
        word.spanish_sentence = example_sentence["spanish"]
        word.english_sentence = example_sentence["english"]
    word.example_sentences = parsed["examples"]
    return word


async def make_queries(
    words: Sequence[types.Word], prompt: str
) -> AsyncIterable[types.Word]:
    async with aiohttp.ClientSession() as session:
        query_futures = {}
        for word in words:
            c_prompt = prompt % {
                "term": word.word,
                "meaning": word.meaning,
                "pos": word.part_of_speech,
            }
            task = asyncio.create_task(
                query_model(session, prompt=c_prompt, params={"max_tokens": 8192})
            )
            query_futures[task] = word

        pending = set(query_futures)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for future in done:
                word = query_futures[future]
                try:
                    parsed, model = parse_results(await future)
                except (ValueError, json.decoder.JSONDecodeError):
                    print(f"Skipping {word.word}")
                    continue

                word.model_used = model
                yield result_to_word(word, parsed)
