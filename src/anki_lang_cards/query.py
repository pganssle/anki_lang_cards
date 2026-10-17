import itertools
import json
import random
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import Any, Protocol

import attrs

from . import _types as types
from . import caches, prompts, runner
from .backends import base


class PromptHandler(Protocol):
    prompt_template: str
    system_prompt: str

    def make_prompt(self, words: Sequence[types.Word]) -> str: ...


@attrs.frozen
class DroppedWords:
    """Words whose request got no usable answer after its last retry."""

    words: Sequence[types.Word]
    reason: str


async def generate_sentences(
    query_runner: runner.QueryRunner,
    words: Sequence[types.Word],
    prompt: PromptHandler,
    *,
    model_name: str,
    words_per_prompt: int | None = None,
    max_invalid_output_retries: int = 0,
) -> AsyncIterator[types.Word | DroppedWords]:
    """Ask the model for example sentences, yielding words as they are filled.

    A request the runner gives up on is yielded as :class:`DroppedWords`, so
    that the caller can report which words got no sentences.

    :param model_name: Recorded as the source of each generated sentence.
    :param words_per_prompt: Defaults to what the runner's backend suggests.
    """
    if words_per_prompt is None:
        words_per_prompt = query_runner.backend.default_words_per_prompt

    requests: dict[base.Request, tuple[types.Word, ...]] = {}
    for word_subset in itertools.batched(words, words_per_prompt):
        request = base.Request(
            prompt=prompt.make_prompt(word_subset),
            system_prompt=prompt.system_prompt,
            json_schema=prompts.MULTI_SENTENCE_RESPONSE_SCHEMA,
        )
        requests[request] = word_subset

    def describe(request: base.Request) -> str:
        return str(tuple(word.word for word in requests[request]))

    citation = types.ModelCitation(model=model_name)
    async for outcome in runner.run_with_retries(
        query_runner,
        tuple(requests),
        _parse_response,
        max_retries=max_invalid_output_retries,
        describe=describe,
    ):
        if isinstance(outcome, runner.Skipped):
            yield DroppedWords(requests[outcome.request], outcome.reason)
            continue
        request, parsed = outcome.request, outcome.value
        parsed_with_sources = [
            {
                **entry,
                "examples": [
                    {**example, "source": citation} for example in entry["examples"]
                ],
            }
            for entry in parsed
        ]
        for word in result_to_words(requests[request], parsed_with_sources):
            word.model_used = model_name
            yield word


def decode_words(text: str) -> Sequence[Mapping[str, Any]]:
    """Decode a ``{"words": [...]}`` response into its list of entries.

    Responses from backends that cannot enforce the schema may also come
    wrapped in a markdown code block or as a bare list, which is accepted.

    :raises ValueError: If the response does not have that shape.
    """
    text = text.strip()
    if text.startswith("```"):
        print("Removing ```json wrapper")
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"Failed to decode text:\n{text}")
        raise
    words = data["words"] if isinstance(data, dict) and "words" in data else data
    if not isinstance(words, list) or not all(
        isinstance(entry, dict)
        and isinstance(entry.get("examples"), list)
        and all(isinstance(example, dict) for example in entry["examples"])
        for entry in words
    ):
        raise ValueError(f"Response is not a list of words with examples:\n{text}")
    return words


def _parse_response(
    request: base.Request, response: caches.ResponseData
) -> Sequence[Mapping[str, Any]]:
    return decode_words(response["text"])


class SentencePromptHandler(PromptHandler):
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template
        self.system_prompt = ""

    def make_prompt(self, words: Sequence[types.Word]) -> str:
        word = words[0]
        return self.prompt_template % {
            "term": word.word,
            "meaning": word.meaning,
            "pos": word.part_of_speech,
        }


class MultiSentencePromptHandler(PromptHandler):
    def __init__(self, prompt_template: str, system_prompt: str = ""):
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    def make_prompt(self, words: Iterable[types.Word]) -> str:
        prompt_data = json.dumps(
            [
                {
                    "term": word.word,
                    "part_of_speech": str(word.part_of_speech),
                    "meaning": word.meaning,
                }
                for word in words
            ],
            ensure_ascii=False,
            indent=4,
        )
        return self.prompt_template % {"data": prompt_data}


def result_to_word(
    word: types.Word, parsed: Mapping[str, str | list[str]]
) -> types.Word:

    random.shuffle(parsed["examples"])
    word.sentences = (
        *word.sentences,
        *map(types.Sentence.from_dict, parsed["examples"]),
    )
    return word


def result_to_words(
    words: Sequence[types.Word], parsed: Sequence[Mapping[str, Any]]
) -> Iterable[types.Word]:
    word_map = {word.word: word for word in words}
    for parsed_word in parsed:
        try:
            yield result_to_word(word_map[parsed_word["term"]], parsed_word)
        except KeyError:
            print(f"Term mismatch for {parsed_word}, skipping.")
