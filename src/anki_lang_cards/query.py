import itertools
import json
import random
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import Any, Protocol

from . import _types as types
from . import caches, prompts, runner
from .backends import base


class PromptHandler(Protocol):
    prompt_template: str
    system_prompt: str

    def make_prompt(self, words: Sequence[types.Word]) -> str: ...


async def generate_sentences(
    query_runner: runner.QueryRunner,
    words: Sequence[types.Word],
    prompt: PromptHandler,
    *,
    model_name: str,
    words_per_prompt: int | None = None,
    max_invalid_output_retries: int = 0,
) -> AsyncIterator[types.Word]:
    """Ask the model for example sentences, yielding words as they are filled.

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

    def parse(
        request: base.Request, response: caches.ResponseData
    ) -> Sequence[tuple[types.Word, Mapping[str, Any]]]:
        return match_entries(requests[request], decode_words(response["text"]))

    citation = types.ModelCitation(model=model_name)
    async for _, matched in runner.run_with_retries(
        query_runner,
        tuple(requests),
        parse,
        max_retries=max_invalid_output_retries,
        describe=describe,
    ):
        for word, entry in matched:
            entry_with_sources = {
                **entry,
                "examples": [
                    {**example, "source": citation} for example in entry["examples"]
                ],
            }
            word = result_to_word(word, entry_with_sources)
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
                    "id": word_id,
                    "term": word.word,
                    "part_of_speech": str(word.part_of_speech),
                    "meaning": word.meaning,
                }
                for word_id, word in enumerate(words)
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


def match_entries(
    words: Sequence[types.Word], entries: Sequence[Mapping[str, Any]]
) -> Sequence[tuple[types.Word, Mapping[str, Any]]]:
    """Pair each entry of a sentence response with the word it answers.

    Entries are matched on the ``id`` the prompt gave each word (its index in
    ``words``) rather than on the term, which homographs share and which the
    model sometimes rewrites.

    :raises ValueError: If an entry's id is missing, unknown or repeated, or
        if any word has no entry, so that the request is retried rather than
        losing the word.
    """
    by_id: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        entry_id = entry.get("id")
        if type(entry_id) is not int or not 0 <= entry_id < len(words):
            raise ValueError(f"Response entry has no valid id: {entry}")
        if entry_id in by_id:
            raise ValueError(f"Response has more than one entry with id {entry_id}")
        by_id[entry_id] = entry

    if missing := [word.word for i, word in enumerate(words) if i not in by_id]:
        raise ValueError(f"Response is missing entries for {missing}")

    matched = tuple((word, by_id[i]) for i, word in enumerate(words))
    for word, entry in matched:
        if entry.get("term") != word.word:
            print(f"Model reported {word.word!r} as {entry.get('term')!r}.")
    return matched
