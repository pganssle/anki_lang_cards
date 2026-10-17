"""Tests for prompt construction in the query module."""

import asyncio
import json
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import ClassVar, Final, Self

from anki_lang_cards import _types as types
from anki_lang_cards import caches, prompts, query, runner
from anki_lang_cards.backends import base

_INPUT_RE: Final = re.compile(r"INPUT:\n-+\n(?P<json>.*)\n+OUTPUT:\n-+", re.DOTALL)


def _make_word(word: str, meaning: str) -> types.Word:
    return types.Word(
        rank=None,
        word=word,
        meaning=meaning,
        part_of_speech="nm",
        article="el",
        sentences=(),
    )


def test_multi_sentence_prompt_is_valid_json() -> None:
    """The INPUT section of the prompt parses as a JSON list matching the
    words, even when a meaning contains quotes."""
    handler = query.MultiSentencePromptHandler(
        prompts.MULTI_SENTENCE_PROMPT_TEMPLATE,
        system_prompt=prompts.MULTI_SENTENCE_SYSTEM_PROMPT,
    )
    words = [
        _make_word("gato", "cat"),
        _make_word("entrecomillado", 'text "in quotes"'),
    ]
    prompt = handler.make_prompt(words)

    match = _INPUT_RE.search(prompt)
    assert match is not None
    parsed = json.loads(match.group("json"))
    assert [entry["term"] for entry in parsed] == ["gato", "entrecomillado"]
    assert parsed[1]["meaning"] == 'text "in quotes"'


class _OneWordBackend:
    """Answers each one-word prompt with a sentence, or rejects it."""

    service: ClassVar[str] = "one-word"
    features: ClassVar[frozenset[base.Feature]] = frozenset()
    default_words_per_prompt: ClassVar[int] = 1

    def __init__(self, rejected: str):
        self.model_name = "one-word"
        self.max_concurrency: int | None = None
        self.rejected = rejected

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def query_params(self) -> Mapping[str, object]:
        return {}

    async def query(self, request: base.Request) -> caches.ResponseData:
        match = _INPUT_RE.search(request.prompt)
        assert match is not None
        (entry,) = json.loads(match.group("json"))
        if entry["term"] == self.rejected:
            raise base.RequestError("HTTP 500: overloaded", status=500)
        text = json.dumps(
            {
                "words": [
                    {
                        "term": entry["term"],
                        "examples": [
                            {
                                "spanish": f"Una frase con {entry['term']}.",
                                "english": "A sentence.",
                            }
                        ],
                    }
                ]
            }
        )
        return {"text": text, "usage": {"input": 1, "output": 1, "details": None}}


def test_generate_sentences_reports_dropped_words(tmp_path: pathlib.Path) -> None:
    """A request the runner gives up on comes back as the words it held, so
    the caller can say which words got no sentences."""
    backend = _OneWordBackend(rejected="perro")
    cacher = caches.QueryCacher(
        cache_type=caches.CacheTypes.SENTENCES,
        model_name="one-word",
        model_service="one-word",
        cache_loc=tmp_path,
    )
    query_runner = runner.QueryRunner(backend, cacher)
    words = [_make_word("gato", "cat"), _make_word("perro", "dog")]
    handler = query.MultiSentencePromptHandler(prompts.MULTI_SENTENCE_PROMPT_TEMPLATE)

    async def collect() -> Sequence[types.Word | query.DroppedWords]:
        async with query_runner:
            return [
                result
                async for result in query.generate_sentences(
                    query_runner, words, handler, model_name="one-word"
                )
            ]

    results = asyncio.run(collect())

    (dropped,) = (r for r in results if isinstance(r, query.DroppedWords))
    assert [word.word for word in dropped.words] == ["perro"]
    assert dropped.reason == "HTTP 500: overloaded"
    (filled,) = (r for r in results if isinstance(r, types.Word))
    assert filled.word == "gato"
    assert [sentence.spanish for sentence in filled.sentences] == [
        "Una frase con gato."
    ]
