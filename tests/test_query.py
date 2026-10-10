"""Tests for prompt construction and response matching in the query module."""

import asyncio
import json
import pathlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Final, Self

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import caches, prompts, query, runner
from anki_lang_cards.backends import base

_INPUT_RE: Final = re.compile(r"INPUT:\n-+\n(?P<json>.*)\n+OUTPUT:\n-+", re.DOTALL)

type Entries = Sequence[Mapping[str, Any]]
type Responder = Callable[[Entries, int], Entries]


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


def test_multi_sentence_prompt_numbers_the_words() -> None:
    """Each word in the prompt carries its index in the batch as its id."""
    handler = query.MultiSentencePromptHandler(prompts.MULTI_SENTENCE_PROMPT_TEMPLATE)
    words = [_make_word("cura", "priest"), _make_word("cura", "cure")]

    match = _INPUT_RE.search(handler.make_prompt(words))

    assert match is not None
    parsed = json.loads(match.group("json"))
    assert [(entry["id"], entry["meaning"]) for entry in parsed] == [
        (0, "priest"),
        (1, "cure"),
    ]


#####
# Matching responses back to words
def _entry(entry_id: object, term: str) -> Mapping[str, Any]:
    return {
        "id": entry_id,
        "term": term,
        "examples": [
            {"spanish": f"Frase {entry_id}.", "english": f"Sentence {entry_id}."}
        ],
    }


def _echo(inputs: Entries) -> Entries:
    return [_entry(entry["id"], entry["term"]) for entry in inputs]


class ResponderBackend:
    """Answers each prompt by passing its INPUT entries to a responder.

    The responder also gets how many times the backend has been queried
    before, so it can misbehave on the first attempt only.
    """

    service: ClassVar[str] = "fake"
    features: ClassVar[frozenset[base.Feature]] = frozenset()
    default_words_per_prompt: ClassVar[int] = 10

    def __init__(self, responder: Responder):
        self.model_name = "fake-model"
        self.max_concurrency = None
        self.responder = responder
        self.calls = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def query_params(self) -> Mapping[str, object]:
        return {}

    async def query(self, request: base.Request) -> caches.ResponseData:
        match = _INPUT_RE.search(request.prompt)
        assert match is not None
        entries = self.responder(json.loads(match.group("json")), self.calls)
        self.calls += 1
        return {
            "text": json.dumps({"words": entries}),
            "usage": {"input": 1, "output": 1, "details": None},
        }


def _generate(
    tmp_path: pathlib.Path,
    words: Sequence[types.Word],
    responder: Responder,
    *,
    max_retries: int = 1,
) -> tuple[Sequence[types.Word], ResponderBackend]:
    backend = ResponderBackend(responder)
    query_runner = runner.QueryRunner(
        backend,
        caches.QueryCacher(
            cache_type=caches.CacheTypes.SENTENCES,
            model_name=backend.model_name,
            model_service=backend.service,
            cache_loc=tmp_path,
        ),
    )
    handler = query.MultiSentencePromptHandler(prompts.MULTI_SENTENCE_PROMPT_TEMPLATE)

    async def collect() -> Sequence[types.Word]:
        async with query_runner:
            return [
                word
                async for word in query.generate_sentences(
                    query_runner,
                    words,
                    handler,
                    model_name=backend.model_name,
                    max_invalid_output_retries=max_retries,
                )
            ]

    return asyncio.run(collect()), backend


def _spanish(word: types.Word) -> frozenset[str]:
    return frozenset(sentence.spanish for sentence in word.sentences)


def test_homographs_get_their_own_sentences(tmp_path: pathlib.Path) -> None:
    """Two words with the same headword in one batch are each yielded once,
    with the sentences generated for them."""
    priest = _make_word("cura", "priest")
    cure = _make_word("cura", "cure")

    words, _ = _generate(tmp_path, [priest, cure], lambda inputs, _: _echo(inputs))

    assert len(words) == 2
    assert {id(word) for word in words} == {id(priest), id(cure)}
    assert _spanish(priest) == {"Frase 0."}
    assert _spanish(cure) == {"Frase 1."}
    assert {word.model_used for word in words} == {"fake-model"}


def test_entries_are_matched_by_id_not_order_or_term(tmp_path: pathlib.Path) -> None:
    """Entries that come back reordered and with a rewritten term are still
    matched to the right words."""
    gato = _make_word("gato", "cat")
    atonito = _make_word("atónito/a", "astonished")

    words, backend = _generate(
        tmp_path,
        [gato, atonito],
        lambda inputs, _: [_entry(1, "atónito"), _entry(0, "gato")],
    )

    assert backend.calls == 1
    assert len(words) == 2
    assert _spanish(gato) == {"Frase 0."}
    assert _spanish(atonito) == {"Frase 1."}


def _missing_entry(inputs: Entries) -> Entries:
    return _echo(inputs)[:1]


def _missing_id(inputs: Entries) -> Entries:
    first, second = _echo(inputs)
    return [first, {key: value for key, value in second.items() if key != "id"}]


def _with_second_id(entry_id: object) -> Callable[[Entries], Entries]:
    def respond(inputs: Entries) -> Entries:
        return [_echo(inputs)[0], _entry(entry_id, inputs[1]["term"])]

    return respond


@pytest.mark.parametrize(
    "bad_response",
    [
        pytest.param(_missing_entry, id="missing-entry"),
        pytest.param(_missing_id, id="missing-id"),
        pytest.param(_with_second_id(2), id="unknown-id"),
        pytest.param(_with_second_id(-1), id="negative-id"),
        pytest.param(_with_second_id(0), id="duplicate-id"),
        pytest.param(_with_second_id("1"), id="string-id"),
        pytest.param(_with_second_id(True), id="bool-id"),
        pytest.param(_with_second_id(None), id="null-id"),
    ],
)
def test_unmatchable_response_is_retried(
    tmp_path: pathlib.Path, bad_response: Callable[[Entries], Entries]
) -> None:
    """A response that cannot be matched to every word is retried rather than
    losing the unmatched words."""
    gato = _make_word("gato", "cat")
    perro = _make_word("perro", "dog")

    def respond(inputs: Entries, attempt: int) -> Entries:
        return bad_response(inputs) if attempt == 0 else _echo(inputs)

    words, backend = _generate(tmp_path, [gato, perro], respond)

    assert backend.calls == 2
    assert len(words) == 2
    assert _spanish(gato) == {"Frase 0."}
    assert _spanish(perro) == {"Frase 1."}


def test_unmatchable_response_is_skipped_after_retries(
    tmp_path: pathlib.Path,
) -> None:
    """Once retries run out, a batch that never matches yields nothing,
    rather than a partial result."""
    gato = _make_word("gato", "cat")
    perro = _make_word("perro", "dog")

    words, backend = _generate(
        tmp_path,
        [gato, perro],
        lambda inputs, _: _missing_entry(inputs),
        max_retries=2,
    )

    assert backend.calls == 3
    assert words == []
    assert gato.sentences == perro.sentences == ()
