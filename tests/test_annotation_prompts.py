"""Tests for packing annotation entries into prompts."""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from anki_lang_cards import _types as types
from anki_lang_cards import annotations_to_csv, prompts, runner
from anki_lang_cards.backends import base

_INPUT_RE: Final = re.compile(r"INPUT:\n-+\n(?P<json>.*)\n+OUTPUT:\n-+", re.DOTALL)
MAX_TOKENS: Final = 2048

_RESPONSE: Final = json.dumps(
    {
        "words": [
            {
                "headword": "palabra",
                "part_of_speech": "nf",
                "meaning": "word",
                "examples": [
                    {"spanish": "Una palabra.", "english": "A word.", "source_id": 1}
                ],
            }
        ]
    }
)
"""A model response citing the first source of whichever prompt it answers."""


def _prompt_input(entry_id: int, context_size: int) -> annotations_to_csv.PromptInput:
    return annotations_to_csv.PromptInput(
        rag_entries={},
        annotation_entry={
            "highlights": [f"palabra{entry_id}"],
            "context": "x" * context_size,
        },
        source=types.BookCitation(title=f"Libro {entry_id}", authors=("Autora",)),
    )


def _input_entries(request: base.Request) -> Sequence[dict]:
    match = _INPUT_RE.search(request.prompt)
    assert match is not None
    return json.loads(match.group("json"))


def _entry_ids(prompt: annotations_to_csv.AnnotationPrompt) -> Sequence[int]:
    """The ``entry_id`` each of the prompt's entries was made with."""
    return [
        int(entry["highlights"][0].removeprefix("palabra"))
        for entry in _input_entries(prompt.request)
    ]


def _estimated_tokens(prompt: annotations_to_csv.AnnotationPrompt) -> float:
    request = prompt.request
    return (len(request.system_prompt) + len(request.prompt)) / 4


def test_last_entry_overflowing_is_not_sent_twice() -> None:
    """Regression: when the final entry overflowed the budget, the last
    prompt used to repeat every entry of the prompt before it."""
    prompts = list(
        annotations_to_csv.generate_prompts(
            [_prompt_input(i, 600) for i in range(5)], max_tokens=MAX_TOKENS
        )
    )

    assert [_entry_ids(prompt) for prompt in prompts] == [[0, 1, 2, 3], [4]]


def test_oversized_entry_gets_its_own_prompt() -> None:
    prompts = list(
        annotations_to_csv.generate_prompts(
            [_prompt_input(0, 100), _prompt_input(1, 20_000), _prompt_input(2, 100)],
            max_tokens=MAX_TOKENS,
        )
    )

    assert [_entry_ids(prompt) for prompt in prompts] == [[0], [1], [2]]


def test_no_entries() -> None:
    with pytest.raises(ValueError, match="No prompt inputs"):
        list(annotations_to_csv.generate_prompts([]))


def test_prompt_halves_concatenate_to_the_full_prompt() -> None:
    """The system prompt carries the static instructions and the dialect;
    together the halves read as one prompt, the dictionary entries leading
    into the input."""
    (prompt,) = annotations_to_csv.generate_prompts(
        [_prompt_input(7, 10)], dialect="Martian Spanish"
    )
    request = prompt.request

    assert request.system_prompt == (
        prompts.BOOK_ANNOTATION_SYSTEM_PROMPT % {"dialect": "Martian Spanish"}
    )
    assert "Martian Spanish" not in request.prompt
    assert _entry_ids(prompt) == [7]


@given(st.lists(st.integers(min_value=0, max_value=6_000), min_size=1, max_size=20))
def test_packing_properties(context_sizes: Sequence[int]) -> None:
    prompt_inputs = [
        _prompt_input(entry_id, size) for entry_id, size in enumerate(context_sizes)
    ]

    prompts = list(
        annotations_to_csv.generate_prompts(prompt_inputs, max_tokens=MAX_TOKENS)
    )

    packed = [_entry_ids(prompt) for prompt in prompts]
    # Every entry is sent exactly once, in order.
    assert [entry_id for ids in packed for entry_id in ids] == list(
        range(len(context_sizes))
    )
    for prompt, ids in zip(prompts, packed):
        # Only an entry alone in its prompt may go over the budget.
        assert len(ids) == 1 or _estimated_tokens(prompt) <= MAX_TOKENS
    # A prompt is only closed when the next entry would not have fit in it.
    for prompt_ids, next_ids in zip(packed, packed[1:]):
        combined = annotations_to_csv.generate_prompts(
            [prompt_inputs[i] for i in (*prompt_ids, next_ids[0])],
            max_tokens=10**9,
        )
        assert _estimated_tokens(next(iter(combined))) > MAX_TOKENS


@given(st.lists(st.integers(min_value=0, max_value=6_000), min_size=1, max_size=20))
def test_sources_are_numbered_within_each_prompt(
    context_sizes: Sequence[int],
) -> None:
    prompt_inputs = [
        _prompt_input(entry_id, size) for entry_id, size in enumerate(context_sizes)
    ]

    for prompt in annotations_to_csv.generate_prompts(
        prompt_inputs, max_tokens=MAX_TOKENS
    ):
        source_ids = [entry["source_id"] for entry in _input_entries(prompt.request)]
        assert source_ids == list(range(1, len(source_ids) + 1))
        # Each source_id maps back to the citation of the entry it labels.
        assert [prompt.sources[source_id] for source_id in source_ids] == [
            prompt_inputs[entry_id].source for entry_id in _entry_ids(prompt)
        ]


@pytest.mark.parametrize("preceding", [0, 1, 4])
def test_prompt_does_not_depend_on_the_rest_of_the_run(preceding: int) -> None:
    """Regression: source IDs came from a process-wide counter, so adding or
    dropping one annotation changed every later prompt, and with it every
    response cache key."""
    # Built separately, as separate runs would.
    alone = [_prompt_input(99, 20_000)]
    in_a_run = [
        *(_prompt_input(i, 10) for i in range(preceding)),
        _prompt_input(99, 20_000),
    ]

    (expected,) = annotations_to_csv.generate_prompts(alone, max_tokens=MAX_TOKENS)
    *_, actual = annotations_to_csv.generate_prompts(in_a_run, max_tokens=MAX_TOKENS)

    assert actual.request == expected.request
    assert actual.request.key == expected.request.key
    assert actual.sources == {1: in_a_run[-1].source}


def test_response_maps_back_to_each_prompts_sources() -> None:
    first, second = annotations_to_csv.generate_prompts(
        [_prompt_input(0, 10), _prompt_input(1, 20_000)], max_tokens=MAX_TOKENS
    )
    citations = [
        sentence.source
        for prompt in (first, second)
        for word in annotations_to_csv.parse_llm_response(
            _RESPONSE, model_used="m", source_mapping=prompt.sources
        )
        for sentence in word.sentences
    ]

    assert citations == [first.sources[1], second.sources[1]]
    assert [citation.title for citation in citations] == ["Libro 0", "Libro 1"]


class _FakeRunner:
    """Answers each distinct request with the same response, like a runner
    that found all of them in the cache."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.queried: list[base.Request] = []

    async def run(
        self,
        requests: Sequence[base.Request],
        *,
        invalidate_cache: bool = False,
        skip_cache: bool = False,
    ) -> AsyncIterator[runner.Success | runner.Failure]:
        for request in dict.fromkeys(requests):
            self.queried.append(request)
            yield runner.Success(request, {"text": self.response_text})


def test_annotation_in_two_books_cites_both() -> None:
    """Regression: the same highlight in two books makes two identical
    prompts, which the runner sends once; the shared response was parsed with
    only the last book's sources, so the first book's citation was lost."""
    books = [
        types.BookCitation(title=f"Libro {i}", authors=("Autora",)) for i in range(2)
    ]
    # Too big to share a prompt, so each book's copy gets its own.
    prompt_inputs = [
        annotations_to_csv.PromptInput(
            rag_entries={},
            annotation_entry={"highlights": ["palabra"], "context": "x" * 20_000},
            source=book,
        )
        for book in books
    ]
    sources_by_request = annotations_to_csv.group_sources_by_request(
        annotations_to_csv.generate_prompts(prompt_inputs, max_tokens=MAX_TOKENS)
    )
    fake_runner = _FakeRunner(_RESPONSE)

    async def collect() -> Sequence[types.Word]:
        return [
            word
            async for words in annotations_to_csv.annotate(
                fake_runner,  # type: ignore[arg-type]
                sources_by_request,
                model_used="m",
                max_retries=0,
            )
            for word in words
        ]

    words = asyncio.run(collect())

    # One request for the two prompts, so the progress count is one.
    assert len(sources_by_request) == 1
    assert len(fake_runner.queried) == 1
    assert [sentence.source for word in words for sentence in word.sentences] == books
