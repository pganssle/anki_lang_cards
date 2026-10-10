"""Tests for packing annotation entries into prompts."""

import json
import re
from collections.abc import Sequence
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from anki_lang_cards import _types as types
from anki_lang_cards import annotations_to_csv, prompts
from anki_lang_cards.backends import base

_INPUT_RE: Final = re.compile(r"INPUT:\n-+\n(?P<json>.*)\n+OUTPUT:\n-+", re.DOTALL)
MAX_TOKENS: Final = 2048


def _prompt_input(source_id: int, context_size: int) -> annotations_to_csv.PromptInput:
    return annotations_to_csv.PromptInput(
        rag_entries={},
        annotation_entry={
            "highlights": [f"palabra{source_id}"],
            "context": "x" * context_size,
            "source_id": source_id,
        },
        source=types.BookCitation(title="Libro", authors=("Autora",)),
    )


def _source_ids(request: base.Request) -> Sequence[int]:
    match = _INPUT_RE.search(request.prompt)
    assert match is not None
    return [entry["source_id"] for entry in json.loads(match.group("json"))]


def _estimated_tokens(request: base.Request) -> float:
    return (len(request.system_prompt) + len(request.prompt)) / 4


def test_last_entry_overflowing_is_not_sent_twice() -> None:
    """Regression: when the final entry overflowed the budget, the last
    prompt used to repeat every entry of the prompt before it."""
    requests = list(
        annotations_to_csv.generate_prompts(
            [_prompt_input(i, 600) for i in range(5)], max_tokens=MAX_TOKENS
        )
    )

    assert [_source_ids(request) for request in requests] == [[0, 1, 2, 3], [4]]


def test_oversized_entry_gets_its_own_prompt() -> None:
    requests = list(
        annotations_to_csv.generate_prompts(
            [_prompt_input(0, 100), _prompt_input(1, 20_000), _prompt_input(2, 100)],
            max_tokens=MAX_TOKENS,
        )
    )

    assert [_source_ids(request) for request in requests] == [[0], [1], [2]]


def test_no_entries() -> None:
    with pytest.raises(ValueError, match="No prompt inputs"):
        list(annotations_to_csv.generate_prompts([]))


def test_prompt_halves_concatenate_to_the_full_prompt() -> None:
    """The system prompt carries the static instructions and the dialect;
    together the halves read as one prompt, the dictionary entries leading
    into the input."""
    (request,) = annotations_to_csv.generate_prompts(
        [_prompt_input(7, 10)], dialect="Martian Spanish"
    )

    assert request.system_prompt == (
        prompts.BOOK_ANNOTATION_SYSTEM_PROMPT % {"dialect": "Martian Spanish"}
    )
    assert "Martian Spanish" not in request.prompt
    assert _source_ids(request) == [7]


@given(st.lists(st.integers(min_value=0, max_value=6_000), min_size=1, max_size=20))
def test_packing_properties(context_sizes: Sequence[int]) -> None:
    prompt_inputs = [
        _prompt_input(source_id, size) for source_id, size in enumerate(context_sizes)
    ]

    requests = list(
        annotations_to_csv.generate_prompts(prompt_inputs, max_tokens=MAX_TOKENS)
    )

    packed = [_source_ids(request) for request in requests]
    # Every entry is sent exactly once, in order.
    assert [source_id for ids in packed for source_id in ids] == list(
        range(len(context_sizes))
    )
    for request, ids in zip(requests, packed):
        # Only an entry alone in its prompt may go over the budget.
        assert len(ids) == 1 or _estimated_tokens(request) <= MAX_TOKENS
    # A prompt is only closed when the next entry would not have fit in it.
    for request_ids, next_ids in zip(packed, packed[1:]):
        combined = annotations_to_csv.generate_prompts(
            [prompt_inputs[i] for i in (*request_ids, next_ids[0])],
            max_tokens=10**9,
        )
        assert _estimated_tokens(next(iter(combined))) > MAX_TOKENS
