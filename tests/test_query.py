"""Tests for prompt construction in the query module."""

import json
import re
from typing import Final

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import prompts, query

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


@pytest.mark.xfail(
    strict=True,
    reason="MultiSentencePromptHandler interpolates words into a JSON "
    "template without escaping, and never wraps the objects in a list, so "
    "the INPUT section is not valid JSON.",
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
