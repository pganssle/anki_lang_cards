"""Tests for the prompt templates.

These mostly guard the %-interpolation: a stray ``%`` introduced into a
template would make formatting raise at runtime, and a missing key would
mean e.g. the configured dialect silently never reaches the model.
"""

from collections.abc import Mapping

import pytest

from anki_lang_cards import prompts


@pytest.mark.parametrize(
    ("template", "keys"),
    [
        (
            prompts.MULTI_SENTENCE_SYSTEM_PROMPT,
            {"dialect": "zz-dialect-zz"},
        ),
        (
            prompts.BOOK_ANNOTATION_PROMPT,
            {
                "dialect": "zz-dialect-zz",
                "input": "zz-input-zz",
                "rag_entries": "zz-rag-zz",
            },
        ),
        (
            prompts.SENTENCE_PROMPT,
            {"term": "zz-term-zz", "pos": "zz-pos-zz", "meaning": "zz-meaning-zz"},
        ),
        (prompts.MULTI_SENTENCE_PROMPT_TEMPLATE, {"data": "zz-data-zz"}),
        (prompts.ORG_TO_CSV_PROMPT_TEMPLATE, {"org_text": "zz-org-zz"}),
    ],
)
def test_templates_format_cleanly(template: str, keys: Mapping[str, str]) -> None:
    formatted = template % keys

    for value in keys.values():
        assert value in formatted
