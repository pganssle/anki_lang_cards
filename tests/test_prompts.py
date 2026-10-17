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
        (prompts.BOOK_ANNOTATION_SYSTEM_PROMPT, {"dialect": "zz-dialect-zz"}),
        (
            prompts.BOOK_ANNOTATION_PROMPT_TEMPLATE,
            {"input": "zz-input-zz", "rag_entries": "zz-rag-zz"},
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


def test_fill_examples_keeps_percent_signs() -> None:
    """A ``%`` in an example sentence must survive the later dialect substitution."""
    template = "In %(dialect)s:\n%(example_input)s\n%(example_output)s\n"

    prompt = prompts.fill_examples(
        template,
        example_input=({"term": "por ciento", "meaning": "percent"},),
        example_output=(
            {
                "term": "por ciento",
                "examples": [
                    {"spanish": "Subió un 20%.", "english": "It rose 20%."},
                ],
            },
        ),
    )
    formatted = prompt % {"dialect": "zz-dialect-zz"}

    assert "In zz-dialect-zz:" in formatted
    assert '"Subió un 20%."' in formatted
    assert '"It rose 20%."' in formatted
