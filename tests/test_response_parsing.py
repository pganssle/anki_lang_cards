"""Tests for decoding model responses and for the response schemas."""

import json
from collections.abc import Iterator, Mapping
from typing import Any, Final

import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import annotations_to_csv, config, prompts, query

ENTRY: Final = {"term": "gato", "examples": [{"spanish": "Gato.", "english": "Cat."}]}


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"words": [ENTRY]}),
        json.dumps([ENTRY]),
        "```json\n" + json.dumps({"words": [ENTRY]}) + "\n```",
        "  ```\n" + json.dumps([ENTRY]) + "```  ",
    ],
)
def test_decode_words(text: str) -> None:
    assert query.decode_words(text) == [ENTRY]


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, I can't produce JSON today.",
        json.dumps({"words": "gato"}),
        json.dumps({"something": [ENTRY]}),
        json.dumps({"words": [{"term": "gato"}]}),
        json.dumps({"words": [{"term": "gato", "examples": ["Gato."]}]}),
        json.dumps(["gato"]),
    ],
)
def test_decode_words_rejects(text: str) -> None:
    # Undecodable JSON raises json.JSONDecodeError, a ValueError subclass.
    with pytest.raises(ValueError, match="Expecting value|not a list of words"):
        query.decode_words(text)


def test_annotation_entry_missing_field_is_retryable() -> None:
    entry = {"highlight": "gatos", "headword": "gato", "examples": []}

    with pytest.raises(ValueError, match="meaning"):
        annotations_to_csv.parse_llm_response(
            json.dumps({"words": [entry]}), model_used="m", source_mapping={}
        )


def test_annotation_response() -> None:
    citation = types.BookCitation(title="Libro", authors=("Autora",), id=7)
    response = {
        "words": [
            {
                "highlight": "gatos",
                "headword": "gato",
                "part_of_speech": "nm",
                "meaning": "cat",
                "article": None,
                "examples": [
                    {"spanish": "Los gatos.", "english": "The cats.", "source_id": 7}
                ],
            }
        ]
    }

    (word,) = annotations_to_csv.parse_llm_response(
        json.dumps(response), model_used="m", source_mapping={7: citation}
    )

    assert word.word == "gato"
    assert word.article is None
    (sentence,) = word.sentences
    assert sentence.source == citation


def _objects(schema: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every object schema nested anywhere in a schema."""
    pending = [schema]
    while pending:
        node = pending.pop()
        if node.get("type") == "object":
            yield node
        pending.extend(node.get("properties", {}).values())
        pending.extend(node.get("anyOf", ()))
        if "items" in node:
            pending.append(node["items"])


@pytest.mark.parametrize(
    "schema",
    [prompts.MULTI_SENTENCE_RESPONSE_SCHEMA, prompts.BOOK_ANNOTATION_RESPONSE_SCHEMA],
)
def test_schemas_meet_api_requirements(schema: Mapping[str, Any]) -> None:
    """Structured outputs need every object closed and fully required."""
    objects = list(_objects(schema))
    assert objects
    for node in objects:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])


def test_sentence_prompt_example_is_a_valid_response() -> None:
    system_prompt = prompts.MULTI_SENTENCE_SYSTEM_PROMPT % {
        "dialect": config.DEFAULT_DIALECT
    }
    example = system_prompt.split("OUTPUT:\n--------------\n")[-1]

    entries = query.decode_words(example)

    assert [entry["term"] for entry in entries] == [
        "casa",
        "rebotar",
        "piso",
        "muñeca",
    ]
    assert [entry["id"] for entry in entries] == [0, 1, 2, 3]
    for entry in entries:
        assert set(entry) == {"id", "term", "examples"}
        assert len(entry["examples"]) == 4


def test_sentence_prompt_example_input_ids_match_output() -> None:
    """The example's input numbers its terms the way make_prompt does, and
    the example output echoes the same ids."""
    system_prompt = prompts.MULTI_SENTENCE_SYSTEM_PROMPT % {
        "dialect": config.DEFAULT_DIALECT
    }
    example_input = system_prompt.split("INPUT:\n----------\n")[-1].split(
        "\n\nOUTPUT:"
    )[0]
    example_output = system_prompt.split("OUTPUT:\n--------------\n")[-1]

    inputs = json.loads(example_input)
    assert [entry["id"] for entry in inputs] == list(range(len(inputs)))
    assert [(entry["id"], entry["term"]) for entry in inputs] == [
        (entry["id"], entry["term"]) for entry in query.decode_words(example_output)
    ]
