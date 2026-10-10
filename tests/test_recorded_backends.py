"""The pipelines against real model APIs, replayed from recordings.

Each backend in ``RECORDED_BACKENDS`` runs the same scenarios, and each
(scenario, backend) pair has its own cassette under
``tests/cassettes/test_recorded_backends/``. To cover a new provider, add an
entry and record its cassettes (see the notes in conftest.py):

    pytest tests/test_recorded_backends.py --record-mode=once

The assertions check the shape of what comes back rather than exact text, so
the same tests pass against a recording and, with ``--disable-recording``,
against the live API.
"""

import asyncio
import os
import pathlib
from collections.abc import Callable, Sequence
from typing import Final

import attrs
import pytest

from anki_lang_cards import _types as types
from anki_lang_cards import annotations_to_csv, caches, config, prompts, query
from anki_lang_cards.backends import anthropic_sdk, base, runner

REPO_CONFIG: Final = pathlib.Path(__file__).parent.parent / "config.toml"


def _anthropic_api_key(live: bool) -> str | None:
    """A placeholder when replaying; otherwise whatever credentials exist."""
    if not live:
        return "replayed-key"
    if "ANTHROPIC_API_KEY" not in os.environ and REPO_CONFIG.exists():
        repo_config = config.Config.from_file(REPO_CONFIG)
        if repo_config.anthropic is not None:
            return repo_config.anthropic.api_key
    return None  # Let the SDK find credentials itself.


def _make_anthropic(model_name: str, live: bool) -> base.Backend:
    poll_interval = 10.0 if live else 0.0
    return anthropic_sdk.AnthropicBackend(
        model_name,
        api_key=_anthropic_api_key(live),
        poll_interval=poll_interval,
        max_poll_interval=6 * poll_interval,
    )


@attrs.frozen
class RecordedBackend:
    spec: str
    model_name: str
    make: Callable[[str, bool], base.Backend]
    """Build the backend from the model name and whether the API is live."""

    def __str__(self) -> str:
        return self.spec


RECORDED_BACKENDS: Final[Sequence[RecordedBackend]] = (
    RecordedBackend(
        spec="anthropic/claude-sonnet-5",
        model_name="claude-sonnet-5",
        make=_make_anthropic,
    ),
)

WORDS: Final = (("gato", "cat", "nm"), ("correr", "to run", "vi"))


@pytest.fixture(params=RECORDED_BACKENDS, ids=str)
def recorded_backend(request: pytest.FixtureRequest) -> RecordedBackend:
    return request.param


def _make_runner(
    recorded: RecordedBackend,
    live: bool,
    cache_loc: pathlib.Path,
    cache_type: caches.CacheTypes,
    *,
    batch: bool,
) -> runner.QueryRunner:
    backend = recorded.make(recorded.model_name, live)
    if batch and base.Feature.BATCH not in backend.features:
        pytest.skip(f"{recorded.spec} cannot batch")
    cacher = caches.QueryCacher(
        cache_type=cache_type,
        model_name=recorded.spec,
        model_service=backend.service,
        cache_loc=cache_loc,
    )
    # One request at a time keeps the order of recorded requests stable,
    # and lets later requests read the prompt cache the first one wrote.
    return runner.QueryRunner(backend, cacher, batch=batch, max_concurrency=1)


def _cached_usage(query_runner: runner.QueryRunner) -> Sequence[caches.UsageData]:
    return [
        query_runner.cacher.load_from_cache_file(path)["response"]["usage"]
        for path in sorted(query_runner.cacher.cache_dir.glob("*.json"))
    ]


def _generate(query_runner: runner.QueryRunner) -> Sequence[types.Word]:
    words = [
        types.Word(
            rank=None,
            word=word,
            meaning=meaning,
            part_of_speech=part_of_speech,
            article=None,
            sentences=(),
        )
        for word, meaning, part_of_speech in WORDS
    ]
    handler = query.MultiSentencePromptHandler(
        prompts.MULTI_SENTENCE_PROMPT_TEMPLATE,
        system_prompt=prompts.MULTI_SENTENCE_SYSTEM_PROMPT
        % {"dialect": config.DEFAULT_DIALECT},
    )

    async def generate() -> Sequence[types.Word]:
        async with query_runner:
            return [
                word
                async for word in query.generate_sentences(
                    query_runner,
                    words,
                    handler,
                    model_name=query_runner.backend.model_name,
                    words_per_prompt=1,
                    max_invalid_output_retries=1,
                )
            ]

    return asyncio.run(generate())


def _assert_sentences(words: Sequence[types.Word], model_name: str) -> None:
    assert sorted(word.word for word in words) == sorted(word for word, *_ in WORDS)
    for word in words:
        assert word.sentences
        for sentence in word.sentences:
            assert sentence.spanish.strip()
            assert sentence.english.strip()
            assert isinstance(sentence.source, types.ModelCitation)
            assert sentence.source.model == model_name


def _assert_prompt_cache_used(
    query_runner: runner.QueryRunner, usage: Sequence[caches.UsageData]
) -> None:
    """The shared system prompt was written to or read from the cache."""
    if base.Feature.PROMPT_CACHING not in query_runner.backend.features:
        return
    cached_tokens = 0
    for entry in usage:
        details = entry["details"]
        assert details is not None
        cached_tokens += details["cache_creation_input_tokens"]
        cached_tokens += details["cache_read_input_tokens"]
    assert cached_tokens > 0


@pytest.mark.vcr
def test_sentences(
    recorded_backend: RecordedBackend, live_api: bool, tmp_path: pathlib.Path
) -> None:
    query_runner = _make_runner(
        recorded_backend, live_api, tmp_path, caches.CacheTypes.SENTENCES, batch=False
    )

    words = _generate(query_runner)

    _assert_sentences(words, recorded_backend.model_name)
    usage = _cached_usage(query_runner)
    assert len(usage) == len(WORDS)
    assert all(entry["input"] > 0 and entry["output"] > 0 for entry in usage)
    _assert_prompt_cache_used(query_runner, usage)


@pytest.mark.vcr
def test_sentences_batched(
    recorded_backend: RecordedBackend, live_api: bool, tmp_path: pathlib.Path
) -> None:
    query_runner = _make_runner(
        recorded_backend, live_api, tmp_path, caches.CacheTypes.SENTENCES, batch=True
    )

    words = _generate(query_runner)

    _assert_sentences(words, recorded_backend.model_name)
    usage = _cached_usage(query_runner)
    assert len(usage) == len(WORDS)
    assert all(entry["details"] and entry["details"]["batch"] for entry in usage)
    assert query_runner.journal.pending() == {}


@pytest.mark.vcr
def test_book_annotations(
    recorded_backend: RecordedBackend, live_api: bool, tmp_path: pathlib.Path
) -> None:
    # Citation IDs otherwise come from a process-wide counter, and the ID is
    # part of the recorded request.
    citation = types.BookCitation(
        title="Libro de Pruebas", authors=("Autora de Ejemplo",), pages=(12,), id=3
    )
    prompt_input = annotations_to_csv.PromptInput(
        rag_entries={"gato": "gato (n)\n  0. cat\n  1. jack (for a car)"},
        annotation_entry={
            "highlights": ["gatos"],
            "context": "Los gatos dormían al sol mientras los niños jugaban.",
            "source_id": citation.id,
        },
        source=citation,
    )
    (request,) = annotations_to_csv.generate_prompts([prompt_input])
    query_runner = _make_runner(
        recorded_backend,
        live_api,
        tmp_path,
        caches.CacheTypes.ANNOTATIONS,
        batch=False,
    )

    def parse(
        request: base.Request, response: caches.ResponseData
    ) -> Sequence[types.Word]:
        return annotations_to_csv.parse_llm_response(
            response["text"],
            model_used=recorded_backend.spec,
            source_mapping={citation.id: citation},
        )

    async def annotate() -> Sequence[types.Word]:
        async with query_runner:
            return [
                word
                async for _, words in runner.run_with_retries(
                    query_runner, [request], parse, max_retries=1, describe=str
                )
                for word in words
            ]

    (word,) = asyncio.run(annotate())

    assert word.word.startswith("gato")
    assert "cat" in word.meaning.lower()
    assert word.sentences
    assert any(sentence.source == citation for sentence in word.sentences)
    _assert_prompt_cache_used(query_runner, _cached_usage(query_runner))
