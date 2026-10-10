"""Tests for backend selection, the query runner and the batch journal."""

import argparse
import asyncio
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, Self

import attrs
import pytest

from anki_lang_cards import caches, config
from anki_lang_cards.backends import base, runner, selection, tgw

#####
# Resolving model specs


@pytest.mark.parametrize(
    ("spec", "service", "expected_service", "expected_model"),
    [
        ("anthropic/claude-sonnet-5", None, "anthropic", "claude-sonnet-5"),
        ("tgw/llama.gguf", None, "tgw", "llama.gguf"),
        ("gpt-5", None, "llm", "gpt-5"),
        ("ollama/llama3", None, "llm", "ollama/llama3"),
        ("anthropic/claude-sonnet-5", "llm", "llm", "anthropic/claude-sonnet-5"),
        ("claude-sonnet-5", "anthropic", "anthropic", "claude-sonnet-5"),
        ("anthropic", None, "llm", "anthropic"),
    ],
)
def test_resolve(
    spec: str, service: str | None, expected_service: str, expected_model: str
) -> None:
    resolved = selection.resolve(spec, service)

    assert resolved.spec == spec
    assert resolved.service == expected_service
    assert resolved.model_name == expected_model
    assert resolved.backend_class.service == expected_service


def test_resolve_unknown_service() -> None:
    with pytest.raises(ValueError, match="Unknown model service"):
        selection.resolve("gpt-5", "openai")


#####
# Choosing features

BATCH = base.Feature.BATCH
CACHING = base.Feature.PROMPT_CACHING


def _never_asked(question: str) -> bool:
    raise AssertionError(f"Unexpectedly asked: {question}")


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        (None, frozenset({BATCH}), frozenset({BATCH})),
        (True, frozenset({BATCH}), frozenset({BATCH})),
        (False, frozenset({BATCH}), frozenset()),
        (None, frozenset(), frozenset()),
        (False, frozenset(), frozenset()),
    ],
)
def test_choose_features_without_asking(
    requested: bool | None,
    available: frozenset[base.Feature],
    expected: frozenset[base.Feature],
) -> None:
    chosen = selection.choose_features(
        [selection.FeatureRequest(BATCH, requested, "--batch")],
        available,
        backend_description="the test backend",
        confirm=_never_asked,
    )
    assert chosen == expected


def test_required_feature_unavailable_accepted() -> None:
    questions: list[str] = []

    def accept(question: str) -> bool:
        questions.append(question)
        return True

    chosen = selection.choose_features(
        [
            selection.FeatureRequest(BATCH, True, "batch = true in [models]"),
            selection.FeatureRequest(CACHING, None),
        ],
        frozenset({CACHING}),
        backend_description="the test backend",
        confirm=accept,
    )

    assert chosen == frozenset({CACHING})
    (question,) = questions
    assert "Batching" in question
    assert "batch = true in [models]" in question
    assert "the test backend" in question


@pytest.mark.parametrize(
    ("feature", "flag"), [(BATCH, "--no-batch"), (CACHING, "--no-prompt-caching")]
)
def test_required_feature_unavailable_declined(
    feature: base.Feature, flag: str
) -> None:
    with pytest.raises(selection.FeatureUnavailableError, match=flag):
        selection.choose_features(
            [selection.FeatureRequest(feature, True, "a flag")],
            frozenset(),
            backend_description="the test backend",
            confirm=lambda question: False,
        )


def test_confirm_without_terminal_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", _never_asked)

    assert not selection.confirm_on_terminal("Continue?")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("YES ", True), ("n", False), ("", False), ("maybe", False)],
)
def test_confirm_on_terminal(
    monkeypatch: pytest.MonkeyPatch, answer: str, expected: bool
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    assert selection.confirm_on_terminal("Continue?") is expected


#####
# Building a runner from the command line and config


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection.add_arguments(parser)
    return parser.parse_args(argv)


def _config(
    *,
    models: config.ModelsConfig | None = None,
    anthropic: config.AnthropicConfig | None = None,
    tgw_config: config.TGWConfig | None = None,
) -> config.Config:
    return config.Config(
        annotations=config.AnnotationsConfig(),
        models=models or config.ModelsConfig(),
        anthropic=anthropic,
        tgw=tgw_config,
    )


TGW_CONFIG = config.TGWConfig(api_base_url="http://localhost:5000/v1/")


@pytest.mark.parametrize(
    ("argv", "conf", "expected"),
    [
        (
            (),
            _config(anthropic=config.AnthropicConfig(default_model="anthropic/x")),
            "anthropic/x",
        ),
        (
            (),
            _config(
                models=config.ModelsConfig(default_model="anthropic/y"),
                anthropic=config.AnthropicConfig(default_model="anthropic/x"),
            ),
            "anthropic/y",
        ),
        (
            ("--model", "gpt-5"),
            _config(models=config.ModelsConfig(default_model="anthropic/y")),
            "gpt-5",
        ),
    ],
)
def test_model_spec_precedence(
    argv: Sequence[str], conf: config.Config, expected: str
) -> None:
    assert selection.model_spec(_parse(*argv), conf) == expected


@pytest.mark.parametrize("argv", [(), ("--model-service", "tgw")])
def test_model_spec_requires_a_model(argv: Sequence[str]) -> None:
    """There is no built-in default model, for any backend."""
    with pytest.raises(selection.ModelNotConfiguredError, match="default_model"):
        selection.model_spec(_parse(*argv), _config(tgw_config=TGW_CONFIG))


@pytest.mark.parametrize(
    ("argv", "models", "batch", "caching"),
    [
        ((), config.ModelsConfig(), True, True),
        (("--no-batch",), config.ModelsConfig(), False, True),
        ((), config.ModelsConfig(batch=False), False, True),
        (("--batch",), config.ModelsConfig(batch=False), True, True),
        (("--no-prompt-caching",), config.ModelsConfig(), True, False),
    ],
)
def test_make_runner_anthropic_features(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: Sequence[str],
    models: config.ModelsConfig,
    batch: bool,
    caching: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    conf = _config(
        models=attrs.evolve(models, default_model="anthropic/claude-sonnet-5"),
        anthropic=config.AnthropicConfig(api_key="key"),
    )

    resolved, query_runner = selection.make_runner(
        _parse(*argv), conf, caches.CacheTypes.SENTENCES, confirm=_never_asked
    )

    assert resolved.service == "anthropic"
    assert query_runner.batch is batch
    assert query_runner.backend.prompt_caching is caching  # type: ignore[attr-defined]
    assert query_runner.cacher.cache_dir == (
        caches.DEFAULT_CACHE_LOC
        / "sentences"
        / "anthropic"
        / "anthropic"
        / "claude-sonnet-5"
    )


def test_make_runner_without_batching_by_default() -> None:
    """An unrequested feature the backend lacks is silently left off."""
    resolved, query_runner = selection.make_runner(
        _parse("--model", "tgw/llama.gguf"),
        _config(tgw_config=TGW_CONFIG),
        caches.CacheTypes.SENTENCES,
        confirm=_never_asked,
    )

    assert resolved.service == "tgw"
    assert not query_runner.batch
    assert query_runner.max_concurrency == TGW_CONFIG.max_connections


@pytest.mark.parametrize("accept", [True, False])
def test_make_runner_required_batching_unavailable(accept: bool) -> None:
    args = _parse("--model", "tgw/llama.gguf")
    conf = _config(models=config.ModelsConfig(batch=True), tgw_config=TGW_CONFIG)

    def confirm(question: str) -> bool:
        assert "batch = true in [models]" in question
        return accept

    if accept:
        _, query_runner = selection.make_runner(
            args, conf, caches.CacheTypes.SENTENCES, confirm=confirm
        )
        assert not query_runner.batch
    else:
        with pytest.raises(selection.FeatureUnavailableError, match="--no-batch"):
            selection.make_runner(
                args, conf, caches.CacheTypes.SENTENCES, confirm=confirm
            )


def test_make_runner_tgw_without_config() -> None:
    with pytest.raises(ValueError, match=r"\[tgw\]"):
        selection.make_runner(
            _parse("--model", "tgw/llama.gguf"),
            _config(),
            caches.CacheTypes.SENTENCES,
            confirm=_never_asked,
        )


#####
# The query runner


def _response(text: str) -> caches.ResponseData:
    return {"text": text, "usage": {"input": 1, "output": 1, "details": None}}


class FakeBackend:
    """Answers each prompt with a scripted sequence of replies.

    A reply is either the response text or an exception to raise. The last
    reply repeats once the script runs out.
    """

    service: ClassVar[str] = "fake"
    features: ClassVar[frozenset[base.Feature]] = frozenset({base.Feature.BATCH})
    default_words_per_prompt: ClassVar[int] = 1

    def __init__(
        self,
        replies: Mapping[str, Sequence[str | Exception]] | None = None,
        *,
        max_concurrency: int | None = None,
        batch_failures: Mapping[str, str] | None = None,
    ):
        self.model_name = "fake-model"
        self.max_concurrency = max_concurrency
        self.replies = dict(replies or {})
        self.batch_failures = dict(batch_failures or {})
        self.queried: list[str] = []
        self.resumed: list[Mapping[str, frozenset[str]]] = []
        self.entered = False

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.entered = False

    def query_params(self) -> Mapping[str, object]:
        return {"fake": True}

    def _reply(self, request: base.Request) -> caches.ResponseData:
        self.queried.append(request.prompt)
        script = self.replies.get(request.prompt, [f"answer to {request.prompt}"])
        reply = script[0]
        if len(script) > 1:
            self.replies[request.prompt] = script[1:]
        if isinstance(reply, Exception):
            raise reply
        return _response(reply)

    async def query(self, request: base.Request) -> caches.ResponseData:
        return self._reply(request)

    async def query_batch(
        self,
        requests: Sequence[base.Request],
        *,
        resume: Mapping[str, frozenset[str]],
        on_submit: base.SubmitCallback,
    ) -> base.BatchResults:
        self.resumed.append(resume)
        job_id = f"job-{len(self.resumed)}"
        on_submit(job_id, frozenset(request.key for request in requests))
        responses: dict[str, caches.ResponseData] = {}
        failures: dict[str, str] = {}
        for request in requests:
            if request.prompt in self.batch_failures:
                failures[request.key] = self.batch_failures[request.prompt]
                continue
            try:
                responses[request.key] = self._reply(request)
            except base.IncompleteResponseError as e:
                failures[request.key] = str(e)
        return base.BatchResults(
            responses=responses,
            failures=failures,
            completed_jobs=frozenset({job_id, *resume}),
        )


def _cacher(tmp_path: pathlib.Path) -> caches.QueryCacher:
    return caches.QueryCacher(
        cache_type=caches.CacheTypes.SENTENCES,
        model_name="fake-model",
        model_service="fake",
        cache_loc=tmp_path,
    )


def _run(
    query_runner: runner.QueryRunner,
    requests: Sequence[base.Request],
    **kwargs: bool,
) -> Mapping[str, runner.Success | runner.Failure]:
    async def collect() -> Mapping[str, runner.Success | runner.Failure]:
        async with query_runner:
            return {
                outcome.request.prompt: outcome
                async for outcome in query_runner.run(requests, **kwargs)
            }

    return asyncio.run(collect())


def _texts(
    outcomes: Mapping[str, runner.Success | runner.Failure],
) -> Mapping[str, str]:
    return {
        prompt: outcome.response["text"]
        for prompt, outcome in outcomes.items()
        if isinstance(outcome, runner.Success)
    }


REQUESTS = (
    base.Request("uno", system_prompt="sys"),
    base.Request("dos", system_prompt="sys"),
)


@pytest.mark.parametrize("batch", [False, True])
def test_runner_caches_responses(tmp_path: pathlib.Path, batch: bool) -> None:
    backend = FakeBackend()
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path), batch=batch)

    first = _run(query_runner, REQUESTS)
    second = _run(query_runner, REQUESTS)

    expected = {"uno": "answer to uno", "dos": "answer to dos"}
    assert _texts(first) == expected
    assert _texts(second) == expected
    assert sorted(backend.queried) == ["dos", "uno"]
    cached = query_runner.cacher.load_from_cache("uno", system_prompt="sys")
    assert cached is not None
    assert cached["input"]["system_prompt"] == "sys"
    assert cached["input"]["query_params"] == {"fake": True, "batch": batch}


def test_runner_deduplicates_requests(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend()
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    outcomes = _run(query_runner, [REQUESTS[0], REQUESTS[0]])

    assert list(outcomes) == ["uno"]
    assert backend.queried == ["uno"]


def test_runner_invalidate_cache(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend({"uno": ["old", "new"]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))
    _run(query_runner, REQUESTS[:1])

    refreshed = _run(query_runner, REQUESTS[:1], invalidate_cache=True)
    cached = _run(query_runner, REQUESTS[:1])

    assert _texts(refreshed) == {"uno": "new"}
    assert _texts(cached) == {"uno": "new"}
    assert backend.queried == ["uno", "uno"]


def test_runner_skip_cache(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend({"uno": ["first", "second"]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    _run(query_runner, REQUESTS[:1], skip_cache=True)
    again = _run(query_runner, REQUESTS[:1], skip_cache=True)

    assert _texts(again) == {"uno": "second"}
    assert query_runner.cacher.load_from_cache("uno", system_prompt="sys") is None


@pytest.mark.parametrize("batch", [False, True])
def test_runner_incomplete_response_not_cached(
    tmp_path: pathlib.Path, batch: bool
) -> None:
    backend = FakeBackend({"uno": [base.IncompleteResponseError("cut off")]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path), batch=batch)

    outcomes = _run(query_runner, REQUESTS)

    assert outcomes["uno"] == runner.Failure(REQUESTS[0], "cut off")
    assert isinstance(outcomes["dos"], runner.Success)
    assert query_runner.cacher.load_from_cache("uno", system_prompt="sys") is None


def test_runner_batch_failures(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend(batch_failures={"uno": "errored: overloaded"})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path), batch=True)

    outcomes = _run(query_runner, REQUESTS)

    assert outcomes["uno"] == runner.Failure(REQUESTS[0], "errored: overloaded")
    assert _texts(outcomes) == {"dos": "answer to dos"}


def test_runner_batch_resumes_and_clears_journal(tmp_path: pathlib.Path) -> None:
    """Jobs left in the journal by a dead run are offered for resuming, and
    every collected job leaves the journal."""
    backend = FakeBackend()
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path), batch=True)
    query_runner.journal.record("job-from-dead-run", frozenset({REQUESTS[0].key}))

    _run(query_runner, REQUESTS)

    assert backend.resumed == [{"job-from-dead-run": frozenset({REQUESTS[0].key})}]
    assert query_runner.journal.pending() == {}


def test_runner_batch_requires_support(tmp_path: pathlib.Path) -> None:
    backend = tgw.TGWBackend("model", api_base_url="http://localhost/")
    with pytest.raises(ValueError, match="cannot batch"):
        runner.QueryRunner(backend, _cacher(tmp_path), batch=True)


@pytest.mark.parametrize(
    ("backend_cap", "requested", "expected"),
    [(None, 5, 5), (2, 5, 2), (8, 5, 5)],
)
def test_runner_concurrency_cap(
    tmp_path: pathlib.Path, backend_cap: int | None, requested: int, expected: int
) -> None:
    query_runner = runner.QueryRunner(
        FakeBackend(max_concurrency=backend_cap),
        _cacher(tmp_path),
        max_concurrency=requested,
    )
    assert query_runner.max_concurrency == expected


def test_runner_enters_backend(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend()
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    async def check() -> None:
        async with query_runner:
            assert backend.entered
        assert not backend.entered

    asyncio.run(check())


#####
# Retries


def _parse_number(request: base.Request, response: caches.ResponseData) -> int:
    return int(response["text"])


def _run_with_retries(
    query_runner: runner.QueryRunner,
    requests: Sequence[base.Request],
    parse: Callable[[base.Request, caches.ResponseData], int] = _parse_number,
    **kwargs: object,
) -> Mapping[str, int]:
    async def collect() -> Mapping[str, int]:
        async with query_runner:
            return {
                request.prompt: parsed
                async for request, parsed in runner.run_with_retries(
                    query_runner,
                    requests,
                    parse,
                    describe=lambda request: request.prompt,
                    **kwargs,  # type: ignore[arg-type]
                )
            }

    return asyncio.run(collect())


@pytest.mark.parametrize("batch", [False, True])
def test_retries_bypass_the_cache(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], batch: bool
) -> None:
    backend = FakeBackend(
        {
            "uno": ["not a number", "1"],
            "dos": [base.IncompleteResponseError("cut off"), "2"],
        }
    )
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path), batch=batch)

    results = _run_with_retries(query_runner, REQUESTS, max_retries=1)

    assert results == {"uno": 1, "dos": 2}
    out = capsys.readouterr().out
    assert "Attempt 1 failed for uno" in out
    assert "Attempt 1 failed for dos: cut off" in out
    assert "Retrying 2 request(s)" in out
    # The good response replaced the bad one in the cache.
    assert _run_with_retries(query_runner, REQUESTS, max_retries=0) == results


def test_retries_give_up(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = FakeBackend({"uno": ["bad"], "dos": ["2"]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    results = _run_with_retries(query_runner, REQUESTS, max_retries=2)

    assert results == {"dos": 2}
    assert backend.queried.count("uno") == 3
    assert "Skipping uno after 3 attempt(s)." in capsys.readouterr().out


def test_retries_respect_skip_cache(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend({"uno": ["bad", "1"]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    results = _run_with_retries(
        query_runner,
        REQUESTS[:1],
        max_retries=1,
        skip_cache=True,
        invalidate_cache=True,
    )

    assert results == {"uno": 1}
    assert not any(query_runner.cacher.cache_dir.glob("*.json"))


#####
# The batch journal


def test_journal_roundtrip(tmp_path: pathlib.Path) -> None:
    journal = base.BatchJournal(tmp_path / "jobs")
    assert journal.pending() == {}

    journal.record("job-a", frozenset({"k1", "k2"}))
    journal.record("job-b", frozenset({"k3"}))
    assert base.BatchJournal(tmp_path / "jobs").pending() == {
        "job-a": frozenset({"k1", "k2"}),
        "job-b": frozenset({"k3"}),
    }

    journal.discard("job-a")
    journal.discard("job-never-recorded")
    assert journal.pending() == {"job-b": frozenset({"k3"})}


def test_request_key_matches_cache_filename(tmp_path: pathlib.Path) -> None:
    request = base.Request("prompt", system_prompt="system")
    cacher = _cacher(tmp_path)

    assert cacher.llm_cache_file_loc(request.prompt, request.system_prompt) == (
        cacher.cache_dir / f"{request.key}.json"
    )
