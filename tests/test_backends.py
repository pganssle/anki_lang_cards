"""Tests for backend selection, the query runner and the batch journal."""

import argparse
import asyncio
import importlib.metadata
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, Self

import aiohttp.test_utils
import aiohttp.web
import attrs
import pytest

from anki_lang_cards import backends, caches, config, model_selection, runner
from anki_lang_cards.backends import base, llm_fallback, tgw

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
    resolved = model_selection.resolve(spec, service)

    assert resolved.spec == spec
    assert resolved.service == expected_service
    assert resolved.model_name == expected_model
    assert resolved.backend_class.service == expected_service


def test_resolve_unknown_service() -> None:
    with pytest.raises(model_selection.UnknownServiceError, match="choose from"):
        model_selection.resolve("gpt-5", "openai")


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
    chosen = model_selection.choose_features(
        [model_selection.FeatureRequest(BATCH, requested, "--batch")],
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

    chosen = model_selection.choose_features(
        [
            model_selection.FeatureRequest(BATCH, True, "batch = true in [models]"),
            model_selection.FeatureRequest(CACHING, None),
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
    with pytest.raises(model_selection.FeatureUnavailableError, match=flag):
        model_selection.choose_features(
            [model_selection.FeatureRequest(feature, True, "a flag")],
            frozenset(),
            backend_description="the test backend",
            confirm=lambda question: False,
        )


def test_confirm_without_terminal_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", _never_asked)

    assert not model_selection.confirm_on_terminal("Continue?")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("YES ", True), ("n", False), ("", False), ("maybe", False)],
)
def test_confirm_on_terminal(
    monkeypatch: pytest.MonkeyPatch, answer: str, expected: bool
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    assert model_selection.confirm_on_terminal("Continue?") is expected


#####
# Building a runner from the command line and config


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    model_selection.add_arguments(parser)
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
    assert model_selection.model_spec(_parse(*argv), conf) == expected


@pytest.mark.parametrize("argv", [(), ("--model-service", "tgw")])
def test_model_spec_requires_a_model(argv: Sequence[str]) -> None:
    """There is no built-in default model, for any backend."""
    with pytest.raises(model_selection.ModelNotConfiguredError, match="default_model"):
        model_selection.model_spec(_parse(*argv), _config(tgw_config=TGW_CONFIG))


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

    resolved, query_runner = model_selection.make_runner(
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
    resolved, query_runner = model_selection.make_runner(
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
        _, query_runner = model_selection.make_runner(
            args, conf, caches.CacheTypes.SENTENCES, confirm=confirm
        )
        assert not query_runner.batch
    else:
        with pytest.raises(model_selection.FeatureUnavailableError, match="--no-batch"):
            model_selection.make_runner(
                args, conf, caches.CacheTypes.SENTENCES, confirm=confirm
            )


def test_make_runner_tgw_without_config() -> None:
    with pytest.raises(model_selection.BackendConfigurationError, match=r"\[tgw\]"):
        model_selection.make_runner(
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


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (base.RequestError("HTTP 400", status=400), "HTTP 400"),
        (base.RequestError("HTTP 429", status=429), "HTTP 429"),
        (base.RequestError("HTTP 500", status=500), "HTTP 500"),
        (base.RequestError("Connection error."), "Connection error."),
        (base.RequestError("ModelError: plugin broke"), "ModelError: plugin broke"),
    ],
)
def test_runner_request_error_fails_only_that_request(
    tmp_path: pathlib.Path, error: Exception, reason: str
) -> None:
    """One request's error leaves the others to complete and be cached."""
    backend = FakeBackend({"uno": [error]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    outcomes = _run(query_runner, REQUESTS)

    assert outcomes["uno"] == runner.Failure(REQUESTS[0], reason)
    assert _texts(outcomes) == {"dos": "answer to dos"}
    assert query_runner.cacher.load_from_cache("uno", system_prompt="sys") is None
    cached = query_runner.cacher.load_from_cache("dos", system_prompt="sys")
    assert cached is not None


@pytest.mark.parametrize(
    "error",
    [
        base.RequestError("bad key", status=401),
        base.RequestError("no access", status=403),
        base.RequestError("no such model", status=404),
        # Not RequestErrors, so bugs rather than failed requests.
        TypeError("bug"),
        KeyError("content"),
        base.ConfigurationError("no API key"),
    ],
)
def test_runner_run_wide_error_aborts(tmp_path: pathlib.Path, error: Exception) -> None:
    """Errors every request would hit stop the run instead of failing each."""
    backend = FakeBackend({"uno": [error]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    with pytest.raises(type(error)):
        _run(query_runner, REQUESTS)


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


def _parse_text(request: base.Request, response: caches.ResponseData) -> str:
    return response["text"]


def _run_with_retries(
    query_runner: runner.QueryRunner,
    requests: Sequence[base.Request],
    parse: Callable[[base.Request, caches.ResponseData], int] = _parse_number,
    **kwargs: object,
) -> Mapping[str, int]:
    return {
        outcome.request.prompt: outcome.value
        for outcome in _run_outcomes(query_runner, requests, parse, **kwargs)
        if isinstance(outcome, runner.Parsed)
    }


def _run_outcomes[T](
    query_runner: runner.QueryRunner,
    requests: Sequence[base.Request],
    parse: Callable[[base.Request, caches.ResponseData], T],
    **kwargs: object,
) -> Sequence[runner.Parsed[T] | runner.Skipped]:
    async def collect() -> Sequence[runner.Parsed[T] | runner.Skipped]:
        async with query_runner:
            return [
                outcome
                async for outcome in runner.run_with_retries(
                    query_runner,
                    requests,
                    parse,
                    describe=lambda request: request.prompt,
                    **kwargs,  # type: ignore[arg-type]
                )
            ]

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


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("bad", "invalid literal for int() with base 10: 'bad'"),
        (base.RequestError("HTTP 500", status=500), "HTTP 500"),
        (base.IncompleteResponseError("cut off"), "cut off"),
    ],
)
def test_retries_give_up(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    reply: str | Exception,
    reason: str,
) -> None:
    """A request that fails its last retry is reported as skipped, with the
    reason its last attempt failed."""
    backend = FakeBackend({"uno": [reply], "dos": ["2"]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    outcomes = _run_outcomes(query_runner, REQUESTS, _parse_number, max_retries=2)

    assert set(outcomes) == {
        runner.Parsed(REQUESTS[1], 2),
        runner.Skipped(REQUESTS[0], reason, attempts=3),
    }
    assert backend.queried.count("uno") == 3
    assert "Skipping uno after 3 attempt(s)." in capsys.readouterr().out


def test_retries_report_the_last_reason(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend(
        {"uno": [base.RequestError("HTTP 500", status=500), "bad"], "dos": ["2"]}
    )
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    outcomes = _run_outcomes(query_runner, REQUESTS, _parse_number, max_retries=1)

    (skipped,) = (o for o in outcomes if isinstance(o, runner.Skipped))
    assert skipped.request == REQUESTS[0]
    assert skipped.reason.startswith("invalid literal")
    assert skipped.attempts == 2


@pytest.mark.parametrize(
    "error",
    [base.RequestError("Connection error."), base.IncompleteResponseError("cut")],
)
def test_every_request_failing_stops_the_run(
    tmp_path: pathlib.Path, error: Exception
) -> None:
    """Nothing getting through in the first round is a problem with the run,
    not with the requests, so there are no retries to pay for."""
    backend = FakeBackend({"uno": [error, "1"], "dos": [error, "2"]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    with pytest.raises(runner.AllRequestsFailedError, match=str(error)) as info:
        _run_outcomes(query_runner, REQUESTS, _parse_number, max_retries=2)

    assert {failure.request for failure in info.value.failures} == set(REQUESTS)
    assert sorted(backend.queried) == ["dos", "uno"]


def test_every_batch_request_failing_stops_the_run(tmp_path: pathlib.Path) -> None:
    backend = FakeBackend(
        batch_failures={"uno": "errored: overloaded", "dos": "errored: overloaded"}
    )
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path), batch=True)

    with pytest.raises(runner.AllRequestsFailedError, match="overloaded"):
        _run_outcomes(query_runner, REQUESTS, _parse_number, max_retries=2)

    assert len(backend.resumed) == 1


def test_every_request_failing_to_parse_is_retried(tmp_path: pathlib.Path) -> None:
    """Unparseable output is the model's doing, and a retry may well fix it."""
    backend = FakeBackend({"uno": ["bad", "1"], "dos": ["bad", "2"]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    assert _run_with_retries(query_runner, REQUESTS, max_retries=1) == {
        "uno": 1,
        "dos": 2,
    }


def test_only_a_retry_round_failing_does_not_stop_the_run(
    tmp_path: pathlib.Path,
) -> None:
    """One persistently bad request is skipped, not escalated, even though
    it is all that is left in the retry rounds."""
    backend = FakeBackend({"uno": [base.RequestError("HTTP 500", status=500)]})
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    outcomes = _run_outcomes(query_runner, REQUESTS, _parse_number, max_retries=1)

    assert runner.Skipped(REQUESTS[0], "HTTP 500", attempts=2) in outcomes


def test_cached_responses_count_as_getting_through(tmp_path: pathlib.Path) -> None:
    """A cached success alongside a failed miss is not a run-wide failure."""
    query_runner = runner.QueryRunner(FakeBackend(), _cacher(tmp_path))
    _run_with_retries(query_runner, REQUESTS[1:], max_retries=0, parse=_parse_text)
    query_runner.backend = FakeBackend({"uno": [base.RequestError("down")]})  # type: ignore[assignment]

    outcomes = _run_outcomes(query_runner, REQUESTS, _parse_text, max_retries=0)

    assert set(outcomes) == {
        runner.Parsed(REQUESTS[1], "answer to dos"),
        runner.Skipped(REQUESTS[0], "down", attempts=1),
    }


def test_retries_request_errors(tmp_path: pathlib.Path) -> None:
    """A request error is retried like incomplete output."""
    backend = FakeBackend(
        {"uno": [base.RequestError("HTTP 500", status=500), "1"], "dos": ["2"]}
    )
    query_runner = runner.QueryRunner(backend, _cacher(tmp_path))

    assert _run_with_retries(query_runner, REQUESTS, max_retries=1) == {
        "uno": 1,
        "dos": 2,
    }


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
# Backends report request errors


class _StatusError(Exception):
    """An SDK error carrying its HTTP status, as the Anthropic SDK's do."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (llm_fallback.llm.ModelError("plugin broke"), None),
        (_StatusError(429), 429),
        (_StatusError(401), 401),
    ],
)
def test_llm_backend_reports_request_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: int | None
) -> None:
    class Model:
        needs_key = None
        supports_schema = False

        async def prompt(self, *args: object, **kwargs: object) -> object:
            raise error

    monkeypatch.setattr(llm_fallback.llm, "get_async_model", lambda name: Model())
    backend = llm_fallback.LLMBackend("fake")

    with pytest.raises(base.RequestError, match=str(error)) as info:
        asyncio.run(backend.query(REQUESTS[0]))

    assert info.value.status == status
    assert info.value.__cause__ is error


def test_llm_backend_passes_other_errors_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error with no HTTP status is not known to be about the request."""

    class Model:
        needs_key = None
        supports_schema = False

        async def prompt(self, *args: object, **kwargs: object) -> object:
            raise KeyError("content")

    monkeypatch.setattr(llm_fallback.llm, "get_async_model", lambda name: Model())

    with pytest.raises(KeyError):
        asyncio.run(llm_fallback.LLMBackend("fake").query(REQUESTS[0]))


def _tgw_app(status: int) -> aiohttp.web.Application:
    async def info(request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.json_response({"model_name": "model"})

    async def completions(request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.Response(status=status, reason="rejected")

    app = aiohttp.web.Application()
    app.router.add_get("/v1/internal/model/info", info)
    app.router.add_post("/v1/completions", completions)
    return app


def test_tgw_backend_reports_http_errors() -> None:
    async def query() -> None:
        async with aiohttp.test_utils.TestServer(_tgw_app(400)) as server:
            backend = tgw.TGWBackend("model", api_base_url=str(server.make_url("/v1/")))
            async with backend:
                await backend.query(REQUESTS[0])

    with pytest.raises(base.RequestError, match="HTTP 400: rejected") as info:
        asyncio.run(query())

    assert info.value.status == 400


def test_tgw_backend_reports_connection_errors() -> None:
    async def query() -> None:
        server = aiohttp.test_utils.TestServer(_tgw_app(200))
        await server.start_server()
        backend = tgw.TGWBackend("model", api_base_url=str(server.make_url("/v1/")))
        async with backend:
            await server.close()
            await backend.query(REQUESTS[0])

    with pytest.raises(base.RequestError, match="ServerDisconnectedError") as info:
        asyncio.run(query())

    assert info.value.status is None


#####
# The batch journal


def test_journal_roundtrip(tmp_path: pathlib.Path) -> None:
    journal = runner.BatchJournal(tmp_path / "jobs")
    assert journal.pending() == {}

    journal.record("job-a", frozenset({"k1", "k2"}))
    journal.record("job-b", frozenset({"k3"}))
    assert runner.BatchJournal(tmp_path / "jobs").pending() == {
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


#####
# The backend registry


@pytest.mark.parametrize(
    ("name", "claims_prefix"), [("anthropic", True), ("tgw", True), ("llm", False)]
)
def test_built_in_backends(name: str, claims_prefix: bool) -> None:
    backend = backends.load(name)

    assert backend.service == name
    assert backend.claims_prefix is claims_prefix
    assert name in backends.names()


def test_unknown_backend() -> None:
    with pytest.raises(backends.UnknownBackendError, match="choose from anthropic"):
        backends.load("openai")


class PluginBackend(FakeBackend):
    """A backend another package might register."""

    service: ClassVar[str] = "plugin"
    claims_prefix: ClassVar[bool] = True

    @classmethod
    def supported_features(cls, model_name: str) -> frozenset[base.Feature]:
        return cls.features

    @classmethod
    def from_config(
        cls, model_name: str, conf: config.Config, features: frozenset[base.Feature]
    ) -> Self:
        backend = cls()
        backend.model_name = model_name
        return backend


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register PluginBackend, and a plugin trying to replace a built-in."""
    entry_points = (
        importlib.metadata.EntryPoint(
            "plugin", f"{__name__}:PluginBackend", backends.ENTRY_POINT_GROUP
        ),
        importlib.metadata.EntryPoint(
            "anthropic", f"{__name__}:PluginBackend", backends.ENTRY_POINT_GROUP
        ),
    )

    def fake_entry_points(*, group: str) -> Sequence[importlib.metadata.EntryPoint]:
        return entry_points if group == backends.ENTRY_POINT_GROUP else ()

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


@pytest.mark.usefixtures("plugin")
def test_plugin_backends_are_found() -> None:
    assert backends.names() == ("anthropic", "tgw", "llm", "plugin")
    assert backends.load("plugin") is PluginBackend
    # A plugin cannot replace a built-in backend.
    assert backends.load("anthropic").__name__ == "AnthropicBackend"

    resolved = model_selection.resolve("plugin/some-model")

    assert resolved.service == "plugin"
    assert resolved.model_name == "some-model"


@pytest.mark.usefixtures("plugin")
def test_make_runner_with_a_plugin_backend() -> None:
    _, query_runner = model_selection.make_runner(
        _parse("--model", "plugin/some-model"),
        _config(),
        caches.CacheTypes.SENTENCES,
        confirm=_never_asked,
    )

    assert isinstance(query_runner.backend, PluginBackend)
    assert query_runner.backend.model_name == "some-model"


def test_anthropic_from_config() -> None:
    conf = _config(
        anthropic=config.AnthropicConfig(
            api_key="key", effort="low", max_tokens=1234, base_url="http://x"
        )
    )

    backend = backends.load("anthropic").from_config(
        "claude-sonnet-5", conf, frozenset()
    )

    assert backend.query_params() == {
        "max_tokens": 1234,
        "effort": "low",
        "prompt_caching": False,
        "structured_output": False,
    }


def test_llm_from_config_passes_the_anthropic_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        needs_key = "anthropic"
        supports_schema = False
        key: str | None = None

    model = Model()
    monkeypatch.setattr(llm_fallback.llm, "get_async_model", lambda name: model)
    conf = _config(anthropic=config.AnthropicConfig(api_key="secret"))

    backends.load("llm").from_config("anthropic/claude-sonnet-5", conf, frozenset())

    assert model.key == "secret"


#####
# Structured output


@pytest.mark.parametrize(
    ("argv", "models", "expected"),
    [
        ((), config.ModelsConfig(), True),
        (("--no-structured-output",), config.ModelsConfig(), False),
        ((), config.ModelsConfig(structured_output=False), False),
    ],
)
def test_make_runner_anthropic_structured_output(
    argv: Sequence[str], models: config.ModelsConfig, expected: bool
) -> None:
    conf = _config(
        models=attrs.evolve(models, default_model="anthropic/claude-sonnet-5"),
        anthropic=config.AnthropicConfig(api_key="key"),
    )

    _, query_runner = model_selection.make_runner(
        _parse(*argv), conf, caches.CacheTypes.SENTENCES, confirm=_never_asked
    )

    assert query_runner.backend.structured_output is expected  # type: ignore[attr-defined]


def test_llm_structured_output_follows_the_model() -> None:
    """Whether the llm fallback can enforce a schema depends on the model."""
    assert base.Feature.STRUCTURED_OUTPUT in (
        llm_fallback.LLMBackend.supported_features("anthropic/claude-sonnet-5")
    )
    backend = llm_fallback.LLMBackend("anthropic/claude-sonnet-5")
    assert backend.structured_output
    assert backend.query_params() == {"structured_output": True}
    assert not llm_fallback.LLMBackend(
        "anthropic/claude-sonnet-5", structured_output=False
    ).structured_output


def test_required_structured_output_unavailable() -> None:
    pytest.importorskip("llm_fake_cards")
    questions: list[str] = []

    def accept(question: str) -> bool:
        questions.append(question)
        return True

    _, query_runner = model_selection.make_runner(
        _parse("--model", "fake-cards", "--structured-output"),
        _config(),
        caches.CacheTypes.SENTENCES,
        confirm=accept,
    )

    (question,) = questions
    assert "Structured output was requested (--structured-output)" in question
    assert query_runner.backend.features == frozenset()
    assert not query_runner.backend.structured_output  # type: ignore[attr-defined]
