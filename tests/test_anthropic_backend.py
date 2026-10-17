"""Edge cases of the Anthropic backend, against a scripted fake of the API.

The happy paths run against recordings of the real API in
test_recorded_backends.py; this covers what a recording can't produce on
demand: truncated and refused responses, errored batch items, resumed and
expired batch jobs.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import anthropic
import attrs
import httpx2
import pytest

from anki_lang_cards.backends import anthropic_sdk, base

JOB_ID: Final = "msgbatch_new"
REQUEST: Final = base.Request("the prompt", system_prompt="the system prompt")


def _message(
    text: str = "hola",
    *,
    stop_reason: str = "end_turn",
    thinking: bool = False,
    cache_creation: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    content: list[Mapping[str, Any]] = [{"type": "text", "text": text}]
    if thinking:
        content.insert(0, {"type": "thinking", "thinking": "", "signature": "sig"})
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation": cache_creation,
        },
    }


def _batch(job_id: str, status: str) -> Mapping[str, Any]:
    return {
        "id": job_id,
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": int(status != "ended"),
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "results_url": (
            f"https://api.anthropic.com/v1/messages/batches/{job_id}/results"
            if status == "ended"
            else None
        ),
        "created_at": "2026-10-10T00:00:00Z",
        "expires_at": "2026-10-11T00:00:00Z",
        "ended_at": None,
        "archived_at": None,
        "cancel_initiated_at": None,
    }


@attrs.define
class FakeAPI:
    """Just enough of the Messages and Message Batches endpoints."""

    message: Mapping[str, Any] = attrs.field(factory=_message)
    batch_results: dict[str, Sequence[Mapping[str, Any]]] = attrs.field(factory=dict)
    """JSONL result lines for each job ID that exists."""

    polls_before_done: int = 1
    sent: list[tuple[str, str, Any]] = attrs.field(factory=list)
    _polls: dict[str, int] = attrs.field(factory=dict)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.sent.append((request.method, path, body))

        if path == "/v1/messages":
            return httpx2.Response(200, json=self.message)
        if path == "/v1/messages/batches":
            self.batch_results[JOB_ID] = [
                {
                    "custom_id": item["custom_id"],
                    "result": {"type": "succeeded", "message": self.message},
                }
                for item in body["requests"]
            ]
            return httpx2.Response(200, json=_batch(JOB_ID, "in_progress"))

        job_id = path.removeprefix("/v1/messages/batches/").removesuffix("/results")
        if job_id not in self.batch_results:
            return httpx2.Response(
                404,
                json={
                    "type": "error",
                    "error": {"type": "not_found_error", "message": "gone"},
                },
            )
        if path.endswith("/results"):
            lines = "\n".join(json.dumps(line) for line in self.batch_results[job_id])
            return httpx2.Response(200, content=lines.encode())
        polls = self._polls[job_id] = self._polls.get(job_id, 0) + 1
        status = "ended" if polls > self.polls_before_done else "in_progress"
        return httpx2.Response(200, json=_batch(job_id, status))

    def bodies(self, method: str, path: str) -> Sequence[Any]:
        return [body for m, p, body in self.sent if (m, p) == (method, path)]


def _backend(api: FakeAPI, **kwargs: Any) -> anthropic_sdk.AnthropicBackend:
    return anthropic_sdk.AnthropicBackend(
        "claude-test",
        api_key="test-key",
        poll_interval=0.0,
        max_poll_interval=0.0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(api.handle)
        ),
        **kwargs,
    )


def _query(backend: anthropic_sdk.AnthropicBackend, request: base.Request) -> Any:
    async def run() -> Any:
        async with backend:
            return await backend.query(request)

    return asyncio.run(run())


def _query_batch(
    backend: anthropic_sdk.AnthropicBackend,
    requests: Sequence[base.Request],
    resume: Mapping[str, frozenset[str]] | None = None,
) -> tuple[base.BatchResults, Mapping[str, frozenset[str]]]:
    submitted: dict[str, frozenset[str]] = {}

    def on_submit(job_id: str, keys: frozenset[str]) -> None:
        submitted[job_id] = keys

    async def run() -> base.BatchResults:
        async with backend:
            return await backend.query_batch(
                requests, resume=resume or {}, on_submit=on_submit
            )

    return asyncio.run(run()), submitted


#####
# One request at a time


def test_query_sends_cached_system_prompt() -> None:
    api = FakeAPI()

    response = _query(_backend(api), REQUEST)

    (body,) = api.bodies("POST", "/v1/messages")
    assert body["model"] == "claude-test"
    assert body["max_tokens"] == anthropic_sdk.DEFAULT_MAX_TOKENS
    assert body["messages"] == [{"role": "user", "content": "the prompt"}]
    assert body["system"] == [
        {
            "type": "text",
            "text": "the system prompt",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert "output_config" not in body
    assert response["text"] == "hola"
    assert response["usage"]["input"] == 10
    assert response["usage"]["output"] == 5
    assert response["usage"]["details"]["cache_creation_input_tokens"] == 100
    assert response["usage"]["details"]["batch"] is False


@pytest.mark.parametrize(
    ("cache_creation", "expected"),
    [
        (
            {"ephemeral_5m_input_tokens": 30, "ephemeral_1h_input_tokens": 70},
            {
                "cache_creation_5m_input_tokens": 30,
                "cache_creation_1h_input_tokens": 70,
            },
        ),
        (None, {}),
    ],
)
def test_query_records_cache_write_lifetimes(
    cache_creation: Mapping[str, int] | None, expected: Mapping[str, int]
) -> None:
    """Writes at the hour-long lifetime cost more, so the split is kept when
    the API reports it."""
    api = FakeAPI(message=_message(cache_creation=cache_creation))

    details = _query(_backend(api), REQUEST)["usage"]["details"]

    lifetimes = {
        key: value for key, value in details.items() if "_5m_" in key or "_1h_" in key
    }
    assert lifetimes == expected


def test_query_options() -> None:
    api = FakeAPI()

    _query(_backend(api, prompt_caching=False, effort="low", max_tokens=123), REQUEST)

    (body,) = api.bodies("POST", "/v1/messages")
    assert body["system"] == [{"type": "text", "text": "the system prompt"}]
    assert body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] == 123


def test_query_without_system_prompt() -> None:
    api = FakeAPI()

    _query(_backend(api), base.Request("just a prompt"))

    (body,) = api.bodies("POST", "/v1/messages")
    assert "system" not in body


def test_query_skips_thinking_blocks() -> None:
    api = FakeAPI(message=_message("[1, 2]", thinking=True))

    assert _query(_backend(api), REQUEST)["text"] == "[1, 2]"


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_query_incomplete_response(stop_reason: str) -> None:
    api = FakeAPI(message=_message("[{", stop_reason=stop_reason))

    with pytest.raises(base.IncompleteResponseError, match=stop_reason):
        _query(_backend(api), REQUEST)


@pytest.mark.parametrize("status", [400, 404])
def test_query_reports_http_errors(status: int) -> None:
    def reject(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, json={"type": "error", "error": {"message": "no"}}
        )

    backend = anthropic_sdk.AnthropicBackend(
        "claude-test",
        api_key="test-key",
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(reject)
        ),
    )

    with pytest.raises(base.RequestError, match=f"{status}") as info:
        _query(backend, REQUEST)

    assert info.value.status == status
    assert isinstance(info.value.__cause__, anthropic.APIStatusError)


def test_query_reports_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    backend = anthropic_sdk.AnthropicBackend(
        "claude-test",
        api_key="test-key",
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(refuse)
        ),
    )
    # The SDK would otherwise retry with a backoff of seconds.
    monkeypatch.setattr(backend._client, "max_retries", 0)

    with pytest.raises(base.RequestError, match="APIConnectionError") as info:
        _query(backend, REQUEST)

    assert info.value.status is None


def test_query_params() -> None:
    backend = _backend(FakeAPI(), effort="medium", prompt_caching=False)

    assert backend.query_params() == {
        "max_tokens": anthropic_sdk.DEFAULT_MAX_TOKENS,
        "effort": "medium",
        "prompt_caching": False,
        "structured_output": True,
    }


SCHEMA: Final = {
    "type": "object",
    "properties": {"words": {"type": "array", "items": {"type": "string"}}},
    "required": ["words"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("kwargs", "expected_output_config"),
    [
        ({}, {"format": {"type": "json_schema", "schema": SCHEMA}}),
        (
            {"effort": "low"},
            {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        ),
        ({"structured_output": False}, None),
    ],
)
def test_query_structured_output(
    kwargs: Mapping[str, Any], expected_output_config: Mapping[str, Any] | None
) -> None:
    api = FakeAPI()

    _query(
        _backend(api, **kwargs),
        base.Request("the prompt", system_prompt="system", json_schema=SCHEMA),
    )

    (body,) = api.bodies("POST", "/v1/messages")
    assert body.get("output_config") == expected_output_config


def test_batch_structured_output() -> None:
    api = FakeAPI()

    _query_batch(_backend(api), [base.Request("uno", json_schema=SCHEMA)])

    (body,) = api.bodies("POST", "/v1/messages/batches")
    (item,) = body["requests"]
    assert item["params"]["output_config"] == {
        "format": {"type": "json_schema", "schema": SCHEMA}
    }


#####
# Batches

OTHER_REQUEST: Final = base.Request("another prompt", system_prompt="the system prompt")


def test_query_batch_submits_and_collects(capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeAPI(polls_before_done=2)
    requests = [base.Request("uno"), base.Request("dos", system_prompt="solo")]

    results, submitted = _query_batch(_backend(api), requests)

    (body,) = api.bodies("POST", "/v1/messages/batches")
    custom_ids = {item["custom_id"] for item in body["requests"]}
    assert custom_ids == {request.key for request in requests}
    assert submitted == {JOB_ID: frozenset(custom_ids)}
    assert set(results.responses) == custom_ids
    assert all(
        response["usage"]["details"]["batch"] is True
        for response in results.responses.values()
    )
    assert results.failures == {}
    assert results.completed_jobs == {JOB_ID}
    assert len(api.bodies("GET", f"/v1/messages/batches/{JOB_ID}")) >= 3
    assert f"Batch job {JOB_ID}: 1 processing" in capsys.readouterr().out
    # No system prompt is shared, so there is nothing to warm up.
    assert api.bodies("POST", "/v1/messages") == []


SHARED_REQUESTS: Final = (
    base.Request("a", system_prompt="shared"),
    base.Request("b", system_prompt="shared"),
    base.Request("c", system_prompt="shared"),
    base.Request("d", system_prompt="solo"),
    base.Request("e"),
)


def test_query_batch_warms_the_prompt_cache() -> None:
    """One request per shared system prompt goes ahead of the batch, writing
    the hour-long cache entry the batched requests will read."""
    api = FakeAPI()

    results, submitted = _query_batch(_backend(api), SHARED_REQUESTS)

    (warm_up,) = api.bodies("POST", "/v1/messages")
    warm_key = min(request.key for request in SHARED_REQUESTS[:3])
    (warm_request,) = [r for r in SHARED_REQUESTS if r.key == warm_key]
    assert warm_up["messages"][0]["content"] == warm_request.prompt
    assert warm_up["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    methods_and_paths = [(method, path) for method, path, _ in api.sent]
    assert methods_and_paths.index(("POST", "/v1/messages")) < methods_and_paths.index(
        ("POST", "/v1/messages/batches")
    )

    (body,) = api.bodies("POST", "/v1/messages/batches")
    batched = {item["custom_id"] for item in body["requests"]}
    assert batched == {request.key for request in SHARED_REQUESTS} - {warm_key}
    for item in body["requests"]:
        if "system" in item["params"]:
            assert item["params"]["system"][0]["cache_control"]["ttl"] == "1h"
    assert submitted == {JOB_ID: frozenset(batched)}
    assert set(results.responses) == {request.key for request in SHARED_REQUESTS}
    assert results.responses[warm_key]["usage"]["details"]["batch"] is False


def test_query_batch_without_prompt_caching() -> None:
    api = FakeAPI()

    _query_batch(_backend(api, prompt_caching=False), SHARED_REQUESTS)

    assert api.bodies("POST", "/v1/messages") == []
    (body,) = api.bodies("POST", "/v1/messages/batches")
    assert len(body["requests"]) == len(SHARED_REQUESTS)
    assert all(
        "cache_control" not in item["params"].get("system", [{}])[0]
        for item in body["requests"]
    )


def test_query_batch_incomplete_warm_up() -> None:
    """A cut-off warm-up response is a failure to retry; the batch still runs."""
    api = FakeAPI(message=_message("[{", stop_reason="max_tokens"))

    results, submitted = _query_batch(_backend(api), SHARED_REQUESTS[:2])

    warm_key = min(request.key for request in SHARED_REQUESTS[:2])
    assert "max_tokens" in results.failures[warm_key]
    assert submitted == {
        JOB_ID: frozenset({r.key for r in SHARED_REQUESTS[:2]} - {warm_key})
    }


def test_query_batch_all_resumed_needs_no_warm_up() -> None:
    api = FakeAPI()
    keys = frozenset(request.key for request in SHARED_REQUESTS[:2])
    api.batch_results["msgbatch_old"] = [
        {"custom_id": key, "result": {"type": "succeeded", "message": _message()}}
        for key in keys
    ]

    results, submitted = _query_batch(
        _backend(api), SHARED_REQUESTS[:2], resume={"msgbatch_old": keys}
    )

    assert api.bodies("POST", "/v1/messages") == []
    assert submitted == {}
    assert set(results.responses) == keys


def test_query_batch_failures() -> None:
    api = FakeAPI()
    api.batch_results["msgbatch_old"] = [
        {
            "custom_id": REQUEST.key,
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                },
            },
        },
        {
            "custom_id": OTHER_REQUEST.key,
            "result": {
                "type": "succeeded",
                "message": _message("[{", stop_reason="max_tokens"),
            },
        },
        {"custom_id": "not-requested", "result": {"type": "expired"}},
    ]
    third = base.Request("third")
    api.batch_results["msgbatch_older"] = [
        {"custom_id": third.key, "result": {"type": "canceled"}}
    ]

    results, submitted = _query_batch(
        _backend(api),
        [REQUEST, OTHER_REQUEST, third],
        resume={
            "msgbatch_old": frozenset(
                {REQUEST.key, OTHER_REQUEST.key, "not-requested"}
            ),
            "msgbatch_older": frozenset({third.key}),
        },
    )

    assert submitted == {}
    assert results.responses == {}
    assert results.failures[REQUEST.key] == "errored: Overloaded"
    assert "max_tokens" in results.failures[OTHER_REQUEST.key]
    assert results.failures[third.key] == "canceled"
    assert "not-requested" not in results.failures
    assert results.completed_jobs == {"msgbatch_old", "msgbatch_older"}


def test_query_batch_resumes_only_relevant_jobs() -> None:
    api = FakeAPI()
    api.batch_results["msgbatch_old"] = [
        {
            "custom_id": REQUEST.key,
            "result": {"type": "succeeded", "message": _message()},
        }
    ]

    results, submitted = _query_batch(
        _backend(api),
        [REQUEST, OTHER_REQUEST],
        resume={
            "msgbatch_old": frozenset({REQUEST.key}),
            "msgbatch_unrelated": frozenset({"someone-else"}),
        },
    )

    # Only the request the old job doesn't hold is submitted again.
    (body,) = api.bodies("POST", "/v1/messages/batches")
    assert [item["custom_id"] for item in body["requests"]] == [OTHER_REQUEST.key]
    assert submitted == {JOB_ID: frozenset({OTHER_REQUEST.key})}
    assert set(results.responses) == {REQUEST.key, OTHER_REQUEST.key}
    assert results.completed_jobs == {"msgbatch_old", JOB_ID}


def test_query_batch_expired_job(capsys: pytest.CaptureFixture[str]) -> None:
    """A resumed job the API no longer has fails its requests, so a retry
    submits them afresh, and leaves the journal."""
    api = FakeAPI()

    results, submitted = _query_batch(
        _backend(api), [REQUEST], resume={"msgbatch_gone": frozenset({REQUEST.key})}
    )

    assert submitted == {}
    assert REQUEST.key in results.failures
    assert results.completed_jobs == {"msgbatch_gone"}
    assert "msgbatch_gone no longer exists" in capsys.readouterr().out
