"""Tests for the Text Generation WebUI backend's retries and response handling.

`aiohttp.ClientSession` is replaced with a fake that serves scripted replies,
so no server is needed.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Final

import aiohttp
import multidict
import pytest
import yarl

from anki_lang_cards import _types as types
from anki_lang_cards.backends import base, tgw

MODEL: Final = "llama.gguf"
REQUEST: Final = base.Request("prompt", system_prompt="sys ")
ENDPOINT: Final = yarl.URL("http://tgw.invalid/v1/completions")


def _completion(finish_reason: str, text: str = "answer") -> Mapping[str, Any]:
    return {
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }


class FakeResponse:
    def __init__(self, reply: int | Mapping[str, Any]):
        self.reply = reply

    def raise_for_status(self) -> None:
        if isinstance(self.reply, int):
            raise aiohttp.ClientResponseError(
                request_info=aiohttp.RequestInfo(
                    ENDPOINT,
                    "POST",
                    multidict.CIMultiDictProxy(multidict.CIMultiDict()),
                ),
                history=(),
                status=self.reply,
            )

    async def json(self) -> Mapping[str, Any]:
        assert not isinstance(self.reply, int)
        return self.reply


class FakeSession:
    """Serves the model info, then the scripted completion replies in order.

    A reply is a status code to fail with or a JSON body.
    """

    def __init__(self, completions: Sequence[int | Mapping[str, Any]]):
        self.completions = list(completions)
        self.posts = 0

    @contextlib.asynccontextmanager
    async def get(self, url: str) -> AsyncIterator[FakeResponse]:
        yield FakeResponse({"model_name": MODEL})

    @contextlib.asynccontextmanager
    async def post(self, url: str, json: object) -> AsyncIterator[FakeResponse]:
        self.posts += 1
        yield FakeResponse(self.completions.pop(0))

    async def __aexit__(self, *args: object) -> None:
        pass


@pytest.fixture(autouse=True)
def no_backoff_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    real_sleep = asyncio.sleep

    async def no_sleep(delay: float, *args: object) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)


def _query(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> Mapping[str, Any]:
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: session)

    async def run() -> Mapping[str, Any]:
        backend = tgw.TGWBackend(
            MODEL, api_base_url=types.Url("http://tgw.invalid/v1/")
        )
        async with backend:
            return await backend.query(REQUEST)

    return asyncio.run(run())


def test_query_returns_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession([_completion("stop")])
    response = _query(monkeypatch, session)
    assert response["text"] == "answer"
    assert response["usage"]["input"] == 3
    assert response["usage"]["output"] == 4


def test_truncated_response_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession([_completion("length")])
    with pytest.raises(base.IncompleteResponseError, match="cut off"):
        _query(monkeypatch, session)


def test_missing_text_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession([{"choices": [{"finish_reason": "error"}]}])
    with pytest.raises(base.IncompleteResponseError, match="No text"):
        _query(monkeypatch, session)


@pytest.mark.parametrize("status", [500, 503, 429, 408])
def test_transient_errors_are_retried(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    session = FakeSession([status, status, _completion("stop")])
    assert _query(monkeypatch, session)["text"] == "answer"
    assert session.posts == 3


def test_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession([500] * (tgw._MAX_TRIES + 1))
    with pytest.raises(aiohttp.ClientResponseError):
        _query(monkeypatch, session)
    assert session.posts == tgw._MAX_TRIES


@pytest.mark.parametrize("status", [400, 401, 404, 422, 499])
def test_client_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    session = FakeSession([status, _completion("stop")])
    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        _query(monkeypatch, session)
    assert exc_info.value.status == status
    assert session.posts == 1
