"""Tests of the requests AnkiConnectClient sends, against a fake HTTP session."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import attrs
import pytest

from anki_lang_cards import anki_import


@attrs.frozen
class _Request:
    action: str
    params: Mapping[str, Any]


class _FakeResponse:
    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    async def json(self) -> Mapping[str, Any]:
        return self._payload


class _FakeSession:
    """Records every posted action and answers it from a scripted mapping.

    ``responses`` maps an Anki Connect action name to the ``result`` the
    server should answer with; an action that is not in the mapping gets an
    error response, as the real server does for an unknown action.
    """

    def __init__(self, responses: Mapping[str, Any]):
        self._responses = responses
        self.requests: list[_Request] = []

    @contextlib.asynccontextmanager
    async def post(
        self, endpoint: str, json: Mapping[str, Any], timeout: float
    ) -> AsyncIterator[_FakeResponse]:
        action = json["action"]
        self.requests.append(_Request(action, json["params"]))
        if action in self._responses:
            yield _FakeResponse({"result": self._responses[action], "error": None})
        else:
            yield _FakeResponse(
                {"result": None, "error": f"unsupported action: {action}"}
            )


def _client(session: _FakeSession) -> anki_import.AnkiConnectClient:
    # The client only ever calls ``session.post``, which the fake provides,
    # so the aiohttp.ClientSession annotation is stricter than it needs.
    return anki_import.AnkiConnectClient(session)  # type: ignore[arg-type]


def _card_ids(*ids: int) -> Sequence[anki_import.CardId]:
    return tuple(anki_import.CardId(card_id) for card_id in ids)


@pytest.mark.parametrize(
    "card_ids",
    [
        pytest.param(_card_ids(5000, 5001), id="tuple"),
        pytest.param(list(_card_ids(5000, 5001)), id="list"),
        pytest.param(frozenset(_card_ids(5000)), id="frozenset"),
    ],
)
def test_unsuspend_cards_sends_flat_card_list(
    card_ids: Sequence[anki_import.CardId],
) -> None:
    session = _FakeSession({"unsuspend": None})
    client = _client(session)

    assert asyncio.run(client.unsuspend_cards(card_ids)) is None
    assert session.requests == [_Request("unsuspend", {"cards": list(card_ids)})]
    # Anki Connect expects a flat list of integer card ids
    (request,) = session.requests
    assert isinstance(request.params["cards"], list)


def test_unsuspend_cards_raises_on_error() -> None:
    session = _FakeSession({})
    client = _client(session)

    with pytest.raises(ValueError, match="unsupported action: unsuspend"):
        asyncio.run(client.unsuspend_cards(_card_ids(5000)))


@pytest.mark.parametrize(
    "card_ids",
    [
        pytest.param(_card_ids(5000, 5001, 5002), id="tuple"),
        pytest.param(frozenset(_card_ids(5000, 5001, 5002)), id="frozenset"),
    ],
)
def test_are_suspended_sends_flat_card_list(
    card_ids: Sequence[anki_import.CardId],
) -> None:
    sent = list(card_ids)
    suspended = {card_id: card_id != 5001 for card_id in sent}
    session = _FakeSession({"areSuspended": [suspended[c] for c in sent]})
    client = _client(session)

    assert asyncio.run(client.are_suspended(card_ids)) == suspended
    assert session.requests == [_Request("areSuspended", {"cards": sent})]
    (request,) = session.requests
    assert isinstance(request.params["cards"], list)


def test_are_suspended_empty() -> None:
    session = _FakeSession({"areSuspended": []})
    client = _client(session)

    assert asyncio.run(client.are_suspended(())) == {}
    assert session.requests == [_Request("areSuspended", {"cards": []})]
