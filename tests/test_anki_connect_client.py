"""Tests of the requests AnkiConnectClient sends, against a fake HTTP session."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from anki_lang_cards import anki_import


class _FakeResponse:
    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    async def json(self) -> Mapping[str, Any]:
        return self._payload


class _FakeSession:
    """Records posted payloads and answers like Anki Connect would."""

    def __init__(self, known_cards: Sequence[int]):
        self.known_cards = frozenset(known_cards)
        self.requests: list[Mapping[str, Any]] = []

    @contextlib.asynccontextmanager
    async def post(
        self, endpoint: str, json: Mapping[str, Any], timeout: float
    ) -> AsyncIterator[_FakeResponse]:
        self.requests.append(json)
        assert json["action"] == "unsuspend"
        cards = json["params"]["cards"]
        # Anki Connect expects a flat list of integer card ids
        assert isinstance(cards, list), cards
        assert all(isinstance(card, int) for card in cards), cards
        assert set(cards) <= self.known_cards, cards
        yield _FakeResponse({"result": None, "error": None})


def test_unsuspend_cards_sends_flat_card_list() -> None:
    session = _FakeSession(known_cards=[5000, 5001])
    client = anki_import.AnkiConnectClient(session)  # type: ignore[arg-type]
    card_ids = (anki_import.CardId(5000), anki_import.CardId(5001))

    assert asyncio.run(client.unsuspend_cards(card_ids)) is None
    assert [request["params"] for request in session.requests] == [
        {"cards": [5000, 5001]}
    ]
