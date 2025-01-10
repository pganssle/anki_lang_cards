import asyncio
import contextlib
import json
from collections.abc import (AsyncIterator, Iterable, Iterator, Mapping,
                             Sequence)
from typing import Any, Final, NewType, TypeAlias

import aiohttp

CardId: TypeAlias = NewType("CardId", int)
AnkiCardInfo: TypeAlias = Mapping[str, Any]
URL: Final[str] = "http://localhost:8765"
AC_VERSION: Final[int] = 6


class AnkiConnectClient:
    def __init__(self, session: aiohttp.ClientSession, endpoint: str = URL):
        self._session = session
        self.endpoint = endpoint

    async def fetch_card(self, note_id: CardId) -> Mapping:
        data = {
            "action": "notesInfo",
            "version": AC_VERSION,
            "params": {"notes": [note_id]},
        }
        async with self._session.post(self.endpoint, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result["result"][0]

    async def get_card_ids(
        self, deck_name: str, query: str | None = None
    ) -> Sequence[CardId]:
        """
        Retrieve a list of card ids corresponding to existing cards.
        Args:
            deck_name (str): The name of the Anki deck to retrieve cards from.
            query (str): Any addtional query to send along with the deck filter
        Returns:
            Sequence of integers representng card ids.
        """
        full_query = f'deck:"{deck_name}"'
        if query:
            full_query += " " + query

        data = {
            "action": "findNotes",
            "version": AC_VERSION,
            "params": {"query": full_query},
        }

        async with self._session.post(URL, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return tuple(map(CardId, result["result"]))

    async def get_anki_cards(
        self, card_list: Iterable[CardId]
    ) -> AsyncIterator[AnkiCardInfo]:
        """
        Retrieve a list of existing cards in the specified Anki deck.

        Args:
            deck_name (str): The name of the Anki deck to retrieve cards from.

        Yields:
            list: A list of dicts, where each dict represents an Anki card with its fields.
        """

        pending = {
            asyncio.create_task(self.fetch_card(card_id), name=str(card_id))
            for card_id in card_list
        }

        errors: dict[CardId, Exception] = {}

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                card_id = CardId(int(task.get_name()))
                try:
                    result = await task
                    yield result
                except Exception as e:
                    errors[card_id] = e

        if errors:
            raise ValueError(f"Failed to fetch cards: {errors}")

    async def update_card_info(
        self, new_card_info: AnkiCardInfo | Sequence[AnkiCardInfo]
    ) -> None:
        if not hasattr(new_card_info, "keys"):
            tasks = [
                asyncio.create_task(self.update_card_info(card_info))
                for card_info in new_card_info
            ]
            await asyncio.gather(*tasks)
            return

        data = {
            "action": "updateNoteFields",
            "version": AC_VERSION,
            "params": {"note": new_card_info},
        }

        async with self._session.post(self.endpoint, json=data) as response:
            response.raise_for_status()


@contextlib.asynccontextmanager
async def create_client(endpoint: str = URL, **kwargs) -> Iterator[AnkiConnectClient]:
    if "connector" not in kwargs:
        kwargs["connector"] = aiohttp.TCPConnector(limit_per_host=5)
    async with aiohttp.ClientSession(**kwargs) as session:
        yield AnkiConnectClient(session, endpoint=endpoint)


async def main():
    async with create_client() as client:
        cards_with_sound = await client.get_card_ids(
            "Spanish::Vocab by Frequency", query="Article:[sound*"
        )

        print(f"Updating {len(cards_with_sound)} cards")
        updated_notes = []
        for card_id in cards_with_sound:
            new_note_info = {
                "id": card_id,
                "fields": {
                    "Article": "",
                    "Spanish Audio": "",
                },
            }

            updated_notes.append(new_note_info)

        await client.update_card_info(updated_notes)
        print("Done")


if __name__ == "__main__":
    asyncio.run(main())
