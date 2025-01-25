import asyncio
import contextlib
import csv
import json
import math
import pathlib
import warnings
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from typing import (Any, Final, Generic, Mapping, NewType, TypeAlias,
                    TypedDict, TypeVar)

import aiohttp

from . import _types as types

NoteId = NewType("NoteId", int)
DeckId = NewType("DeckId", int)
JsonStr = NewType("JsonStr", str)
URL: Final[str] = "http://localhost:8765"
AC_VERSION: Final[int] = 6

T = TypeVar("T", bound=str)


class Field(TypedDict, Generic[T]):
    value: T
    order: int


StrField: TypeAlias = Field[str]
JsonStrField: TypeAlias = Field[JsonStr]


class NoteFields(TypedDict, total=False):
    Rank: StrField
    Word: StrField
    Part_of_Speech: StrField
    Definition: StrField
    Spanish: StrField
    English: StrField
    Freq: StrField
    Image: StrField
    Article: StrField
    Sentence_Source: JsonStrField
    Note: StrField
    Extra_Sentences: JsonStrField
    model_used: StrField


class NoteData(TypedDict, total=False):
    noteId: NoteId
    profile: str
    tags: Sequence[str]
    fields: NoteFields
    modelName: str
    mod: int
    cards: Sequence[int]


class DeckStats(TypedDict):
    deck_id: DeckId
    name: str
    new_count: int
    learn_count: int
    review_count: int
    total_in_deck: int


class AnkiConnectClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        endpoint: str = URL,
        timeout=300,
        retries=3,
    ):
        self._session = session
        self._retries = retries
        self.endpoint = endpoint
        self._timeout = timeout

    async def _post(self, data: Mapping) -> Mapping:
        assert self._retries >= 0
        for i in range(self._retries + 1):
            try:
                async with self._session.post(
                    self.endpoint, json=data, timeout=self._timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result
            except aiohttp.ClientResponseError:
                raise
            except aiohttp.ClientConnectionError as e:
                last_error = e
                continue

        raise last_error

    async def get_deck_stats(self, deck_name: str) -> DeckStats:
        data = {
            "action": "getDeckStats",
            "version": AC_VERSION,
            "params": {"decks": [deck_name]},
        }
        result = await self._post(data)
        return next(iter(result["result"].values()))

    async def fetch_note(self, note_id: NoteId) -> NoteData:
        data = {
            "action": "notesInfo",
            "version": AC_VERSION,
            "params": {"notes": [note_id]},
        }
        result = await self._post(data=data)
        return result["result"][0]

    async def get_note_ids(
        self, deck_name: str, query: str | None = None
    ) -> Sequence[NoteId]:
        """
        Retrieve a list of note ids corresponding to existing notes.
        Args:
            deck_name (str): The name of the Anki deck to retrieve notes from.
            query (str): Any addtional query to send along with the deck filter
        Returns:
            Sequence of integers representng note ids.
        """
        full_query = f'deck:"{deck_name}"'
        if query:
            full_query += " " + query

        data = {
            "action": "findNotes",
            "version": AC_VERSION,
            "params": {"query": full_query},
        }

        result = await self._post(data=data)
        return tuple(map(NoteId, result["result"]))

    async def get_anki_notes(
        self, note_list: Iterable[NoteId]
    ) -> AsyncIterator[NoteData]:
        """
        Retrieve a list of existing notes in the specified Anki deck.

        Args:
            deck_name (str): The name of the Anki deck to retrieve notes from.

        Yields:
            list: A list of dicts, where each dict represents an Anki note with its fields.
        """

        pending = {
            asyncio.create_task(self.fetch_note(note_id), name=str(note_id))
            for note_id in note_list
        }

        errors: dict[NoteId, Exception] = {}
        retries: dict[NoteId, int] = {}

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                note_id = NoteId(int(task.get_name()))
                try:
                    result = await task
                    yield result
                except Exception as e:
                    if retries.setdefault(note_id, 0) < 3:
                        pending.add(
                            asyncio.create_task(
                                self.fetch_note(note_id), name=str(note_id)
                            )
                        )
                        retries[note_id] += 1
                    else:
                        errors[note_id] = e

        if errors:
            raise ValueError(f"Failed to fetch notes: {errors}")

    async def update_note_info(
        self, new_note_info: NoteData | Sequence[NoteData]
    ) -> None:
        if not hasattr(new_note_info, "keys"):
            tasks = [
                asyncio.create_task(self.update_note_info(note_info))
                for note_info in new_note_info
            ]
            await asyncio.gather(*tasks)
            return

        data = {
            "action": "updateNoteFields",
            "version": AC_VERSION,
            "params": {"note": new_note_info},
        }

        result = self._post(data)
        if result["error"] != None:
            raise ValueError(result["error"])
        return result

    async def retrieve_matching_words(
        self,
        deck_name: str,
        words: Sequence[types.Word],
    ) -> Mapping[NoteId, types.Word]:
        def make_query(word: types.Word):
            word_query = f'"Word:{word.word}"'
            if word.rank:
                rank_query = f'"Rank:{word.rank}"'
                return f"({word_query} or {rank_query})"
            else:
                return f"({word_query})"

        query = " or ".join(make_query(word) for word in words)
        if not query:
            return {}
        ids = await self.get_note_ids(deck_name, query=query)

        output = {}
        async for note_data in self.get_anki_notes(ids):
            output[note_data["noteId"]] = self.word_from_anki_connect_response(
                note_data
            )

        return output

    def word_from_anki_connect_response(self, resp: NoteData) -> types.Word:
        """Creates a `Word` object from an AnkiConnect response representing a Note."""
        fields: NoteFields = resp.get("fields", {})

        def get_field_value(field_name: str) -> str | None:
            """Helper function to safely extract the value of a field."""
            field: Field = fields.get(field_name, None)
            return field["value"] if field else None

        def fix_json_field(json_str: JsonStr) -> JsonStr:
            json_str = json_str.replace('""', '"')
            json_str = json_str.replace("&quot;", '"')
            json_str = json_str.replace("&nbsp;", " ")
            json_str = json_str.replace("<br>", "\n")
            return json_str

        def parse_json_field(field_name: str) -> Sequence[Any] | None:
            """Helper function to parse JSON fields into a Python object."""
            field_value = get_field_value(field_name)
            if field_value:
                try:
                    return json.loads(fix_json_field(field_value))
                except json.JSONDecodeError:
                    return None
            return None

        sentence_source = get_field_value("Sentence Source")
        if sentence_source:
            sentence_citation: types.Citation | None = types.Citation.from_json(
                fix_json_field(sentence_source)
            )
        else:
            sentence_citation = None

        def parse_sentences() -> Sequence[dict[str, str | types.Citation]]:
            data = parse_json_field("Extra Sentences") or ()
            sentences: list[dict[str, str | types.Citation]] = []
            for sentence in data:
                source_dict = sentence.get("source", None)
                if not source_dict:
                    sentences.append(sentence)
                else:
                    source = types.Citation.from_dict(sentence["source"])
                    new_sentence = {**sentence, "source": source}
                    sentences.append(new_sentence)
            return sentences

        rank_str = get_field_value("Rank")

        return types.Word(
            rank=int(rank_str) if rank_str else None,
            word=get_field_value("Word") or "",
            meaning=get_field_value("Definition") or "",
            part_of_speech=get_field_value("Part_of_Speech") or "",
            article=get_field_value("Article"),
            spanish_sentence=get_field_value("Spanish"),
            english_sentence=get_field_value("English"),
            example_sentences=parse_sentences(),
            model_used=get_field_value("model_used"),
            sentence_source=sentence_citation,
        )

    async def find_highest_rank(
        self, deck_name: str, batch_size: int = 1000, max_range: int = 1000
    ) -> int:
        """Estimate the highest rank.

        Arguments:
            max_range: Consider something the "highest rank" if there are no
            cards `max_range` above it. E.g. if there is a gap of size 5 and
            `max_range` is 4 or 5, return the last element before the 5-element
            gap. If `max_range` is 6 or higher, find a gap large enough to
            accomodate it.

        Returns:
            Returns the largest existing rank that has no cards for at least
            `max_range` numbers.
        """
        # We can assume that there are no duplicates, meaning that the minimum
        # rank is going to be the total number of notes in the deck.
        #
        # TODO: total_in_deck might be the number of /cards/ in the deck,
        # which complicates things if we ever add more than one card for
        # each note.
        deck_stats = await self.get_deck_stats(deck_name)
        lower_bound = deck_stats["total_in_deck"] - 1

        # Step 2: Query regex patterns in batches.
        batch_size = 1000  # Define batch size for regex ranges.
        largest_rank: int = 0
        start = lower_bound
        end = lower_bound + batch_size

        while True:
            regex_pattern = _generate_regex_patterns(start, end)
            query = f'"Rank:re:{regex_pattern}"'
            note_ids = await self.get_note_ids(deck_name, query=query)
            if largest_rank and not len(note_ids) and end >= largest_rank + max_range:
                break

            async for note in self.get_anki_notes(note_ids):
                try:
                    rank = int(note["fields"]["Rank"]["value"])
                except ValueError as e:
                    warnings.warn(
                        f"Error loading rank field, continuing: {e}", RuntimeWarning
                    )
                    continue
                largest_rank = max(rank, largest_rank)
            start = end
            end = end + batch_size

        return largest_rank


@contextlib.asynccontextmanager
async def create_client(endpoint: str = URL, **kwargs) -> Iterator[AnkiConnectClient]:
    if "connector" not in kwargs:
        kwargs["connector"] = aiohttp.TCPConnector(limit_per_host=5)
    async with aiohttp.ClientSession(**kwargs) as session:
        yield AnkiConnectClient(session, endpoint=endpoint)


def _load_anki_word(row: Sequence[str]) -> types.Word:
    (
        rank,
        word,
        part_of_speech,
        meaning,
        spanish_sentence,
        english_sentence,
        frequency,
        _image,
        article,
        _spanish_audio,
        _note,
        extra_sentences,
        model_used,
        _tags,
    ) = row

    if extra_sentences:
        try:
            extra_sentences = extra_sentences.replace('""', '"')
            extra_sentences = extra_sentences.replace("&quot;", '"')
            extra_sentences = extra_sentences.replace("&nbsp;", " ")
            extra_sentences = tuple(json.loads(extra_sentences))
        except:
            print(f"Failed to load {row}")
            raise
    return types.Word(
        rank=int(rank),
        word=word,
        part_of_speech=part_of_speech,
        meaning=meaning,
        article=None,
        spanish_sentence=spanish_sentence or None,
        english_sentence=english_sentence or None,
        example_sentences=extra_sentences or (),
        model_used=model_used,
    )


def load_anki_words(in_path: pathlib.Path) -> Sequence[types.Word]:
    """Loads from an Anki .txt export"""
    out = []
    with open(in_path, "rt") as f:
        # Find the first entry in the tab-separated file
        while True:
            line_start = f.tell()
            line = f.readline()
            if not line.startswith("#"):
                f.seek(line_start)
                break

        reader = csv.reader(f, dialect="excel-tab")
        for row in reader:
            try:
                out.append(_load_anki_word(row))
            except ValueError:
                print(row)
                raise

        return out


def _generate_regex_patterns(start: int, end: int) -> str:
    """Generate a single regex pattern for matching numbers within a range."""

    def _floor_to(number: int, digit: int) -> int:
        m = 10 ** (digit - 1)
        return m * (number // m)

    def _ceil_to(number: int, digit: int) -> int:
        m = 10 ** (digit - 1)
        return m * int(math.ceil(number / m))

    def helper(left: int, right: int) -> list[tuple[int, int]]:
        if left > right:
            raise ValueError(
                f"End of range must be greater than beginning ({left}, {right})"
            )

        digits = int(math.log10(right))
        new_right = None
        for i in range(1, digits + 1):
            if right % 10**i:
                potential_new_right = _floor_to(right, i + 1)
                if potential_new_right > left:
                    new_right = potential_new_right
                else:
                    digits = i - 1
                break

        # Now we want to start looking at ceilings from the left
        if new_right is None:
            for i in range(digits - 1, 0, -1):
                if (left % 10**i) - (left % 10 ** (i - 1)):
                    potential_new_right = _ceil_to(left, i + 1)
                    if potential_new_right < right:
                        new_right = potential_new_right
                    break

        if new_right is not None:
            return helper(left, new_right) + [(new_right, right)]
        else:
            return [(left, right)]

    ranges = helper(start, end)
    regexes = []
    for left, right in ranges:
        if right - left == 1:
            regexes.append(str(left))
        else:
            left_str = str(left)
            right_str = str(right)
            regex_components = []
            all_zeros = True
            for c1, c2 in zip(reversed(left_str), reversed(right_str)):
                if all_zeros and c2 == "0":
                    if c1 == "0":
                        regex_components.append("[0-9]")
                    elif c2 == "0":
                        regex_components.append(f"[{c1}-9]")
                        all_zeros = False
                elif c1 == c2 or int(c1) == int(c2) - 1:
                    regex_components.append(c1)
                    all_zeros=False
                else:
                    regex_components.append(f"[{c1}-{int(c2)-1}]")
                    all_zeros=False
            regexes.append("".join(reversed(regex_components)))
    return "(" + "|".join(regexes) + ")"
