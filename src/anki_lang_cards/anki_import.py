import asyncio
import contextlib
import csv
import html
import io
import itertools
import json
import math
import pathlib
import warnings
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from typing import Any, Final, NewType, TypeAlias, TypedDict

import aiohttp

from . import _types as types
from . import headword

NoteId = NewType("NoteId", int)
DeckId = NewType("DeckId", int)
CardId = NewType("CardId", int)
JsonStr = NewType("JsonStr", str)
URL: Final[str] = "http://localhost:8765"
AC_VERSION: Final[int] = 6


def _related_query(word: types.Word) -> str:
    """One word's search terms, exact plus the prefixes variants share.

    The marker-stripped prefix finds deck spellings that add a valency marker
    in either direction; the slash-notation stem prefix finds the other
    shorthand for the same gendered pair. The stem prefix is only used when
    slash notation is present — prefixing on an arbitrary short headword would
    drag in half the deck.
    """
    terms = [f'"Word:{word.word}"']
    if word.rank:
        terms.append(f'"Rank:{word.rank}"')
    stripped = headword.strip_valency_markers(word.word).strip()
    if stripped:
        terms.append(f'"Word:{stripped}*"')
    if "/" in stripped:
        stem = stripped.partition("/")[0].strip()
        if stem:
            terms.append(f'"Word:{stem}*"')
    return "(" + " or ".join(dict.fromkeys(terms)) + ")"


class Field[T: str](TypedDict):
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
    cards: Sequence[CardId]


class CardData(TypedDict, total=False):
    cardId: CardId

    profile: str
    tags: Sequence[str]
    fields: NoteFields
    modelName: str
    mod: int
    cards: Sequence[CardId]


class DeckStats(TypedDict):
    deck_id: DeckId
    name: str
    new_count: int
    learn_count: int
    review_count: int
    total_in_deck: int


def _word_to_note_data(
    note_id: NoteId, word: types.Word | tuple[types.Word, Mapping[str, str]]
) -> NoteData:
    if not isinstance(word, types.Word):
        word, extra_fields = word
    else:
        extra_fields = {}
    fields = {
        "Word": word.word,
        "Definition": word.meaning,
        # The field is named "Part-of-Speech" in the notetype; AnkiConnect
        # silently ignores unknown field names, so getting this wrong means
        # the part of speech never updates.
        "Part-of-Speech": str(word.part_of_speech),
    }
    if word.rank is not None:
        fields["Rank"] = str(word.rank)

    if word.article is not None:
        fields["Article"] = word.article

    if word.sentences:
        fields["Spanish"] = word.spanish_sentence
        fields["English"] = word.english_sentence
        if word.sentence_source:
            fields["Sentence Source"] = types.json_for_anki_field(
                word.sentence_source.to_dict()
            )
        else:
            fields["Sentence Source"] = ""

    fields = {**fields, **extra_fields}
    if word.example_sentences:
        # TODO: Refactor `WORD` or something to expose just this thing we need
        fields["Extra Sentences"] = word.to_csv_line()[11]

    return {"id": note_id, "fields": fields}


def _reescape_interior_quotes(json_str: str, max_repairs: int = 200) -> str:
    """Re-insert backslashes for interior quotes that lost their escapes.

    Some hand-edited fields had their ``\\"`` escapes stripped to bare
    quotes (``"english": ""Very good," he said."``). Each round parses the
    candidate and, on failure, escapes the closest quote before the error
    position — the quote that prematurely terminated the string — until
    the field parses or the repair budget runs out.
    """
    candidate = json_str
    for _ in range(max_repairs):
        try:
            json.loads(candidate)
        except json.JSONDecodeError as e:
            quote_pos = candidate.rfind('"', 0, e.pos)
            if quote_pos <= 0:
                break
            candidate = candidate[:quote_pos] + "\\" + candidate[quote_pos:]
        else:
            break
    return candidate


def _replace_br(json_str: str) -> str:
    """Replace ``<br>`` with a newline that keeps the JSON parseable.

    Inside a quoted string a raw newline is an invalid control character,
    so the two-character ``\\n`` escape is substituted (parsing to a real
    newline in the content); outside strings ``<br>`` is inter-token
    debris and becomes plain whitespace.

    The in/out-of-string classification tracks quotes with escape
    awareness, so it is only as trustworthy as the input's quoting: it
    must run on text whose quote structure has already been repaired.
    """
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(json_str):
        if json_str.startswith("<br>", i):
            out.append("\\n" if in_string else "\n")
            i += 4
            continue
        char = json_str[i]
        if in_string and char == "\\":
            out.append(json_str[i : i + 2])
            i += 2
            continue
        if char == '"':
            in_string = not in_string
        out.append(char)
        i += 1
    return "".join(out)


def _json_repair_candidates(json_str: str) -> Iterator[str]:
    """Yield progressively more aggressively repaired forms of a JSON field.

    The first candidate is the field verbatim; the second undoes the HTML
    escaping historically applied by Anki imports; the third undoes the
    hand-applied ``\\"`` -> ``\\\\"`` doubling that used to be necessary for
    escapes to survive the card template's JS string parsing; the fourth
    collapses doubled quotes; and the last re-escapes interior quotes whose
    backslashes were stripped entirely. The later rewrites are destructive
    on valid JSON (e.g. the doubled-quote collapse corrupts legitimately
    empty strings), which is why they are only ever applied as a fallback.

    ``<br>`` replacement is applied separately to each candidate rather
    than once up front: _replace_br decides between ``\\n`` (in-string)
    and plain whitespace (between tokens) by tracking quotes, so it has
    to run after the quoting damage a given candidate repairs, not
    before.
    """
    yield json_str
    unescaped = (
        json_str.replace("&quot;", '"').replace("&nbsp;", " ")
        # &amp; must come last so it unwraps double escaping one layer at
        # a time.
        .replace("&amp;", "&")
    )
    yield _replace_br(unescaped)
    dedoubled = unescaped.replace('\\\\"', '\\"')
    yield _replace_br(dedoubled)
    yield _replace_br(dedoubled.replace('""', '"'))
    yield _replace_br(_reescape_interior_quotes(dedoubled))


def parse_json_field_with_repair(json_str: str) -> Any:
    """Parse a JSON-bearing Anki field, repairing HTML-escape mangling.

    Valid JSON is parsed as-is; mangled fields go through repairs of
    increasing aggressiveness.

    :raises json.JSONDecodeError: If no candidate parses (the error comes
        from the most-repaired candidate).
    """
    last_error: json.JSONDecodeError | None = None
    for candidate in _json_repair_candidates(json_str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e
    assert last_error is not None
    raise last_error


def _unescape_entities(text: str) -> str:
    """Unescape HTML entities to a fixed point and normalize nbsp.

    Repeated bad imports can stack an ``&amp;`` layer each time, so a
    single ``html.unescape`` pass is not always enough. The iteration cap
    is purely a safety net; this converges in one or two passes.
    """
    for _ in range(10):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text.replace("\xa0", " ")


def _unescape_json_strings(obj: Any) -> Any:
    """Recursively unescape HTML entities in the string values of ``obj``.

    Entity damage inside JSON string values (``didn&#x27;t``) leaves the
    field parseable, so the structural repairs never see it; the content
    still needs unescaping after parsing. Keys are left alone: they are
    schema (``spanish``, ``english``, ...), not imported content.
    """
    if isinstance(obj, str):
        return _unescape_entities(obj)
    if isinstance(obj, list):
        return [_unescape_json_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _unescape_json_strings(value) for key, value in obj.items()}
    return obj


def canonicalize_text_field(field_value: str) -> str:
    """Undo the HTML-entity escaping imports apply to plain-text fields.

    Importing with "allow HTML" disabled makes Anki HTML-escape the field
    content (``'`` becomes ``&#x27;``, ``"`` becomes ``&quot;``, ...).
    ``<`` is kept escaped: a literal ``<`` in a field could be swallowed
    as a tag when the card template renders.
    """
    text = _unescape_entities(field_value.replace("<br>", " "))
    return text.replace("<", "&lt;").strip()


def canonicalize_sentence_source(field_value: str) -> str:
    """Repair a ``Sentence Source`` field and re-serialize it canonically.

    Repairs both structural damage (escapes broken by imports or hand
    edits) and entity damage inside the string values, which leaves the
    JSON parseable but the content mangled.

    :raises json.JSONDecodeError: If the field cannot be repaired.
    """
    return types.json_for_anki_field(
        _unescape_json_strings(parse_json_field_with_repair(field_value))
    )


def canonicalize_extra_sentences(field_value: str) -> str:
    """Repair an ``Extra Sentences`` field and re-serialize it canonically.

    Beyond the escape and in-string entity repairs, this also unwraps the
    historical damage where a sentence's ``source`` was stored as a JSON
    string rather than an object.

    :raises json.JSONDecodeError: If the field cannot be repaired.
    """
    sentences = parse_json_field_with_repair(field_value)
    fixed = []
    for sentence in sentences:
        source = sentence.get("source")
        if isinstance(source, str) and source:
            sentence = {**sentence, "source": parse_json_field_with_repair(source)}
        fixed.append(sentence)
    return types.json_for_anki_field(_unescape_json_strings(fixed))


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

    def _unwrap_result(self, result: Mapping) -> ...:
        if result["error"]:
            # TODO: Custom error class
            raise ValueError(f"Anki Connect Error: {result['error']}")
        return result["result"]

    async def _post(self, data: Mapping) -> Mapping:
        assert self._retries >= 0
        for _ in range(self._retries + 1):
            try:
                async with self._session.post(
                    self.endpoint, json=data, timeout=self._timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
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
        (note,) = await self.fetch_notes((note_id,))
        return note

    async def fetch_notes(self, note_ids: Sequence[NoteId]) -> Sequence[NoteData]:
        data = {
            "action": "notesInfo",
            "version": AC_VERSION,
            "params": {"notes": list(note_ids)},
        }
        result = await self._post(data=data)
        return result["result"]

    async def are_suspended(self, card_ids: Sequence[CardId]) -> Mapping[CardId, bool]:
        data = {
            "action": "areSuspended",
            "version": AC_VERSION,
            "params": {"cards": card_ids},
        }
        raw_result = await self._post(data=data)
        result = self._unwrap_result(raw_result)

        return dict(zip(card_ids, result, strict=True))

    async def unsuspend_cards(self, card_ids: Sequence[CardId]) -> bool:
        data = {
            "action": "unsuspend",
            "version": AC_VERSION,
            "params": {"cards": [card_ids]},
        }
        result = await self._post(data=data)
        return self._unwrap_result(result)

    async def delete_notes(self, note_ids: Sequence[NoteId]) -> None:
        data = {
            "action": "deleteNotes",
            "version": AC_VERSION,
            "params": {"notes": list(note_ids)},
        }
        result = await self._post(data=data)
        return self._unwrap_result(result)

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
            full_query += f" and ({query})"

        data = {
            "action": "findNotes",
            "version": AC_VERSION,
            "params": {"query": full_query},
        }

        result = await self._post(data=data)
        return tuple(map(NoteId, self._unwrap_result(result)))

    async def notes_mod_times(self, note_ids: Sequence[NoteId]) -> Mapping[NoteId, int]:
        """The last-modified time of each note, in one cheap request.

        This is what makes caching fetched notes safe: any edit — including
        just opening a note in Anki's editor, which normalizes its HTML —
        bumps the note's mod, so an unchanged mod means the cached copy is
        the current copy.
        """
        data = {
            "action": "notesModTime",
            "version": AC_VERSION,
            "params": {"notes": list(note_ids)},
        }
        result = await self._post(data=data)
        return {
            NoteId(entry["noteId"]): int(entry["mod"])
            for entry in self._unwrap_result(result)
        }

    async def get_anki_notes(
        self, note_list: Iterable[NoteId], batch_size: int = 1
    ) -> AsyncIterator[NoteData]:
        """
        Retrieve existing notes by id, yielding them as they arrive.

        Args:
            note_list: The ids of the notes to fetch.
            batch_size: How many notes to request per ``notesInfo`` call.

        Yields:
            A dict per note, each representing an Anki note with its
            fields. Notes are yielded in completion order, not input
            order.
        """

        def _spawn(batch: tuple[NoteId, ...]) -> asyncio.Task[Sequence[NoteData]]:
            return asyncio.create_task(self.fetch_notes(batch))

        batches: dict[asyncio.Task[Sequence[NoteData]], tuple[NoteId, ...]] = {
            _spawn(batch): batch for batch in itertools.batched(note_list, batch_size)
        }
        pending = set(batches)

        errors: dict[tuple[NoteId, ...], Exception] = {}
        retries: dict[tuple[NoteId, ...], int] = {}

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                batch = batches.pop(task)
                try:
                    for note in await task:
                        yield note
                except Exception as e:  # noqa: BLE001
                    if retries.setdefault(batch, 0) < 3:
                        retries[batch] += 1
                        new_task = _spawn(batch)
                        batches[new_task] = batch
                        pending.add(new_task)
                    else:
                        errors[batch] = e

        if errors:
            raise ValueError(f"Failed to fetch notes: {errors}")

    async def update_note_info_from_words(
        self, replacements: Mapping[NoteId, tuple[types.Word, Mapping[str, str]]]
    ) -> None:
        new_note_info = []
        for note_id, word in replacements.items():
            new_note_info.append(_word_to_note_data(note_id, word))

        if len(new_note_info) == 1:
            new_note_info = new_note_info[0]
        await self.update_note_info(new_note_info)

    async def update_note_info(
        self, new_note_info: NoteData | Sequence[NoteData]
    ) -> None:
        if not hasattr(new_note_info, "keys"):
            tasks = [
                asyncio.create_task(self.update_note_info(note_info))
                for note_info in new_note_info
            ]
            await asyncio.gather(*tasks)
            return None

        data = {
            "action": "updateNoteFields",
            "version": AC_VERSION,
            "params": {"note": new_note_info},
        }

        result = await self._post(data)
        if result["error"] is not None:
            raise ValueError(result["error"])
        return result

    async def retrieve_matching_words(
        self,
        deck_name: str,
        words: Sequence[types.Word],
        *,
        cache: "NoteCache | None" = None,
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
        return await fetch_note_words(
            self, ids, cache=cache, description="Fetching matching notes"
        )

    async def retrieve_related_words(
        self,
        deck_name: str,
        words: Sequence[types.Word],
        *,
        cache: "NoteCache | None" = None,
    ) -> Mapping[NoteId, types.Word]:
        """Fetch notes matching these words or any variant spelling of them.

        `retrieve_matching_words` asks for exact spellings, and duplicate
        detection matches looser than that: "seguir el rollo" and "seguir el
        rollo (a alguien)" are one lookup key, as are "mirón/a" and
        "mirón/ona". A deck note under a variant spelling has to be fetched
        before it can be flagged, so each word is also searched as a prefix of
        its marker-stripped form — which finds spellings that add a marker —
        and, for slash notation, of its stem, which finds the other way of
        writing the same pair. The extra notes a prefix drags in are harmless:
        they share no lookup key with anything, so nothing pairs with them.
        """
        query = " or ".join(_related_query(word) for word in words)
        if not query:
            return {}
        ids = await self.get_note_ids(deck_name, query=query)
        return await fetch_note_words(
            self, ids, cache=cache, description="Fetching related notes"
        )

    def word_from_anki_connect_response(self, resp: NoteData) -> types.Word:
        """Creates a `Word` object from an AnkiConnect response representing a Note."""
        fields: NoteFields = resp.get("fields", {})

        def get_field_value(field_name: str) -> str | None:
            """Helper function to safely extract the value of a field."""
            field: Field = fields.get(field_name, None)
            return field["value"] if field else None

        def parse_json_field(field_name: str) -> Sequence[Any] | None:
            """Helper function to parse JSON fields into a Python object."""
            field_value = get_field_value(field_name)
            if field_value:
                try:
                    return parse_json_field_with_repair(field_value)
                except json.JSONDecodeError:
                    return None
            return None

        sentence_source = get_field_value("Sentence Source")
        if sentence_source:
            sentence_citation: types.Citation | None = types.Citation.from_dict(
                parse_json_field_with_repair(sentence_source)
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
        spanish_sentence = get_field_value("Spanish")
        english_sentence = get_field_value("English")
        example_sentences = parse_sentences()

        return types.Word(
            rank=int(rank_str) if rank_str else None,
            word=get_field_value("Word") or "",
            meaning=get_field_value("Definition") or "",
            part_of_speech=get_field_value("Part-of-Speech") or "",
            article=get_field_value("Article"),
            sentences=load_sentences(
                spanish_sentence, english_sentence, sentence_citation, example_sentences
            ),
            model_used=get_field_value("model_used"),
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
            regex_pattern = _generate_regex_pattern(start, end)
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


DEFAULT_NOTE_CACHE: Final[pathlib.Path] = pathlib.Path(".cache/anki_notes.json")
_FETCH_BATCH_SIZE: Final[int] = 50
"""Notes per notesInfo request. One request per note is what made fetching
slow; batches keep the request count proportional to the deck, not the note
count."""


def _word_to_row(word: types.Word) -> str:
    buffer = io.StringIO()
    csv.writer(buffer).writerow(word.to_csv_line())
    return buffer.getvalue().strip()


def _word_from_row(row: str) -> types.Word:
    buffer = io.StringIO()
    csv.writer(buffer).writerow(types.Word.make_csv_header())
    (word,) = types.Word.words_from_csv("\n".join((buffer.getvalue().strip(), row)))
    return word


class NoteCache:
    """Words fetched from Anki, valid while each note's mod is unchanged.

    Fetching a deck's notes is by far the slowest thing the tools do, and
    most notes have not changed since the last run. Each entry records the
    note's mod at fetch time; `notes_mod_times` asks the live deck for all
    current mods in one request, and only notes whose mod moved — or that
    were never seen — are fetched for real. Entries for deleted notes are
    never requested again and simply linger.
    """

    def __init__(self, cache_path: pathlib.Path):
        self.cache_path: Final = cache_path
        self._entries: dict[int, tuple[int, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for note_id, entry in data.items():
            self._entries[int(note_id)] = (int(entry["mod"]), entry["row"])

    def save(self) -> None:
        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(note_id): {"mod": mod, "row": row}
            for note_id, (mod, row) in sorted(self._entries.items())
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False))

    def get(self, note_id: NoteId, mod: int | None) -> types.Word | None:
        """The cached word, but only while the note still has this mod."""
        if mod is None:
            return None
        entry = self._entries.get(int(note_id))
        if entry is None or entry[0] != mod:
            return None
        try:
            return _word_from_row(entry[1])
        except (ValueError, KeyError):
            return None

    def put(self, note_id: NoteId, mod: int, word: types.Word) -> None:
        self._entries[int(note_id)] = (mod, _word_to_row(word))

    def __len__(self) -> int:
        return len(self._entries)


async def fetch_note_words(
    client: AnkiConnectClient,
    note_ids: Sequence[NoteId],
    *,
    cache: NoteCache | None = None,
    description: str = "Fetching notes",
) -> Mapping[NoteId, types.Word]:
    """Fetch notes as Words, reusing the cache and showing progress.

    With a cache, one `notesModTime` request decides which notes actually
    need fetching; the rest are read back from disk and said so. The fetch
    itself is batched and reported on a progress bar, since a deck-sized
    fetch is the sort of wait that looks like a hang without one.
    """
    import tqdm

    note_ids = list(note_ids)
    output: dict[NoteId, types.Word] = {}
    to_fetch: Sequence[NoteId] = note_ids
    mods: Mapping[NoteId, int] = {}

    if cache is not None and note_ids:
        mods = await client.notes_mod_times(note_ids)
        misses: list[NoteId] = []
        for note_id in note_ids:
            word = cache.get(note_id, mods.get(note_id))
            if word is None:
                misses.append(note_id)
            else:
                output[note_id] = word
        to_fetch = misses
        if output:
            print(
                f"  {len(output)} of {len(note_ids)} notes unchanged since "
                "the last fetch, read from cache"
            )

    if to_fetch:
        with tqdm.tqdm(total=len(to_fetch), desc=description, unit="note") as bar:
            async for note in client.get_anki_notes(
                to_fetch, batch_size=_FETCH_BATCH_SIZE
            ):
                note_id = NoteId(note["noteId"])
                word = client.word_from_anki_connect_response(note)
                output[note_id] = word
                if cache is not None and note_id in mods:
                    cache.put(note_id, mods[note_id], word)
                bar.update(1)
        if cache is not None:
            cache.save()

    return output


def load_sentences(
    spanish_sentence: str,
    english_sentence: str,
    sentence_source: types.Citation | str | None,
    example_sentences: Mapping[str, str | Mapping],
) -> Sequence[types.Sentence]:
    if isinstance(sentence_source, str):
        if sentence_source:
            sentence_source = types.Citation.from_json(sentence_source)
        else:
            sentence_source = None

    # A note with no first sentence must not grow a Sentence full of Nones —
    # comparing or merging one of those crashes far from here.
    first: tuple[types.Sentence, ...] = ()
    if spanish_sentence is not None:
        first = (
            types.Sentence(
                spanish=spanish_sentence,
                english=english_sentence,
                source=sentence_source,
            ),
        )
    return (*first, *map(types.Sentence.from_dict, example_sentences))


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
        sentence_citation,
        _note,
        extra_sentences,
        model_used,
        _tags,
    ) = row

    if extra_sentences:
        try:
            extra_sentences = tuple(parse_json_field_with_repair(extra_sentences))
        except:
            print(f"Failed to load {row}")
            raise

    return types.Word(
        rank=int(rank),
        word=word,
        part_of_speech=part_of_speech,
        meaning=meaning,
        article=None,
        sentences=load_sentences(
            spanish_sentence or None,
            english_sentence or None,
            sentence_citation or None,
            extra_sentences or (),
        ),
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


def _generate_regex_pattern(start: int, end: int) -> str:
    """Generate a single regex pattern for matching numbers within a range."""
    ranges = _generate_regex_ranges(start, end)
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
                        if c1 == "9":
                            regex_components.append("9")
                        else:
                            regex_components.append(f"[{c1}-9]")
                        all_zeros = False
                elif c1 == c2 or int(c1) == (int(c2) - 1) % 10:
                    regex_components.append(c1)
                    all_zeros = False
                else:
                    regex_components.append(f"[{c1}-{int(c2)-1}]")
                    all_zeros = False
            regexes.append("".join(reversed(regex_components)))
    if len(regexes) > 1:
        return "(" + "|".join(regexes) + ")"
    elif len(regexes) == 1:
        return regexes[0]
    else:
        return ""


def _generate_regex_ranges(start: int, end: int) -> list[tuple[int, int]]:
    def _get_lsd(n: int) -> int:
        """Retrieves the position of the least significant digit."""
        assert n != 0
        str_n = str(n)
        for i, c in enumerate(reversed(str_n)):  # noqa: RET503
            if c != "0":
                return i
        raise AssertionError("A non-zero integer must contain a non-zero digit")

    def _floor_to(number: int, digit: int) -> int:
        m = 10 ** (digit - 1)
        return m * (number // m)

    def _ceil_to(number: int, digit: int) -> int:
        m = 10 ** (digit - 1)
        return m * int(math.ceil(number / m))

    # We need to take our `start` and `end` and break them into three
    # sections: 1, round up to the next least significant digit on the
    # left, round down to the next least significant digit to the right,
    # then repeat for the range in the middle. Rounding up or down by exactly
    # one always lets us generate a regex.
    range_stack = []
    left, right = start, end
    left_done = False
    right_done = False
    while not (right_done and left_done):
        if not left_done:
            l_lsd = _get_lsd(left)
            potential_new_left = _ceil_to(left, l_lsd + 2)
            if potential_new_left > right or left == potential_new_left:
                left_done = True
            else:
                range_stack.append((left, potential_new_left))
                left = potential_new_left

        if not right_done:
            r_lsd = _get_lsd(right)
            potential_new_right = _floor_to(right, r_lsd + 2)
            if potential_new_right < left or right == potential_new_right:
                right_done = True
            else:
                range_stack.append((potential_new_right, right))
                right = potential_new_right

        if left_done and right_done and left != right:
            range_stack.append((left, right))

    return range_stack
