"""Tests for the mod-time note cache and the batched, cached fetch."""

import asyncio
from collections.abc import Mapping

from anki_lang_cards import _types as types
from anki_lang_cards import anki_import


class TestNoteCache:
    """Fetched notes are reused while their mod times stand still."""

    @staticmethod
    def make_word(rank: int, word: str, meaning: str) -> types.Word:
        return types.Word(
            rank=rank,
            word=word,
            meaning=meaning,
            part_of_speech="nm",
            article=None,
            sentences=(),
        )

    def test_a_cached_note_survives_a_reload(self, tmp_path) -> None:
        path = tmp_path / "notes.json"
        cache = anki_import.NoteCache(path)
        word = self.make_word(100, "zanja", "trench")
        cache.put(anki_import.NoteId(1), 555, word)
        cache.save()

        reloaded = anki_import.NoteCache(path)
        assert reloaded.get(anki_import.NoteId(1), 555) == word

    def test_a_changed_mod_invalidates_the_entry(self, tmp_path) -> None:
        cache = anki_import.NoteCache(tmp_path / "notes.json")
        cache.put(anki_import.NoteId(1), 555, self.make_word(100, "zanja", "trench"))

        assert cache.get(anki_import.NoteId(1), 556) is None
        assert cache.get(anki_import.NoteId(1), None) is None

    def test_a_corrupt_cache_file_is_treated_as_empty(self, tmp_path) -> None:
        path = tmp_path / "notes.json"
        path.write_text("not json{")
        assert len(anki_import.NoteCache(path)) == 0


class FakeFetchClient:
    """Just enough client for fetch_note_words: mods and batched fetches."""

    def __init__(self, words: Mapping[int, types.Word], mods: Mapping[int, int]):
        self.words = dict(words)
        self.mods = dict(mods)
        self.mod_requests = 0
        self.fetched: list[int] = []
        self.batch_sizes: list[int] = []

    async def notes_mod_times(self, note_ids):
        self.mod_requests += 1
        return {nid: self.mods[nid] for nid in note_ids}

    async def get_anki_notes(self, note_ids, batch_size=1):
        self.batch_sizes.append(batch_size)
        for nid in note_ids:
            self.fetched.append(nid)
            yield {"noteId": nid}

    def word_from_anki_connect_response(self, note):
        return self.words[note["noteId"]]


class TestFetchNoteWords:
    @staticmethod
    def client() -> FakeFetchClient:
        words = {
            nid: TestNoteCache.make_word(nid, f"palabra{nid}", f"meaning {nid}")
            for nid in (1, 2, 3)
        }
        return FakeFetchClient(words, {1: 10, 2: 20, 3: 30})

    def test_a_second_run_fetches_nothing(self, tmp_path) -> None:
        client = self.client()
        cache = anki_import.NoteCache(tmp_path / "notes.json")
        ids = [anki_import.NoteId(n) for n in (1, 2, 3)]

        first = asyncio.run(anki_import.fetch_note_words(client, ids, cache=cache))
        assert len(first) == 3
        assert sorted(client.fetched) == [1, 2, 3]

        again = asyncio.run(
            anki_import.fetch_note_words(
                client, ids, cache=anki_import.NoteCache(cache.cache_path)
            )
        )
        assert again == first
        assert sorted(client.fetched) == [1, 2, 3], "nothing was refetched"

    def test_only_the_changed_note_is_refetched(self, tmp_path) -> None:
        client = self.client()
        cache = anki_import.NoteCache(tmp_path / "notes.json")
        ids = [anki_import.NoteId(n) for n in (1, 2, 3)]
        asyncio.run(anki_import.fetch_note_words(client, ids, cache=cache))
        client.fetched.clear()

        client.mods[2] = 21
        client.words[2] = TestNoteCache.make_word(2, "palabra2", "edited meaning")
        result = asyncio.run(
            anki_import.fetch_note_words(
                client, ids, cache=anki_import.NoteCache(cache.cache_path)
            )
        )

        assert client.fetched == [2]
        assert result[anki_import.NoteId(2)].meaning == "edited meaning"

    def test_without_a_cache_everything_is_fetched_in_batches(self) -> None:
        client = self.client()
        ids = [anki_import.NoteId(n) for n in (1, 2, 3)]

        result = asyncio.run(anki_import.fetch_note_words(client, ids, cache=None))

        assert len(result) == 3
        assert client.mod_requests == 0
        assert all(size > 1 for size in client.batch_sizes)
