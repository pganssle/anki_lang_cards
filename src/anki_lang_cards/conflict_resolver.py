"""Module for handling word conflicts."""

import copy
import csv
import io
import itertools
import json
import pathlib
from collections.abc import (
    Collection,
    Iterable,
    Iterator,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import Final, Self, TypeAlias

import attrs

from . import _types as types
from . import headword
from . import part_of_speech as pos

_BucketKeyType: TypeAlias = tuple[str, pos.BasePoS]


def _write_to_csv(lines: Iterable[Sequence[object]]) -> str:
    """Render rows as CSV text, for the cache serialization below."""
    buffer = io.StringIO()
    csv.writer(buffer).writerows(lines)
    return buffer.getvalue().strip()


class WordLookup:
    """Represents a collection of words grouped by their defining characteristics.

    This structure is used to deduplicate word objects. It abstracts away the
    definition of what makes two words the same, but roughly speaking the word
    should be considered "the same" if the "front" of the Anki card would be
    identical (or close enough to identical as to make for a bad memorization
    experience).

    For example:

       dulce (nm), "candy, sweets"
       dulce (adj), "sweet"

    Should be considered different words because the front of the card can be
    "el dulce" for the noun and "dulce" for the adjective (the part of speech
    is shown on the card but not read out, and it's hard to ask people to pay
    close attention to this since it so rarely matters).

    Similarly, you can differentiate some things by their gender, like

        el cura (nm), "priest"
        la cura (nf), "cure"

    But you do not want to have different cards for:

        el piso (nm), "apartment"
        el piso (nm), "floor"

    Since these can only be differentiated in context.

    Words are bucketed on the characteristics that admit no wildcards — the
    canonicalized headword (see `headword.lookup_key`, which collapses spellings
    that the card template renders identically, such as "mirón/ona" and
    "mirón/a") and the base part of speech — and then split into compatibility
    groups *within* a bucket using `PartOfSpeech.__eq__`. This is deliberately
    not a plain hash key: `PartOfSpeech` equality treats an unspecified
    characteristic as a wildcard, so "n" compares equal to both "nm" and "nf"
    while "nm" and "nf" compare unequal. A hash-based key cannot express that
    relation, and using one causes "n" and "nm" to land in different groups
    despite comparing equal.

    Because equality is not transitive, group membership requires compatibility
    with *every* existing member rather than just the first one, and the
    grouping therefore depends on insertion order. Given "nm", "n", "nf" in that
    order the result is [["nm", "n"], ["nf"]]; given "n", "nm", "nf" it is
    [["n", "nm"], ["nf"]]. Either way the incompatible words stay apart, which
    is the property that matters here.

    Iterating over this data structure gives an iterable of groups of words.
    """

    def __init__(self, words: Iterable[types.Word]):
        self._buckets: MutableMapping[
            _BucketKeyType, MutableSequence[MutableSequence[types.Word]]
        ] = {}
        if words:
            for word in words:
                self.add(word)

    def merge(self, other: Self) -> Self:
        """Create a new WordLookup by merging two existing objects."""
        return type(self)(itertools.chain(*self, *other))

    def lookup(self, word: types.Word) -> Sequence[types.Word]:
        """Look up a word in the data structure."""
        group = self._find_group(word)
        return () if group is None else tuple(group)

    def add(self, word: types.Word) -> None:
        """Add a word to the data structure."""
        group = self._find_group(word)
        if group is None:
            self._buckets.setdefault(self._bucket_key(word), []).append([word])
        else:
            group.append(word)

    def __len__(self) -> int:
        return sum(len(groups) for groups in self._buckets.values())

    def __contains__(self, value: types.Word) -> bool:
        return self._find_group(value) is not None

    def __iter__(self) -> Iterator[Sequence[types.Word]]:
        return iter(
            group for groups in self._buckets.values() for group in tuple(groups)
        )

    def _bucket_key(self, word: types.Word) -> _BucketKeyType:
        return (headword.lookup_key(word.word), word.part_of_speech.base_pos)

    def _find_group(self, word: types.Word) -> MutableSequence[types.Word] | None:
        """Find the compatibility group a word belongs to, if there is one."""
        for group in self._buckets.get(self._bucket_key(word), ()):
            if all(member.part_of_speech == word.part_of_speech for member in group):
                return group
        return None


@attrs.frozen
class MergeOperation:
    """Represents a serialiable merge operation (resolving conflicts)."""

    original_words: tuple[types.Word, ...]
    result: types.Word

    # TODO: Use cattrs for this instead
    def to_dict(self) -> dict:
        return {
            "originals": _write_to_csv([w.to_csv_line() for w in self.original_words]),
            "result": _write_to_csv([self.result.to_csv_line()]),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        header_line = _write_to_csv((types.Word.make_csv_header(),))
        originals = types.Word.words_from_csv(
            "\n".join((header_line, data["originals"]))
        )
        result = types.Word.words_from_csv("\n".join((header_line, data["result"])))
        if len(result) != 1:
            raise ValueError(f"Bad merge operation dict: {data}")
        return cls(original_words=originals, result=result[0])


class MergeCache:
    """Manages caching of merge operations."""

    _KeyType: TypeAlias = tuple[types.Word, ...]

    def __init__(self, cache_path: pathlib.Path):
        self.cache_path: Final = cache_path
        self._cache: dict = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Loads the contents of the cache from disk."""
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                data = json.load(f)
            self._cache = {}
            for op in data:
                merge_op = MergeOperation.from_dict(op)
                self._cache[tuple(sorted(merge_op.original_words))] = merge_op

    def save_cache(self) -> None:
        """Save the contents of the cache to disk."""
        json_text = json.dumps([op.to_dict() for op in self._cache.values()])
        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json_text)

    def get_merge(self, words: Collection[types.Word]) -> MergeOperation | None:
        """Get a merge operation matching a collection of words."""
        key = self._key(words)
        if key in self._cache:
            return self._cache[key]
        return None

    def add_merge(self, operation: MergeOperation) -> None:
        """Adds a merge operation to the cache"""
        key = self._key(operation.original_words)
        self._cache[key] = operation

    def __contains__(self, words: Collection[types.Word]) -> bool:
        key = self._key(words)
        return key in self._cache

    def _key(self, words: Collection[types.Word]) -> _KeyType:
        """Calculates the lookup key from the collection of words."""
        return tuple(sorted(words))


def _reconcile_parts_of_speech(
    first: pos.PartOfSpeech, second: pos.PartOfSpeech
) -> pos.PartOfSpeech:
    """Find a part of speech that covers both, keeping as much as it can.

    Dropping straight to the bare base loses the gender, which is on the front
    of the card: merging "nm/f" with "nf" gave a plain "n" and the article went
    with it. A gender that claims both forms already covers a specific one, so
    keep it.
    """
    if isinstance(first, pos.Noun) and isinstance(second, pos.Noun):
        subsuming = {pos.Gender.FORM_DEPENDENT, pos.Gender.COMMON, pos.Gender.AMBIGUOUS}
        for keep, other in ((first, second), (second, first)):
            if keep.gender in subsuming and other.gender != pos.Gender.UNSPECIFIED:
                return keep
    return pos.base_part_of_speech(first.base_pos)


def auto_merge_words(words: Collection[types.Word]) -> types.Word:
    """Automatically merge a sequence of words according to the specified rules."""
    # Split and clean meanings
    meanings: list[list[str]] = []
    is_verb = any(w.part_of_speech.base_pos == pos.BasePoS.VERB for w in words)
    # Only restore the infinitive marker if it was there to begin with. A group
    # can hold a verb phrase beside a preposition — "de no ser por" was labelled
    # both — and prefixing an unrelated gloss produces "to if it weren't for".
    had_infinitive = False
    for word in words:
        word_meanings = [m.strip() for m in word.meaning.split(",")]
        if is_verb:
            had_infinitive = had_infinitive or any(
                m.startswith("to ") for m in word_meanings
            )
            word_meanings = [m[3:] if m.startswith("to ") else m for m in word_meanings]
        meanings.append(word_meanings)

    # Calculate unique meanings maintaining order
    seen: set[str] = set()
    merged_meanings: list[str] = []

    # First, handle words with ranks in order
    ranked_words = [(w, m) for w, m in zip(words, meanings) if w.rank is not None]
    ranked_words.sort(key=lambda x: x[0].rank or float("inf"))

    for _, word_meanings in ranked_words:
        for meaning in word_meanings:
            if meaning not in seen:
                seen.add(meaning)
                merged_meanings.append(meaning)

    # Then handle unranked words in alternating fashion
    unranked_meanings = [m for w, m in zip(words, meanings) if w.rank is None]

    if unranked_meanings:
        max_len = max(len(m) for m in unranked_meanings)
        for i in range(max_len):
            for word_meanings in unranked_meanings:
                if i < len(word_meanings) and word_meanings[i] not in seen:
                    seen.add(word_meanings[i])
                    merged_meanings.append(word_meanings[i])

    # Add "to" back if it's a verb that was glossed as one
    if is_verb and had_infinitive and merged_meanings:
        merged_meanings[0] = "to " + merged_meanings[0]

    # TODO: Move this to its own function, possibly on Word
    # Create merged word
    ranks = [w.rank for w in words if w.rank is not None]
    new_meaning = ", ".join(merged_meanings)
    new_rank = min(ranks) if ranks else None
    base_word = attrs.evolve(
        next(iter(words)), rank=min(ranks) if ranks else None, meaning=new_meaning
    )
    merged_words = (
        attrs.evolve(word, rank=new_rank, meaning=new_meaning) for word in words
    )
    base_word = next(merged_words)
    for word in merged_words:
        if base_word.part_of_speech != word.part_of_speech:
            reconciled = _reconcile_parts_of_speech(
                base_word.part_of_speech, word.part_of_speech
            )
            base_word.part_of_speech = reconciled
            word.part_of_speech = reconciled
        base_word.update(word)
    return base_word


def merge_word_set(words: Collection[types.Word]) -> types.Word:
    """Create one word from a non-conflicting set of words."""
    if len(words) == 0:
        raise ValueError("Cannot merge empty set of words.")
    wi = iter(words)

    if len(words) == 1:
        return next(wi)

    out = copy.copy(next(wi))
    for next_word in wi:
        out.update(next_word)

    return out
