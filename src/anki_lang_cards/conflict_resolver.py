""" Module for handling word conflicts."""

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
    Set,
)
from typing import Any, Final, Self, TypeAlias

import attrs

from . import _types as types
from . import part_of_speech as pos

_WordKeyType: TypeAlias = tuple[str, pos.BasePoS]


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

    Iterating over this data structure gives an iterable of groups of words.
    """

    def __init__(self, words: Iterable[types.Word]):
        self._words: MutableMapping[_WordKeyType, MutableSequence[types.Word]] = {}
        if words:
            for word in words:
                self.add(word)

    def merge(self, other: Self) -> Self:
        """Create a new WordLookup by merging two existing objects."""
        return type(self)(itertools.chain(*self, *other))

    def lookup(self, word: types.Word) -> Sequence[types.Word]:
        """Look up a word in the data structure."""
        return self._words.get(self._key(word), ())

    def add(self, word: types.Word) -> None:
        """Add a word to the data structure."""
        k = self._key(word)
        if k not in self._words:
            self._words[k] = [word]
        else:
            self._words[k].append(word)

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, value: types.Word) -> bool:
        return self._key(value) in self._words

    def __iter__(self) -> Iterator[Sequence[types.Word]]:
        return iter(self._words.values())

    def _key(self, word: types.Word) -> _WordKeyType:
        return (word.word, word.part_of_speech.base_pos)


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


def find_existing_words(
    new_words: Iterable[types.Word],
    existing_words: WordLookup,
) -> Iterable[tuple[types.Word, Sequence[types.Word]]]:
    """Finds words that already exist in the word lookup object.

    Yields:
        A tuple containing the original word and the group of existing words.
    """
    for word in new_words:
        if found := existing_words.lookup(word):
            yield (word, found)


def find_conflicting_word_sets(
    new_words: Iterable[types.Word], existing_words: WordLookup
) -> Iterable[Set[types.Word]]:
    """Find sets of words with the same term and part of speech that are unequal.

    This mostly comes down to the same words with slightly different definitions,
    words with different ranks, and words with the same base part of speech but
    different characteristics (e.g. "vtr" vs "v").

    Arguments:
        new_words: New words that may be incompatible with the contents of the
            existing words.

        existing_words: An indexed lookup object containing sets of words grouped
            by the same characteristics that are used to consider two words
            "the same" for purposes of finding conflicts.

    Yields:
        Immutable sets containing the groups of incompatible word objects

    """
    for word, matching_words in find_existing_words(new_words, existing_words):
        if any(mw != word for mw in matching_words):
            yield frozenset({word, *matching_words})


def find_merged_outputs(
    conflict_sets: Iterable[Set[types.Word]],
    output_words: WordLookup,
    merge_cache: MergeCache,
) -> Iterable[MergeOperation | Set[types.Word]]:
    """Find conflicts that have already been resolved and output.

    This is for the situation where earlier runs of the program have involved
    resolving a conflict in word defintiions and the value has been sent to
    the output. This would normally invalidate the merge operation cache,
    because now the merged output is an additional conflicting word to merge
    with the other words, so we want to check if the output in `output_words`
    already exists in the merge cache.

    Arguments:
        conflict_sets: An output from `find_conflicting_word_sets` which has
            been run on the inputs to be resolved *excluding* `output_words`.
        output_words: A WordLookup calculated from the output of an earlier
            run.
        merge_cache: The cached results of previous merges.

    Yields:
        Yields the merged words from the word cache.
    """
    for word_set in conflict_sets:
        if (
            merge_operation := merge_cache.get_merge(word_set)
        ) is not None and merge_operation.result in output_words:
            ow_set = output_words.lookup(merge_operation.result)
            if len(ow_set) == 1 and next(iter(ow_set)) == merge_operation.result:
                yield merge_operation
                continue
        yield word_set


def filter_merged_outputs(
    conflict_sets: Iterable[Set[types.Word]],
    output_words: WordLookup,
    merge_cache: MergeCache,
) -> Iterable[Set[types.Word]]:
    """Wrapper around `find_merged_outputs` that removes already merged outputs."""
    for set_or_merge_op in find_merged_outputs(
        conflict_sets, output_words, merge_cache
    ):
        if not isinstance(set_or_merge_op, MergeOperation):
            yield set_or_merge_op


def print_conflicts(
    to_merge: Iterable[Set[types.Word]],
    max_words: int | None = None,
    *,
    print_func=print,
) -> None:
    """Print out the set of conflicting words in a way that is somewhat readable."""
    if max_words is None:
        limit = itertools.repeat(None)
    else:
        limit = itertools.repeat(None, max_words)

    def render_word(word: types.Word) -> str:
        if word.part_of_speech.base_pos == pos.BasePoS.NOUN and word.article:
            output = [f"{word.article} {word.word}"]
        else:
            output = [f"{word.word}"]

        output.append(f"({word.part_of_speech})")
        output.append(f"{word.meaning} ({len(word.sentences)}s)")
        return " ".join(output)

    for word_set, _ in zip(to_merge, limit):
        if len(word_set) == 1:
            continue
        if len(word_set) == 2:
            w1, w2 = word_set
            print_func(f"  {render_word(w1)} !=")
            print_func(f"    {render_word(w2)}")
        else:
            wi = iter(word_set)
            w = next(wi)

            print_func(f"  {render_word(w)} !=")
            for w in wi:
                print_func(f"    {render_word(w)}")


def _write_to_csv(lines: Sequence[Sequence[Any]]) -> str:
    f = io.StringIO()
    writer = csv.writer(f)
    writer.writerows(lines)
    f.seek(0)
    return f.read()
